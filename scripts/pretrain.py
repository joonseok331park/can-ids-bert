"""Pre-train the public CAN-BERT teacher prototype."""

from __future__ import annotations

import argparse
import math
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import BertConfig, get_linear_schedule_with_warmup

from core.dataset import MLMDataset
from core.tokenizer import CANSequencer, CANTokenizer
from models.teacher import CANBertForMaskedLM
from utils.data_loader import load_can_data


CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_TYPE = "can-bert-pretrain"


@dataclass(frozen=True)
class TrainEpochResult:
    average_raw_loss: float
    microbatches: int
    optimizer_steps: int
    skipped_optimizer_steps: int
    global_optimizer_step: int


@dataclass(frozen=True)
class ResumeState:
    start_epoch: int
    global_optimizer_step: int
    mode: str


class IncompatibleCheckpointError(RuntimeError):
    """Raised when a versioned checkpoint cannot continue the current run."""


def _seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--vocab_path", required=True)
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--dataset_type", default="candump")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume_from_checkpoint",
        help=(
            "Continue optimizer, scheduler, scaler, epoch, and global-step state. "
            "Legacy model-only checkpoints are loaded as an explicit warm start."
        ),
    )
    checkpoint_group.add_argument(
        "--warm_start_from_checkpoint",
        help="Load model weights only and start a new training schedule",
    )
    parser.add_argument("--seq_len", type=int, default=126)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Total number of epochs, including completed resumed epochs",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--mask_prob", type=float, default=0.45)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--intermediate", type=int, default=512)
    parser.add_argument(
        "--num_workers", type=int, default=max((os.cpu_count() or 1) // 2, 0)
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient_accumulation_steps must be at least 1")
    return args


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _setup_ddp() -> Tuple[int, int, torch.device]:
    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    return rank, local_rank, device


def _build_loader(
    args: argparse.Namespace,
    tokenizer: CANTokenizer,
    rank: int = 0,
    world_size: int = 1,
) -> Tuple[DataLoader, int]:
    frame = load_can_data(args.data_path, dataset_type=args.dataset_type)
    sequences = CANSequencer(tokenizer, seq_len=args.seq_len, stride=1).transform(
        frame
    )
    if not sequences:
        raise RuntimeError("Sequencer produced no sequences")

    dataset = MLMDataset(sequences, tokenizer, mask_prob=args.mask_prob)
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError(
            "Training dataloader is empty; reduce batch size or provide more data"
        )
    if rank == 0:
        print(
            f"[INFO] loaded {len(sequences)} sequences in {len(loader)} microbatches"
        )
    return loader, len(loader)


def _build_model(
    tokenizer: CANTokenizer, args: argparse.Namespace
) -> CANBertForMaskedLM:
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate,
        max_position_embeddings=args.seq_len,
    )
    return CANBertForMaskedLM(config)


def _optimizer_updates_per_epoch(
    microbatches_per_epoch: int, gradient_accumulation_steps: int
) -> int:
    if microbatches_per_epoch < 1:
        raise ValueError("microbatches_per_epoch must be at least 1")
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    return math.ceil(microbatches_per_epoch / gradient_accumulation_steps)


def _build_optim_sched(
    model: torch.nn.Module,
    args: argparse.Namespace,
    microbatches_per_epoch: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    updates_per_epoch = _optimizer_updates_per_epoch(
        microbatches_per_epoch, args.gradient_accumulation_steps
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=updates_per_epoch * args.epochs,
    )
    return optimizer, scheduler


def _build_grad_scaler(device: torch.device) -> Any:
    enabled = device.type == "cuda"
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model behind DDP and compile wrappers."""
    while True:
        if isinstance(model, DDP):
            model = model.module
            continue
        original = getattr(model, "_orig_mod", None)
        if isinstance(original, torch.nn.Module):
            model = original
            continue
        return model


def _gradient_sync_context(model: torch.nn.Module, should_update: bool):
    no_sync = getattr(model, "no_sync", None)
    if not should_update and callable(no_sync):
        return no_sync()
    return nullcontext()


def _optimizer_step_was_applied(scaler: Any, scale_before: float) -> bool:
    """Use GradScaler's scale transition to detect an overflow-skipped step."""
    return float(scaler.get_scale()) >= scale_before


def _train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LambdaLR,
    scaler: Any,
    device: torch.device,
    epoch: int,
    end_epoch: int,
    rank: int,
    gradient_accumulation_steps: int = 1,
    global_optimizer_step: int = 0,
) -> TrainEpochResult:
    """Train one epoch and step the scheduler only after applied updates."""
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    microbatch_count = len(loader)
    if microbatch_count == 0:
        raise ValueError("Training dataloader is empty")

    model.train()
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    progress = (
        tqdm(loader, desc=f"Epoch {epoch + 1}/{end_epoch}", leave=False)
        if rank == 0
        else loader
    )

    raw_loss_total = 0.0
    optimizer_steps = 0
    skipped_optimizer_steps = 0
    optim.zero_grad(set_to_none=True)

    for step, batch in enumerate(progress):
        group_start = (step // gradient_accumulation_steps) * gradient_accumulation_steps
        group_size = min(
            gradient_accumulation_steps, microbatch_count - group_start
        )
        should_update = (
            (step + 1) % gradient_accumulation_steps == 0
            or step + 1 == microbatch_count
        )
        batch = {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in batch.items()
        }

        with _gradient_sync_context(model, should_update):
            with _autocast_context(device):
                raw_loss = model(**batch)[0]
                scaled_loss = raw_loss / group_size
            scaler.scale(scaled_loss).backward()

        if should_update:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = float(scaler.get_scale())
            scaler.step(optim)
            scaler.update()
            if _optimizer_step_was_applied(scaler, scale_before):
                sched.step()
                optimizer_steps += 1
                global_optimizer_step += 1
            else:
                skipped_optimizer_steps += 1
            optim.zero_grad(set_to_none=True)

        raw_loss_value = float(raw_loss.detach().item())
        raw_loss_total += raw_loss_value
        if rank == 0:
            if step % 100 == 0:
                wandb.log(
                    {
                        "train/raw_loss": raw_loss_value,
                        "train/scaled_loss": float(scaled_loss.detach().item()),
                        "train/lr": sched.get_last_lr()[0],
                        "train/epoch": epoch,
                        "train/microbatch": step,
                        "train/global_optimizer_step": global_optimizer_step,
                    }
                )
            if isinstance(progress, tqdm):
                progress.set_postfix(loss=f"{raw_loss_value:.4f}")

    result = TrainEpochResult(
        average_raw_loss=raw_loss_total / microbatch_count,
        microbatches=microbatch_count,
        optimizer_steps=optimizer_steps,
        skipped_optimizer_steps=skipped_optimizer_steps,
        global_optimizer_step=global_optimizer_step,
    )
    if rank == 0:
        print(
            f"[EPOCH {epoch + 1}] raw_loss={result.average_raw_loss:.4f}, "
            f"optimizer_steps={result.optimizer_steps}, "
            f"overflow_skips={result.skipped_optimizer_steps}"
        )
    return result


def _model_config(model: torch.nn.Module) -> dict[str, Any]:
    config = getattr(_unwrap_model(model), "config", None)
    if config is None or not hasattr(config, "to_dict"):
        return {}
    return config.to_dict()


def _save_checkpoint(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LambdaLR,
    scaler: Any,
    epoch: int,
    global_optimizer_step: int,
    out_dir: Path,
    rank: int,
    training_config: Mapping[str, Any],
) -> Path | None:
    """Save a versioned, fully resumable checkpoint on rank zero."""
    if rank != 0:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"can-bert-pretrained-epoch-{epoch + 1}.pt"
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "checkpoint_type": CHECKPOINT_TYPE,
            "epoch": epoch + 1,
            "global_optimizer_step": global_optimizer_step,
            "model_state_dict": _unwrap_model(model).state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": _model_config(model),
            "training_config": dict(training_config),
        },
        checkpoint_path,
    )
    print(f"[INFO] checkpoint saved -> {checkpoint_path}")
    return checkpoint_path


def _legacy_model_state(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise IncompatibleCheckpointError("Checkpoint payload is not a mapping")
    for key in ("model_state_dict", "model", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise IncompatibleCheckpointError("Checkpoint does not contain model weights")


def _move_value_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {
            key: _move_value_to_device(nested, device)
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_move_value_to_device(nested, device) for nested in value]
    if isinstance(value, tuple):
        return tuple(_move_value_to_device(nested, device) for nested in value)
    return value


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for parameter, state in optimizer.state.items():
        optimizer.state[parameter] = _move_value_to_device(state, device)


def _validate_training_config(
    stored: Mapping[str, Any], expected: Mapping[str, Any] | None
) -> None:
    if expected is None:
        return
    mismatches = [
        key
        for key, expected_value in expected.items()
        if stored.get(key) != expected_value
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={stored.get(key)!r}, current={expected[key]!r}"
            for key in mismatches
        )
        raise IncompatibleCheckpointError(
            "Training schedule is incompatible with the checkpoint: " + details
        )


def _validate_model_config(
    stored: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    architecture_keys = (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "max_position_embeddings",
    )
    mismatches = [
        key for key in architecture_keys if stored.get(key) != current.get(key)
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={stored.get(key)!r}, current={current.get(key)!r}"
            for key in mismatches
        )
        raise IncompatibleCheckpointError(
            "Model architecture is incompatible with the checkpoint: " + details
        )


def _load_checkpoint(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LambdaLR,
    scaler: Any,
    ckpt_path: str | Path,
    device: torch.device,
    rank: int,
    expected_training_config: Mapping[str, Any] | None = None,
    force_warm_start: bool = False,
) -> ResumeState:
    """Load true-resume state or explicitly fall back to legacy warm start."""
    checkpoint_path = Path(ckpt_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    schema_version = (
        checkpoint.get("schema_version") if isinstance(checkpoint, Mapping) else None
    )
    if force_warm_start or schema_version is None:
        state = _legacy_model_state(checkpoint)
        try:
            _unwrap_model(model).load_state_dict(state, strict=True)
        except RuntimeError as error:
            raise IncompatibleCheckpointError(
                f"Warm-start model weights are incompatible: {error}"
            ) from error
        mode = "forced-warm-start" if force_warm_start else "legacy-warm-start"
        if rank == 0:
            print(
                f"[WARM START] loaded model weights from {checkpoint_path}; "
                "optimizer, scheduler, scaler, epoch, and global step were reset"
            )
        return ResumeState(0, 0, mode)

    if schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise IncompatibleCheckpointError(
            f"Unsupported checkpoint schema {schema_version!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    required = {
        "checkpoint_type",
        "epoch",
        "global_optimizer_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "model_config",
        "training_config",
    }
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise IncompatibleCheckpointError(
            "Versioned checkpoint is incomplete: " + ", ".join(missing)
        )
    if checkpoint["checkpoint_type"] != CHECKPOINT_TYPE:
        raise IncompatibleCheckpointError(
            f"Unexpected checkpoint type: {checkpoint['checkpoint_type']!r}"
        )
    _validate_training_config(
        checkpoint["training_config"], expected_training_config
    )
    _validate_model_config(checkpoint["model_config"], _model_config(model))

    try:
        _unwrap_model(model).load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        optim.load_state_dict(checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optim, device)
        sched.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    except (KeyError, RuntimeError, ValueError) as error:
        raise IncompatibleCheckpointError(
            f"Versioned checkpoint state is incompatible: {error}"
        ) from error

    start_epoch = int(checkpoint["epoch"])
    global_step = int(checkpoint["global_optimizer_step"])
    if start_epoch < 0 or global_step < 0:
        raise IncompatibleCheckpointError(
            "Checkpoint epoch and global optimizer step must be non-negative"
        )
    if rank == 0:
        print(
            f"[TRUE RESUME] epoch={start_epoch}, "
            f"global_optimizer_step={global_step}, source={checkpoint_path}"
        )
    return ResumeState(start_epoch, global_step, "true-resume")


def _training_config(
    args: argparse.Namespace, microbatches_per_epoch: int
) -> dict[str, int | float]:
    updates_per_epoch = _optimizer_updates_per_epoch(
        microbatches_per_epoch, args.gradient_accumulation_steps
    )
    return {
        "microbatches_per_epoch": microbatches_per_epoch,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "updates_per_epoch": updates_per_epoch,
        "total_epochs": args.epochs,
        "total_optimizer_steps": updates_per_epoch * args.epochs,
        "warmup_steps": args.warmup_steps,
        "learning_rate": args.learning_rate,
    }


def main() -> None:
    args = _parse_args()
    is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_ddp:
        rank, local_rank, device = _setup_ddp()
        world_size = int(os.environ["WORLD_SIZE"])
        _seed_everything(args.seed + rank)
    else:
        rank, local_rank, world_size = 0, -1, 1
        device = _device()
        _seed_everything(args.seed)

    if rank == 0:
        wandb.init(
            project="CAN-IDS-DDP-Pretrain",
            config=vars(args),
            mode="offline",
            name=f"ddp-{world_size}gpu" if is_ddp else "single-process",
        )

    tokenizer = CANTokenizer()
    tokenizer.load_vocab(args.vocab_path)
    loader, microbatches_per_epoch = _build_loader(
        args, tokenizer, rank, world_size
    )
    model = _build_model(tokenizer, args).to(device)
    if hasattr(torch, "compile") and torch.__version__.startswith("2."):
        model = torch.compile(model)
        if rank == 0:
            print("[INFO] model compiled")

    optim, sched = _build_optim_sched(model, args, microbatches_per_epoch)
    scaler = _build_grad_scaler(device)
    training_config = _training_config(args, microbatches_per_epoch)

    resume_state = ResumeState(0, 0, "new-run")
    if args.resume_from_checkpoint:
        resume_state = _load_checkpoint(
            model,
            optim,
            sched,
            scaler,
            args.resume_from_checkpoint,
            device,
            rank,
            expected_training_config=training_config,
        )
    elif args.warm_start_from_checkpoint:
        resume_state = _load_checkpoint(
            model,
            optim,
            sched,
            scaler,
            args.warm_start_from_checkpoint,
            device,
            rank,
            force_warm_start=True,
        )

    if resume_state.start_epoch > args.epochs:
        raise ValueError(
            f"Checkpoint epoch {resume_state.start_epoch} exceeds --epochs {args.epochs}"
        )

    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    global_optimizer_step = resume_state.global_optimizer_step
    for epoch in range(resume_state.start_epoch, args.epochs):
        result = _train_epoch(
            model,
            loader,
            optim,
            sched,
            scaler,
            device,
            epoch,
            args.epochs,
            rank,
            args.gradient_accumulation_steps,
            global_optimizer_step,
        )
        global_optimizer_step = result.global_optimizer_step
        if is_ddp:
            dist.barrier()
        _save_checkpoint(
            model,
            optim,
            sched,
            scaler,
            epoch,
            global_optimizer_step,
            Path(args.output_dir),
            rank,
            training_config,
        )
        if is_ddp:
            dist.barrier()

    if rank == 0:
        print("[INFO] pre-training finished")
        wandb.finish()
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
