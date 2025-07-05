# scripts/pretrain.py (모든 문제 해결된 최종 버전)
# -*- coding: utf-8 -*-
"""
CAN‑BERT 사전 훈련 스크립트 (기본 DDP 지원)

▪ 8-GPU DDP 분산 훈련 지원
▪ DistributedSampler 사용 (Map-style Dataset)
▪ find_unused_parameters=True로 BertModel pooler 미사용 문제 해결
▪ 점진적 체크포인트 재개 지원
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm.auto import tqdm
from transformers import BertConfig, get_linear_schedule_with_warmup

from core.dataset import MLMDataset
from core.tokenizer import CANTokenizer, CANSequencer
from models.teacher import CANBertForMaskedLM
from utils.data_loader import load_can_data


# --------------------------------------------------------------------------- #
# 1. DDP 및 준비 유틸리티                                                      #
# --------------------------------------------------------------------------- #

def _seed_everything(seed: int = 42):
    """모든 랜덤 시드 고정 (재현성)"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _parse_args() -> argparse.Namespace:
    """명령행 인수 파싱"""
    ap = argparse.ArgumentParser("CAN‑BERT pre‑training (DDP 지원)")

    # DDP 관련
    ap.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")

    # 필수 I/O
    ap.add_argument("--data_path", required=True, help="Training data path")
    ap.add_argument("--vocab_path", required=True, help="Vocabulary file path")
    ap.add_argument("--output_dir", default="checkpoints", help="Output directory")
    ap.add_argument("--dataset_type", default="candump", help="Dataset type")
    ap.add_argument("--resume_from_checkpoint", default=None, help="Checkpoint to resume from")

    # 학습 하이퍼파라미터
    ap.add_argument("--seq_len", type=int, default=126, help="Sequence length")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU")
    ap.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    ap.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    ap.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    ap.add_argument("--mask_prob", type=float, default=0.15, help="Mask probability")

    # Teacher 모델 크기
    ap.add_argument("--hidden_size", type=int, default=256, help="Hidden size")
    ap.add_argument("--num_layers", type=int, default=4, help="Number of layers")
    ap.add_argument("--num_heads", type=int, default=1, help="Number of attention heads")
    ap.add_argument("--intermediate", type=int, default=512, help="Intermediate size")

    # 시스템
    ap.add_argument("--num_workers", type=int, default=2, help="DataLoader workers")

    return ap.parse_args()


def _device() -> torch.device:
    """기본 디바이스 반환"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _setup_ddp() -> Tuple[int, int, torch.device]:
    """
    DDP 환경 초기화
    Returns:
        rank: 글로벌 rank
        local_rank: 로컬 rank  
        device: CUDA 디바이스
    """
    dist.init_process_group(backend="nccl")
    
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    print(f"[RANK {rank}] Initialized DDP: rank={rank}, local_rank={local_rank}, world_size={world_size}")
    return rank, local_rank, device


def _build_loader(
    args: argparse.Namespace, 
    tokenizer: CANTokenizer, 
    rank: int = 0, 
    world_size: int = 1
) -> Tuple[DataLoader, int]:
    """
    기본 DDP DataLoader 생성 (DistributedSampler 사용)
    
    Args:
        args: 명령행 인수
        tokenizer: CANTokenizer
        rank: 현재 프로세스의 글로벌 rank
        world_size: 전체 프로세스 수
    Returns:
        DataLoader, 총 스텝 수
    """
    # 데이터 로드 및 시퀀스 생성
    df = load_can_data(args.data_path, dataset_type=args.dataset_type)
    sequencer = CANSequencer(tokenizer, seq_len=args.seq_len, stride=1)
    seqs = sequencer.transform(df)
    
    if not seqs:
        raise RuntimeError("Sequencer produced no sequences.")

    if rank == 0:
        print(f"[INFO] Total sequences loaded: {len(seqs)}")

    # MLM 데이터셋 생성
    ds = MLMDataset(seqs, tokenizer, mask_prob=args.mask_prob)
    
    # DDP용 DistributedSampler 생성 (Map-style Dataset이므로 가능)
    if world_size > 1:
        sampler = DistributedSampler(
            ds, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True,
            drop_last=True
        )
        shuffle = False  # sampler가 있으면 shuffle=False
    else:
        sampler = None
        shuffle = True   # 단일 GPU에서는 shuffle=True
    
    # DataLoader 생성
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    total_steps = len(loader)
    if rank == 0:
        print(f"[INFO] DataLoader created: {total_steps} steps per epoch")
    
    return loader, total_steps


def _build_model(
    tokenizer: CANTokenizer, args: argparse.Namespace
) -> CANBertForMaskedLM:
    """모델 생성"""
    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate,
        max_position_embeddings=args.seq_len,
    )
    return CANBertForMaskedLM(cfg)


def _build_optim_sched(
    model: CANBertForMaskedLM,
    args: argparse.Namespace,
    steps_per_epoch: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Optimizer & Scheduler 생성"""
    optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=steps_per_epoch * args.epochs,
    )
    return optim, sched


# --------------------------------------------------------------------------- #
# 2. DDP 훈련 루프                                                             #
# --------------------------------------------------------------------------- #

def _train_epoch(
    model: DDP,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LambdaLR,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    end_epoch: int,
    rank: int,
) -> None:
    """한 에포크 훈련"""
    model.train()
    
    # DistributedSampler의 에포크 설정 (셔플링 다양화)
    if hasattr(loader.sampler, 'set_epoch'):
        loader.sampler.set_epoch(epoch)
    
    # 진행률 표시는 rank 0에서만
    if rank == 0:
        prog = tqdm(loader, desc=f"Epoch {epoch+1}/{end_epoch}", leave=False)
    else:
        prog = loader
    
    total_loss = 0.0
    num_steps = 0
    
    for step, batch in enumerate(prog):
        # 배치를 GPU로 이동
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        
        optim.zero_grad()
        
        # Mixed Precision Training
        with autocast():
            loss = model(**batch)[0]
        
        # 그래디언트 스케일링 및 역전파
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()
        sched.step()
        
        total_loss += loss.item()
        num_steps += 1
        
        # 로그는 rank 0에서만
        if rank == 0:
            if step % 100 == 0:
                wandb.log({
                    "train_loss": loss.item(), 
                    "lr": sched.get_last_lr()[0],
                    "epoch": epoch,
                    "step": step
                })
            if isinstance(prog, tqdm):
                prog.set_postfix(loss=f"{loss.item():.4f}")
    
    # 에포크 평균 손실 계산
    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    
    if rank == 0:
        print(f"[EPOCH {epoch+1}] Average Loss: {avg_loss:.4f}")


def _save_checkpoint(
    model: DDP,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    out_dir: Path,
    rank: int,
) -> Path:
    """체크포인트 저장 (rank 0에서만)"""
    if rank != 0:
        return None
    
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"can-bert-pretrained-epoch-{epoch+1}.pt"
    
    # DDP 모델의 실제 모델 state_dict 저장
    torch.save(
        {
            "epoch": epoch + 1,  # 다음 epoch부터 재개
            "model": model.module.state_dict(),  # DDP에서는 .module 필요
            "optim_state": optim.state_dict(),
            "sched_state": sched.state_dict(),
        },
        ckpt_path,
    )
    
    print(f"[INFO] Checkpoint saved → {ckpt_path}")
    return ckpt_path


def _load_checkpoint(
    model: CANBertForMaskedLM,
    optim: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LambdaLR,
    ckpt_path: str,
    device: torch.device,
    rank: int,
) -> int:
    """체크포인트 로드"""
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    if rank == 0:
        print(f"[INFO] Loading checkpoint from {ckpt_path}")
    
    # 모든 rank에서 체크포인트 로드
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # 모델 상태 로드
    model.load_state_dict(ckpt["model"])
    
    # Optimizer, Scheduler는 새로운 데이터에 맞춰 재생성되므로
    # 이전 상태는 로드하지 않음 (사용자 요구사항에 따라)
    
    start_epoch = ckpt["epoch"]
    if rank == 0:
        print(f"[INFO] Resumed from epoch {start_epoch}")
    
    return start_epoch


# --------------------------------------------------------------------------- #
# 3. 메인 함수                                                                #
# --------------------------------------------------------------------------- #

def main() -> None:
    """메인 함수"""
    args = _parse_args()
    
    # DDP 환경 체크 및 초기화
    is_ddp = "WORLD_SIZE" in os.environ
    
    if is_ddp:
        rank, local_rank, device = _setup_ddp()
        world_size = int(os.environ["WORLD_SIZE"])
        # 각 rank마다 다른 시드로 데이터 다양성 확보
        _seed_everything(args.seed + rank)
    else:
        rank, world_size = 0, 1
        device = _device()
        _seed_everything(args.seed)
        print("[INFO] Running in single GPU mode")
    
    # W&B 초기화 (rank 0에서만)
    if rank == 0:
        wandb.init(
            project="CAN-IDS-DDP-Pretrain", 
            config=vars(args), 
            mode="offline",
            name=f"ddp-{world_size}gpu" if is_ddp else "single-gpu"
        )
    
    # 토크나이저 로드
    tokenizer = CANTokenizer()
    tokenizer.load_vocab(args.vocab_path)
    
    # 데이터로더 생성
    loader, steps_per_epoch = _build_loader(args, tokenizer, rank, world_size)
    
    # 모델 생성 및 GPU로 이동
    model = _build_model(tokenizer, args).to(device)
    
    # Optimizer & Scheduler 생성
    optim, sched = _build_optim_sched(model, args, steps_per_epoch)
    
    # Mixed Precision Scaler
    scaler = GradScaler() if device.type == "cuda" else None
    
    # 체크포인트 로드
    start_epoch = 0
    if args.resume_from_checkpoint:
        start_epoch = _load_checkpoint(
            model, optim, sched, args.resume_from_checkpoint, device, rank
        )
    
    # DDP로 모델 래핑 (find_unused_parameters=True로 BertModel pooler 문제 해결)
    if is_ddp:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=True  # ← 핵심: BertModel pooler 미사용 문제 해결
        )
        if rank == 0:
            print("[INFO] Model wrapped with DDP (find_unused_parameters=True)")
    
    # 훈련 루프
    target_epoch = start_epoch + args.epochs
    
    if rank == 0:
        print(f"[INFO] Training from epoch {start_epoch} to {target_epoch - 1}")
        print(f"[INFO] Steps per epoch: {steps_per_epoch}")
        print(f"[INFO] Total training steps: {steps_per_epoch * args.epochs}")
    
    for epoch in range(start_epoch, target_epoch):
        # 훈련
        _train_epoch(
            model, loader, optim, sched, scaler, 
            device, epoch, target_epoch, rank
        )
        
        # DDP 동기화 (모든 프로세스가 에포크 완료 대기)
        if is_ddp:
            dist.barrier()
        
        # 체크포인트 저장 (rank 0에서만)
        _save_checkpoint(model, optim, sched, epoch, Path(args.output_dir), rank)
        
        # 다시 동기화
        if is_ddp:
            dist.barrier()
    
    if rank == 0:
        print("[INFO] Pre‑training finished.")
        wandb.finish()
    
    # DDP 정리
    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()