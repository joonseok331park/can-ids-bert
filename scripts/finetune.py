# scripts/finetune.py
# -*- coding: utf-8 -*-
"""Fine-tune the public teacher prototype for four-class classification."""

from __future__ import annotations

import argparse
import hashlib
import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import BertConfig, get_linear_schedule_with_warmup

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import wandb

from core.classification_dataset import ClassificationDataset
from core.classes import CLASS_LABELS, CLASS_NAMES, NUM_CLASSES
from core.tokenizer import CANTokenizer
from models.teacher_classifier import CANBertForClassification
from scripts.pretrain import (
    CHECKPOINT_SCHEMA_VERSION as PRETRAIN_CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_TYPE as PRETRAIN_CHECKPOINT_TYPE,
    MODEL_CONFIG_REQUIRED_KEYS as PRETRAIN_MODEL_CONFIG_REQUIRED_KEYS,
)


class IncompatiblePretrainedCheckpointError(RuntimeError):
    """Raised when fine-tuning cannot safely consume pretrained weights."""


def _sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.casefold()
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise IncompatiblePretrainedCheckpointError(f"{label} must be a mapping")
    return value


def _normalize_model_key(key: str) -> str:
    """Remove wrapper prefixes that do not belong to the saved architecture."""
    prefixes = ("module.", "_orig_mod.")
    while key.startswith(prefixes):
        key = key.split(".", 1)[1]
    return key


def _load_pretrained_bert(
    model: CANBertForClassification,
    model_state: Mapping[str, Any],
) -> int:
    """Load a complete, shape-compatible BERT body or fail explicitly."""
    normalized_state: dict[str, Any] = {}
    for key, value in model_state.items():
        if not isinstance(key, str):
            raise IncompatiblePretrainedCheckpointError(
                "Pretrained checkpoint contains a non-string model key"
            )
        normalized = _normalize_model_key(key)
        if normalized in normalized_state:
            raise IncompatiblePretrainedCheckpointError(
                f"Pretrained checkpoint contains duplicate model key {normalized!r}"
            )
        normalized_state[normalized] = value

    has_wrapped_bert_body = any(
        key.startswith("bert.") for key in normalized_state
    )
    candidates = (
        {
            key.removeprefix("bert."): value
            for key, value in normalized_state.items()
            if key.startswith("bert.")
        }
        if has_wrapped_bert_body
        else normalized_state
    )

    expected = model.bert.state_dict()
    missing = sorted(set(expected) - set(candidates))
    unexpected = sorted(set(candidates) - set(expected))
    malformed = sorted(
        key for key, value in candidates.items() if not torch.is_tensor(value)
    )
    shape_mismatches = sorted(
        key
        for key in set(expected) & set(candidates)
        if torch.is_tensor(candidates[key])
        and candidates[key].shape != expected[key].shape
    )
    if missing or unexpected or malformed or shape_mismatches:
        details = []
        if missing:
            details.append("missing keys=" + ", ".join(missing))
        if unexpected:
            details.append("unexpected keys=" + ", ".join(unexpected))
        if malformed:
            details.append("non-tensor values=" + ", ".join(malformed))
        if shape_mismatches:
            details.append("shape mismatches=" + ", ".join(shape_mismatches))
        raise IncompatiblePretrainedCheckpointError(
            "Pretrained checkpoint does not contain an exact compatible BERT body: "
            + "; ".join(details)
        )

    try:
        result = model.bert.load_state_dict(candidates, strict=True)
    except RuntimeError as error:
        raise IncompatiblePretrainedCheckpointError(
            f"Pretrained BERT state failed strict validation: {error}"
        ) from error
    if result.missing_keys or result.unexpected_keys:
        raise IncompatiblePretrainedCheckpointError(
            "Pretrained BERT state failed strict validation"
        )
    return len(candidates)


def _legacy_model_state(checkpoint: Any) -> Mapping[str, Any]:
    checkpoint = _require_mapping(checkpoint, "Legacy checkpoint payload")
    for key in ("model_state_dict", "model", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise IncompatiblePretrainedCheckpointError(
        "Legacy checkpoint does not contain model weights"
    )


def _versioned_model_state(
    checkpoint: Any,
    model: CANBertForClassification,
    vocab_path: str | Path,
) -> Mapping[str, Any]:
    """Validate the versioned pretraining contract before returning weights."""
    checkpoint = _require_mapping(checkpoint, "Pretrained checkpoint payload")
    if "schema_version" not in checkpoint:
        raise IncompatiblePretrainedCheckpointError(
            "Schema-less checkpoints are rejected by --pretrained_checkpoint; "
            "use --legacy_pretrained_checkpoint for an explicit weights-only load"
        )
    if checkpoint["schema_version"] != PRETRAIN_CHECKPOINT_SCHEMA_VERSION:
        raise IncompatiblePretrainedCheckpointError(
            f"Unsupported pretrained checkpoint schema "
            f"{checkpoint['schema_version']!r}; expected "
            f"{PRETRAIN_CHECKPOINT_SCHEMA_VERSION}"
        )
    if checkpoint.get("checkpoint_type") != PRETRAIN_CHECKPOINT_TYPE:
        raise IncompatiblePretrainedCheckpointError(
            "Unexpected pretrained checkpoint type: "
            f"{checkpoint.get('checkpoint_type')!r}"
        )

    training_config = _require_mapping(
        checkpoint.get("training_config"),
        "Pretrained checkpoint training configuration",
    )
    stored_vocab_sha256 = training_config.get("vocab_sha256")
    if not _valid_sha256(stored_vocab_sha256):
        raise IncompatiblePretrainedCheckpointError(
            "Pretrained checkpoint training configuration has missing or invalid "
            "vocab_sha256"
        )
    current_vocab_sha256 = _sha256_file(vocab_path)
    if stored_vocab_sha256 != current_vocab_sha256:
        raise IncompatiblePretrainedCheckpointError(
            "Vocabulary content is incompatible with the pretrained checkpoint: "
            f"checkpoint vocab_sha256={stored_vocab_sha256}, "
            f"current vocab_sha256={current_vocab_sha256}"
        )

    stored_model_config = _require_mapping(
        checkpoint.get("model_config"),
        "Pretrained checkpoint model configuration",
    )
    current_model_config = model.config.to_dict()
    current_model_config.setdefault(
        "position_embedding_type",
        getattr(model.config, "position_embedding_type", "absolute"),
    )
    stored_missing = [
        key
        for key in PRETRAIN_MODEL_CONFIG_REQUIRED_KEYS
        if key not in stored_model_config
    ]
    current_missing = [
        key
        for key in PRETRAIN_MODEL_CONFIG_REQUIRED_KEYS
        if key not in current_model_config
    ]
    if stored_missing or current_missing:
        details = []
        if stored_missing:
            details.append("checkpoint missing " + ", ".join(stored_missing))
        if current_missing:
            details.append("current model missing " + ", ".join(current_missing))
        raise IncompatiblePretrainedCheckpointError(
            "Pretrained model architecture is incomplete: " + "; ".join(details)
        )
    mismatches = [
        key
        for key in PRETRAIN_MODEL_CONFIG_REQUIRED_KEYS
        if stored_model_config[key] != current_model_config[key]
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={stored_model_config[key]!r}, "
            f"current={current_model_config[key]!r}"
            for key in mismatches
        )
        raise IncompatiblePretrainedCheckpointError(
            "Pretrained model architecture is incompatible: " + details
        )

    return _require_mapping(
        checkpoint.get("model_state_dict"),
        "Pretrained checkpoint model_state_dict",
    )


def _load_finetune_pretrained_checkpoint(
    model: CANBertForClassification,
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    *,
    legacy: bool = False,
) -> int:
    """Load a validated schema-v2 checkpoint or an explicit legacy warm start."""
    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    checkpoint = torch.load(source, map_location="cpu")
    if legacy:
        warnings.warn(
            "Loading a legacy pretrained checkpoint without schema, vocabulary "
            "lineage, or model-config verification; applying compatible BERT "
            "weights only",
            RuntimeWarning,
            stacklevel=2,
        )
        model_state = _legacy_model_state(checkpoint)
    else:
        model_state = _versioned_model_state(
            checkpoint, model, vocab_path
        )
    return _load_pretrained_bert(model, model_state)


def _finetune_checkpoint_payload(
    model: CANBertForClassification,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_f1: float,
    config: BertConfig,
) -> Dict:
    """Build a weights-only-safe fine-tuning checkpoint payload."""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_f1": val_f1,
        "config": config.to_dict(),
    }


def calculate_class_weights(dataset: ClassificationDataset) -> torch.Tensor:
    """Calculate inverse-frequency weights for the fixed class set."""
    labels = list(dataset.labels)
    if not labels:
        raise ValueError("Cannot calculate class weights for an empty dataset")
    invalid = sorted(set(labels) - set(CLASS_LABELS))
    if invalid:
        raise ValueError(f"Training split contains invalid labels: {invalid}")
    class_counts = np.bincount(labels, minlength=NUM_CLASSES)
    missing = [
        CLASS_NAMES[index]
        for index, count in enumerate(class_counts)
        if count == 0
    ]
    if missing:
        raise ValueError(
            "Training split is missing required classes: " + ", ".join(missing)
        )
    total_samples = len(labels)
    weights = total_samples / (NUM_CLASSES * class_counts)

    print("Class distribution and weights:")
    for index, (count, weight) in enumerate(zip(class_counts, weights)):
        print(
            f"  {CLASS_NAMES[index]} ({index}): "
            f"{count} samples, weight: {weight:.4f}"
        )

    return torch.FloatTensor(weights)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """Compute metrics over all four labels, including absent predictions."""
    if not y_true:
        raise ValueError("Cannot compute metrics for an empty target set")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    invalid_targets = sorted(set(y_true) - set(CLASS_LABELS))
    invalid_predictions = sorted(set(y_pred) - set(CLASS_LABELS))
    if invalid_targets or invalid_predictions:
        raise ValueError(
            "Metrics received labels outside the fixed class set: "
            f"targets={invalid_targets}, predictions={invalid_predictions}"
        )

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, labels=CLASS_LABELS, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, labels=CLASS_LABELS, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, labels=CLASS_LABELS, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, labels=CLASS_LABELS, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, labels=CLASS_LABELS, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, labels=CLASS_LABELS, average='macro', zero_division=0),
    }

    precision_per_class = precision_score(
        y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0
    )
    f1_per_class = f1_score(
        y_true, y_pred, labels=CLASS_LABELS, average=None, zero_division=0
    )

    for index, class_name in enumerate(CLASS_NAMES):
        metrics[f'precision_{class_name}'] = precision_per_class[index]
        metrics[f'recall_{class_name}'] = recall_per_class[index]
        metrics[f'f1_{class_name}'] = f1_per_class[index]

    return metrics


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Return a fixed 4-by-4 confusion matrix."""
    if not y_true:
        raise ValueError("Cannot compute a confusion matrix for an empty target set")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    invalid_targets = sorted(set(y_true) - set(CLASS_LABELS))
    invalid_predictions = sorted(set(y_pred) - set(CLASS_LABELS))
    if invalid_targets or invalid_predictions:
        raise ValueError(
            "Confusion matrix received labels outside the fixed class set: "
            f"targets={invalid_targets}, predictions={invalid_predictions}"
        )
    return confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> Tuple[float, Dict[str, float], List[int], List[int]]:
    """Evaluate a non-empty loader with the fixed class metric set."""
    if len(dataloader) == 0:
        raise ValueError("Evaluation dataloader is empty")
    model.eval()
    total_loss = 0.0
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels_on_device = batch["labels"].to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=None if criterion is not None else labels_on_device,
            )

            loss = (
                criterion(outputs.logits, labels_on_device)
                if criterion is not None
                else outputs.loss
            )
            total_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            
            predictions.extend(preds)
            ground_truths.extend(labels)
    
    if not ground_truths:
        raise ValueError("Evaluation dataloader produced no samples")
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(ground_truths, predictions)
    
    return avg_loss, metrics, predictions, ground_truths


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CAN-BERT for 4-class classification")
    
    parser.add_argument("--train_data_dir", required=True, help="Training data directory")
    parser.add_argument("--val_data_dir", required=True, help="Validation data directory") 
    parser.add_argument("--test_data_dir", required=True, help="Test data directory")
    parser.add_argument("--vocab_path", required=True, help="Vocabulary file path")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--pretrained_checkpoint",
        help=(
            "Schema-v2 CAN-BERT pretraining checkpoint whose vocabulary and "
            "forward model configuration match this run"
        ),
    )
    checkpoint_group.add_argument(
        "--legacy_pretrained_checkpoint",
        help=(
            "Explicitly load a schema-less legacy checkpoint as strictly "
            "compatible BERT weights only"
        ),
    )
    
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=126, help="Sequence length")
    parser.add_argument("--body_lr", type=float, default=2e-6, help="Learning rate for BERT body")
    parser.add_argument("--head_lr", type=float, default=5e-5, help="Learning rate for classification head")
    parser.add_argument(
        "--short_file_policy",
        choices=("error", "skip"),
        default="skip",
        help="How to handle a source file too short for one sequence",
    )

    parser.add_argument("--wandb_project", default="can-bert-finetune", help="Wandb project name")
    parser.add_argument("--wandb_run_name", default=None, help="Wandb run name")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    use_wandb = True
    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode="offline",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "body_lr": args.body_lr,
                "head_lr": args.head_lr,
                "model_architecture": "CAN-BERT",
                "num_classes": 4,
            }
        )
        print("[INFO] Wandb initialized in offline mode")
    except Exception as e:
        print(f"[WARNING] Wandb initialization failed: {e}")
        print("[INFO] Continuing without wandb logging")
        use_wandb = False
    
    print("[INFO] Loading tokenizer...")
    tokenizer = CANTokenizer()
    tokenizer.load_vocab(args.vocab_path)
    
    print("[INFO] Loading datasets...")
    train_dataset = ClassificationDataset(
        args.train_data_dir, tokenizer, args.seq_len, args.short_file_policy
    )
    val_dataset = ClassificationDataset(
        args.val_data_dir, tokenizer, args.seq_len, args.short_file_policy
    )
    test_dataset = ClassificationDataset(
        args.test_data_dir, tokenizer, args.seq_len, args.short_file_policy
    )
    
    class_weights = calculate_class_weights(train_dataset).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("[INFO] Initializing model...")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=1,
        intermediate_size=512,
        num_labels=NUM_CLASSES,
        max_position_embeddings=args.seq_len,
    )
    model = CANBertForClassification(config, num_labels=NUM_CLASSES).to(device)
    
    checkpoint_path = (
        args.pretrained_checkpoint or args.legacy_pretrained_checkpoint
    )
    legacy_checkpoint = args.legacy_pretrained_checkpoint is not None
    print(f"[INFO] Loading pretrained weights from {checkpoint_path}")
    loaded_tensors = _load_finetune_pretrained_checkpoint(
        model,
        checkpoint_path,
        args.vocab_path,
        legacy=legacy_checkpoint,
    )
    print(f"[INFO] Loaded {loaded_tensors} pretrained BERT tensors")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer_grouped_parameters = [
        {
            'params': model.bert.parameters(),
            'lr': args.body_lr,
            'name': 'bert_body'
        },
        {
            'params': model.classifier.parameters(),
            'lr': args.head_lr,
            'name': 'classification_head'
        }
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    print("[INFO] Starting training...")
    best_val_f1 = float("-inf")
    best_model_path = None
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_ground_truths = []
        
        if len(train_loader) == 0:
            raise ValueError("Training dataloader is empty")
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad(set_to_none=True)
            labels_on_device = batch["labels"].to(device)
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=None,
            )

            loss = criterion(outputs.logits, labels_on_device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            train_predictions.extend(preds)
            train_ground_truths.extend(labels)
        
        avg_train_loss = train_loss / len(train_loader)
        train_metrics = compute_metrics(train_ground_truths, train_predictions)
        
        val_loss, val_metrics, val_predictions, val_ground_truths = evaluate_model(
            model, val_loader, device, criterion
        )
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_metrics['f1_weighted']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1_weighted']:.4f}")
        
        if use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": avg_train_loss,
                "train/f1_weighted": train_metrics['f1_weighted'],
                "train/accuracy": train_metrics['accuracy'],
                "val/loss": val_loss,
                "val/f1_weighted": val_metrics['f1_weighted'],
                "val/accuracy": val_metrics['accuracy'],
                "learning_rate": scheduler.get_last_lr()[0],
            })
        
        if val_metrics['f1_weighted'] > best_val_f1:
            best_val_f1 = val_metrics['f1_weighted']
            best_model_path = Path(args.output_dir) / "pilot-finetuned-best.pt"
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(
                _finetune_checkpoint_payload(
                    model, optimizer, epoch + 1, best_val_f1, config
                ),
                best_model_path,
            )
            
            print(f"[INFO] New best model saved: {best_model_path}")
    
    print("\n[INFO] Final evaluation on test set...")
    if best_model_path and best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics, test_predictions, test_ground_truths = evaluate_model(
        model, test_loader, device, criterion
    )
    
    print("\n=== FINAL TEST RESULTS ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    
    cm = compute_confusion_matrix(test_ground_truths, test_predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nClassification Report:")
    print(
        classification_report(
            test_ground_truths,
            test_predictions,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )
    
    if use_wandb:
        wandb.log({
            "test/loss": test_loss,
            "test/accuracy": test_metrics['accuracy'],
            "test/f1_weighted": test_metrics['f1_weighted'],
            "test/f1_macro": test_metrics['f1_macro'],
            "best_val_f1": best_val_f1,
        })
        wandb.finish()
    print(f"\n[INFO] Training completed. Best model saved at: {best_model_path}")


if __name__ == "__main__":
    main()
