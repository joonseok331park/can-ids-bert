# scripts/finetune.py
# -*- coding: utf-8 -*-
"""CAN-BERT 4클래스 분류용 미세 조정 스크립트."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple
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
from core.tokenizer import CANTokenizer
from models.teacher_classifier import CANBertForClassification


def calculate_class_weights(dataset: ClassificationDataset) -> torch.Tensor:
    """
    클래스 불균형 대응을 위한 가중치 계산
    """
    labels = [dataset.labels[i] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # 역수 가중치 계산 (작은 클래스에 더 큰 가중치)
    weights = total_samples / (len(class_counts) * class_counts)
    
    print("Class distribution and weights:")
    class_names = ['Benign', 'DoS', 'Fuzzy', 'Malfunction']
    for i, (count, weight) in enumerate(zip(class_counts, weights)):
        print(f"  {class_names[i]} ({i}): {count} samples, weight: {weight:.4f}")
    
    return torch.FloatTensor(weights)


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    종합 평가 지표 계산
    """
    class_names = ['Benign', 'DoS', 'Fuzzy', 'Malfunction']
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    # 클래스별 지표
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
    
    return metrics


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float], List[int], List[int]]:
    """
    모델 평가 및 지표 계산
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    ground_truths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            
            total_loss += outputs.loss.item()
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            
            predictions.extend(preds)
            ground_truths.extend(labels)
    
    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(ground_truths, predictions)
    
    return avg_loss, metrics, predictions, ground_truths


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune CAN-BERT for 4-class classification")
    
    # 필수 인자
    parser.add_argument("--train_data_dir", required=True, help="Training data directory")
    parser.add_argument("--val_data_dir", required=True, help="Validation data directory") 
    parser.add_argument("--test_data_dir", required=True, help="Test data directory")
    parser.add_argument("--vocab_path", required=True, help="Vocabulary file path")
    parser.add_argument("--resume_from_checkpoint", required=True, help="Pretrained checkpoint path")
    
    # 선택적 인자
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=126, help="Sequence length")
    parser.add_argument("--body_lr", type=float, default=2e-6, help="Learning rate for BERT body")
    parser.add_argument("--head_lr", type=float, default=5e-5, help="Learning rate for classification head")
    
    # wandb 설정
    parser.add_argument("--wandb_project", default="can-bert-finetune", help="Wandb project name")
    parser.add_argument("--wandb_run_name", default=None, help="Wandb run name")
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # wandb 초기화 (오프라인 모드)
    use_wandb = True
    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode="offline",  # 오프라인 모드로 실행
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
    
    # 토크나이저 로드
    print("[INFO] Loading tokenizer...")
    tokenizer = CANTokenizer()
    tokenizer.load_vocab(args.vocab_path)
    
    # 데이터셋 로드
    print("[INFO] Loading datasets...")
    train_dataset = ClassificationDataset(args.train_data_dir, tokenizer, args.seq_len)
    val_dataset = ClassificationDataset(args.val_data_dir, tokenizer, args.seq_len)
    test_dataset = ClassificationDataset(args.test_data_dir, tokenizer, args.seq_len)
    
    # 클래스 가중치 계산
    class_weights = calculate_class_weights(train_dataset).to(device)
    
    # 데이터로더 생성
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 모델 설정
    print("[INFO] Initializing model...")
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=1,
        intermediate_size=512,
        num_labels=4,  # 4클래스 분류
        max_position_embeddings=args.seq_len,
    )
    model = CANBertForClassification(config, num_labels=4).to(device)
    
    # 사전 훈련된 가중치 로드 (선별적 로딩)
    print(f"[INFO] Loading pretrained weights from {args.resume_from_checkpoint}")
    checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
    
    # BERT 몸통 가중치만 추출
    bert_state_dict = {}
    model_state = checkpoint.get("model", checkpoint)  # "model" 키 확인, 없으면 전체 사용
    
    for key, value in model_state.items():
        if key.startswith("bert."):
            # "bert." 접두사 제거
            new_key = key.replace("bert.", "")
            bert_state_dict[new_key] = value
    
    # BERT 몸통에만 가중치 로드 (분류 헤드는 무작위 초기화 유지)
    model.bert.load_state_dict(bert_state_dict, strict=False)
    print("[INFO] Successfully loaded pretrained BERT weights")
    
    # 손실 함수 (클래스 불균형 대응)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 옵티마이저 (차등 학습률)
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
    
    # 스케줄러
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # 훈련 루프
    print("[INFO] Starting training...")
    best_val_f1 = 0.0
    best_model_path = None
    
    for epoch in range(args.epochs):
        # 훈련
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_ground_truths = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # 예측값 수집
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            train_predictions.extend(preds)
            train_ground_truths.extend(labels)
        
        # 훈련 지표 계산
        avg_train_loss = train_loss / len(train_loader)
        train_metrics = compute_metrics(train_ground_truths, train_predictions)
        
        # 검증
        val_loss, val_metrics, val_predictions, val_ground_truths = evaluate_model(
            model, val_loader, device
        )
        
        # 로깅
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train F1: {train_metrics['f1_weighted']:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_metrics['f1_weighted']:.4f}")
        
        # wandb 로깅
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
        
        # 최적 모델 저장
        if val_metrics['f1_weighted'] > best_val_f1:
            best_val_f1 = val_metrics['f1_weighted']
            best_model_path = Path(args.output_dir) / f"pilot-finetuned-best.pt"
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'config': config,
            }, best_model_path)
            
            print(f"[INFO] New best model saved: {best_model_path}")
    
    # 최종 테스트 평가
    print("\n[INFO] Final evaluation on test set...")
    if best_model_path and best_model_path.exists():
        # 최적 모델 로드
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_metrics, test_predictions, test_ground_truths = evaluate_model(
        model, test_loader, device
    )
    
    # 최종 결과 출력
    print("\n=== FINAL TEST RESULTS ===")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test F1 (weighted): {test_metrics['f1_weighted']:.4f}")
    print(f"Test F1 (macro): {test_metrics['f1_macro']:.4f}")
    
    # 혼동행렬 출력
    cm = confusion_matrix(test_ground_truths, test_predictions)
    print("\nConfusion Matrix:")
    print(cm)
    
    # 클래스별 상세 리포트
    class_names = ['Benign', 'DoS', 'Fuzzy', 'Malfunction']
    print("\nClassification Report:")
    print(classification_report(test_ground_truths, test_predictions, target_names=class_names))
    
    # wandb 최종 로깅
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
