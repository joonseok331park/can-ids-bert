# scripts/run_full_training.py
# -*- coding: utf-8 -*-
"""
CAN-BERT 다중 파트 자동화 사전 훈련 스크립트 (DDP 지원)

▪ 22개로 분할된 데이터 파일을 순회하며 8-GPU DDP 훈련
▪ torchrun을 사용하여 각 데이터 파트마다 분산 훈련 실행
▪ 항상 처음부터 새로운 훈련 시작 (체크포인트 재개 없음)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def _parse_args() -> argparse.Namespace:
    """명령행 인수 파싱"""
    ap = argparse.ArgumentParser("Automated multi-part CAN-BERT pre-training (DDP)")
    ap.add_argument("--parts_dir", default="data/aggregated_parts", type=str, 
                   help="Directory containing data parts")
    ap.add_argument("--checkpoint_dir", default="checkpoints", type=str,
                   help="Directory for checkpoints and vocab")
    ap.add_argument("--epochs_per_part", default=5, type=int,
                   help="Number of epochs per data part")
    ap.add_argument("--max_parts", type=int, default=None,
                   help="Maximum number of parts to process (for testing)")
    ap.add_argument("--num_workers", type=int, default=4, 
                   help="DataLoader subprocess count per GPU")
    ap.add_argument("--batch_size", type=int, default=32,
                   help="Batch size per GPU (will be distributed across 8 GPUs)")
    ap.add_argument("--nproc_per_node", type=int, default=8,
                   help="Number of GPUs to use")
    return ap.parse_args()


def _discover_parts(parts_dir: Path, limit: int | None) -> List[Path]:
    """데이터 파트 파일들을 발견하고 정렬"""
    parts = sorted([p for p in parts_dir.glob("part_*") if p.is_file() and not p.suffix])
    if limit is not None:
        parts = parts[:limit]
    if not parts:
        raise FileNotFoundError(f"[ERR] {parts_dir}에 part_* 데이터 파일이 없습니다.")
    return parts




def _run_command(cmd: list[str]) -> None:
    """서브프로세스 명령어 실행"""
    print("▶ " + " ".join(cmd))
    try:
        result = subprocess.run(cmd, check=True, shell=False, capture_output=True, text=True)
        print(f"✓ Command completed successfully")
        if result.stdout:
            print(f"STDOUT: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Command failed with return code {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        raise


def main():
    """메인 함수"""
    args = _parse_args()
    parts_dir = Path(args.parts_dir).expanduser()
    ckpt_dir = Path(args.checkpoint_dir).expanduser()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    print("=== CAN-BERT DDP 분산 훈련 자동화 스크립트 시작 ===")
    print(f"📁 데이터 디렉토리: {parts_dir}")
    print(f"💾 체크포인트 디렉토리: {ckpt_dir}")
    print(f"🔥 GPU 수: {args.nproc_per_node}")
    print(f"📦 GPU당 배치 크기: {args.batch_size}")
    
    # 어휘 파일 체크 (필요시 생성)
    vocab_path = ckpt_dir / "vocab.json"
    if not vocab_path.exists():
        print(f"[WARN] {vocab_path} not found. Running build_vocab first.")
        vocab_cmd = [
            sys.executable, "-m", "scripts.build_vocab", 
            "--input", "data/HCRL_dataset/train_aggregated.log", 
            "--output_dir", str(ckpt_dir)
        ]
        _run_command(vocab_cmd)
        
        if not vocab_path.exists():
            raise FileNotFoundError(f"Vocabulary file {vocab_path} was not created successfully.")
        print(f"✓ Vocabulary file created: {vocab_path}")

    # 데이터 파트 발견
    parts = _discover_parts(parts_dir, args.max_parts)
    print(f"[INFO] 총 {len(parts)}개의 데이터 파트를 대상으로 훈련을 시작합니다.")
    
    # 항상 처음부터 새로운 훈련 시작
    print("[INFO] 새로운 훈련을 시작합니다 (기존 체크포인트 무시).")

    # 각 데이터 파트에 대해 DDP 훈련 실행
    for idx, part in enumerate(parts):
        start_epoch_for_part = idx * args.epochs_per_part
        end_epoch_for_part = start_epoch_for_part + args.epochs_per_part
        
        
        print(f"\n{'='*60}")
        print(f"🚀 Part {idx+1}/{len(parts)}: {part.name}")
        print(f"📊 목표 epoch: {start_epoch_for_part} → {end_epoch_for_part-1}")
        print(f"{'='*60}")
        
        # torchrun 명령어 구성
        cmd_prefix = [
            "torchrun", 
            "--standalone", 
            f"--nproc_per_node={args.nproc_per_node}",
            "-m", "scripts.pretrain"
        ]
        
        cmd_args = [
            "--data_path", str(part),
            "--vocab_path", str(vocab_path),
            "--output_dir", str(ckpt_dir),
            "--seq_len", "126",
            "--batch_size", str(args.batch_size),  # GPU당 배치 크기
            "--epochs", str(args.epochs_per_part),
            "--learning_rate", "1e-3",
            "--num_workers", str(args.num_workers),
            "--dataset_type", "candump",
            "--seed", "42",  # 재현성을 위한 시드
        ]
        
        
        # DDP 훈련 실행
        full_cmd = cmd_prefix + cmd_args
        try:
            _run_command(full_cmd)
        except subprocess.CalledProcessError as e:
            print(f"❌ Part {part.name} 훈련 실패!")
            print(f"❌ 오류 코드: {e.returncode}")
            sys.exit(1)
        
        print(f"✅ {part.name} 훈련 완료")

    print("\n" + "="*60)
    print("🎉 모든 파트 학습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()