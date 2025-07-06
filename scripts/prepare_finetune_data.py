#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
미세 조정용 데이터 준비 스크립트
CAN-MIRGU 데이터를 4클래스로 분류하여 train/validation/test 폴더에 배치
"""

import os
import shutil
from pathlib import Path
import random
from typing import Dict, List, Tuple

def classify_attack_files(attack_dir: Path) -> Dict[str, List[Path]]:
    """
    공격 파일들을 4클래스로 분류
    
    Returns:
        Dict[str, List[Path]]: 클래스별 파일 목록
    """
    classification = {
        'Benign': [],
        'DoS': [],
        'Fuzzy': [],
        'Malfunction': []
    }
    
    # Real_attacks 디렉토리 파일들 분류
    real_attacks_dir = attack_dir / 'Real_attacks'
    if real_attacks_dir.exists():
        for file_path in real_attacks_dir.glob('*.log'):
            filename = file_path.name.lower()
            if 'dos' in filename:
                classification['DoS'].append(file_path)
            elif 'fuzz' in filename:
                classification['Fuzzy'].append(file_path)
            else:
                classification['Malfunction'].append(file_path)
    
    # Masquerade_attacks와 Suspension_attacks도 Malfunction으로 분류
    for subdir in ['Masquerade_attacks', 'Suspension_attacks']:
        subdir_path = attack_dir / subdir
        if subdir_path.exists():
            for file_path in subdir_path.glob('*.log'):
                classification['Malfunction'].append(file_path)
    
    return classification

def get_benign_files(benign_dir: Path) -> List[Path]:
    """
    정상 데이터 파일들 수집
    """
    benign_files = []
    for day_dir in benign_dir.glob('Day_*'):
        for file_path in day_dir.glob('*.log'):
            benign_files.append(file_path)
    return benign_files

def split_files(files: List[Path], train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    파일들을 train/validation/test로 분할
    """
    random.seed(42)  # 재현 가능한 결과를 위해
    files_shuffled = files.copy()
    random.shuffle(files_shuffled)
    
    n_total = len(files_shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_files = files_shuffled[:n_train]
    val_files = files_shuffled[n_train:n_train + n_val]
    test_files = files_shuffled[n_train + n_val:]
    
    return train_files, val_files, test_files

def copy_files_to_split(files: List[Path], target_dir: Path, class_name: str):
    """
    파일들을 대상 디렉토리로 복사 (심볼릭 링크 사용으로 효율성 향상)
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    for i, file_path in enumerate(files):
        target_file = target_dir / f"{class_name}_{i+1:03d}_{file_path.stem}.log"
        try:
            # 심볼릭 링크 생성 (디스크 공간 절약)
            if not target_file.exists():
                os.symlink(file_path.absolute(), target_file)
                print(f"Linked: {file_path.name} -> {target_file.name}")
        except OSError:
            # 심볼릭 링크 실패 시 복사
            shutil.copy2(file_path, target_file)
            print(f"Copied: {file_path.name} -> {target_file.name}")

def main():
    # 데이터 경로 설정
    dataset_dir = Path('dataset/CAN-MIRGU(train)')
    finetune_data_dir = Path('data/finetune_data')
    
    # 클래스별 파일 분류
    print("=== 데이터 파일 분류 ===")
    
    # 공격 데이터 분류
    attack_dir = dataset_dir / 'Attack'
    attack_classification = classify_attack_files(attack_dir)
    
    # 정상 데이터 수집
    benign_dir = dataset_dir / 'Benign'
    benign_files = get_benign_files(benign_dir)
    attack_classification['Benign'] = benign_files
    
    # 클래스별 파일 개수 출력
    for class_name, files in attack_classification.items():
        print(f"{class_name}: {len(files)} files")
    
    # 각 클래스별로 train/validation/test 분할
    print("\n=== 데이터 분할 및 복사 ===")
    
    for class_name, files in attack_classification.items():
        if not files:
            print(f"Warning: No files found for class {class_name}")
            continue
            
        # 파일 분할
        train_files, val_files, test_files = split_files(files)
        
        print(f"\n{class_name} 클래스:")
        print(f"  Train: {len(train_files)} files")
        print(f"  Validation: {len(val_files)} files")
        print(f"  Test: {len(test_files)} files")
        
        # 파일 복사
        copy_files_to_split(train_files, finetune_data_dir / 'train', class_name)
        copy_files_to_split(val_files, finetune_data_dir / 'validation', class_name)
        copy_files_to_split(test_files, finetune_data_dir / 'test', class_name)
    
    print("\n=== 데이터 준비 완료 ===")
    print(f"데이터가 {finetune_data_dir} 디렉토리에 준비되었습니다.")
    
    # 최종 통계 출력
    for split in ['train', 'validation', 'test']:
        split_dir = finetune_data_dir / split
        if split_dir.exists():
            file_count = len(list(split_dir.glob('*.log')))
            print(f"{split}: {file_count} files")

if __name__ == "__main__":
    main()