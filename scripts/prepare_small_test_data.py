#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
작은 테스트 데이터셋 생성 스크립트 (파일럿 테스트용)
"""

import os
import shutil
from pathlib import Path
import random

def create_small_dataset():
    """
    파일럿 테스트를 위한 작은 데이터셋 생성
    각 클래스당 적은 수의 라인만 추출하여 빠른 테스트 가능
    """
    
    # 소스 데이터 경로
    dataset_dir = Path('dataset/CAN-MIRGU(train)')
    target_dir = Path('data/finetune_data')
    
    # 대상 디렉토리 생성
    for split in ['train', 'validation', 'test']:
        (target_dir / split).mkdir(parents=True, exist_ok=True)
    
    # 각 클래스별 소스 파일 선택
    source_files = {
        'Benign': [
            dataset_dir / 'Benign/Day_1/Benign_day1_file1.log',
            dataset_dir / 'Benign/Day_1/Benign_day1_file2.log'
        ],
        'DoS': [
            dataset_dir / 'Attack/Real_attacks/DoS_attack.log'
        ],
        'Fuzzy': [
            dataset_dir / 'Attack/Real_attacks/Fuzzing_random_IDs.log',
            dataset_dir / 'Attack/Real_attacks/Fuzzing_valid_IDs.log'
        ],
        'Malfunction': [
            dataset_dir / 'Attack/Real_attacks/EMS_attack.log',
            dataset_dir / 'Attack/Real_attacks/Gear_shifter_attack_1.log'
        ]
    }
    
    # 각 클래스별로 작은 샘플 생성
    for class_name, files in source_files.items():
        print(f"Processing {class_name} class...")
        
        for i, source_file in enumerate(files):
            if not source_file.exists():
                print(f"Warning: {source_file} does not exist")
                continue
                
            # 원본 파일에서 처음 1000라인만 읽기
            with open(source_file, 'r', encoding='utf-8') as f:
                lines = []
                for j, line in enumerate(f):
                    if j >= 1000:  # 처음 1000라인만
                        break
                    lines.append(line)
            
            if not lines:
                continue
                
            # train/validation/test 분할 (7:2:1)
            random.seed(42)
            random.shuffle(lines)
            
            n_total = len(lines)
            n_train = int(n_total * 0.7)
            n_val = int(n_total * 0.2)
            
            train_lines = lines[:n_train]
            val_lines = lines[n_train:n_train + n_val]
            test_lines = lines[n_train + n_val:]
            
            # 파일 저장
            if train_lines:
                train_file = target_dir / 'train' / f"{class_name}_{i+1:03d}.log"
                with open(train_file, 'w', encoding='utf-8') as f:
                    f.writelines(train_lines)
                print(f"  Created train file: {train_file.name} ({len(train_lines)} lines)")
            
            if val_lines:
                val_file = target_dir / 'validation' / f"{class_name}_{i+1:03d}.log"
                with open(val_file, 'w', encoding='utf-8') as f:
                    f.writelines(val_lines)
                print(f"  Created validation file: {val_file.name} ({len(val_lines)} lines)")
            
            if test_lines:
                test_file = target_dir / 'test' / f"{class_name}_{i+1:03d}.log"
                with open(test_file, 'w', encoding='utf-8') as f:
                    f.writelines(test_lines)
                print(f"  Created test file: {test_file.name} ({len(test_lines)} lines)")
    
    # 최종 통계
    print("\n=== Final Statistics ===")
    for split in ['train', 'validation', 'test']:
        split_dir = target_dir / split
        file_count = len(list(split_dir.glob('*.log')))
        
        total_lines = 0
        for file_path in split_dir.glob('*.log'):
            with open(file_path, 'r', encoding='utf-8') as f:
                total_lines += sum(1 for _ in f)
        
        print(f"{split}: {file_count} files, {total_lines} total lines")

if __name__ == "__main__":
    create_small_dataset()