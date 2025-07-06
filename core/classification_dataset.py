# core/classification_dataset.py
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd

from core.tokenizer import CANTokenizer
from utils.data_loader import load_classification_data


class ClassificationDataset(Dataset):
    """
    미세 조정을 위한 분류 데이터셋.
    디렉토리에서 클래스별 파일들을 로드하고 시퀀스로 변환합니다.
    """
    
    def __init__(self, data_dir: str | Path, tokenizer: CANTokenizer, seq_len: int = 126):
        """
        데이터셋을 초기화합니다.
        
        Args:
            data_dir: 데이터 디렉토리 경로 (train/validation/test)
            tokenizer: CAN 토크나이저
            seq_len: 시퀀스 길이 (기본값: 126)
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        # 데이터 로드
        print(f"Loading data from {self.data_dir}")
        self.df = load_classification_data(self.data_dir)
        
        # 시퀀스와 라벨 생성
        self.sequences, self.labels = self._make_sequences()
        
        print(f"Generated {len(self.sequences)} sequences")

    def _make_sequences(self) -> Tuple[List[List[int]], List[int]]:
        """
        DataFrame을 시퀀스와 라벨로 변환
        """
        frames = []
        
        # CAN 프레임 토큰화
        for _, row in self.df.iterrows():
            cid = row['CAN_ID']
            payload = row['Data']
            label = row['Label']
            
            # CAN ID를 정수로 변환 후 오프셋 적용
            id_tok = str(int(cid, 16) + self.tokenizer.id_offset)
            
            # 프레임 토큰: [ID] + [페이로드 바이트들] + [패딩]
            frame_tokens = [id_tok] + payload + ["<VOID>"] * (8 - len(payload))
            frames.append((frame_tokens, label))
        
        # 슬라이딩 윈도우로 시퀀스 생성
        frames_per_seq = self.seq_len // 9  # 9는 프레임당 토큰 수 (ID + 8 바이트)
        sequences = []
        labels = []
        
        for i in range(len(frames) - frames_per_seq + 1):
            window = frames[i:i + frames_per_seq]
            
            # 윈도우 내 모든 토큰을 하나의 시퀀스로 결합
            tokens = []
            for frame_tokens, _ in window:
                tokens.extend(frame_tokens)
            
            # 토큰화
            encoded_tokens = self.tokenizer.encode(tokens)
            
            # 시퀀스 길이 맞추기 (패딩 또는 절단)
            if len(encoded_tokens) < self.seq_len:
                # 패딩
                encoded_tokens.extend([self.tokenizer.get_pad_token_id()] * (self.seq_len - len(encoded_tokens)))
            else:
                # 절단
                encoded_tokens = encoded_tokens[:self.seq_len]
            
            sequences.append(encoded_tokens)
            
            # 윈도우 내 라벨 결정: 하나라도 공격이 있으면 해당 클래스로 분류
            window_labels = [label for _, label in window]
            # 가장 높은 우선순위 라벨 선택 (Malfunction > Fuzzy > DoS > Benign)
            if 3 in window_labels:
                final_label = 3  # Malfunction
            elif 2 in window_labels:
                final_label = 2  # Fuzzy
            elif 1 in window_labels:
                final_label = 1  # DoS
            else:
                final_label = 0  # Benign
                
            labels.append(final_label)
        
        return sequences, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # attention_mask 생성 (패딩이 아닌 토큰은 1, 패딩은 0)
        pad_token_id = self.tokenizer.get_pad_token_id()
        attention_mask = [1 if token_id != pad_token_id else 0 for token_id in sequence]
        
        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }