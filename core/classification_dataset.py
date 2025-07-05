# core/classification_dataset.py
import torch
from torch.utils.data import Dataset
from typing import List, Dict

class ClassificationDataset(Dataset):
    """
    미세 조정을 위한 분류 데이터셋.
    인코딩된 시퀀스와 해당 시퀀스의 라벨을 반환합니다.
    """
    def __init__(self, sequences: List[List[int]], labels: List[int]):
        """
        데이터셋을 초기화합니다.

        Args:
            sequences (List[List[int]]): 정수 시퀀스의 리스트.
            labels (List[int]): 각 시퀀스에 해당하는 라벨의 리스트.
        """
        self.sequences = sequences
        self.labels = labels
        self.seq_len = len(sequences[0]) if sequences else 0

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # 패딩이 없다고 가정하고 attention_mask는 모두 1로 생성
        attention_mask = [1] * self.seq_len

        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }