# core/dataset.py
# -*- coding: utf-8 -*-
"""
Masked‑Language‑Model(MLM) 학습용 Torch Dataset
· CANSequencer가 생성한 정수 시퀀스에 동적 마스킹 적용
"""

from __future__ import annotations

import random
from typing import Dict, List, Sequence

import torch

from core.tokenizer import CANTokenizer, SpecialToken


class MLMDataset(torch.utils.data.Dataset):
    """
    CAN 시퀀스 → (input_ids, attention_mask, labels) 튜플 생성
    """

    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        tokenizer: CANTokenizer,
        mask_prob: float = 0.45,
    ) -> None:
        self.sequences = list(sequences)
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob

        # 캐싱
        self._mask_id = self.tokenizer.token_to_id[SpecialToken.MASK]
        self._special_ids = {
            self.tokenizer.token_to_id[tk]
            for tk in SpecialToken
            if tk in self.tokenizer.token_to_id
        }

    # ---------------- Dataset 표준 메서드 ---------------- #

    def __len__(self) -> int:  # noqa: Dunder length
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # noqa: Dunder get item
        seq = list(self.sequences[idx])  # shallow copy
        labels = [-100] * len(seq)  # -100 → loss ignore_index

        cand_idx = [
            i for i, tid in enumerate(seq) if tid not in self._special_ids
        ]
        num_mask = min(len(cand_idx), max(1, int(len(cand_idx) * self.mask_prob)))
        for i in random.sample(cand_idx, num_mask):
            labels[i] = seq[i]
            rand = random.random()
            if rand < 0.8:
                seq[i] = self._mask_id
            elif rand < 0.9:
                # 10% 랜덤 토큰 치환(특수 토큰 제외 범위)
                lo = max(self._special_ids) + 1
                seq[i] = random.randint(lo, self.tokenizer.vocab_size - 1)
            # else: 그대로 두어 10%는 "원본 유지"

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "attention_mask": torch.ones(len(seq), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
