# core/tokenizer.py
# -*- coding: utf-8 -*-
"""
UGRP 프로젝트 – CAN 버스 IDS 연구용 토크나이저·시퀀서 모듈

● Jo & Kim (2024)의 '오프셋 기반 통합 어휘집' 방식을 구현
● 코드 식별자‧API는 영어, 주석‧docstring은 한국어
● PEP 8 + 타입 힌트 준수
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


# -------------------------- 1. 공용 상수 -------------------------- #

class SpecialToken(str, Enum):
    PAD = "<PAD>"
    UNK = "<UNK>"
    MASK = "<MASK>"
    VOID = "<VOID>"
    CLS = "<CLS>"
    SEP = "<SEP>"


DEFAULT_DATA_TOKENS: Sequence[str] = tuple(f"{i:02X}" for i in range(256))


def _ensure_hex_bytes(data_bytes: Sequence[str]) -> List[str]:
    """16진수 2자리로 이뤄진 페이로드 바이트 검증/정규화"""
    cleaned: List[str] = []
    for b in data_bytes:
        b_up = b.upper()
        if (
            len(b_up) != 2
            or any(ch not in "0123456789ABCDEF" for ch in b_up)
        ):
            raise ValueError(f"잘못된 데이터 바이트: {b_up}")
        cleaned.append(b_up)
    return cleaned


# ------------------------ 2. CANTokenizer ------------------------ #

class CANTokenizer:
    """Jo & Kim (2024) 방식의 토크나이저"""

    def __init__(
        self,
        id_offset: int | None = None,
        include_cls_sep: bool = False,
    ) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        # 2‑1) 특수 토큰 추가
        self._add_tokens(list(SpecialToken))
        if not include_cls_sep:                       # CLS/SEP 제거(선택)
            self._drop_tokens([SpecialToken.CLS, SpecialToken.SEP])

        # 2‑2) 00~FF 페이로드 토큰 추가
        self._add_tokens(DEFAULT_DATA_TOKENS)

        # 2‑3) ID 오프셋 결정
        self._id_offset: int = (
            id_offset if id_offset is not None else len(self.token_to_id)
        )

    # ---------- 사전 관리 내부 메서드 ---------- #
    def _add_tokens(self, tokens: Iterable[str]) -> None:
        for tk in tokens:
            if tk not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[tk] = idx
                self.id_to_token[idx] = tk

    def _drop_tokens(self, tokens: Iterable[str]) -> None:
        for tk in tokens:
            if tk in self.token_to_id:
                idx = self.token_to_id.pop(tk)
                self.id_to_token.pop(idx, None)

    # ---------- 공개 프로퍼티 ---------- #
    @property
    def id_offset(self) -> int:
        return self._id_offset

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # ---------- 주요 API ---------- #
    def build_vocab(self, df: pd.DataFrame) -> None:
        """DataFrame 내 고유 CAN ID ↗ ID 토큰 추가"""
        uniq_ids = sorted(
            {str(can_id).upper() for can_id in df["CAN_ID"].unique()},
            key=lambda can_id: int(can_id, 16),
        )
        id_tokens = [str(int(i, 16) + self._id_offset) for i in uniq_ids]
        self._add_tokens(id_tokens)

    def save_vocab(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(self.token_to_id, fp, ensure_ascii=False, indent=4)

    def load_vocab(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fp:
            self.token_to_id = json.load(fp)
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, tokens: Sequence[str]) -> List[int]:
        unk = self.token_to_id[SpecialToken.UNK]
        return [self.token_to_id.get(tk, unk) for tk in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.id_to_token.get(i, SpecialToken.UNK) for i in ids]


# ------------------------ 3. CANSequencer ------------------------ #

class CANSequencer:
    """
    CAN 프레임 → 고정 길이 시퀀스로 변환 (슬라이딩 윈도우)
    """

    def __init__(
        self,
        tokenizer: CANTokenizer,
        seq_len: int = 128,
        stride: int = 1,
        pad_to_seq_len: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.stride = stride
        self.pad_to_seq_len = pad_to_seq_len

    # ------------------ 내부 헬퍼 ------------------ #
    def _frame_to_tokens(self, can_id: str, payload: Sequence[str]) -> List[str]:
        id_token = str(int(can_id, 16) + self.tokenizer.id_offset)
        payload_clean = _ensure_hex_bytes(payload)

        if len(payload_clean) > 8:
            raise ValueError("CAN Payload 길이는 최대 8 byte입니다.")
        if len(payload_clean) < 8:
            payload_clean += [SpecialToken.VOID] * (8 - len(payload_clean))

        return [id_token, *payload_clean]

    # ------------------ 공개 API ------------------ #
    def transform(self, df: pd.DataFrame) -> List[List[int]]:
        """DataFrame → 정수 시퀀스 목록"""

        # 1) 프레임 단위 토큰화
        frame_tokens = [
            self._frame_to_tokens(cid, pl)
            for cid, pl in zip(df["CAN_ID"].values, df["Data"].values)
        ]

        # 2) 토큰 스트림 (NumPy 배열) ★ FIX: list → ndarray 유지
        token_stream = np.fromiter(
            (tk for frame in frame_tokens for tk in frame),
            dtype=object,
        )

        # 3) 슬라이딩 윈도우 생성
        sequences: List[List[str]] = []
        for start in range(0, len(token_stream) - self.seq_len + 1, self.stride):
            window = token_stream[start : start + self.seq_len].tolist()
            sequences.append(window)

        # 4) PAD & 인코딩
        if self.pad_to_seq_len and sequences:
            pad_id = self.tokenizer.token_to_id[SpecialToken.PAD]
            seq_np = np.full(
                (len(sequences), self.seq_len), pad_id, dtype=np.int32
            )
            for i, seq in enumerate(sequences):
                enc = self.tokenizer.encode(seq)
                seq_np[i, : len(enc)] = enc
            return seq_np.tolist()

        return [self.tokenizer.encode(seq) for seq in sequences]


# ---------------- 4. 모듈 단독 실행 시 테스트 ---------------- #

if __name__ == "__main__":
    sample = {
        "CAN_ID": ["01A", "02B", "01A", "03C"],
        "Data": [
            ["11", "22", "33", "44", "55", "66", "77", "88"],
            ["AA", "BB", "CC", "DD", "EE", "FF", "00", "11"],
            ["99", "88", "77", "66", "55", "44", "33", "22"],
            ["DE", "AD", "BE", "EF", "CA", "FE", "BA", "BE"],
        ],
    }
    df_sample = pd.DataFrame(sample)

    tokenizer = CANTokenizer()
    tokenizer.build_vocab(df_sample)
    print(f"Vocab size: {tokenizer.vocab_size}")

    sequencer = CANSequencer(tokenizer, seq_len=10, stride=1)
    encoded_sequences = sequencer.transform(df_sample)

    # 확인
    print("첫 번째 시퀀스 디코딩:")
    print(tokenizer.decode(encoded_sequences[0]))
