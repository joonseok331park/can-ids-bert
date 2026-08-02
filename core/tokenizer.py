# core/tokenizer.py
# -*- coding: utf-8 -*-
"""Tokenizer and fixed-length sequencer for classic CAN frames."""

from __future__ import annotations

import json
from enum import Enum
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


class SpecialToken(str, Enum):
    PAD = "<PAD>"
    UNK = "<UNK>"
    MASK = "<MASK>"
    VOID = "<VOID>"
    CLS = "<CLS>"
    SEP = "<SEP>"


DEFAULT_DATA_TOKENS: Sequence[str] = tuple(f"{i:02X}" for i in range(256))


def _ensure_hex_bytes(data_bytes: Sequence[str]) -> List[str]:
    """Validate and canonicalize payload bytes."""
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


class CANTokenizer:
    """Offset-based vocabulary for payload bytes and observed CAN IDs."""

    def __init__(
        self,
        id_offset: int | None = None,
        include_cls_sep: bool = False,
    ) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        self._add_tokens(list(SpecialToken))
        if not include_cls_sep:
            self._drop_tokens([SpecialToken.CLS, SpecialToken.SEP])

        self._add_tokens(DEFAULT_DATA_TOKENS)

        self._id_offset: int = (
            id_offset if id_offset is not None else len(self.token_to_id)
        )

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

    @property
    def id_offset(self) -> int:
        return self._id_offset

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def add_can_ids(self, can_ids: Iterable[str]) -> int:
        """Validate CAN IDs and add their offset tokens in numeric order.

        Equivalent spellings such as ``0a`` and ``00A`` map to the same token.
        The return value is the number of tokens newly added to the vocabulary.
        """
        numeric_ids: set[int] = set()
        for value in can_ids:
            canonical = str(value).strip().upper()
            if not canonical or len(canonical) > 8:
                raise ValueError(f"Invalid CAN ID: {value!r}")
            if any(ch not in "0123456789ABCDEF" for ch in canonical):
                raise ValueError(f"Invalid CAN ID: {value!r}")
            numeric = int(canonical, 16)
            if numeric > 0x1FFFFFFF:
                raise ValueError(f"CAN ID exceeds the 29-bit limit: {value!r}")
            numeric_ids.add(numeric)

        before = self.vocab_size
        self._add_tokens(
            str(numeric + self._id_offset) for numeric in sorted(numeric_ids)
        )
        return self.vocab_size - before

    def build_vocab(self, df: pd.DataFrame) -> None:
        """Add all CAN IDs from a parsed DataFrame."""
        self.add_can_ids(df["CAN_ID"].unique())

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


class CANSequencer:
    """Convert parsed frames into fixed-length sliding token windows."""

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

    def _frame_to_tokens(self, can_id: str, payload: Sequence[str]) -> List[str]:
        id_token = str(int(can_id, 16) + self.tokenizer.id_offset)
        payload_clean = _ensure_hex_bytes(payload)

        if len(payload_clean) > 8:
            raise ValueError("CAN Payload 길이는 최대 8 byte입니다.")
        if len(payload_clean) < 8:
            payload_clean += [SpecialToken.VOID] * (8 - len(payload_clean))

        return [id_token, *payload_clean]

    def transform(self, df: pd.DataFrame) -> List[List[int]]:
        """Convert a parsed DataFrame into integer token sequences."""
        frame_tokens = [
            self._frame_to_tokens(cid, pl)
            for cid, pl in zip(df["CAN_ID"].values, df["Data"].values)
        ]

        token_stream = np.fromiter(
            (tk for frame in frame_tokens for tk in frame),
            dtype=object,
        )

        sequences: List[List[str]] = []
        for start in range(0, len(token_stream) - self.seq_len + 1, self.stride):
            window = token_stream[start : start + self.seq_len].tolist()
            sequences.append(window)

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

    print("첫 번째 시퀀스 디코딩:")
    print(tokenizer.decode(encoded_sequences[0]))
