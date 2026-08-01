"""File-boundary-safe dataset for four-class fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from core.classes import CLASS_LABELS
from core.tokenizer import CANSequencer, CANTokenizer, SpecialToken
from utils.data_loader import load_classification_data


class ClassificationDataset(Dataset):
    """Create windows within individual labeled source files.

    ``short_file_policy="error"`` rejects a file that cannot form one full
    frame window. ``"skip"`` records and skips such files, but still rejects a
    dataset when every source file is skipped.
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: CANTokenizer,
        seq_len: int = 126,
        short_file_policy: str = "skip",
    ) -> None:
        if seq_len < 9:
            raise ValueError("seq_len must be at least 9 tokens")
        if short_file_policy not in {"error", "skip"}:
            raise ValueError("short_file_policy must be 'error' or 'skip'")

        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.short_file_policy = short_file_policy
        self.df = load_classification_data(self.data_dir)
        self.skipped_short_files: list[str] = []
        self.sequence_sources: list[dict[str, int | str]] = []
        self.sequences, self.labels = self._make_sequences()
        self.sequence_count_by_label = {
            label: self.labels.count(label) for label in CLASS_LABELS
        }
        self.sequence_count_by_file = {
            source_file: sum(
                item["source_file"] == source_file
                for item in self.sequence_sources
            )
            for source_file in sorted(
                self.df["SourceFile"].unique(), key=lambda value: value.casefold()
            )
        }
        if self.skipped_short_files:
            print(
                f"Skipped {len(self.skipped_short_files)} short source file(s): "
                + ", ".join(self.skipped_short_files)
            )
        print(
            f"Loaded {self.df['SourceFile'].nunique()} source file(s); "
            f"generated {len(self.sequences)} sequence(s)"
        )

    def _make_sequences(self) -> Tuple[List[List[int]], List[int]]:
        frames_per_sequence = self.seq_len // 9
        sequencer = CANSequencer(self.tokenizer, seq_len=self.seq_len)
        sequences: List[List[int]] = []
        labels: List[int] = []

        source_files = sorted(
            self.df["SourceFile"].unique(), key=lambda value: value.casefold()
        )
        for source_file in source_files:
            group = self.df[self.df["SourceFile"] == source_file].sort_values(
                "SourceLine", kind="stable"
            )
            file_labels = group["Label"].unique().tolist()
            if len(file_labels) != 1:
                raise ValueError(
                    f"Source file {source_file!r} has multiple labels: {file_labels}"
                )
            if len(group) < frames_per_sequence:
                message = (
                    f"Source file {source_file!r} has {len(group)} valid frames; "
                    f"at least {frames_per_sequence} are required"
                )
                if self.short_file_policy == "error":
                    raise ValueError(message)
                self.skipped_short_files.append(source_file)
                continue

            frame_tokens = [
                sequencer._frame_to_tokens(row.CAN_ID, row.Data)
                for row in group.itertuples(index=False)
            ]
            source_lines = group["SourceLine"].tolist()
            for start in range(len(frame_tokens) - frames_per_sequence + 1):
                tokens = [
                    token
                    for frame in frame_tokens[start : start + frames_per_sequence]
                    for token in frame
                ]
                encoded = self.tokenizer.encode(tokens[: self.seq_len])
                pad_id = self.tokenizer.token_to_id[SpecialToken.PAD]
                encoded.extend([pad_id] * (self.seq_len - len(encoded)))
                sequences.append(encoded)
                labels.append(int(file_labels[0]))
                self.sequence_sources.append(
                    {
                        "source_file": source_file,
                        "start_line": int(source_lines[start]),
                        "end_line": int(
                            source_lines[start + frames_per_sequence - 1]
                        ),
                    }
                )

        if not sequences:
            detail = (
                f"; skipped short files: {', '.join(self.skipped_short_files)}"
                if self.skipped_short_files
                else ""
            )
            raise ValueError(f"No classification sequences were generated{detail}")
        return sequences, labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        pad_id = self.tokenizer.token_to_id[SpecialToken.PAD]
        attention_mask = [int(token_id != pad_id) for token_id in sequence]
        return {
            "input_ids": torch.tensor(sequence, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
