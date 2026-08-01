"""Build a tokenizer vocabulary with the canonical candump parser."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

from core.tokenizer import CANTokenizer
from utils.data_loader import ParseStats, parse_candump_line


def _collect_unique_ids(path: Path) -> tuple[Set[str], ParseStats]:
    """Return canonical IDs and scan counts, rejecting all-invalid input."""
    if not path.is_file():
        raise FileNotFoundError(path)

    unique_ids: Set[str] = set()
    total_lines = 0
    rejected_lines = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            total_lines += 1
            record = parse_candump_line(line)
            if record is None:
                rejected_lines += 1
                continue
            unique_ids.add(record["CAN_ID"])

    stats = ParseStats(
        total_lines=total_lines,
        valid_lines=total_lines - rejected_lines,
        rejected_lines=rejected_lines,
    )
    if not unique_ids:
        raise ValueError(
            f"No valid candump frames in {path} "
            f"(total={stats.total_lines}, rejected={stats.rejected_lines})"
        )
    return unique_ids, stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/HCRL_dataset/train_aggregated.log"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/vocab.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ids, stats = _collect_unique_ids(args.data_file)
    print(
        "[INFO] parser counts: "
        f"valid={stats.valid_lines:,}, rejected={stats.rejected_lines:,}, "
        f"total={stats.total_lines:,}"
    )

    tokenizer = CANTokenizer()
    tokenizer.add_can_ids(ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_vocab(str(args.output))
    print(
        f"[INFO] vocab saved -> {args.output} "
        f"(CAN IDs={len(ids):,}, size={tokenizer.vocab_size:,})"
    )


if __name__ == "__main__":
    main()
