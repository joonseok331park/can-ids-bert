"""Canonical candump parsing and deterministic dataset loading."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from core.classes import CLASS_NAMES, CLASS_TO_LABEL


_LINE_RE = re.compile(
    r"\((?P<ts>\d+\.\d+)\)\s+"
    r"[A-Z0-9_.:-]+\s+"
    r"(?P<cid>[0-9A-F]{1,8})#(?P<payload>[0-9A-F]*)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParseStats:
    """Line counts produced by a canonical candump scan."""

    total_lines: int
    valid_lines: int
    rejected_lines: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def parse_candump_line(line: str) -> Optional[Dict]:
    """Parse one complete classic-CAN candump line.

    IDs and payload bytes are returned in uppercase. Frames outside the 29-bit
    ID range, payloads longer than eight bytes, odd-length payloads, and lines
    with trailing text are rejected.
    """
    match = _LINE_RE.fullmatch(line.strip())
    if not match:
        return None

    can_id = match.group("cid").upper()
    payload_hex = match.group("payload").upper()
    if int(can_id, 16) > 0x1FFFFFFF:
        return None
    if len(payload_hex) % 2 or len(payload_hex) > 16:
        return None

    data = [
        payload_hex[index : index + 2]
        for index in range(0, len(payload_hex), 2)
    ]
    return {
        "Timestamp": float(match.group("ts")),
        "CAN_ID": can_id,
        "DLC": len(data),
        "Data": data,
        "Label": 0,
    }


def _parse(line: str) -> Optional[Dict]:
    """Backward-compatible alias for the public parser."""
    return parse_candump_line(line)


def load_can_data(path: str | Path, dataset_type: str = "candump") -> pd.DataFrame:
    """Load a candump file and reject files with no valid frames."""
    if dataset_type != "candump":
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: List[Dict] = []
    total_lines = 0
    rejected_lines = 0
    with path.open(encoding="utf-8") as stream:
        for source_line, line in enumerate(
            tqdm(stream, desc=f"Parsing {path.name}"), start=1
        ):
            total_lines += 1
            record = parse_candump_line(line)
            if record is None:
                rejected_lines += 1
                continue
            record["SourceLine"] = source_line
            rows.append(record)

    stats = ParseStats(
        total_lines=total_lines,
        valid_lines=len(rows),
        rejected_lines=rejected_lines,
    )
    if not rows:
        raise ValueError(
            f"No valid candump frames in {path} "
            f"(total={stats.total_lines}, rejected={stats.rejected_lines})"
        )

    frame = pd.DataFrame(rows).astype(
        {
            "Timestamp": "float64",
            "CAN_ID": "string",
            "DLC": "int16",
            "Data": "object",
            "Label": "int8",
            "SourceLine": "int64",
        }
    )
    frame.attrs["parse_stats"] = stats.to_dict()
    return frame[
        ["Timestamp", "CAN_ID", "DLC", "Data", "Label", "SourceLine"]
    ]


def _classification_label(path: Path) -> tuple[str, int]:
    matches = [
        (name, label)
        for name, label in CLASS_TO_LABEL.items()
        if path.name.casefold().startswith(f"{name}_".casefold())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Unclassified file {path.name!r}; expected one of "
            + ", ".join(f"{name}_*.log" for name in CLASS_NAMES)
        )
    return matches[0]


def load_classification_data(data_dir: str | Path) -> pd.DataFrame:
    """Load labeled files in stable path order while retaining provenance."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    files = sorted(
        data_dir.rglob("*.log"),
        key=lambda path: path.relative_to(data_dir).as_posix().casefold(),
    )
    if not files:
        raise ValueError(f"No .log files found in {data_dir}")

    loaded: list[pd.DataFrame] = []
    for file_path in files:
        _, label = _classification_label(file_path)
        frame = load_can_data(file_path)
        frame["Label"] = label
        frame["SourceFile"] = file_path.relative_to(data_dir).as_posix()
        loaded.append(frame)

    combined = pd.concat(loaded, ignore_index=True)
    combined = combined.sort_values(
        ["SourceFile", "SourceLine"], kind="stable"
    ).reset_index(drop=True)
    return combined[
        [
            "Timestamp",
            "CAN_ID",
            "DLC",
            "Data",
            "Label",
            "SourceFile",
            "SourceLine",
        ]
    ]
