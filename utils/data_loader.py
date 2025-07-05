# utils/data_loader.py
# -*- coding: utf-8 -*-
"""
candump → 표준 DataFrame 변환 (가변 DLC, 무패딩) – 새 tokenizer 호환
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

_LINE_RE = re.compile(
    r"\((?P<ts>\d+\.\d+)\)\s+\w+\s+(?P<cid>[0-9A-F]+)#(?P<payload>[0-9A-F]*)"
)


def _parse(line: str) -> Optional[Dict]:
    m = _LINE_RE.match(line)
    if not m:
        return None
    ts = float(m.group("ts"))
    cid = m.group("cid")
    payload_hex = m.group("payload")
    data = [
        payload_hex[i : i + 2].upper()
        for i in range(0, len(payload_hex), 2)
        if payload_hex
    ]
    return {
        "Timestamp": ts,
        "CAN_ID": cid,
        "DLC": len(data),
        "Data": data,  # 패딩 X – Sequencer 단계에서 VOID 패딩
        "Label": 0,  # 사전 훈련용 기본 = 정상
    }


def load_can_data(path: str | Path, dataset_type: str = "candump") -> pd.DataFrame:
    """
    candump 로그 → 표준 스키마 DataFrame
    Columns: Timestamp | CAN_ID | DLC | Data(List[str]) | Label
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    rows: List[Dict] = []
    with path.open(encoding="utf-8") as fp:
        total = sum(1 for _ in fp)
        fp.seek(0)
        for line in tqdm(fp, total=total, desc=f"Parsing {path.name}"):
            rec = _parse(line)
            if rec:
                rows.append(rec)

    df = pd.DataFrame(rows).astype(
        {
            "Timestamp": "float64",
            "CAN_ID": "string",
            "DLC": "int16",
            "Data": "object",
            "Label": "int8",
        }
    )
    return df[["Timestamp", "CAN_ID", "DLC", "Data", "Label"]]
