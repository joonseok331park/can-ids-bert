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
    r"\((?P<ts>\d+\.\d+)\)\s+"
    r"[A-Z0-9_.:-]+\s+"
    r"(?P<cid>[0-9A-F]{1,8})#(?P<payload>[0-9A-F]*)",
    re.IGNORECASE,
)


def _parse(line: str) -> Optional[Dict]:
    m = _LINE_RE.fullmatch(line.strip())
    if not m:
        return None
    ts = float(m.group("ts"))
    cid = m.group("cid").upper()
    payload_hex = m.group("payload")
    if int(cid, 16) > 0x1FFFFFFF:
        return None
    if len(payload_hex) % 2 or len(payload_hex) > 16:
        return None
    data = [
        payload_hex[i : i + 2].upper()
        for i in range(0, len(payload_hex), 2)
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


def load_classification_data(data_dir: str | Path) -> pd.DataFrame:
    """
    분류용 데이터 디렉토리 로드 (여러 클래스 파일들)
    
    Args:
        data_dir: 데이터 디렉토리 경로 (Benign_*.log, DoS_*.log, Fuzzy_*.log, Malfunction_*.log 파일들 포함)
        
    Returns:
        pd.DataFrame: 모든 파일의 데이터가 합쳐진 DataFrame with proper labels
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_dir}")
    
    # 클래스 레이블 매핑
    class_mapping = {
        'Benign': 0,
        'DoS': 1, 
        'Fuzzy': 2,
        'Malfunction': 3
    }
    
    all_data = []
    
    # 각 클래스별 파일들 처리
    for class_name, label in class_mapping.items():
        class_files = list(data_dir.glob(f"{class_name}_*.log"))
        print(f"Found {len(class_files)} files for {class_name} class")
        
        for file_path in class_files:
            try:
                # 개별 파일 로드
                df = load_can_data(file_path)
                # 라벨 설정
                df['Label'] = label
                all_data.append(df)
                print(f"Loaded {len(df)} records from {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
    
    if not all_data:
        raise ValueError(f"No valid data files found in {data_dir}")
    
    # 모든 데이터 합치기
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total loaded records: {len(combined_df)}")
    
    # 클래스별 분포 출력
    class_counts = combined_df['Label'].value_counts().sort_index()
    print("Class distribution:")
    for label, count in class_counts.items():
        class_name = [k for k, v in class_mapping.items() if v == label][0]
        print(f"  {class_name} ({label}): {count} records")
    
    return combined_df
