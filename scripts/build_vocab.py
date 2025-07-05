# scripts/build_vocab.py
# -*- coding: utf-8 -*-
"""
대용량 candump 로그 → vocab.json (동적 id_offset 사용)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Set

from tqdm import tqdm

from core.tokenizer import CANTokenizer

_LINE_RE = re.compile(r"\(\d+\.\d+\)\s+\w+\s+([0-9A-F]+)#")


def _collect_unique_ids(path: Path) -> Set[str]:
    uniq: Set[str] = set()
    with path.open(encoding="utf-8") as fp:
        total = sum(1 for _ in fp)
        fp.seek(0)
        for line in tqdm(fp, total=total, desc="Scanning IDs"):
            m = _LINE_RE.match(line)
            if m:
                uniq.add(m.group(1))
    return uniq


def main() -> None:
    data_file = Path("data/HCRL_dataset/train_aggregated.log")
    out_dir = Path("checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = out_dir / "vocab.json"

    ids = _collect_unique_ids(data_file)
    print(f"[INFO] unique CAN IDs: {len(ids):,}")

    tok = CANTokenizer()  # 00~FF 데이터 토큰 및 특수 토큰 자동 포함
    tok._add_tokens(  # 내부 메서드 사용 허용 – 프로젝트 컨벤션
        [str(int(cid, 16) + tok.id_offset) for cid in ids]
    )
    tok.save_vocab(vocab_path)
    print(f"[INFO] vocab saved → {vocab_path}  (size={tok.vocab_size:,})")


if __name__ == "__main__":
    main()
