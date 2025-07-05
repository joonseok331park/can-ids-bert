# scripts/split_data_mp.py
"""
멀티프로세싱 기반 candump 로그 분할 스크립트.

* 한 번의 파일 스캔으로 모든 줄의 바이트 오프셋을 기록
* offset 리스트를 N개 파티션으로 나누어 프로세스 풀에 배분
* 각 워커는 input 파일을 seek() 후 담당 영역만 읽어 part_XX에 기록
"""

from __future__ import annotations

import os
import multiprocessing as mp
from typing import List, Tuple
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────────────────────
# 파라미터
# ────────────────────────────────────────────────────────────────────────────────
INPUT_FILE = os.path.join('data', 'HCRL_dataset', 'train_aggregated.log')
OUTPUT_DIR = os.path.join('data', 'aggregated_parts')
LINES_PER_FILE = 5_000_000          # 5 M lines
NUM_WORKERS = max(mp.cpu_count() - 1, 1)  # 시스템 코어 수 - 1 (여유 코어)

# ────────────────────────────────────────────────────────────────────────────────
# 유틸리티
# ────────────────────────────────────────────────────────────────────────────────
def collect_offsets(file_path: str) -> List[int]:
    """
    파일을 1회 스캔하면서 **행의 시작 바이트 오프셋**을 모두 수집한다.
    반환값: offset 리스트 (len == 총 라인 수 + 1)  ─ 맨 끝 바이트 포함
    """
    offsets: List[int] = [0]
    with open(file_path, 'rb') as fp:                      # 바이너리 모드(오프셋 정확성)
        while fp.readline():               # readline() → '\n' 포함
            offsets.append(fp.tell())      # 다음 라인의 시작 위치
    return offsets


def build_tasks(offsets: List[int]) -> List[Tuple[int, int, str]]:
    """
    오프셋 리스트를 기반으로 (start, end, part_path) 튜플 목록 생성.
    각 튜플은 워커가 담당할 바이트 범위를 의미.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_lines = len(offsets) - 1
    part_count = (total_lines + LINES_PER_FILE - 1) // LINES_PER_FILE

    tasks: List[Tuple[int, int, str]] = []
    for idx in range(part_count):
        start_idx = idx * LINES_PER_FILE               # line index
        end_idx = min((idx + 1) * LINES_PER_FILE, total_lines)
        start_byte = offsets[start_idx]
        end_byte = offsets[end_idx]
        part_path = os.path.join(OUTPUT_DIR, f'part_{idx:02d}')
        tasks.append((start_byte, end_byte, part_path))
    return tasks


def split_worker(args: Tuple[int, int, str]) -> None:
    """
    워커 프로세스: input 파일의 [start_byte:end_byte) 범위를 읽어 part 파일 생성.
    """
    start_byte, end_byte, part_path = args
    with open(INPUT_FILE, 'rb') as fin, open(part_path, 'wb') as fout:
        fin.seek(start_byte)
        remaining = end_byte - start_byte
        # 4 MB 버퍼 단위 copy
        buffer_size = 4 * 1024 * 1024
        while remaining > 0:
            chunk = fin.read(min(buffer_size, remaining))
            if not chunk:
                break
            fout.write(chunk)
            remaining -= len(chunk)


# ────────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Step 1/3 ▶ '{INPUT_FILE}' 행 오프셋 수집 중…")
    offsets = collect_offsets(INPUT_FILE)

    total_lines = len(offsets) - 1
    print(f"    ↳ 총 {total_lines:,} lines, {len(offsets) - 1:,} offsets 수집 완료")

    print("Step 2/3 ▶ 워커 작업 목록 생성…")
    tasks = build_tasks(offsets)
    print(f"    ↳ part 파일 수: {len(tasks):,} (worker {NUM_WORKERS}개 사용)")

    print("Step 3/3 ▶ 멀티프로세싱 분할 시작…")
    with mp.Pool(processes=NUM_WORKERS) as pool:
        list(tqdm(pool.imap_unordered(split_worker, tasks),
                  total=len(tasks),
                  desc="파일 분할 진행"))
    print("✔ 분할 완료!  결과는:", OUTPUT_DIR)


if __name__ == '__main__':
    mp.freeze_support()        # Windows 지원
    main()
