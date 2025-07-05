# scripts/aggregate_data.py
import os, glob
from tqdm import tqdm

def main():
    source_dir = os.path.join('dataset', 'CAN-MIRGU(train)', 'Benign')
    output_dir = os.path.join('data', 'HCRL_dataset')
    output_filename = os.path.join(output_dir, 'train_aggregated.log')

    os.makedirs(output_dir, exist_ok=True)
    log_files = glob.glob(os.path.join(source_dir, '**', '*.log'), recursive=True)

    if not log_files:
        print(f"오류: '{source_dir}'에서 .log 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(log_files)}개 .log 파일 병합 시작 -> {output_filename}")
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        for filename in tqdm(log_files, desc="파일 병합 중"):
            with open(filename, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read().strip() + '\n')
    print("병합 완료!")

if __name__ == '__main__':
    main()