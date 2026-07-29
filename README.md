# CAN-BERT 기반 CAN 버스 침입 탐지 연구 프로토타입

차량 내부 네트워크의 `candump` 로그를 파싱하고, CAN ID와 페이로드를 토큰화해 BERT 기반 Masked Language Model(MLM)을 사전 학습한 뒤 4개 클래스 분류로 미세 조정하는 연구용 코드입니다.

이 저장소는 데이터 처리와 모델 실험을 위한 프로토타입입니다. 학습 데이터, 학습 완료 가중치, 벤치마크 결과, 차량 탑재용 런타임은 포함하지 않으며 실제 차량 환경에서의 성능이나 실시간 배포 가능성을 주장하지 않습니다.

## 현재 구현 범위

| 영역 | 구현 내용 |
| --- | --- |
| 로그 처리 | `candump` 형식의 timestamp, CAN ID, payload 파싱 |
| 토큰화 | 특수 토큰, `00`–`FF` 바이트 토큰, CAN ID 오프셋, `<VOID>` 패딩 |
| 시퀀싱 | CAN 프레임을 고정 길이 토큰 시퀀스로 변환 |
| 사전 학습 | PyTorch·Transformers 기반 CAN-BERT 교사 모델과 MLM 데이터셋 |
| 미세 조정 | Benign, DoS, Fuzzy, Malfunction 4개 클래스 분류 |
| 학습 보조 | 로그 병합, 어휘 생성, 데이터 분할, 체크포인트 저장, 오프라인 W&B 로깅 |

지식 증류와 경량 학생 모델은 참고 논문을 바탕으로 검토한 후속 방향이며, 현재 저장소에는 구현되어 있지 않습니다.

## 처리 흐름

```text
candump logs
  -> scripts.aggregate_data
  -> scripts.build_vocab
  -> scripts.split_data
  -> scripts.pretrain
  -> scripts.prepare_finetune_data
  -> scripts.finetune
```

주요 모듈은 다음과 같습니다.

```text
core/
  tokenizer.py                 CAN 토크나이저와 시퀀서
  dataset.py                   MLM 데이터셋
  classification_dataset.py    4클래스 분류 데이터셋
models/
  teacher.py                   MLM 교사 모델
  teacher_classifier.py        분류 헤드가 포함된 교사 모델
scripts/
  aggregate_data.py            정상 로그 병합
  build_vocab.py               데이터 기반 어휘 생성
  split_data.py                대용량 로그 분할
  pretrain.py                  MLM 사전 학습
  prepare_finetune_data.py     분류용 데이터 분할
  finetune.py                  4클래스 미세 조정 및 테스트 평가
utils/
  data_loader.py               candump 파서
```

## 환경 설정

### 요구 환경

- Python 3.10 이상
- Git
- 사전 학습을 실행하려면 CUDA를 사용할 수 있는 NVIDIA GPU 권장
- `torchrun` 분산 학습 경로는 Linux와 NCCL 환경을 전제로 함

GPU 환경에 맞는 PyTorch 설치 명령은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 확인하세요.

### 저장소 설치

```bash
git clone https://github.com/joonseok331park/can-ids-bert.git
cd can-ids-bert
python -m venv .venv
```

가상 환경을 활성화합니다.

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

의존성을 설치합니다. 아래 `torch` 명령은 기본 예시이며, CUDA 학습 환경에서는 앞서 안내한 공식 설치 명령으로 대체할 수 있습니다.

```bash
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

데이터 없이 소스의 Python 문법을 확인하려면 다음 명령을 사용할 수 있습니다.

```bash
python -m compileall core models scripts utils
```

## 데이터 준비

원본 데이터는 저장소에 포함되어 있지 않습니다. 사용 권한을 확보한 CAN-MIRGU 로그를 다음 구조에 배치해야 합니다.

```text
dataset/
└── CAN-MIRGU(train)/
    ├── Benign/
    │   └── Day_*/
    │       └── *.log
    └── Attack/
        ├── Real_attacks/
        │   └── *.log
        ├── Masquerade_attacks/
        │   └── *.log
        └── Suspension_attacks/
            └── *.log
```

파서는 다음과 같은 `candump` 로그 행을 입력으로 받습니다.

```text
(1613599955.394625) can0 0C8#0000000000000000
```

### 사전 학습 데이터 생성

아래 명령은 인자 없이 저장소의 기본 경로를 사용합니다.

```bash
python -m scripts.aggregate_data
python -m scripts.build_vocab
python -m scripts.split_data
```

생성되는 주요 산출물은 다음과 같습니다.

```text
data/HCRL_dataset/train_aggregated.log
data/aggregated_parts/part_00
data/aggregated_parts/part_01
...
checkpoints/vocab.json
```

저장소에 포함된 `checkpoints/vocab.json`은 코드 구조 확인용 어휘 스냅샷입니다. 실제 학습에서는 사용하는 데이터로 `scripts.build_vocab`을 다시 실행하는 편이 안전합니다.

## 학습

### MLM 사전 학습

`scripts.pretrain`은 단일 프로세스 실행과 `torchrun` 실행 인자를 모두 제공하지만, 현재 혼합 정밀도·체크포인트 경로는 CUDA/DDP 사용을 중심으로 작성되어 있습니다. 다음은 분할 데이터 하나를 GPU 한 장에서 확인하는 명령 예시입니다.

```bash
torchrun --standalone --nproc_per_node=1 -m scripts.pretrain --data_path data/aggregated_parts/part_00 --vocab_path checkpoints/vocab.json --output_dir checkpoints --epochs 1 --batch_size 32 --num_workers 4
```

저장되는 체크포인트 이름은 다음 형식입니다.

```text
checkpoints/can-bert-pretrained-epoch-1.pt
```

### 4클래스 미세 조정

먼저 원본 데이터 파일을 train, validation, test 디렉토리로 나눕니다.

```bash
python -m scripts.prepare_finetune_data
```

그다음 사전 학습 체크포인트를 지정해 분류 모델을 학습합니다.

```bash
python -m scripts.finetune --train_data_dir data/finetune_data/train --val_data_dir data/finetune_data/validation --test_data_dir data/finetune_data/test --vocab_path checkpoints/vocab.json --resume_from_checkpoint checkpoints/can-bert-pretrained-epoch-1.pt --output_dir checkpoints
```

`scripts.finetune`은 각 epoch의 학습·검증 지표를 출력하고, 마지막에 test split을 평가합니다. 가장 높은 validation weighted F1 체크포인트는 다음 경로에 저장됩니다.

```text
checkpoints/pilot-finetuned-best.pt
```

## 재현 범위와 제한

- `dataset/`, 생성된 `data/`, 학습 체크포인트(`*.pt`)는 Git 추적 대상이 아닙니다.
- 공개 저장소에는 학습 결과 수치가 없으므로 정확도, F1, 지연 시간에 대한 결과를 제시하지 않습니다.
- `scripts.run_full_training`은 여러 데이터 파트를 순회하기 위한 실험용 자동화 스크립트입니다. 위 빠른 시작에서는 입력과 체크포인트가 명시적인 `scripts.pretrain` 경로를 사용합니다.
- `scripts.evaluate`는 별도의 탐색용 평가 스크립트이며, 문서화된 경로에서는 `scripts.finetune`에 포함된 test 평가를 사용합니다.
- 이 코드는 연구용 프로토타입으로, 안전이 중요한 실제 차량 시스템에 바로 배포하기 위한 구현이 아닙니다.

## 참고 논문

1. Natasha Alkhatib, Maria Mushtaq, Hadi Ghauch, Jean-Luc Danger, [“CAN-BERT do it? Controller Area Network Intrusion Detection System based on BERT Language Model”](https://arxiv.org/abs/2210.09439), 2022.
2. Hyunjun Jo, Deok-Hwan Kim, [“Intrusion Detection Using Transformer in Controller Area Network”](https://doi.org/10.1109/ACCESS.2024.3452634), *IEEE Access*, vol. 12, pp. 121932–121946, 2024.
3. Pengzhou Cheng, Lei Hua, Haobin Jiang, Gongshen Liu, [“LSF-IDM: Automotive Intrusion Detection Model with Lightweight Attribution and Semantic Fusion”](https://arxiv.org/abs/2308.01237), 2023.

첫 번째와 두 번째 논문은 CAN 메시지를 언어 모델 입력으로 구성하고 MLM·Transformer 기반 이상 탐지를 탐색하는 데 참고했습니다. 세 번째 논문의 지식 증류 접근은 경량화 후속 연구 방향으로 참고했으며, 현재 코드의 완료 기능으로 표시하지 않습니다.
