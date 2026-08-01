# CAN-BERT 기반 CAN 버스 침입 탐지 연구 프로토타입

`candump` 로그의 CAN ID와 페이로드를 토큰화해 BERT masked language model(MLM)을 사전 학습하고, Benign·DoS·Fuzzy·Malfunction 네 클래스로 미세 조정하는 연구용 코드입니다. 공개 범위는 teacher model 실험 파이프라인까지이며, 차량 탑재용 제품이나 검증된 실시간 탐지기를 제공하지 않습니다.

## 구현 범위

- `candump` 행 파싱과 CAN ID·0–8 byte 페이로드 검증
- 특수 토큰, 256개 byte 토큰, 데이터 기반 CAN ID 토큰으로 구성한 어휘
- 고정 길이 시퀀스와 MLM 동적 마스킹
- Transformers 기반 BERT MLM teacher와 4-class classifier
- 데이터 병합·분할, 어휘 생성, 사전 학습, 미세 조정 스크립트

학습 데이터, 학습된 가중치, 성능 결과, student model, knowledge distillation, ONNX export 및 배포 런타임은 이 저장소에 포함하지 않습니다.

## 코드 구성

```text
core/       tokenizer, sequencer, MLM/classification datasets
models/     MLM teacher and four-class classifier
scripts/    data preparation, pretraining, and fine-tuning entry points
utils/      candump parser and data loading
tests/      synthetic parser, tokenizer, dataset, and model smoke tests
```

처리 흐름은 다음과 같습니다.

```text
candump logs
  -> scripts.aggregate_data
  -> scripts.build_vocab
  -> scripts.split_data
  -> scripts.pretrain
  -> scripts.prepare_finetune_data
  -> scripts.finetune
```

## 환경과 최소 검증

- Python 3.10 이상
- 학습에는 PyTorch와 `requirements.txt`의 패키지 필요
- 문서화한 분산 사전 학습 경로는 Linux, CUDA, NCCL 환경을 전제로 함

GPU 환경에 맞는 PyTorch를 [공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라 먼저 설치한 뒤 나머지 의존성을 설치합니다.

```bash
python -m venv .venv
# activate the environment for your shell
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements.txt
```

데이터가 없어도 다음 검사를 실행할 수 있습니다.

```bash
python -m compileall core models scripts utils
python -m unittest discover -s tests -v
```

테스트는 작은 합성 입력으로 parser/tokenizer/dataset 동작과 teacher model의 CPU forward shape를 확인합니다. 실제 데이터 학습이나 성능 재현을 대신하지 않습니다.

## 데이터와 학습 산출물

원본 CAN 로그는 데이터 사용 권한과 크기 때문에 공개 저장소에 넣지 않습니다. `dataset/`, 생성된 `data/`, 어휘와 모델 체크포인트가 저장되는 `checkpoints/`, W&B 로그는 Git에서 제외됩니다. 어휘는 사용자가 권한을 보유한 입력 데이터에서 다시 생성해야 합니다.

예상 입력 행:

```text
(1613599955.394625) can0 0C8#0000000000000000
```

기본 데이터 준비 명령:

```bash
python -m scripts.aggregate_data
python -m scripts.build_vocab
python -m scripts.split_data
```

생성한 어휘와 입력 경로를 명시해 단일 프로세스 분산 실행 흐름을 확인할 수 있습니다.

```bash
torchrun --standalone --nproc_per_node=1 -m scripts.pretrain \
  --data_path data/aggregated_parts/part_00 \
  --vocab_path checkpoints/vocab.json \
  --output_dir checkpoints \
  --epochs 1 --batch_size 32 --num_workers 4
```

4-class 데이터 분할과 미세 조정:

```bash
python -m scripts.prepare_finetune_data
python -m scripts.finetune \
  --train_data_dir data/finetune_data/train \
  --val_data_dir data/finetune_data/validation \
  --test_data_dir data/finetune_data/test \
  --vocab_path checkpoints/vocab.json \
  --resume_from_checkpoint checkpoints/can-bert-pretrained-epoch-1.pt \
  --output_dir checkpoints
```

## Project lineage

이 저장소는 public teacher-model prototype입니다. 2025년 DGIST UGRP 팀 연구에서는 별도 산출물로 binary student model과 ONNX 평가까지 확장했습니다. 그 후속 package는 source–model parity, 팀 기여 범위, 데이터·모델 재배포 권리 검토가 끝나지 않아 이 저장소에 포함하지 않습니다. 따라서 이 저장소가 teacher-to-student 전체 파이프라인을 제공하거나 후속 결과를 재현한다고 해석해서는 안 됩니다.

## 검증 범위와 한계

- CI와 로컬 smoke test는 합성 입력의 파싱·토큰화·모델 forward만 검사합니다.
- 공개 데이터, 가중치, 실험 로그가 없으므로 정확도, F1, 지연 시간 또는 일반화 성능을 주장하지 않습니다.
- CPU 단독 학습과 대규모 데이터 학습은 지원·검증된 실행 경로가 아닙니다.
- 이 코드는 안전이 중요한 실제 차량 시스템에 바로 배포하기 위한 구현이 아닙니다.
- 저장소에는 명시적인 소프트웨어 라이선스가 없습니다. 팀·원 코드 권리 확인 전에는 재사용 허가를 전제로 하지 마세요.

## References

1. Natasha Alkhatib et al., [“CAN-BERT do it? Controller Area Network Intrusion Detection System based on BERT Language Model”](https://arxiv.org/abs/2210.09439), 2022.
2. Hyunjun Jo and Deok-Hwan Kim, [“Intrusion Detection Using Transformer in Controller Area Network”](https://doi.org/10.1109/ACCESS.2024.3452634), *IEEE Access*, 2024.
3. Pengzhou Cheng et al., [“LSF-IDM: Automotive Intrusion Detection Model with Lightweight Attribution and Semantic Fusion”](https://arxiv.org/abs/2308.01237), 2023.

앞의 두 연구는 CAN 메시지의 language-model representation과 Transformer 기반 탐지를 설계할 때 참고했습니다. 세 번째 연구의 distillation 접근은 후속 팀 연구 방향의 참고 자료이며, 현재 공개 코드의 구현 범위가 아닙니다.
