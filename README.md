# CAN-BERT 기반 CAN 버스 침입 탐지 연구 프로토타입

`candump` 로그의 CAN ID와 페이로드를 토큰화해 BERT masked language model(MLM)을 사전 학습하고, Benign·DoS·Fuzzy·Malfunction 네 클래스로 미세 조정하는 연구용 코드입니다. 공개 범위는 teacher model 실험 파이프라인까지이며, 차량 탑재용 제품이나 검증된 실시간 탐지기를 제공하지 않습니다.

학습 데이터, 학습된 가중치, 성능 결과, student model, knowledge distillation, ONNX export 및 배포 런타임은 이 저장소에 포함하지 않습니다.

## 구현 범위

- 11-bit/29-bit CAN ID, 0–8 byte payload, 완전한 행 일치를 검사하는 공통 `candump` parser
- 특수 토큰, 256개 byte 토큰, 검증된 CAN ID 토큰으로 구성한 어휘
- 고정 길이 시퀀스와 MLM 동적 마스킹
- Transformers 기반 BERT MLM teacher와 고정 4-class classifier
- 파일 경계를 유지하는 분류 시퀀스와 source file/line provenance
- 실제 optimizer update에 맞춘 gradient accumulation, scheduler, AMP overflow 처리
- versioned full-state resume와 legacy model-only warm start
- 정렬된 데이터 병합, 파일 단위 split, SHA-256 manifest

```text
core/       tokenizer, sequencer, MLM/classification datasets, class constants
models/     MLM teacher and four-class classifier
scripts/    data preparation, pretraining, and fine-tuning entry points
utils/      canonical candump parser and data loading
tests/      smoke tests and training/data integrity tests
```

## 환경

- Python 3.12
- 학습에는 PyTorch와 `requirements.txt`의 패키지 필요
- 분산 사전 학습 경로는 Linux, CUDA, NCCL 환경을 전제로 함

GPU 환경에 맞는 PyTorch를 [공식 설치 안내](https://pytorch.org/get-started/locally/)에 따라 먼저 설치합니다. CPU 전용 의존성 검증 예시는 다음과 같습니다.

```bash
python -m venv .venv
# activate the environment for your shell
python -m pip install --upgrade pip
python -m pip install torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

## Canonical parser와 vocabulary

모든 로더와 vocabulary builder는 `utils.data_loader.parse_candump_line`을 사용합니다. parser는 CAN ID와 payload를 대문자로 정규화하고, 29-bit 상한·payload 길이·행 끝까지 검사합니다. 스캔 결과는 valid/rejected/total count로 보고되며, 유효한 frame이 하나도 없으면 중단합니다. Vocabulary에는 공개 API `CANTokenizer.add_can_ids()`로 검증된 ID만 추가합니다.

이 parser로 다시 만든 canonical vocabulary는 이전 정규식 기반 vocabulary와 token index 또는 hash가 달라질 수 있습니다. 새 vocabulary와 기존 checkpoint를 혼용하지 말고 별도 artifact lineage로 관리합니다.

예상 입력 행:

```text
(1613599955.394625) can0 0C8#0000000000000000
```

## 결정론적 데이터 준비

원본 CAN 로그는 데이터 사용 권한과 크기 때문에 공개 저장소에 넣지 않습니다. `dataset/`, 생성된 `data/`, `checkpoints/`, W&B 로그는 Git에서 제외됩니다.

병합 입력은 정규화된 상대 경로 순으로 처리합니다. 출력 디렉터리가 비어 있지 않으면 기본 동작은 실패이며, 기존 생성물의 교체를 의도한 경우에만 `--overwrite`를 사용합니다.

```bash
python -m scripts.aggregate_data \
  --source-dir "dataset/CAN-MIRGU(train)/Benign" \
  --output-file data/HCRL_dataset/train_aggregated.log

python -m scripts.build_vocab \
  --data-file data/HCRL_dataset/train_aggregated.log \
  --output checkpoints/vocab.json

python -m scripts.split_data
```

`aggregate_manifest.json`에는 정렬된 source relative path, source SHA-256, output SHA-256가 기록됩니다.

미세 조정 split은 source file 단위로 수행합니다. 각 클래스에 최소 세 파일이 있어야 하며, train/validation/test에 각 클래스가 하나 이상 포함되고 한 source가 두 split에 중복되지 않는지 확인합니다.

```bash
python -m scripts.prepare_finetune_data \
  --dataset-dir "dataset/CAN-MIRGU(train)" \
  --output-dir data/finetune_data \
  --seed 42 --train-ratio 0.7 --val-ratio 0.15 \
  --link-mode copy
```

`split_manifest.json`에는 class, split, source relative path, target relative path, materialization 방식, source SHA-256, ratio와 클래스별 유효 seed가 기록됩니다. 유효 seed는 기본 seed에 고정된 `CLASS_NAMES` index를 더해 계산합니다. `Real_attacks` 파일명은 `dos`, `fuzz`, `malfunction` 중 정확히 한 규칙에 일치해야 합니다. `Masquerade_attacks`와 `Suspension_attacks` 디렉터리는 Malfunction으로 분류합니다. 그 밖의 파일은 자동으로 Malfunction에 넣지 않고 오류로 보고합니다.

정렬·최소 개수·고정 seed 규칙을 적용한 split은 이전 스크립트의 결과와 달라질 수 있으므로 기존 실험과 별도 lineage로 취급합니다.

같은 출력의 재생성을 명시하려면 위 데이터 준비 명령에 `--overwrite`를 추가합니다. 이때 기존 트리 전체가 manifest에 기록된 파일과 정확히 일치해야 하며, 중첩된 미등록 파일이나 누락된 파일이 있으면 중단합니다. 새 split은 형제 staging 디렉터리에 모두 만든 뒤 교체하므로 생성 실패 시 기존 출력은 유지됩니다.

## 사전 학습과 체크포인트

`gradient_accumulation_steps`가 loader 길이를 나누어 떨어뜨리지 않아도 각 epoch의 마지막 잔여 microbatch 묶음을 update합니다. Loss log는 축적용 scaled loss와 원래 raw loss를 구분합니다. Gradient clipping은 optimizer update 직전에 수행하며 scheduler와 global optimizer step은 AMP overflow로 건너뛴 update에서는 증가하지 않습니다. DDP의 update가 아닌 microbatch에는 `no_sync()`를 사용합니다.

새 실행 예시입니다. `--epochs`는 재개 전후를 합친 총 epoch 수입니다.

```bash
torchrun --standalone --nproc_per_node=1 -m scripts.pretrain \
  --data_path data/aggregated_parts/part_00 \
  --vocab_path checkpoints/vocab.json \
  --output_dir checkpoints \
  --epochs 5 --batch_size 32 --num_workers 4 \
  --gradient_accumulation_steps 4
```

Schema version 2 체크포인트는 model, optimizer, scheduler, AMP scaler, 다음 epoch, global optimizer step, model/training config를 저장합니다. Training config에는 vocabulary와 dataset 내용의 SHA-256, masking 확률, batch size, seed, world size, AMP 여부, sequence 길이와 dataset type이 포함됩니다. Resume은 이 값과 forward semantics에 영향을 주는 model config가 모두 같고 scheduler step이 global optimizer step과 일치할 때만 허용됩니다. 같은 입력과 설정으로 계속하려면 다음과 같이 full-state resume을 사용합니다.

```bash
torchrun --standalone --nproc_per_node=1 -m scripts.pretrain \
  --data_path data/aggregated_parts/part_00 \
  --vocab_path checkpoints/vocab.json \
  --output_dir checkpoints \
  --epochs 5 --batch_size 32 --num_workers 4 \
  --gradient_accumulation_steps 4 \
  --resume_from_checkpoint checkpoints/can-bert-pretrained-epoch-2.pt
```

이전 model-only 또는 schema 없는 체크포인트는 optimizer/scheduler 상태가 호환된다고 간주하지 않습니다. `--resume_from_checkpoint`로 읽으면 명시적인 legacy warm start log를 남기며 새 schedule로 시작합니다. 이를 처음부터 의도한 경우에는 `--warm_start_from_checkpoint`를 사용합니다.

```bash
python -m scripts.pretrain \
  --data_path data/aggregated_parts/part_00 \
  --vocab_path checkpoints/vocab.json \
  --epochs 5 \
  --warm_start_from_checkpoint checkpoints/legacy-model-only.pt
```

체크포인트는 Python, NumPy, PyTorch의 RNG 상태를 저장하지 않으므로 bitwise-identical continuation을 주장하지 않습니다.

## 4-class 미세 조정

분류 시퀀스는 source file 경계를 넘지 않습니다. 각 source file은 파일명에서 결정된 하나의 label만 가지며 sequence provenance에는 source file과 시작/끝 line이 남습니다. 한 sequence를 만들기에 짧은 파일은 기본적으로 제외하며 파일 수를 보고합니다. 엄격히 중단하려면 `--short_file_policy error`를 명시합니다. 모든 파일이 제외되면 중단합니다.

Class name과 label은 항상 `Benign=0`, `DoS=1`, `Fuzzy=2`, `Malfunction=3`입니다. 학습 split에 한 클래스라도 없으면 class weight 계산을 중단합니다. Metric과 confusion matrix는 sparse prediction에서도 고정 label 네 개와 4×4 shape를 사용합니다.

```bash
python -m scripts.finetune \
  --train_data_dir data/finetune_data/train \
  --val_data_dir data/finetune_data/validation \
  --test_data_dir data/finetune_data/test \
  --vocab_path checkpoints/vocab.json \
  --pretrained_checkpoint checkpoints/can-bert-pretrained-epoch-5.pt \
  --output_dir checkpoints \
  --short_file_policy skip
```

## 검증

Smoke test는 parser/tokenizer/dataset의 기본 동작과 teacher/classifier CPU forward shape를 확인합니다.

```bash
python -m unittest tests.test_data_pipeline tests.test_models -v
```

Training/data integrity test는 accumulation 잔여 update, scheduler/global-step 수, AMP overflow, DDP `no_sync`, checkpoint round trip·한 update 연속성·입력 hash 불일치 거부, file boundary, missing class, empty loader, deterministic manifest·안전한 overwrite 정책을 합성 입력으로 확인합니다.

```bash
python -m unittest \
  tests.test_training_integrity \
  tests.test_checkpoints \
  tests.test_classification_integrity \
  tests.test_data_preparation -v
```

전체 검사:

```bash
python -m compileall core models scripts utils tests
python -m unittest discover -s tests -v
```

이 검사는 실제 데이터 학습이나 성능 재현을 대신하지 않습니다.

## Project lineage

이 저장소는 public teacher-model prototype입니다. 2025년 DGIST UGRP 팀 연구에서는 별도 산출물로 binary student model과 ONNX 평가까지 확장했습니다. 그 후속 package는 source–model parity, 팀 기여 범위, 데이터·모델 재배포 권리 검토가 끝나지 않아 이 저장소에 포함하지 않습니다. 따라서 이 저장소가 teacher-to-student 전체 파이프라인을 제공하거나 후속 결과를 재현한다고 해석해서는 안 됩니다.

## 검증 범위와 한계

- CI와 로컬 검사는 합성 입력 기반 smoke/integrity test입니다.
- 공개 데이터, 가중치, 실험 로그가 없으므로 정확도, F1, 지연 시간 또는 일반화 성능을 주장하지 않습니다.
- CPU 테스트는 작은 단위 입력의 코드 경로 검증이며 CPU 대규모 학습 지원을 의미하지 않습니다.
- 실제 차량 시스템에 바로 배포하기 위한 구현이 아닙니다.
- 저장소에는 명시적인 소프트웨어 라이선스가 없습니다. 팀·원 코드 권리 확인 전에는 재사용 허가를 전제로 하지 마세요.

## References

1. Natasha Alkhatib et al., [“CAN-BERT do it? Controller Area Network Intrusion Detection System based on BERT Language Model”](https://arxiv.org/abs/2210.09439), 2022.
2. Hyunjun Jo and Deok-Hwan Kim, [“Intrusion Detection Using Transformer in Controller Area Network”](https://doi.org/10.1109/ACCESS.2024.3452634), *IEEE Access*, 2024.
3. Pengzhou Cheng et al., [“LSF-IDM: Automotive Intrusion Detection Model with Lightweight Attribution and Semantic Fusion”](https://arxiv.org/abs/2308.01237), 2023.

앞의 두 연구는 CAN 메시지의 language-model representation과 Transformer 기반 탐지를 설계할 때 참고했습니다. 세 번째 연구의 distillation 접근은 후속 팀 연구 방향의 참고 자료이며, 현재 공개 코드의 구현 범위가 아닙니다.
