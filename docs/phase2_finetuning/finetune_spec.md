# 기능 명세서: CAN-BERT 파일럿 미세 조정 스크립트 (finetune.py)

## 1. 개요
- **목표:** 사전 훈련된 `epoch-14.pt` 모델을 활용하여, 정상/공격 레이블이 부여된 CAN 시퀀스 데이터로 '침입 탐지 다중 클래스 분류(Multi-class Classification)' 모델을 만든다.
- **핵심 모델 아키텍처:** `models.teacher_classifier.CANBertForClassification`
- **핵심 기술:** 전이 학습 (Transfer Learning), 미세 조정 (Fine-tuning)

## 2. 데이터 파이프라인
- **입력 데이터 구조:** `data/finetune_data/` 디렉토리 아래의 `train`, `validation` 폴더를 각각 입력으로 받는다. 각 폴더에는 `Benign`, `DoS`, `Fuzzy`, `Malfunction` 데이터가 별도의 `.csv` 또는 `.log` 파일로 존재한다.
- **레이블 정의 (Label Definition):** 클래스는 다음과 같이 정수 값으로 명확히 매핑한다.
  - `Benign`: 0
  - `DoS`: 1
  - `Fuzzy`: 2
  - `Malfunction`: 3
- **데이터셋 클래스 (`ClassificationDataset`):** `torch.utils.data.Dataset`을 상속받는 새로운 `ClassificationDataset` 클래스를 `core/dataset.py`에 정의한다.
  - `__init__(self, data_dir, tokenizer, seq_len)`:
    - `data_dir` 내의 모든 로그/CSV 파일을 읽어들인다.
    - 파일명을 기준으로 각 데이터 라인에 레이블(0~3)을 부여한다.
    - 모든 (시퀀스, 레이블) 쌍을 메모리에 리스트 형태로 로드한다.
  - `__len__(self)`: 전체 데이터 샘플의 수를 반환한다.
  - `__getitem__(self, idx)`:
    - `idx`에 해당하는 CAN 시퀀스를 토크나이징하고 `seq_len`에 맞게 패딩/절단한다.
    - 최종적으로 모델 입력 형식에 맞는 딕셔너리를 반환한다: `{'input_ids': torch.Tensor, 'attention_mask': torch.Tensor, 'labels': torch.LongTensor}`.

## 3. 모델 아키텍처 및 가중치
- **선별적 가중치 로딩 (Selective Weight Loading):**
  - `--resume_from_checkpoint` 인자로 받은 `epoch-14.pt` 파일의 `state_dict`를 로드한다.
  - 로드된 `state_dict`는 `CANBertForMaskedLM`의 가중치이므로, 이 중 BERT '몸통'에 해당하는 부분(`bert.*`)의 가중치만 추출한다.
  - 새로 생성된 `CANBertForClassification` 인스턴스에 추출된 '몸통' 가중치만 주입한다. 모델의 `classifier` 레이어(분류 헤드)는 가중치가 로드되지 않은, 무작위 초기화 상태를 유지해야 한다.

## 4. 훈련 및 평가 로직
- **손실 함수 (Loss Function):** `torch.nn.CrossEntropyLoss`를 사용한다. **(개선 제안: 클래스 불균형 문제를 완화하기 위해 각 클래스의 샘플 수 역수에 비례하는 `weight` 텐서를 계산하여 손실 함수에 전달하는 로직을 추가한다.)**
- **옵티마이저 (Optimizer):** `torch.optim.AdamW`를 사용한다.
- **차등 학습률 (Differential Learning Rate):** **(필수 구현)** 옵티마이저에 파라미터 그룹을 두 개 전달한다.
  - **그룹 1 (BERT Body):** `model.bert.parameters()` - 사전 학습된 지식이 급격히 변하지 않도록 매우 낮은 학습률을 적용한다. (e.g., `2e-6`)
  - **그룹 2 (Classifier Head):** `model.classifier.parameters()` - 처음부터 학습해야 하므로 상대적으로 높은 학습률을 적용한다. (e.g., `5e-5`)
- **평가 지표 (Evaluation Metrics):** 매 epoch 종료 후, validation set에 대해 아래 지표들을 계산한다.
  - Accuracy (전체 정확도)
  - Precision, Recall, F1-Score (각 클래스별, 그리고 가중 평균(weighted average))

## 5. 실행 및 결과물
- **커맨드라인 인터페이스 (CLI):** `argparse`를 통해 다음을 인자로 받는다.
  - `--train_data_dir`, `--val_data_dir`: 훈련/검증 데이터 경로
  - `--vocab_path`: `vocab.json` 경로
  - `--resume_from_checkpoint`: `epoch-14.pt` 경로
  - `--output_dir`: 결과물이 저장될 디렉토리
  - `--epochs`, `--batch_size`, `--body_lr`, `--head_lr` 등 핵심 하이퍼파라미터
- **체크포인트 저장 (Checkpoint Saving):** 훈련 과정 중 **가장 높은 validation F1-Score (weighted average)를 기록한 epoch의 모델** 가중치를 `output_dir`에 `finetuned-best-f1.pt`라는 이름으로 저장한다.
- **로깅 (Logging):** `wandb`를 연동하여 모든 하이퍼파라미터와 epoch별 훈련/검증 손실, 평가 지표들을 기록한다.