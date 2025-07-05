# 프로젝트 마스터 플랜 및 실행 로드맵 (v1.1)

> **Revision History:**
>
> -   v1.0 -> v1.1: Phase 2.3 (파일럿 미세 조정)의 상세 기술 명세서(finetune_spec_v2.2) 내용을 통합하여 단일 마스터 플랜으로 강화.

-   **최종 목표:** 실제 차량용으로 배포 가능한, 고성능/고효율의 CAN 버스 침입 탐지 시스템(IDS) 개발
-   **핵심 방법론:** `CAN-BERT`(교사) 사전 훈련 → `LSF-IDM` 기반 지식 증류 → 경량 `Student` 모델 완성

---

### **Phase 2: 교사(Teacher) 모델 개발 및 완성**

#### **Phase 2.3: 파일럿 미세 조정 및 파이프라인 검증 - `[현재 진행 중]`**

##### **2.3.1. 목표 및 성공 기준**

-   **목표:** 보유 중인 불완전 사전 훈련 모델(`epoch-14.pt`, `mask_ratio=0.15` 조건)을 사용하여, **End-to-End 미세 조정 파이프라인의 기술적 건전성을 검증**하고, 향후 모델 개선을 위한 **초기 성능 베이스라인을 확보**한다.
-   **성공 기준 (Success Criteria):**
    1.  `[ ]` `finetune.py` 스크립트가 데이터 로딩부터 훈련, 검증, 모델 저장까지 **오류 없이 완주**한다.
    2.  `[ ]` `wandb`에 훈련/검증 손실, 학습률, 각 평가 지표가 **정상적으로 로깅**된다.
    3.  `[ ]` `evaluate.py`를 통해 테스트셋에 대한 **최종 성능 리포트(F1, Precision, Recall, Confusion Matrix 등)가 성공적으로 산출**된다.

##### **2.3.2. 데이터 파이프라인**

-   **입력 데이터 구조:** `data/finetune_data/` 디렉토리 아래의 `train`, `validation`, `test` 폴더를 각각 입력으로 받는다. 각 폴더에는 `Benign`, `DoS`, `Fuzzy`, `Malfunction` 데이터가 별도의 `.csv` 또는 `.log` 파일로 존재해야 한다.
-   **레이블 정의 (Label Definition):** 클래스는 다음과 같이 정수 값으로 명확히 매핑한다.
    -   `Benign`: 0
    -   `DoS`: 1
    -   `Fuzzy`: 2
    -   `Malfunction`: 3
-   **데이터셋 클래스 (`ClassificationDataset`):** `torch.utils.data.Dataset`을 상속받는 새로운 `ClassificationDataset` 클래스를 `core/dataset.py`에 구현한다.
    -   `[ ]` `__init__(self, data_dir, tokenizer, seq_len)`:
        -   `data_dir` 내의 모든 로그/CSV 파일을 읽어들인다.
        -   파일명을 기준으로 각 데이터 라인에 레이블(0~3)을 부여한다.
        -   모든 (시퀀스, 레이블) 쌍을 메모리에 리스트 형태로 로드한다.
    -   `[ ]` `__len__(self)`: 전체 데이터 샘플 수를 반환한다.
    -   `[ ]` `__getitem__(self, idx)`:
        -   `idx`에 해당하는 CAN 시퀀스를 토크나이징하고 `seq_len`에 맞게 패딩/절단한다.
        -   최종적으로 모델 입력 형식에 맞는 딕셔너리를 반환한다: `{'input_ids': torch.Tensor, 'attention_mask': torch.Tensor, 'labels': torch.LongTensor}`.

##### **2.3.3. 모델 아키텍처 및 가중치**

-   **선별적 가중치 로딩 (Selective Weight Loading):**
    -   `[ ]` `--resume_from_checkpoint` 인자로 받은 `epoch-14.pt` 파일의 `state_dict`를 로드한다.
    -   **(주의) 이 체크포인트는 `mask_ratio=0.15` 조건으로 사전 훈련되었음을 명확히 인지한다.**
    -   `[ ]` 로드된 `state_dict`는 `CANBertForMaskedLM`의 가중치이므로, 이 중 BERT '몸통'에 해당하는 부분(`bert.*`)의 가중치만 추출한다.
    -   `[ ]` 새로 생성된 `CANBertForClassification` 인스턴스에 추출된 '몸통' 가중치만 주입한다. 모델의 `classifier` 레이어(분류 헤드)는 무작위 초기화 상태를 유지해야 한다.

##### **2.3.4. 훈련 및 평가 로직**

-   **손실 함수 (Loss Function):** `[ ]` 클래스 불균형 문제를 다루기 위해 가중치(weight) 적용이 가능한 `torch.nn.CrossEntropyLoss`를 사용한다. `train` 데이터셋의 클래스별 샘플 수의 역수를 계산하여 `weight` 텐서를 생성하고, 손실 함수 초기화 시 전달하는 로직을 추가한다.
-   **옵티마이저 (Optimizer):** `[ ]` `torch.optim.AdamW`를 사용한다.
-   **차등 학습률 (Differential Learning Rate):**
    -   `[ ]` **(필수 구현)** 옵티마이저에 두 개의 파라미터 그룹을 명시적으로 전달한다.
        -   **그룹 1 (BERT Body):** `model.bert.parameters()` - 낮은 학습률 적용. **(권장: `2e-6` ~ `1e-5`)**
        -   **그룹 2 (Classifier Head):** `model.classifier.parameters()` - 상대적으로 높은 학습률 적용. **(권장: `5e-5` ~ `1e-4`)**
-   **평가 지표 (Evaluation Metrics):** `[ ]` `scikit-learn`을 활용하여 아래 지표들을 계산한다: Accuracy, Precision, Recall, F1-Score (각 클래스별, 'weighted' 평균), Confusion Matrix.

##### **2.3.5. 실행 및 결과물**

-   **커맨드라인 인터페이스 (CLI):** `[ ]` `argparse`를 통해 다음을 인자로 받는다: `--train_data_dir`, `--val_data_dir`, `--test_data_dir`, `--vocab_path`, `--resume_from_checkpoint`, `--output_dir`, `--epochs`, `--batch_size`, `--seq_len`, `--body_lr`, `--head_lr`.
-   **체크포인트 저장:** `[ ]` 가장 높은 **validation F1-Score (weighted average)를 기록한 epoch**의 모델을 `output_dir`에 `pilot-finetuned-best.pt`로 저장한다.
-   **로깅:** `[ ]` `wandb`에 하이퍼파라미터와 모든 훈련/검증 결과를 기록한다. 특히, 사용된 교사 모델의 조건(`pretrain_mask_ratio: 0.15`)을 명시적으로 저장한다.

---

#### **Phase 2.4: 선택적-점진적 사전 훈련 - `[계획]`**

-   **[ ] 목표:** `epoch-14.pt` 모델을 `CAN-BERT` 논문의 최적 조건으로 강화하여, 최종 교사 모델의 기반을 완성한다. (Phase 2.3에서 확보된 베이스라인을 기준으로 성능 향상을 목표로 함)
-   **[ ] 입력:** `epoch-14.pt` 체크포인트, 나머지 17.5개 데이터 파트.
-   **[ ] 핵심 과업 (Checklist):**
    -   `[ ]` `scripts/pretrain.py` 스크립트 실행 시 `--mask_prob`를 **`0.45`**로 설정한다.
    -   `[ ]` 학습률(LR), Epoch 등 다른 학습 스케줄도 `CAN-BERT` 논문을 참고하여 최적의 값으로 재설정한다.
    -   `[ ]` 모든 데이터 파트에 대한 추가 사전 훈련을 실행한다.
-   **[ ] 핵심 산출물:** `teacher-pretrained-full.pt` (완전하게 사전 훈련된 교사 모델)

#### **Phase 2.5: 대조 학습 기반 퍼지 공격 특화 - `[계획]`**

-   **[ ] 목표:** `Fuzzy` 공격에 대한 교사 모델의 분별력을 집중적으로 강화한다.
-   **[ ] 입력:** `teacher-pretrained-full.pt`, `data/finetune_data`
-   **[ ] 핵심 과업 (Checklist):**
    -   `[ ]` `core/dataset.py`에 대조 학습을 위한 데이터셋 클래스를 새로 정의한다.
    -   `[ ]` `SimCLR` 스타일의 대조 손실 함수를 구현한다.
    -   `[ ]` `scripts/contrastive_finetune.py` 라는 새로운 훈련 스크립트를 개발한다.
    -   `[ ]` `Fuzzy` 공격 데이터와 `Benign` 데이터를 사용하여 대조 학습을 실행한다.
-   **[ ] 핵심 산출물:** `teacher-final-specialized.pt` (Fuzzy 공격에 특화된 최종 교사 모델)

---

### **Phase 3: 학생(Student) 모델 개발 및 지식 증류**

#### **Phase 3.1: 학생 모델 아키텍처 설계 및 구현 - `[계획]`**

-   **[ ] 목표:** `LSF-IDM` 논문을 기반으로 경량 학생 모델의 아키텍처를 구현한다.
-   **[ ] 입력:** `LSF-IDM (Cheng et al., 2023)` 논문.
-   **[ ] 핵심 과업 (Checklist):**
    -   `[ ]` `LSF-IDM` 논문의 BiLSTM 또는 경량 Transformer 구조를 분석한다.
    -   `[ ]` `models/student.py` 파일을 생성하고, 분석한 경량 아키텍처를 PyTorch 코드로 구현한다.
-   **[ ] 핵심 산출물:** `models/student.py` 코드 파일.

#### **Phase 3.2: 다중 교사 지식 증류 실행 - `[계획]`**

-   **[ ] 목표:** 완성된 교사 모델들의 지식을 경량 학생 모델에 효율적으로 이전(증류)한다.
-   **[ ] 입력:** `pilot-finetuned-best.pt` (초기 교사), `teacher-final-specialized.pt` (최종 교사), 대규모 정상 데이터.
-   **[ ] 핵심 과업 (Checklist):**
    -   `[ ]` `scripts/distill.py` 라는 새로운 지식 증류 스크립트를 개발한다.
    -   `[ ]` 스크립트 내에 지식 증류 손실 함수를 구현한다.
    -   `[ ]` '다중 교사(Multi-Teacher)' 증류 전략을 구현한다.
-   **[ ] 핵심 산출물:** `student-final-distilled.pt` (모든 지식이 증류된 최종 경량 모델)

---

### **Phase 4: 최종 평가 및 발표**

-   **[ ] 목표:** 완성된 경량 학생 모델의 성능을 종합적으로 평가하고, 연구 결과를 정리한다.
-   **[ ] 입력:** `student-final-distilled.pt`, 전체 `test` 데이터셋.
-   **[ ] 핵심 과업 (Checklist):**
    -   `[ ]` `scripts/evaluate.py`를 사용하여 최종 학생 모델의 성능을 종합적으로 측정한다.
    -   `[ ]` 교사 모델, 베이스라인 모델들과의 성능을 비교 분석표로 정리한다.
    -   `[ ]` 모델의 추론 속도와 메모리 사용량을 측정한다.
    -   `[ ]` (선택 사항) `Fuzzy-fallback` 규칙을 구현하고, 적용 시 성능 변화를 분석한다.
    -   `[ ]` 모든 실험 결과와 분석 내용을 종합하여 최종 연구 보고서 또는 발표 자료를 작성한다.
-   **[ ] 핵심 산출물:** 최종 성능 분석 리포트, 연구 보고서/발표 자료.