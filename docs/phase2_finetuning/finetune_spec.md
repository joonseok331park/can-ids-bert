# 기능 명세서 v2.2: CAN-BERT 파일럿 미세 조정 및 파이프라인 검증

> **Revision History:** > v2.1 -> v2.2: 사전 훈련 모델의 mask_ratio(0.15) 조건을 명시하고, 이를 실험 기록의 일부로 포함하도록 수정.

## 1. 개요

-   **목표:** 보유 중인 불완전 사전 훈련 모델(`epoch-14.pt`, **mask_ratio=0.15** 조건)을 사용하여, **End-to-End 미세 조정 파이프라인의 기술적 건전성을 검증**하고, 향후 모델 개선을 위한 **초기 성능 베이스라인을 확보**한다.
-   **핵심 모델 아키텍처:** `models.teacher_classifier.CANBertForClassification`
-   **핵심 기술:** 전이 학습 (Transfer Learning), 미세 조정 (Fine-tuning)
-   **성공 기준 (Success Criteria):**
    1.  `finetune.py` 스크립트가 데이터 로딩부터 훈련, 검증, 모델 저장까지 **오류 없이 완주**한다.
    2.  `wandb`에 훈련/검증 손실, 학습률, 각 평가 지표가 **정상적으로 로깅**된다.
    3.  `evaluate.py`를 통해 테스트셋에 대한 **최종 성능 리포트(F1, Precision, Recall, Confusion Matrix 등)가 성공적으로 산출**된다.

## 2. 데이터 파이프라인

-   **입력 데이터 구조:** `data/finetune_data/` 디렉토리 아래의 `train`, `validation`, `test` 폴더를 각각 입력으로 받는다. 각 폴더에는 `Benign`, `DoS`, `Fuzzy`, `Malfunction` 데이터가 별도의 `.csv` 또는 `.log` 파일로 존재한다.
-   **레이블 정의 (Label Definition):** 클래스는 다음과 같이 정수 값으로 명확히 매핑한다.
    -   `Benign`: 0
    -   `DoS`: 1
    -   `Fuzzy`: 2
    -   `Malfunction`: 3
-   **데이터셋 클래스 (`ClassificationDataset`):** `torch.utils.data.Dataset`을 상속받는 새로운 `ClassificationDataset` 클래스를 `core/dataset.py`에 정의한다.
    -   `__init__(self, data_dir, tokenizer, seq_len)`:
        -   `data_dir` 내의 모든 로그/CSV 파일을 읽어들인다.
        -   파일명을 기준으로 각 데이터 라인에 레이블(0~3)을 부여한다.
        -   모든 (시퀀스, 레이블) 쌍을 메모리에 리스트 형태로 로드한다.
    -   `__len__(self)`: 전체 데이터 샘플의 수를 반환한다.
    -   `__getitem__(self, idx)`:
        -   `idx`에 해당하는 CAN 시퀀스를 토크나이징하고 `seq_len`에 맞게 패딩/절단한다.
        -   최종적으로 모델 입력 형식에 맞는 딕셔너리를 반환한다: `{'input_ids': torch.Tensor, 'attention_mask': torch.Tensor, 'labels': torch.LongTensor}`.

## 3. 모델 아키텍처 및 가중치

-   **선별적 가중치 로딩 (Selective Weight Loading):**
    -   `--resume_from_checkpoint` 인자로 받은 `epoch-14.pt` 파일의 `state_dict`를 로드한다.
    -   **(주의) 이 체크포인트는 `mask_ratio=0.15` 조건으로 사전 훈련되었음을 명확히 인지한다.** 이는 논문에서 제시한 최적값(0.45)과 다르며, 본 파일럿 테스트의 주요 변수 중 하나이다.
    -   로드된 `state_dict`는 `CANBertForMaskedLM`의 가중치이므로, 이 중 BERT '몸통'에 해당하는 부분(`bert.*`)의 가중치만 추출한다.
    -   새로 생성된 `CANBertForClassification` 인스턴스에 추출된 '몸통' 가중치만 주입한다. 모델의 `classifier` 레이어(분류 헤드)는 가중치가 로드되지 않은, 무작위 초기화 상태를 유지해야 한다.

## 4. 훈련 및 평가 로직

-   **손실 함수 (Loss Function):** 클래스 불균형 문제를 다루기 위해 가중치(weight) 적용이 가능한 `torch.nn.CrossEntropyLoss`를 사용한다. `train` 데이터셋의 클래스별 샘플 수의 역수를 계산하여 `weight` 텐서를 생성하고, 손실 함수 초기화 시 전달하는 로직을 추가한다.
-   **옵티마이저 (Optimizer):** `torch.optim.AdamW`를 사용한다.
-   **차등 학습률 (Differential Learning Rate):** **(필수 구현)** 옵티마이저에 파라미터 그룹을 두 개 전달한다.
    -   **그룹 1 (BERT Body):** `model.bert.parameters()` - 사전 학습된 지식이 급격히 변하지 않도록 매우 낮은 학습률을 적용한다. (e.g., `2e-6`)
    -   **그룹 2 (Classifier Head):** `model.classifier.parameters()` - 처음부터 학습해야 하므로 상대적으로 높은 학습률을 적용한다. (e.g., `5e-5`)
-   **평가 지표 (Evaluation Metrics):** 매 epoch 종료 후, validation set에 대해 `scikit-learn` 라이브러리를 활용하여 아래 지표들을 계산한다.
    -   Accuracy (전체 정확도)
    -   Precision, Recall, F1-Score (각 클래스별, 그리고 가중 평균(weighted average))
    -   Confusion Matrix (혼동 행렬)

## 5. 실행 및 결과물

-   **커맨드라인 인터페이스 (CLI):** `argparse`를 통해 다음을 인자로 받는다.
    -   `--train_data_dir`, `--val_data_dir`, `--test_data_dir`
    -   `--vocab_path`, `--resume_from_checkpoint`, `--output_dir`
    -   `--epochs`, `--batch_size`, `--seq_len`
    -   `--body_lr`, `--head_lr` (차등 학습률을 위한 인자)
-   **체크포인트 저장 (Checkpoint Saving):** 훈련 과정 중 **가장 높은 validation F1-Score (weighted average)를 기록한 epoch**의 모델 `state_dict`를 `output_dir`에 `pilot-finetuned-best.pt`라는 이름으로 저장한다.
-   **로깅 (Logging):** `wandb`를 연동하여 모든 하이퍼파라미터와 epoch별 훈련/검증 손실, 평가 지표들을 기록한다. **특히, 이번 실험에 사용된 교사 모델의 사전 훈련 조건(e.g., `'pretrain_mask_ratio': 0.15`)을 `wandb` config에 명시적으로 저장하여 재현성을 확보한다.**

## 6. 본 파일럿의 한계 및 후속 단계

-   **명시적 한계 (Explicit Limitations):**
    -   본 파일럿으로 생성된 모델(`pilot-finetuned-best.pt`)은 파이프라인 검증 및 베이스라인 측정용이며, **최종 배포용이 아님**을 명확히 인지한다.
    -   교사 모델의 **사전 훈련량(20.5%)이 부족**하고, **마스킹 비율(15%)이 논문 최적치(45%)보다 낮아**, 사전 훈련된 표현력 자체가 제한적일 수 있다.
-   **후속 단계와의 연계 (Link to Next Steps):**
    -   본 파일럿의 성공적인 완료는 **'Phase 2.4: 선택적-점진적 사전 훈련'**(`m=0.45` 적용)과 **'Phase 2.5: 대조 학습 기반 특화 훈련'**으로 나아가기 위한 전제 조건이다.
    -   여기서 얻은 성능 베이스라인은, Phase 2.4와 2.5를 통해 모델이 얼마나 개선되었는지를 측정하는 중요한 척도가 될 것이다. (e.g., `m=0.15` 모델 대비 `m=0.45` 모델의 성능 향상 폭)

    ## 6. 본 파일럿의 한계 및 후속 단계

-   **명시적 한계 (Explicit Limitations):**
    -   본 파일럿으로 생성된 모델(`pilot-finetuned-best.pt`)은 파이프라인 검증 및 베이스라인 측정용이며, **최종 배포용이 아님**을 명확히 인지한다.
    -   교사 모델의 **사전 훈련량(20.5%)과 마스킹 비율(15%)** 이 부족하여 표현력이 제한적이다.

-   **후속 단계와의 연계 (Link to Next Steps):**
    -   본 파일럿의 성공적인 완료는 `Phase 2.4`와 `Phase 2.5`로 나아가기 위한 전제 조건이다.
    -   여기서 얻은 성능 베이스라인은, 후속 단계를 통해 모델이 얼마나 개선되었는지를 측정하는 중요한 척도가 될 것이다.

-   **[미래 실행 계획 - Action Items]**
    -   **[ ] (Phase 2.4) 사전 훈련 재개 시 `mask_prob` 값 변경:**
        -   **대상 스크립트:** `scripts/pretrain.py`
        -   **변경 전:** `--mask_prob 0.15` (기본값)
        -   **변경 후:** `--mask_prob 0.45` (논문 최적치)
        -   **목적:** 교사 모델의 표현력 및 최종 성능 극대화