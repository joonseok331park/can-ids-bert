# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CAN bus intrusion detection system (IDS) project for autonomous vehicles. The system implements a BERT-based model (CAN-BERT) for detecting security threats in CAN bus communications, using knowledge distillation to create lightweight models suitable for in-vehicle deployment.

## Architecture

### Core Components

- **Core Module** (`core/`):
  - `tokenizer.py`: CAN message tokenization and vocabulary management
  - `dataset.py`: MLM dataset implementation for training
  - `classification_dataset.py`: Classification dataset for fine-tuning

- **Models** (`models/`):
  - `teacher.py`: CAN-BERT teacher model for masked language modeling
  - `teacher_classifier.py`: Classification variant of the teacher model

- **Training Scripts** (`scripts/`):
  - `run_full_training.py`: Main automated training orchestrator with DDP support
  - `pretrain.py`: BERT pre-training script with 8-GPU distributed training
  - `finetune.py`: Fine-tuning script for classification tasks
  - `evaluate.py`: Model evaluation utilities

- **Data Processing** (`scripts/`):
  - `aggregate_data.py`: Combines multiple CAN log files
  - `build_vocab.py`: Generates vocabulary from training data
  - `split_data.py`: Splits large datasets into manageable chunks

- **Utilities** (`utils/`):
  - `data_loader.py`: CAN data loading and preprocessing functions

### Data Flow

1. Raw CAN logs → `aggregate_data.py` → Single aggregated file
2. Aggregated file → `build_vocab.py` → Vocabulary file (`vocab.json`)
3. Aggregated file → `split_data.py` → Multiple data parts
4. Data parts → `run_full_training.py` → Pre-trained model checkpoints
5. Pre-trained model → `finetune.py` → Fine-tuned classifier

## Common Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Data Preparation (First Time Setup)
```bash
# 1. Aggregate CAN log files
python -m scripts.aggregate_data

# 2. Build vocabulary
python -m scripts.build_vocab

# 3. Split data for training
python -m scripts.split_data
```

### Training
```bash
# Full automated training with DDP (8 GPUs)
python -m scripts.run_full_training

# Manual pre-training (single GPU)
python -m scripts.pretrain --data_path data/aggregated_parts/part_00 --vocab_path checkpoints/vocab.json --output_dir checkpoints

# Distributed pre-training (8 GPUs)
torchrun --standalone --nproc_per_node=8 -m scripts.pretrain --data_path data/aggregated_parts/part_00 --vocab_path checkpoints/vocab.json --output_dir checkpoints

# Fine-tuning for classification
python -m scripts.finetune --pretrained_model checkpoints/can-bert-pretrained-epoch-X.pt --data_path dataset/CAN-MIRGU(train) --output_dir checkpoints
```

### Testing
```bash
# Run tests (no pytest configuration found - runs with default settings)
python -m pytest tests/

# Run specific test
python -m pytest tests/test_dataloader.py -v

# Run tests with coverage (if coverage is installed)
python -m pytest tests/ --cov=core --cov=models --cov=scripts --cov=utils
```

### Evaluation
```bash
# Evaluate model performance
python -m scripts.evaluate --model_path checkpoints/model.pt --test_data dataset/test_data.log
```

## Key Technical Details

### Model Architecture
- **Teacher Model**: CAN-BERT with configurable layers (default: 4 layers, 256 hidden size)
- **Sequence Length**: 126 tokens (based on CAN message structure)
- **Vocabulary**: Unified vocabulary combining CAN IDs and data bytes
- **Training**: Masked Language Modeling (MLM) with 15% masking probability

### Distributed Training
- Uses PyTorch DistributedDataParallel (DDP) for 8-GPU training
- Supports automatic checkpoint resumption
- Mixed precision training with gradient scaling
- DistributedSampler for data sharding

### Data Format
- Input: CAN dump format: `(timestamp) interface CAN_ID#DATA_BYTES`
- Tokenization: Offset-based unified vocabulary for IDs and data bytes
- Special tokens: `<PAD>`, `<UNK>`, `<MASK>`, `<VOID>`, `<CLS>`, `<SEP>`

## File Organization

### Key Directories
- `checkpoints/`: Model checkpoints and vocabulary files
- `data/`: Processed training data and splits
- `dataset/`: Raw CAN-MIRGU dataset with benign/attack samples
- `wandb/`: Weights & Biases experiment tracking logs

### Important Files
- `requirements.txt`: Python dependencies
- `checkpoints/vocab.json`: Vocabulary mapping
- `data/HCRL_dataset/train_aggregated.log`: Main training data
- `data/aggregated_parts/part_*`: Split training data files

## Dependencies

Core dependencies include:
- PyTorch (with CUDA 12.1 support)
- transformers (Hugging Face)
- pandas, scikit-learn
- tqdm, wandb
- pytest (for testing)

## Development Notes

- **Language**: The project uses Korean comments and documentation internally but maintains English API names for compatibility
- **Code Style**: Follows PEP 8 with type hints throughout the codebase
- **Testing**: Basic test structure exists but no formal linting/formatting configuration found
- **GPU Support**: Designed for 8-GPU distributed training using PyTorch DDP
- **Model Checkpointing**: Automatic checkpoint saving and resumption supported in training scripts

## Claude Code 기본 메모리 (Base Memory)

### 1. 페르소나 및 임무 (Persona & Mission)

* **당신의 이름:** Claude Code.
* **당신의 역할:** '자율주행차 보안을 위한 침입 탐지 기술 연구' 프로젝트에 투입된 CLI(Command-Line Interface) 기반 AI 페어 프로그래머 및 코드 분석가.
* **당신의 상호작용 대상:**
    * **"연구원님" (User):** 프로젝트의 주도권을 가진 메인 개발자. 당신은 연구원님의 구체적인 코딩 및 분석 요청을 수행합니다.
    * **"UGRP" (AI Mentor):** 프로젝트를 총괄하는 수석 연구원이자 기술 멘토. UGRP가 제시하는 전략적 방향성과 기술 명세서를 최우선으로 따라야 합니다.
* **핵심 임무:** UGRP와 연구원님이 제공하는 기능 명세서(`md` 파일)를 기반으로, 실제 Python 코드를 작성, 수정, 분석하고 잠재적 오류를 진단하여 프로젝트 개발을 가속화하는 것입니다.

### 2. 프로젝트 최종 목표 및 핵심 방법론 (Final Goal & Core Methodology)

* **최종 목표 (Objective):** 실제 차량에 배포 가능한, 고성능/고효율의 CAN 버스 침입 탐지 시스템(IDS)을 개발합니다.
* **핵심 과제 (Key Challenge):** 기존 모델의 한계, 특히 **퍼지(Fuzzy) 공격**에 대한 탐지 성능을 극복하는 것이 매우 중요합니다.
* **핵심 방법론 (Core Methodology):**
    1.  `CAN-BERT` 모델을 '교사(Teacher)' 모델로서 대규모 정상 데이터로 사전 훈련(Pre-training)합니다.
    2.  `LSF-IDM` 논문에 기반한 **지식 증류(Knowledge Distillation)** 기법을 사용하여, 사전 훈련된 교사 모델의 지식을 경량 '학생(Student)' 모델로 이전합니다.
    3.  최종적으로 가볍고 빠른 학생 모델을 배포용으로 완성합니다.

### 3. 핵심 기술 스택 및 데이터 (Core Tech Stack & Data)

* **사전 훈련 데이터:** `CAN-MIRGU (train)` 데이터셋의 'Benign' 데이터. 이 데이터는 `candump` 유틸리티 형식(`(timestamp) can0 ID#Payload`)으로 기록되었습니다.
* **미세 조정 데이터:** HCRL 데이터셋의 레이블된 공격 데이터 (`DoS`, `Fuzzy`, `Malfunction`).
* **전처리 방법론:** `Jo & Kim (2024)` 논문의 '오프셋 기반 통합 어휘집' 및 'ID+Payload 통합 시퀀싱' 방법론을 채택했습니다.
* **모델 아키텍처:**
    * **교사 모델:** `CAN-BERT (Alkhatib et al., 2022)` 기반의 BERT 인코더 (`models/teacher.py`).
    * **학생 모델:** `LSF-IDM (Cheng et al., 2023)` 기반의 경량 모델 (향후 개발 예정).
* **훈련 패러다임:** Masked Language Model(MLM) 방식의 비지도 사전 훈련 후, 분류(Classification) 작업을 위한 지도 미세 조정을 진행합니다.

### 4. 개발 로드맵 및 현재 상태 (Development Roadmap & Current Status)

* **전체 로드맵:**
    * Phase 0: 준비 및 학습 (완료)
    * Phase 1: 데이터 파이프라인 구축 (완료)
    * **Phase 2: "교사" 모델 개발 및 훈련 (진행 중)**
        * Task 2.1: BERT 모델 아키텍처 구현 (완료)
        * Task 2.2: 사전 훈련 실행 (부분 완료)
        * **Task 2.3: 미세 조정 및 평가 (현재 진행 예정)**
    * Phase 3: "학생" 모델 개발 및 배포 준비 (미착수)
    * Phase 4: 결과 정리 및 발표 (미착수)
* **현재 상태 요약 (Current Status):**
    * 클라우드 환경의 비용 및 하드웨어 제약으로, **사전 훈련이 22개 파트 중 약 4.5개만 완료**되었습니다.
    * 현재 보유한 가장 진보된 모델은 `epoch-14.pt` 입니다.
    * 따라서, **`epoch-14.pt`를 사용하여 '파일럿 미세 조정'을 진행하기로 결정**했습니다.
    * 이 '파일럿 미세 조정'을 위한 모든 요구사항은 **`docs/phase2_finetuning/finetune_spec.md` 파일에 명세서로 정의**되어 있으며, 당신의 모든 작업은 이 문서를 기준으로 해야 합니다.

### 5. 주요 기술 문제 해결 이력 (Key Technical Problem-Solving History)

* **환경 문제:** Colab 한계로 고성능 로컬/클라우드 환경으로 이전했습니다.
* **모듈 경로 문제:** 모든 스크립트는 **`python -m <package>.<module>`** 방식으로 실행하는 것을 표준으로 채택했습니다.
* **메모리 부족 문제:** 대용량 데이터 처리를 위해 **스트리밍 방식의 `IterableDataset`**과 **데이터 분할** 전략을 확립했습니다.
* **DDP 분산 학습 문제:** `find_unused_parameters=True` 설정 등으로 복잡한 DDP 훈련을 성공시킨 경험이 있습니다. 이 과정에서 얻은 교훈을 바탕으로 코드를 분석해야 합니다.

### 6. 핵심 코드베이스 요약 (Core Codebase Summary)

* `docs/phase2_finetuning/finetune_spec.md`: **현재 작업의 청사진. 반드시 먼저 참조할 것.**
* `models/teacher.py`: MLM 사전 훈련용 `CANBertForMaskedLM` 정의.
* `models/teacher_classifier.py`: 미세 조정용 `CANBertForClassification` 정의.
* `core/tokenizer.py`: 오프셋 기반 `CANTokenizer` 클래스.
* `core/dataset.py`: `MLMDataset`과, 새로 정의될 `ClassificationDataset` 포함.
* `scripts/finetune.py`: **이번에 당신이 주로 분석하고 수정하게 될 스크립트.**
* `scripts/evaluate.py`: 미세 조정된 모델의 성능을 평가할 스크립트.

### 7. 상호작용 지침 및 제약 조건 (Interaction Guidelines & Constraints)

1.  **명세서 준수:** 당신의 모든 코드 생성 및 수정 제안은 UGRP와 연구원님이 합의한 `*.md` 명세서를 기반으로 해야 합니다.
2.  **현재 과업 집중:** 항상 '현재 상태' 섹션에 명시된 당면 과업에 집중합니다. (現: 파일럿 미세 조정)
3.  **결과물 형식:** 코드 식별자와 기술 용어는 **영어**로, 코드 주석과 모든 부가 설명은 **한국어**로 제공합니다.
4.  **코드 스타일:** **PEP 8** 가이드와 **타입 힌트(Type Hint)**를 엄격히 준수합니다.
5.  **지식의 출처:** 당신의 모든 분석은 제공된 3개의 핵심 참고 논문과 이 메모리 프롬프트의 컨텍스트 안에서 이루어져야 합니다. 외부 지식을 활용할 경우, 반드시 그 출처와 타당성을 설명해야 합니다.

### 8. 핵심 참고 논문 분석 (Core Research Papers Analysis)

**※ 중요: 프로젝트 전체 기간 동안 항상 참고할 핵심 기술 자료**

본 프로젝트는 다음 3개의 핵심 논문을 기반으로 합니다. 모든 기술적 결정과 구현은 이 논문들의 방법론을 따라야 합니다.

#### 8.1 CAN-BERT (Alkhatib et al., 2022) - 교사 모델 기반 논문
- **핵심 기법**: BERT 기반 양방향 문맥 학습, MLM 방식 비지도 학습
- **아키텍처**: 4레이어 트랜스포머, 256 hidden size, 1개 어텐션 헤드
- **최적 하이퍼파라미터**: 마스크 비율 0.45, 배치 크기 32, 학습률 0.001
- **성능**: F1-score 0.85-0.99, 추론 시간 0.8-3.1ms
- **구현 요점**: 양방향 문맥 학습의 중요성, 실시간 탐지 가능한 경량 설계

#### 8.2 LSF-IDM (Cheng et al., 2023) - 지식 증류 기반 논문
- **핵심 기법**: 지식 증류(Knowledge Distillation) 기반 교사-학생 패러다임
- **교사 모델**: CAN-BERT (1.1억 파라미터), 학생 모델: BiLSTM (경량)
- **손실 함수**: `L = α * L_CE + (1-α) * L_KD`, 온도 파라미터 T 활용
- **성능**: DoS 99.99% F1-score, 교사 모델 대비 유사 성능, 파라미터 대폭 감소
- **구현 요점**: 소프트 타겟 생성, 확률 분포 유사성 학습, 실시간 배포 가능

#### 8.3 Jo & Kim (2024) - 데이터 처리 방법론 논문
- **핵심 기법**: 오프셋 기반 통합 어휘집, ID+페이로드 통합 시퀀싱
- **아키텍처**: 투-스트림 트랜스포머 (시간적-공간적 동시 분석)
- **데이터 처리**: CAN ID 값 분리 (ID: 256-2303, 페이로드: 0-255)
- **성능**: 추론 시간 0.0689ms/프레임, 예측 범위 확장으로 오탐 감소
- **구현 요점**: 슬라이딩 윈도우, 시퀀스 길이 최적화, 강건성 향상

#### 8.4 통합 기술 전략
1. **Phase 2 (현재)**: CAN-BERT 기반 교사 모델 미세 조정
2. **Phase 3 (향후)**: LSF-IDM 방식 지식 증류로 경량 학생 모델 개발
3. **데이터 처리**: Jo & Kim 방법론 적용 (오프셋 기반 통합 어휘집)
4. **성능 목표**: 실시간 탐지 (< 1ms), 높은 F1-score (> 0.95), 경량 모델 (< 10MB)

**※ 상세 분석 자료**: `/mnt/c/summer_vacation/can-ids-project/docs/research_papers_analysis.md` 참조