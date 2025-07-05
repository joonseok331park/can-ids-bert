# 자율주행차 보안을 위한 CAN 버스 침입 탐지 시스템

## 1. 프로젝트 개요

본 프로젝트는 3개의 핵심 참고 논문에 기반하여, 실제 배포 가능한 고성능/고효율의 차량용 CAN 버스 침입 탐지 시스템(IDS) 개발을 목표로 합니다. `CAN-BERT` 모델을 교사(Teacher)로 사전 훈련하고, 지식 증류(Knowledge Distillation)를 통해 경량 학생(Student) 모델을 최종적으로 개발합니다.

data_loader.py -> tokenizer.py -> aggregate_data.py -> build_vocab.py -> split_data.py -> run_full_training.py -> pretrain.py -> teacher.py -> dataset.py

## 2. 핵심 기술 및 참고 자료

* **언어/프레임워크:** Python, PyTorch, Hugging Face Transformers
* **핵심 방법론:** Masked Language Model (MLM), Knowledge Distillation
* **참고 논문:**
    1.  [cite_start]`CAN-BERT do it?...` (Alkhatib et al., 2022) [cite: 1]
    2.  [cite_start]`Intrusion Detection Using Transformer...` (Jo & Kim, 2024) [cite: 2]
    3.  [cite_start]`LSF-IDM...` (Cheng et al., 2023) [cite: 3]

## 3. 권장 디렉토리 구조

프로젝트의 안정적인 임포트(import)를 위해 각 하위 폴더에 `__init__.py` 파일(내용은 비어있음)을 반드시 포함해야 합니다.

can-ids-project/
├── checkpoints/
├── core/
│   ├── init.py
│   ├── dataset.py
│   └── tokenizer.py
├── data/
│   ├── aggregated_parts/
│   └── HCRL_dataset/
├── dataset/
│   └── CAN-MIRGU(train)/
│       └── Benign/
├── models/
│   ├── init.py
│   └── teacher.py
├── scripts/
│   ├── init.py
│   ├── aggregate_data.py
│   ├── build_vocab.py
│   ├── run_full_training.py
│   ├── split_data.py
│   └── pretrain.py
├── utils/
│   ├── init.py
│   └── data_loader.py
├── requirements.txt
└── README.md

## 4. 로컬 개발 환경 설정 가이드

새로운 데스크탑에서 아래 순서대로 환경을 설정합니다.

### 4-1. 사전 요구사항

1.  **NVIDIA 드라이버:** GPU에 맞는 최신 버전을 설치합니다.
2.  **CUDA Toolkit:** **CUDA 12.1** 버전을 [NVIDIA 공식 사이트](https://developer.nvidia.com/cuda-12-1-0-download-archive)에서 다운로드하여 설치합니다. (`exe (local)` 버전 권장)
3.  **Python:** Python 3.9 이상 버전을 설치합니다.
4.  **Git:** 코드 버전 관리를 위해 [Git을 설치](https://git-scm.com/downloads)합니다.

### 4-2. 프로젝트 설정

1.  **프로젝트 클론 (또는 복사):**
    터미널을 열고, 작업할 위치로 이동한 뒤 프로젝트를 가져옵니다.
    ```bash
    git clone [your-repository-url]
    cd can-ids-project
    ```

2.  **파이썬 가상 환경 생성 및 활성화:**
    프로젝트 폴더 내에서 아래 명령어를 실행하여 독립된 환경을 만듭니다.
    ```bash
    # 가상 환경 생성 (최초 1회)
    python -m venv venv

    # 가상 환경 활성화 (터미널을 새로 켤 때마다 실행)
    # Windows:
    .\venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```
    터미널 프롬프트 앞에 `(venv)`가 표시되면 성공입니다.

3.  **필수 라이브러리 설치:**
    가상 환경이 활성화된 터미널에서 아래 **2단계**를 순서대로 실행합니다.

    **1단계: PyTorch 설치 (CUDA 12.1 호환 버전)**
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
    ```

    **2단계: 나머지 라이브러리 설치**
    ```bash
    pip install -r requirements.txt
    ```

## 5. 실행 워크플로우

환경 설정이 완료된 후, 아래 순서대로 스크립트를 실행합니다.

### 5-1. 데이터 준비 (최초 1회 실행)

1.  **원본 데이터 위치:** `dataset/CAN-MIRGU(train)/Benign/` 폴더에 원본 `.log` 파일들을 위치시킵니다.

2.  **데이터 병합:** 흩어져 있는 로그 파일들을 하나의 큰 파일로 합칩니다.
    ```bash
    python -m scripts.aggregate_data
    ```

3.  **어휘집 생성:** 병합된 데이터를 기반으로 `vocab.json` 파일을 생성합니다.
    ```bash
    python -m scripts.build_vocab
    ```

4.  **훈련 데이터 분할:** 메모리 효율적인 훈련을 위해 병합된 파일을 다시 작은 조각으로 나눕니다.
    ```bash
    python -m scripts.split_data
    ```

### 5-2. 교사 모델 사전 훈련

모든 데이터 준비가 완료되었습니다. 아래 자동화 스크립트를 실행하여 전체 데이터에 대한 사전 훈련을 시작합니다.
```bash
python -m scripts.run_full_training