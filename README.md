```markdown
# 🏗️ 구조물 안정성 예측 AI (Monthly Dacon)

**2026.03 월간 데이콘: 구조물 안정성 물리 추론 AI 경진대회** 본 프로젝트는 2가지 시점(Front, Top)의 이미지를 활용하여 구조물의 안정성(Stable/Unstable)을 예측하는 Multi-View ResNet 모델입니다.

## 📌 주요 기능
* **Multi-View Input:** 정면(Front)과 상단(Top) 이미지를 결합하여 3차원적 특징 학습
* **Robust Augmentation:** 실험실 환경(Train)과 실제 환경(Dev/Test)의 조명/각도 차이를 극복하기 위한 강력한 데이터 증강 적용
* **Best Model Checkpoint:** 학습 중 검증(Validation) 손실이 가장 낮은 모델 자동 저장
* **Cross-Platform:** Windows(CPU/CUDA) 및 WSL2(Linux) 환경 지원

---

## 📂 디렉토리 구조
```text
26.03_monthly_dacon/
├── data/                  # (Git 제외) 원본 데이터 폴더
│   ├── train/             # 학습 이미지 및 영상
│   ├── dev/               # 검증 이미지 (무작위 환경)
│   ├── test/              # 평가 이미지
│   ├── train.csv
│   └── dev.csv
├── models/                # (Git 제외) 학습된 모델(.pth) 저장소
├── main.py                # 학습 및 추론 실행 코드
├── requirements.txt       # 필수 라이브러리 목록
└── README.md              # 프로젝트 설명

```

---

## ⚙️ 환경 설정 (Installation)

사용하는 운영체제와 그래픽카드(GPU) 버전에 따라 설치 방법이 다릅니다.

### 1️⃣ 가상환경 생성 (공통)

Python 3.10 버전을 권장합니다.

```bash
# Conda 사용 시
conda create -n ai_structure_reasoning python=3.10 -y
conda activate ai_structure_reasoning

```

### 2️⃣ PyTorch 설치 (⚠️ 중요)

#### 🅰️ Case A: RTX 50 시리즈 사용자 (WSL2 / Linux 권장)

RTX 5060, 5080 등 **Blackwell 아키텍처(sm_120)** 기반의 최신 GPU를 사용하는 경우, 일반 버전은 작동하지 않습니다. **반드시 Nightly(개발자) 버전**을 설치해야 합니다.

* **권장 환경:** WSL2 (Ubuntu) 또는 Linux
* **설치 명령어:**

```bash
# CUDA 12.6 (또는 12.4) 호환 Nightly 빌드 설치
pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu126](https://download.pytorch.org/whl/nightly/cu126)

```

> **주의:** 설치 후 `python -c "import torch; print(torch.cuda.is_available())"` 명령어로 `True`가 나오는지 확인하세요.

#### 🅱️ Case B: RTX 30/40 시리즈 또는 CPU 사용자 (Windows/Linux)

일반적인 환경에서는 Stable(정식) 버전을 설치합니다.

```bash
# Stable 버전 설치
pip install torch torchvision torchaudio

```

> **참고:** RTX 50 시리즈 사용자가 Windows에서 이 버전을 설치하면 `no kernel image` 에러가 발생하므로, `main.py`에서 `device='cpu'`로 설정해야 합니다.

### 3️⃣ 나머지 라이브러리 설치

```bash
pip install -r requirements.txt

```

---

## 🚀 실행 방법 (Usage)

환경 설정이 완료되면 아래 명령어로 학습과 추론을 한 번에 실행할 수 있습니다.

```bash
python main.py

```

### 실행 프로세스

1. **데이터 로드:** `data/` 폴더에서 이미지 로드
2. **학습 (Training):** 설정된 Epoch만큼 학습 진행 (Augmentation 적용)
3. **모델 저장:** 검증셋(dev) 기준 최고 성능 모델을 `models/best_model.pth`로 저장
4. **추론 (Inference):** 최고 성능 모델을 다시 불러와 `test` 데이터 예측
5. **결과 생성:** `submission.csv` 파일 생성

---

## 🛠️ 문제 해결 (Troubleshooting)

**Q. `RuntimeError: CUDA error: no kernel image is available` 에러가 발생해요.**

* **원인:** PyTorch 버전이 현재 그래픽카드의 아키텍처를 지원하지 않아서 발생합니다.
* **해결:**
1. (추천) WSL2 환경에서 **Case A**의 Nightly 버전을 재설치합니다.
2. (임시) `main.py` 파일의 device 설정을 `device = torch.device("cpu")`로 변경하여 CPU로 실행합니다.



**Q. `IndentationError` 또는 구문 오류가 발생해요.**

* 코드 복사/붙여넣기 과정에서 들여쓰기가 깨진 경우입니다. `main.py`의 함수 정의 부분을 확인해주세요.

---

## 📝 License

This project is based on the Dacon competition dataset.

```

```
