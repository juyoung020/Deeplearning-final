# Week 3 — Language Velocity Model

자연어 locomotion command를 입력받아 quadruped robot(Go2)의 velocity command `[vx, vy, yaw_rate]`를 생성하는 딥러닝 모델.

---

## 역할 분담

| 담당 | 이름 | 파일 |
|------|------|------|
| A — 모델 아키텍처 | 김주영 | `language_velocity_model.py` |
| B — 학습 파이프라인 | 박성현 | `train_language_velocity.py`, `configs/week3_siglip_velocity.json` |
| C — 추론 + 시뮬레이터 | 변민석 | `infer_language_velocity.py`, `go2_physics_teleop.py` |
| D — 문서화 + 데모 | 한준태 | `demo.py`, `README.md` |

---

## Model Architecture

```
locomotion text (e.g. "move forward 50cm")
        ↓
SigLIP2-L/16-256 text encoder  (freeze — 학습 안 함)
        ↓
  text embedding (1024 dim)
        ↓
  Linear(1024 → 128) + ReLU
        ↓
  Linear(128 → 3)
        ↓
  output: [vx, vy, yaw_rate]
```

- SigLIP2 text encoder는 고정(freeze) — 이미지 없이 텍스트 임베딩만 추출
- MLP만 학습: 1024-dim 임베딩 → 3D velocity

---

## Dataset

기본 11개 command(과제 명세 기준)에서 paraphrase 확장 + backward 추가로 **총 46개**.

| 카테고리 | 학습 예시 | Target [vx, vy, yaw] |
|----------|-----------|----------------------|
| Forward | move/go/walk forward 25~100cm, advance | [0.25~1.0, 0, 0] |
| Backward | move/go backward 25~100cm, reverse, back up | [-0.25~-1.0, 0, 0] |
| Turn Left | turn/rotate/spin left 15~45 degree | [0, 0, π/12 ~ π/4] |
| Turn Right | turn/rotate/spin right 15~45 degree | [0, 0, -π/12 ~ -π/4] |
| Stop | stop, halt, stay, don't move | [0, 0, 0] |

velocity target 설계 기준: `velocity = distance / 1초` (시뮬레이터가 커맨드당 1초 실행)

---

## Prerequisites

```bash
conda create -n goodnav python=3.10
conda activate goodnav
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pandas scipy==1.10.1
pip install -r requirements-week3.txt

# Isaac Sim 설정
cd <ISAACSIM_ROOT>
source setup_conda_env.sh
```

---

## Training

```bash
python train_language_velocity.py --config configs/week3_siglip_velocity.json
```

| Hyperparameter | Value |
|----------------|-------|
| Training steps | 3000 |
| Batch size | 32 |
| Learning rate | 0.001 |
| LR Scheduler | CosineAnnealingLR (→ 1e-5) |
| Weight decay | 1e-4 |
| Grad clip norm | 1.0 |
| Loss | MSE |

### 학습 결과

| 지표 | 값 |
|------|-----|
| MSE | ≈ 0.00000000 |
| MAE | ≈ 0.00000004 |
| MAE (vx / vy / yaw) | ≈ 0 / 0 / 0 |

checkpoint는 `checkpoints/language_velocity_mlp.pt`에 저장됨.

---

## Inference

```bash
# 전체 default command 일괄 확인
python infer_language_velocity.py --all

# 단일 command
python infer_language_velocity.py --text "go forward 50cm"

# 결과를 JSON으로 저장
python infer_language_velocity.py --all --output runs/result.json

# 대화형 모드
python infer_language_velocity.py --interactive
```

---

## Simulator Execution

```bash
source /home/ad06/miniconda3/etc/profile.d/conda.sh && \
conda activate goodnav && \
cd /home/ad06/isaacsim/isaac-sim-standalone-4.5.0-linux-x86_64 && \
source setup_conda_env.sh && \
export PYTHONPATH=/home/ad06/IsaacLab/source/isaaclab:$PYTHONPATH && \
cd /home/ad06/isaacsim/IAmGoodNavigator && \
python demo.py --task fine --index 0 --work_dir ./myresults --agent go2 --go2-controller physics
```

`checkpoints/language_velocity_mlp.pt`가 있으면 플래그 없이도 학습된 모델이 자동으로 로드됨.

---

## Supported Commands

시뮬레이터 실행 후 터미널에 입력. 각 커맨드는 **1초** 동안 실행됨.

### Forward
```
move forward 25cm / 50cm / 75cm / 1m
go forward 25cm / 50cm / 75cm / 1 meter
walk forward 25 centimeters / 50 centimeters / 1 meter
advance 50cm / 1 meter
```

### Backward
```
move backward 25cm / 50cm / 75cm / 1m
go backward 25cm / 50cm / 75cm / 1 meter
reverse 25 centimeters / 50 centimeters / 1 meter
back up 50cm
```

### Turn Left
```
turn left 15 / 30 / 45 degree
rotate left 15 / 30 / 45 degrees
spin left 30 / 45 degrees
```

### Turn Right
```
turn right 15 / 30 / 45 degree
rotate right 15 / 30 / 45 degrees
spin right 30 / 45 degrees
```

### Stop
```
stop / halt / stay / don't move
```

### 특수 커맨드
```
finish    → 미션 완료 (평가 팝업)
```

---

## File Structure

```
IAmGoodNavigator/
├── language_velocity_model.py      # A: SigLIP2 인코더 + VelocityMLP + 추론기
├── train_language_velocity.py      # B: 학습 파이프라인
├── infer_language_velocity.py      # C: 추론 CLI
├── go2_physics_teleop.py           # C: 시뮬레이터 연동
├── demo.py                         # D: Isaac Sim 시각화 + 평가
├── configs/
│   └── week3_siglip_velocity.json  # B: 하이퍼파라미터 + 데이터셋 정의
├── checkpoints/
│   └── language_velocity_mlp.pt    # 학습된 모델 가중치
├── requirements-week3.txt          # transformers, safetensors
└── README_week3.md                 # 이 파일
```

---

## Tested Environment

- OS: Ubuntu 24.04
- GPU: RTX 4090 (Driver 570.195.03, CUDA 12.8)
- Isaac Sim: 4.5.0
- Python: 3.10
- PyTorch: 2.5.1
