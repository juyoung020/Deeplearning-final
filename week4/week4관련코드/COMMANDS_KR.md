# Week 4 명령어 정리 (한글)

VLN-VERSE next-action 예측 (Qwen3-VL-2B + LoRA) 작업에 쓰는 명령어 모음입니다.
모든 명령은 repo 루트 `/home/ad06/isaacsim/git/Deeplearning-final` 기준입니다.

---

## 0. 공통 환경 준비 (매번 먼저)

```bash
source /home/ad06/miniconda3/etc/profile.d/conda.sh
conda activate goodnav
cd /home/ad06/isaacsim/git/Deeplearning-final
```

- `goodnav` 환경에 torch 2.5.1+cu118 / transformers 5.9.0(Qwen3-VL 지원) / peft / accelerate / pyarrow 가 이미 설치되어 있습니다.
- GPU: RTX 4090 24GB, bf16 지원.

---

## 1. 의존성 설치 (최초 1회, 이미 완료됨)

```bash
pip install -r week4/requirements_week4.txt
```

> torch / transformers 는 이미 깔려 있으므로 새로 깔거나 업그레이드하지 마세요.
> 추가로 필요한 것: `peft`, `accelerate`, `pyarrow` (필수) / `wandb`, `imageio[ffmpeg]` (선택).

설치 확인:

```bash
python -c "import torch,transformers,peft,accelerate,pyarrow; \
print('torch',torch.__version__,'tf',transformers.__version__,'cuda',torch.cuda.is_available())"
```

---

## 2. Smoke test (학습 전 동작 검증)

```bash
# (A) 다운로드 없이 순수 로직 검증 — 빠름
python week4/smoke_test_week4.py --only tier1

# (B) 모델까지 받아 전체 파이프라인 검증 (prompt/label-mask/forward/loss/generation/소규모 학습)
python week4/smoke_test_week4.py --with-model
```

---

## 3. 데이터 다운로드 (선택적, 기본 dry-run)

HF repo: `Eyz/VLNVerse_data`. 전체 fine-train ≈ 65 GiB 이므로 **반드시 선택적으로** 받으세요.

```bash
# (A) 먼저 dry-run 으로 무엇을 받을지 확인 (실제 다운로드 안 함)
python week4/download_vlnverse.py --dest data/VLNVerse_data --max-episodes 50

# (B) 에피소드 N개만 실제 다운로드 (rgb만, depth 제외 → 용량 절반)
python week4/download_vlnverse.py --dest data/VLNVerse_data --max-episodes 50 --no-dry-run

# (C) 전체 train 다운로드 (디스크 충분할 때만!)
python week4/download_vlnverse.py --dest data/VLNVerse_data --no-dry-run

# depth 도 같이 받고 싶으면 --include-depth 추가
```

> ⚠️ 현재 디스크가 97%(~13GB)로 빠듯합니다. 전체 다운로드 전 공간 확보 필요.

---

## 4. 학습 (메인)

```bash
# 기본(baseline) 학습
python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json

# 중간에 꺼졌을 때 이어서 학습 (가장 최근 latest 체크포인트부터)
python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json --resume

# 빠른 시험 학습 (소량 샘플/에폭)
python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json \
    --max-episodes 20 --max-samples 200 --epochs 1

# 중간 저장 주기 바꾸기 (optimizer step 단위, 기본 200)
python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json \
    --save-every-steps 100
```

### 중간 저장 / 이어하기 동작
- `save_every_steps`(기본 200) 마다 `outputs/<run>/checkpoints/latest/` 에 중간 저장.
- **Ctrl-C 나 크래시가 나도** 그 직전까지 학습한 내용이 `latest/` 에 비상 저장됨.
- `--resume` 를 붙이면 epoch 중간 지점부터 **정확히** 이어서 학습 (이미 본 배치는 건너뜀).
- epoch 끝마다 `checkpoints/epoch_XXX/` 저장, validation loss 최저일 때 `best_adapter/` 저장.

산출물 위치:
```
outputs/week4/baseline/
├── checkpoints/
│   ├── latest/            ← 중간/비상 저장 (이어하기용)
│   ├── epoch_001/ ...
├── best_adapter/         ← validation 기준 최고 모델
├── predictions/          ← epoch별 val 예측/지표
├── logs/train_log.jsonl  ← W&B 없을 때 로그
└── config.json
```

---

## 5. 평가 (offline)

```bash
python week4/eval_week4_offline.py \
    --config week4/configs/week4_qwen3vl_lora.json \
    --adapter outputs/week4/baseline/best_adapter \
    --max-samples 500
```
→ overall accuracy / per-class accuracy / invalid output rate 출력 + `predictions/offline_eval.json` 저장.

---

## 6. 추론 (단일 샘플)

```bash
python week4/infer_week4_action.py \
    --adapter outputs/week4/baseline/best_adapter \
    --instruction "Turn right toward the door" \
    --images frame0.jpg frame1.jpg
```
> `--adapter` 생략 시 base 모델 zero-shot.

---

## 7. Ablation 실험

```bash
# A. history 간격 넓히기 [I_{t-3}, I_t]
python week4/train_week4_qwen3vl.py --config week4/configs/week4_ablation_history_stride3.json

# B. trajectory-aware (이전 action/odometry 추가, leakage-safe)
python week4/train_week4_qwen3vl.py --config week4/configs/week4_ablation_trajectory_aware.json
```

---

## 8. IsaacSim closed-loop (additive, 기존 demo 재사용)

먼저 `start.md` 의 Isaac Sim 환경을 activate 한 뒤:

```bash
python demo.py --task fine --index 0 --agent go2 --go2-controller physics \
    --go2-physics-week4-vln-checkpoint outputs/week4/baseline/best_adapter
```

- `--go2-physics-week4-vln-checkpoint ""` → base 모델 zero-shot.
- `--go2-physics-week4-vln-trajectory-aware` → trajectory-aware 입력 사용.
- Week 4 commander 켜지면 Week 2/3 language prompt 는 자동 비활성.
- ⚠️ egocentric RGB capture(`Week4EgocentricCapture`)는 실제 sim 안에서 별도 검증 필요. 실패 시 검은 프레임 fallback.

---

## 9. 디버그 / 모니터링

```bash
# 학습 로그 실시간 보기
tail -f outputs/week4/baseline/logs/train_log.jsonl

# GPU 사용량
nvidia-smi

# 디스크 여유
df -h /home/ad06
```

---

## 빠른 순서 요약

1. `conda activate goodnav` → repo 루트로 이동
2. `python week4/smoke_test_week4.py --only tier1` (검증)
3. `python week4/download_vlnverse.py --dest data/VLNVerse_data --max-episodes 50 --no-dry-run` (데이터)
4. `python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json` (학습)
5. 끊기면 `... --resume` 로 이어하기
6. `python week4/eval_week4_offline.py --adapter outputs/week4/baseline/best_adapter ...` (평가)
