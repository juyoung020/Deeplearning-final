# Week 4 실행 명령어 정리 (현재 상태 기준)

repo: `/home/ad06/isaacsim/git/Deeplearning-final`
학습 데이터: `data/VLNVerse_data` (3,963 에피소드, 받음)
학습된 모델:
- baseline: `outputs/week4/baseline/best_adapter`
- trajectory-aware: `outputs/week4/ablation_trajectory_aware/best_adapter`

---

## 0. 공통 (항상 먼저)

```bash
source /home/ad06/miniconda3/etc/profile.d/conda.sh
conda activate goodnav
cd /home/ad06/isaacsim/git/Deeplearning-final
```

---

## 1. 학습 (Week 4 LoRA)

```bash
# baseline (이미 완료. 재학습 시)
python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json \
    --batch-size 6 --grad-accum-steps 3

# 중간에 끊겼을 때 이어하기
python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json \
    --batch-size 6 --grad-accum-steps 3 --resume

# ablation: trajectory-aware (지난 위치+이미지)  ← 완료됨
python week4/train_week4_qwen3vl.py --config week4/configs/week4_ablation_trajectory_aware.json \
    --batch-size 6 --grad-accum-steps 3

# ablation: history-stride3  [I_{t-3}, I_t]  ← 미실행
python week4/train_week4_qwen3vl.py --config week4/configs/week4_ablation_history_stride3.json \
    --batch-size 6 --grad-accum-steps 3

# 빠른 시험 (소량)
python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json \
    --max-episodes 50 --epochs 1
```

주요 override 플래그: `--batch-size --grad-accum-steps --epochs --max-episodes --max-samples --save-every-steps --resume --dataset-root --split-json`

---

## 2. 평가 (offline, full val)

```bash
# baseline
python week4/eval_week4_offline.py --config week4/configs/week4_qwen3vl_lora.json \
    --adapter outputs/week4/baseline/best_adapter --max-samples 20000 \
    --out outputs/week4/baseline/predictions/offline_eval_fullval.json

# trajectory-aware
python week4/eval_week4_offline.py --config week4/configs/week4_ablation_trajectory_aware.json \
    --adapter outputs/week4/ablation_trajectory_aware/best_adapter --max-samples 20000 \
    --out outputs/week4/ablation_trajectory_aware/predictions/offline_eval_fullval.json
```

---

## 3. 추론 (단일 샘플)

```bash
python week4/infer_week4_action.py --adapter outputs/week4/baseline/best_adapter \
    --instruction "Turn right toward the door" --images frame0.jpg frame1.jpg
# --adapter 생략 시 base 모델 zero-shot
```

---

## 4. 그래프 / 보고서 자료

```bash
# loss/accuracy 그래프 생성 → outputs/week4/<run>/plots/
python week4/make_plots.py --run outputs/week4/baseline
python week4/make_plots.py --run outputs/week4/ablation_trajectory_aware
```

---

## 5. Smoke test (검증)

```bash
python week4/smoke_test_week4.py --only tier1     # 무다운로드
python week4/smoke_test_week4.py --with-model     # 모델 받아 전체 검증
```

---

## 6. 데이터 다운로드 (이미 받음. 재다운로드/추가 시)

```bash
# dry-run (무엇을 받을지만)
python week4/download_vlnverse.py --dest data/VLNVerse_data --max-episodes 50
# 실제 (rgb만, 16스레드 병렬). HF 토큰 있으면 빠름.
python week4/download_vlnverse.py --dest data/VLNVerse_data --no-dry-run --workers 16
```

---

## 7. IsaacSim 시뮬레이터 실행

### 7-1. 기존 데모 (WASD / GT 등)
```bash
# Isaac Sim 환경까지 한 번에
source /home/ad06/miniconda3/etc/profile.d/conda.sh && conda activate goodnav && \
cd /home/ad06/isaacsim/isaac-sim-standalone-4.5.0-linux-x86_64 && \
source setup_conda_env.sh && \
export PYTHONPATH=/home/ad06/IsaacLab/source/isaaclab:$PYTHONPATH && \
cd /home/ad06/isaacsim/git/Deeplearning-final && \
python demo.py --task fine --index 0 --work_dir ./myresults --agent go2 --go2-controller physics
```

### 7-2. Week 4 VLN 모델로 closed-loop 주행
위 환경 활성화 후:
```bash
python demo.py --task fine --index 0 --agent go2 --go2-controller physics \
    --go2-physics-week4-vln-checkpoint outputs/week4/baseline/best_adapter
```
- trajectory-aware 모델로: `--go2-physics-week4-vln-checkpoint outputs/week4/ablation_trajectory_aware/best_adapter --go2-physics-week4-vln-trajectory-aware`
- base zero-shot: `--go2-physics-week4-vln-checkpoint ""`
- ⚠️ egocentric RGB capture는 실sim에서 별도 검증 필요(실패 시 검은 프레임 fallback).

---

## 8. 정리 (공용 컴퓨터 — 나중에 삭제)

```bash
bash week4/CLEANUP.sh         # 무엇이 지워질지 미리보기
bash week4/CLEANUP.sh --yes   # 데이터/모델캐시/outputs 삭제 (코드/문서는 보존)
```

---

## 빠른 순서 요약
1. `conda activate goodnav` + repo 이동 (§0)
2. 학습: `python week4/train_week4_qwen3vl.py --config ... --batch-size 6 --grad-accum-steps 3` (§1)
3. 평가: `python week4/eval_week4_offline.py --adapter outputs/week4/<run>/best_adapter ...` (§2)
4. 그래프: `python week4/make_plots.py --run outputs/week4/<run>` (§4)
5. 시뮬: §7 (Isaac 환경 활성화 후 demo.py)
```
