# Week 4·5 — VLN 다음 행동 예측 (Qwen3-VL-2B + LoRA) · 모델2(균형형)

자연어 지시 + 1인칭 RGB로 사족보행 로봇 Go2를 움직이는 **Vision-Language Navigation** 파이프라인입니다.
사전학습 멀티모달 모델 `Qwen/Qwen3-VL-2B-Instruct`를 **LoRA**로 미세조정해, 직전·현재 화면 `[I_{t-1}, I_t]`와
내비게이션 지시문을 입력받아 **네 가지 행동 중 하나**를 문장으로 생성하고, IsaacSim에서 `Stop`까지 실제로 주행시킵니다(closed-loop).

이 폴더는 **모델2(균형형)** — 데이터 불균형(전진 67% · 정지 2%)을 재샘플링으로 완화한 학습 라인의 코드입니다.

---

## 1. 행동 공간 (정확히 네 개)

| 모델 출력 | 속도 `[vx, vy, wz]` | 지속 |
|---|---|---|
| `Move forward 25cm` | `[1.0, 0.0, 0.0]` m/s | 0.25 s |
| `Turn left 15 degree` | `[0.0, 0.0, 1.047]` rad/s | 0.25 s |
| `Turn right 15 degree` | `[0.0, 0.0, -1.047]` rad/s | 0.25 s |
| `Stop` | `[0.0, 0.0, 0.0]` | 종료 |

형식이 어긋난 출력은 `Move forward 25cm`로 폴백하고, 그 비율(invalid rate)을 기록합니다.

---

## 2. 무엇을 했나 (요약)

1. **실데이터 진단** — VLN-VERSE parquet의 `observation.action` 라벨과 실제 `rgb.npy`를 직접 열어 행동 분포가
   **전진 67% / 좌 16% / 우 15% / 정지 2%** 로 심하게 치우쳐 있음을 확인. 그냥 학습하면 "무조건 전진"으로 붕괴.
2. **데이터셋 빌드** — 타임스텝마다 `{instruction, images:[I_{t-1}, I_t], answer}` 한 샘플 생성(`t=0 → [I_0, I_0]`). 총 19.5만 샘플.
3. **균형 재샘플링(모델2 핵심)** — `balance_power 0.7`로 회전·정지를 더 자주 뽑아 불균형 완화.
   강도 0.0~1.0 스윕 결과 **1.0은 제자리 회전**, **중간값 0.7**이 전진·회전을 모두 살림.
4. **폭발 방지 프리플라이트** — 본학습 전 작은 데이터로 loss 마스킹·메모리·재현성·LoRA 부착 위치를 자동 점검.
5. **본학습** — 전량 데이터로 6 epoch 학습. 검증은 **epoch 1이 정점**(macro 0.742 · 정지 0.875) 뒤 곧장 악화 →
   회전 데이터를 많이 섞을수록 소수 클래스를 빨리 외워 **과적합이 더 일찍** 옴. best는 epoch 1로 저장.
6. **closed-loop 평가** — IsaacSim에서 SR/OSR/SPL/nDTW/Goal Dist + invalid rate 측정.

평가 지표는 전진에 가려지는 전체 정확도 대신, **네 행동을 고르게 보는 macro 정확도 + 정지 recall**을 1급 기준으로 삼았습니다.

---

## 3. 실행 환경 (이 머신 기준)

```powershell
# 경로의 공백("C:\Users\user one\...") 때문에 conda 런처가 깨져, env 인터프리터를 직접 호출합니다.
$PY = "C:\Users\user one\anaconda3\envs\goodnav\python.exe"
$env:HF_HUB_DISABLE_SYMLINKS = "1"   # Windows: 심볼릭 링크 대신 복사(관리자 권한 불필요)
```

`goodnav`(python 3.10): PyTorch 2.11.0+cu128(Blackwell sm_120), transformers 5.x, peft, pyarrow, opencv.
closed-loop(IsaacSim)는 별도 `isaaclab` env에서 실행합니다.

---

## 4. 파이프라인

### ① 데이터 다운로드 (필요한 에피소드만 선택적으로)

```powershell
& $PY download_vlnverse.py --root C:\Deeplearning-final\VLNVerse_data --split fine_train
# closed-loop 평가용 씬 (≈53 GB 추가)
& $PY download_vlnverse.py --root C:\Deeplearning-final\VLNVerse_data --split fine_val --scenes
```
`depth.npy`는 건너뜀, fine_train RGB만 ≈28 GB. 끊겨도 이어받는 보조 스크립트: `download_resilient.py`.

### ② 학습 JSONL 빌드 (parquet `observation.action` + `rgb.npy`)

```powershell
& $PY build_week4_dataset.py `
  --vlnverse-root C:\Deeplearning-final\VLNVerse_data `
  --split-json C:\Deeplearning-final\VLNVerse_data\raw_data\final_splits\fine_train.json.gz `
  --out data\week4_train.jsonl --frames-dir data\frames
```

### ③ 본학습 전 프리플라이트 (전 항목 PASS여야 발사)

```powershell
& $PY preflight.py --data data\week4_train.jsonl --image-root data\frames
```
loss 마스킹 단위검증 · overfit-tiny sanity · 에피소드 누수 assert · VRAM 헤드룸 · resume sanity 등을 자동 점검.

### ④ LoRA 미세조정 — 모델2(균형형) 레시피

```powershell
& $PY train_week4_lora.py `
  --data data\week4_train.jsonl --image-root data\frames `
  --val-episodes-json data\eval\vlnverse_closed_loop_eval_20episodes.json `
  --output adapters\week4_lora `
  --epochs 6 --balance-power 0.7 --select-metric macro `
  --batch-size 4 --grad-accum 4 --wandb
```
- LoRA `q_proj`·`v_proj`만 (전체의 0.15% = 320만 파라미터), r=16 / α=32 / dropout=0.05.
- AdamW · lr=1e-4 · cosine + 3% warmup · bf16 · gradient checkpointing.
- **에피소드 단위 train/val 분할**(같은 집을 학습·검증에 섞지 않음. frame 단위 분할 금지).
- epoch마다 checkpoint 저장(중단 시 `--resume`로 재개), **macro 기준 best**를 `best/`로 저장.
- 검증 로그: overall + 클래스별 정확일치 정확도, val loss.

### ⑤ closed-loop 평가 (IsaacSim, isaaclab env)

```bash
# IAmGoodNavigator 디렉토리에서, isaaclab env 활성화 상태로:
python demo.py --task fine --index 0 --work_dir ./myresults --agent go2 --go2-controller physics \
  --go2-physics-week4-vln-checkpoint <...>/adapters/week4_lora/best
```
지표(`metrics.py`): SR(목표 3m 이내) · OSR · SPL · nDTW · Goal Dist + invalid rate.
> ⚠️ RTX 5070 Ti(Blackwell)에서는 시뮬을 CPU 파이프라인으로 돌려야 렌더 프리즈를 피합니다(모델만 GPU).
> 자세한 내용은 `REPO_SPECIFIC_WEEK4_NOTES.md` 참고.

---

## 5. 파일 구성

| 파일 | 역할 |
|---|---|
| `actions.py` | 행동 매핑 · 시스템 프롬프트 · 출력 파싱(공용 코어) |
| `download_vlnverse.py` / `download_resilient.py` | VLN-VERSE 선택적 다운로드 / 재시도 다운로드 |
| `build_week4_dataset.py` | parquet + rgb.npy → 학습 JSONL + 프레임 추출 |
| `preflight.py` | 본학습 전 폭발 방지 자동 점검 |
| `train_week4_lora.py` | **LoRA 미세조정**(균형 재샘플링·macro best·resume) |
| `infer_week4_action.py` | 오프라인 추론(greedy, max_new_tokens=8) |
| `vln_commander.py` / `vlm_server.py` | closed-loop 시뮬레이터 연동(추론 서버) |
| `metrics.py` | SR/OSR/SPL/nDTW/GoalDist 계산 |
| `smoke_test_*.py` | 데이터셋 · 로직 · VLM · IsaacSim 렌더 스모크 테스트 |

---

## 6. 검증 현황

| 점검 | 상태 |
|---|---|
| cu128 / Blackwell sm_120 | ✅ |
| 로직 스모크(4행동·속도·t=0 중복·max_steps) | ✅ |
| 실제 Qwen3-VL 추론(in-vocab 행동 생성) | ✅ |
| 데이터셋 빌더(합성 + 실제 에피소드) | ✅ |
| LoRA 학습(실제 에피소드, 320만 trainable = q_proj+v_proj) | ✅ |
| metrics SR/OSR/SPL/nDTW/GoalDist | ✅ |
| 본학습 6 epoch 완료(best epoch 1: macro 0.742 · 정지 0.875 · invalid 0%) | ✅ |
| IsaacSim closed-loop(Blackwell, CPU 시뮬 파이프라인) | ✅ |

> 통합 계약과 IsaacSim 트러블슈팅은 `REPO_SPECIFIC_WEEK4_NOTES.md` 참고.
