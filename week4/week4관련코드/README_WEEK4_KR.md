# Week 4 — VLN-VERSE 다음 행동 예측 (Qwen3-VL-2B + LoRA) [한글]

전체 내비게이션 instruction + 1인칭(egocentric) RGB 이미지 이력 `[I_{t-1}, I_t]` 을 입력받아
`Qwen/Qwen3-VL-2B-Instruct` 를 LoRA(`q_proj`/`v_proj`만)로 fine-tuning 하여, 다음 행동을
아래 4개 canonical 문자열 중 정확히 하나로 생성합니다.

```
Move forward 25cm
Turn right 15 degree
Turn left 15 degree
Stop
```

Week 1~3 코드는 건드리지 않았고, 새 코드는 전부 `week4/` 에 있습니다. closed-loop 연결을 위해
`week3/demo.py`, `week3/go2_physics_teleop.py` 에 **추가(additive)** 변경만 했습니다(모두 `###4` 주석).

## 파일 구성

| 파일 | 역할 |
|---|---|
| `week4_actions.py` | canonical action, id↔text 매핑, 출력 parser, IsaacSim command 변환기 (의존성 없음) |
| `vlnverse_dataset.py` | split 파싱, 에피소드 경로 해석, parquet/RGB(npy) 읽기, history 선택(`t=0`→`[I_0,I_0]`), **episode 단위 split**, synthetic 데이터셋 |
| `trajectory_features.py` | leakage-safe 상대변위/이전행동 텍스트 (ablation용) |
| `qwen3vl_prompt.py` | chat-template 메시지 빌더 + **정답 토큰만 라벨 마스킹** collator |
| `qwen3vl_lora_model.py` | 모델/processor 로드, LoRA `q_proj`/`v_proj` (vision tower 제외 & frozen 검증) |
| `week4_config.py` | JSON config → dataclass |
| `train_week4_qwen3vl.py` | 학습 루프, grad accum, cosine+warmup, **중간저장/이어하기/비상저장**, best-adapter, W&B/JSONL |
| `eval_week4_offline.py` | accuracy / class별 accuracy / invalid 비율 |
| `infer_week4_action.py` | `Week4ActionPredictor` + CLI |
| `vln_commander.py` | IsaacSim용 `QwenVLNActionCommander` |
| `download_vlnverse.py` | HF 선택적 다운로드 (**기본 dry-run**) |
| `smoke_test_week4.py` | Tier1(무다운로드) + Tier2(모델 필요) 검증 |
| `configs/` | baseline + ablation 2종 |

## 핵심 규칙 (과제 명세)

- **프롬프트**: 반드시 `processor.apply_chat_template` 사용. Qwen special token 을 문자열로 직접 넣지 않음.
  학습 시 `add_generation_prompt=False`(정답 포함), 추론 시 `add_generation_prompt=True`(정답 미포함).
- **Loss**: assistant 정답 토큰(+종료토큰)에만 적용. system/이미지/instruction/패딩은 `-100` 마스킹. 라벨 수동 shift 금지.
- **LoRA**: `q_proj`/`v_proj`만. vision encoder/base LM/tokenizer/임베딩은 freeze. (실측: language_model 56개 모듈, 학습 파라미터 약 0.15%, vision frozen 확인됨)
- **Validation**: train 데이터에서 **episode 단위** split만 허용(frame 단위 금지). `fine_val.json.gz` 는 학습 중 validation 에 쓰지 않음(최종 closed-loop 용).
- **Action 매핑**: `0→Stop, 1→Move forward 25cm, 2→Turn left 15 degree, 3→Turn right 15 degree`.
- **이미지 순서**: 항상 oldest→newest.

## 실제 VLN-VERSE 데이터 형식 (직접 확인함)

- split `raw_data/final_splits/fine_train.json.gz` = `{"episodes":[...]}`.
- 각 레코드: `scan`(예 `kujiale_0254`), `episode_id`(예 `kujiale_0254_23_4`, scan 접두사 포함), `instruction.instruction_text`.
- **실제 디렉터리** = `traj_data/vlnverse/<scan>/<episode_id 에서 "<scan>_" 제거>` (예 `kujiale_0254/23_4`).
- RGB 는 `videos/chunk-000/observation.images.rgb/rgb.npy`, shape `(T,256,256,3)` uint8 (mp4 아님).
- parquet pose 컬럼(`observation.robot_position` 등)은 문자열 `"[x,y,z]"`. `observation.action` 은 int64, 마지막 행은 이미 0(Stop).

## 설치 / 데이터 (사용자 작업)

```bash
conda activate goodnav
pip install -r week4/requirements_week4.txt   # peft, accelerate, pyarrow (+ wandb, imageio)
```

데이터(선택적, dry-run 먼저):

```bash
python week4/download_vlnverse.py --dest data/VLNVerse_data --max-episodes 50            # dry-run
python week4/download_vlnverse.py --dest data/VLNVerse_data --max-episodes 50 --no-dry-run
```

전체 fine-train ≈ 65 GiB. closed-loop fine_val 은 200GB 이상 필요 → `vlnverse_fine_val_top53_scans.json`(53 scan / 232 episode) subset 사용.

## Smoke test

```bash
python week4/smoke_test_week4.py --only tier1     # 무다운로드 (통과 확인됨)
python week4/smoke_test_week4.py --with-model     # 모델 받아 forward/loss/generation/학습 검증 (통과 확인됨)
```

## 학습 / 평가 / 추론

```bash
python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json
python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json --resume   # 이어하기

python week4/eval_week4_offline.py --config week4/configs/week4_qwen3vl_lora.json \
    --adapter outputs/week4/baseline/best_adapter --max-samples 500

python week4/infer_week4_action.py --adapter outputs/week4/baseline/best_adapter \
    --instruction "Turn right toward the door" --images f0.jpg f1.jpg
```

### 중간저장 / 이어하기 (중요)
- `save_every_steps`(기본 200) 마다 `checkpoints/latest/` 에 저장.
- **Ctrl-C / 크래시 시에도** 직전까지 학습분이 `latest/` 에 비상 저장됨.
- `--resume` 로 **epoch 중간 지점부터 정확히** 이어서 학습(이미 처리한 배치는 건너뜀, 재학습 없음).
- epoch 마다 `checkpoints/epoch_XXX/`, val loss 최저일 때 `best_adapter/` 저장.
- (검증됨) step2 저장 → step3 크래시 → 비상저장 → 재개 시 앞 3배치 skip → step8 까지 완주.

## closed-loop (IsaacSim, additive)

```bash
python demo.py --task fine --index 0 --agent go2 --go2-controller physics \
    --go2-physics-week4-vln-checkpoint outputs/week4/baseline/best_adapter
```
`--go2-physics-week4-vln-checkpoint ""` 는 base 모델 zero-shot. Week 4 commander 가 켜지면 Week 2/3 prompt 비활성.
egocentric RGB capture(`Week4EgocentricCapture`)는 실제 sim 에서 별도 검증 필요(실패 시 검은 프레임 fallback).

## Ablation

- `week4_ablation_history_stride3.json` — `[I_{t-3}, I_t]` vs baseline `[I_{t-1}, I_t]`.
- `week4_ablation_trajectory_aware.json` — leakage-safe 이전 action/odometry 이력 추가.

## 누수(leakage) 방지 (trajectory-aware)

- target `observation.action[t]`, 미래 pose/action, goal 거리, success 라벨은 **절대** 입력에 넣지 않음.
- 절대 좌표 대신 현재 프레임 기준 상대 변위 사용.
- closed-loop 에서는 GT 가 아니라 로봇 odometry + 이전에 실제 실행한 action 만 사용.
- `trajectory_features.py` / `vln_commander.py` 에서 강제, `smoke_test` 의 `test_trajectory_leakage` 로 검증.

> 명령어 모음은 `COMMANDS_KR.md` 참고. AI 사용 prompt 기록은 `AI_PROMPTS_USED_KR.md` (영문: `AI_PROMPTS_USED.md`).
