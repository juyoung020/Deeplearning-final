# Week 4 — VLN-VERSE Next-Action Prediction (Qwen3-VL-2B + LoRA)

Predict the next navigation action from a full instruction + egocentric RGB
history `[I_{t-1}, I_t]`, using `Qwen/Qwen3-VL-2B-Instruct` fine-tuned with LoRA
on `q_proj`/`v_proj` only. Output is exactly one of four canonical strings:

```
Move forward 25cm
Turn right 15 degree
Turn left 15 degree
Stop
```

Week 1–3 code is untouched; everything new lives in `week4/`, with small
**additive** hooks in `week3/demo.py` and `week3/go2_physics_teleop.py` for the
closed-loop commander (all marked `###4`).

## Files

| File | Purpose |
|---|---|
| `week4_actions.py` | canonical actions, id↔text mapping, output parser, IsaacSim command converter (dependency-free) |
| `vlnverse_dataset.py` | split parsing, episode resolution, parquet/RGB reading, history selection (`t=0`→`[I_0,I_0]`), episode-level split, synthetic dataset |
| `trajectory_features.py` | leakage-safe relative-displacement / previous-action text (ablation) |
| `qwen3vl_prompt.py` | chat-template message builders + collator with **answer-only label masking** |
| `qwen3vl_lora_model.py` | model + processor loader, LoRA on `q_proj`/`v_proj` (vision tower excluded & verified frozen) |
| `week4_config.py` | JSON config → dataclasses |
| `train_week4_qwen3vl.py` | training loop, grad-accum, cosine+warmup, ckpt/resume, best-adapter, W&B/JSONL |
| `eval_week4_offline.py` | accuracy / per-class accuracy / invalid-output rate |
| `infer_week4_action.py` | `Week4ActionPredictor` + CLI |
| `vln_commander.py` | `QwenVLNActionCommander` for IsaacSim (decision-time image buffer, Stop handling, invalid counting) |
| `download_vlnverse.py` | selective HF download (**dry-run by default**) |
| `smoke_test_week4.py` | Tier 1 (no download) + Tier 2 (needs model) tests |
| `configs/` | baseline + 2 ablations (history-stride-3, trajectory-aware) |

## Setup (needs user approval — installs + downloads)

```bash
conda activate goodnav
pip install -r week4/requirements_week4.txt          # peft, accelerate, pyarrow, (wandb, imageio)
```

`torch`/`transformers` are already present in `goodnav` (torch 2.5.1+cu118,
transformers 5.9.0 with Qwen3-VL support). RTX 4090, bf16 supported.

### Data (selective, dry-run first)

```bash
python week4/download_vlnverse.py --dest data/VLNVerse_data --max-episodes 5            # dry-run
python week4/download_vlnverse.py --dest data/VLNVerse_data --max-episodes 5 --no-dry-run
```

Full fine-train ≈ 65 GiB; closed-loop fine_val needs ≫200 GB. Use
`vlnverse_fine_val_top53_scans.json` (53 scans / 232 episodes) for final eval.

## Smoke tests

```bash
# Tier 1 — no downloads, pure python (already passing):
python week4/smoke_test_week4.py --only tier1

# Tier 2 — downloads Qwen3-VL-2B; checks prompt/label-mask/forward/loss/gen/train:
python week4/smoke_test_week4.py --with-model
```

## Train / eval / infer

```bash
python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json
python week4/train_week4_qwen3vl.py --config week4/configs/week4_qwen3vl_lora.json --resume
python week4/train_week4_qwen3vl.py --config ... --max-samples 16 --epochs 1   # quick run

python week4/eval_week4_offline.py --config week4/configs/week4_qwen3vl_lora.json \
    --adapter outputs/week4/baseline/best_adapter --max-samples 500

python week4/infer_week4_action.py --adapter outputs/week4/baseline/best_adapter \
    --instruction "Turn right toward the door" --images f0.jpg f1.jpg
```

## Closed-loop in IsaacSim (additive)

```bash
# (run inside the Isaac Sim env per start.md), then:
python demo.py --task fine --index 0 --agent go2 --go2-controller physics \
    --go2-physics-week4-vln-checkpoint outputs/week4/baseline/best_adapter
```

`--go2-physics-week4-vln-checkpoint ""` uses the base model zero-shot. When the
Week 4 commander is active, the Week 2/3 language prompt is disabled. Validate
egocentric RGB capture (`Week4EgocentricCapture`) before trusting closed-loop
numbers — see REPO_SPECIFIC_WEEK4_NOTES §6.

## Ablations

- `week4_ablation_history_stride3.json` — `[I_{t-3}, I_t]` vs baseline `[I_{t-1}, I_t]`.
- `week4_ablation_trajectory_aware.json` — adds leakage-safe action/odometry history.

## Anti-leakage guarantees (trajectory-aware)

- Never includes the target `observation.action[t]`, future pose/action, goal
  distance, or success label.
- Absolute world coordinates → current-frame relative displacement.
- Closed-loop uses robot odometry + previously executed actions only (no GT).
  Enforced in `trajectory_features.py` / `vln_commander.py` and checked by
  `smoke_test_week4.test_trajectory_leakage`.
