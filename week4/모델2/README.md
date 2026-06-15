# Week 4 — VLN Next-Action Prediction with Qwen3-VL-2B (LoRA)

Closed-loop Vision-Language Navigation: a LoRA-fine-tuned `Qwen/Qwen3-VL-2B-Instruct`
takes the egocentric RGB history `[I_{t-1}, I_t]` + the navigation instruction and
emits one of four actions, executed in IsaacSim until `Stop`.

Implements the official assignment spec ("Deep Learning Term Project Week 4").

## Action space (exactly four)

| Model output | Velocity `[vx, vy, wz]` | Duration |
|---|---|---|
| `Move forward 25cm` | `[1.0, 0.0, 0.0]` m/s | 0.25 s |
| `Turn left 15 degree` | `[0.0, 0.0, 1.047]` rad/s | 0.25 s |
| `Turn right 15 degree` | `[0.0, 0.0, -1.047]` rad/s | 0.25 s |
| `Stop` | `[0.0, 0.0, 0.0]` | terminate |

Invalid output → `Move forward 25cm` (recorded as invalid-output rate).

## Environment (this machine)

```powershell
# all commands use the env interpreter directly (the Anaconda launcher is broken by
# the space in "C:\Users\user one\..."); never rely on `conda activate` / .exe shims.
$PY = "C:\Users\user one\anaconda3\envs\goodnav\python.exe"
$env:HF_HUB_DISABLE_SYMLINKS = "1"   # Windows: copy instead of symlink (no admin needed)
```

`goodnav` (python 3.10) has PyTorch 2.11.0+cu128 (Blackwell sm_120), transformers 5.x,
peft, pyarrow, opencv. Verified: `cuda.is_available()=True`, Qwen3-VL inference OK.

## Pipeline

### 1. Download the data (selective — matched episodes only)

```powershell
& $PY download_vlnverse.py --root C:\Deeplearning-final\VLNVerse_data --split fine_train
# eval scenes (for closed-loop): adds ~53 GB
& $PY download_vlnverse.py --root C:\Deeplearning-final\VLNVerse_data --split fine_val --scenes
```
Skips `depth.npy`; fine_train rgb-only ≈ 28 GB.

### 2. Build the training JSONL (from parquet `observation.action` + `rgb.npy`)

```powershell
& $PY build_week4_dataset.py `
  --vlnverse-root C:\Deeplearning-final\VLNVerse_data `
  --split-json C:\Deeplearning-final\VLNVerse_data\raw_data\final_splits\fine_train.json.gz `
  --out data\week4_train.jsonl --frames-dir data\frames
```
One sample per timestep: `{instruction, images:[I_{t-1},I_t], answer}`; t=0 → `[I_0,I_0]`.

### 3. LoRA fine-tune (q_proj, v_proj only)

```powershell
& $PY train_week4_lora.py `
  --data data\week4_train.jsonl --image-root data\frames `
  --val-episodes-json data\eval\vlnverse_closed_loop_eval_20episodes.json `
  --output adapters\week4_lora --epochs 5 --wandb
```
LoRA r=16 α=32 dropout=0.05, AdamW lr=1e-4, cosine + 3% warmup, bf16, grad-checkpointing.
Episode-level train/val split (10% + the eval-json `val` episodes; frame-level split is
forbidden). Per-epoch checkpoint + `adapters\week4_lora\best` (highest val accuracy).
Validation logs overall + per-class exact-match accuracy.

### 4. Closed-loop evaluation in IsaacSim

```powershell
# from the IAmGoodNavigator dir, with the isaaclab env (Isaac Sim) active:
python demo.py --task fine --index 0 --work_dir ./myresults --agent go2 --go2-controller physics `
  --go2-physics-week4-vln-checkpoint C:\Deeplearning-final\week4\adapters\week4_lora\best
```
Metrics (`metrics.py`): SR, OSR, SPL, nDTW, Goal Dist + invalid-output rate, over the 232
top-53 val episodes and the 20 train/val eval episodes.

## Required ablation (≥1; spec section 15)

Flags on `demo.py` wire directly to ablations — no code change:
`--go2-physics-week4-vln-history-count` (1 vs 2 vs 4), `--...-history-stride` (frame gap),
`--...-trajectory-aware` (prompt change), or retrain with different LoRA r/α.

## Verification status

| Check | Status |
|---|---|
| cu128 / Blackwell sm_120 | ✅ |
| logic smoke (4-action, velocities, t=0 dup, max_steps) | ✅ |
| real Qwen3-VL inference (in-vocab actions) | ✅ |
| dataset builder (synthetic + real episode kujiale_0254/23_4) | ✅ |
| LoRA train (real episode, 3.21M trainable = q_proj+v_proj) | ✅ |
| metrics SR/OSR/SPL/nDTW/GoalDist | ✅ |
| Isaac Sim RTX render on Blackwell | ⏳ (see REPO_SPECIFIC_WEEK4_NOTES.md) |

## Report metric table (fill after eval)

| Model | SR | OSR | SPL | nDTW | Goal Dist | Invalid % |
|---|---|---|---|---|---|---|
| Zero-shot Qwen3-VL-2B | | | | | | |
| LoRA (baseline) | | | | | | |
| LoRA + ablation | | | | | | |

See `REPO_SPECIFIC_WEEK4_NOTES.md` for the integration contract and the Isaac Sim /
Blackwell compatibility notes.
