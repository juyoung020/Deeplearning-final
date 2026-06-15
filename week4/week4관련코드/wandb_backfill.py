"""Backfill a W&B run from an already-finished Week 4 training run's logs.

Satisfies the assignment's W&B Logging requirement (train loss / validation loss
/ training validation accuracy / learning rate / epoch) WITHOUT retraining, by
replaying ``logs/train_log.jsonl`` into a wandb run.

Runs offline by default (no login needed): produces a local run under
``wandb/`` that can later be uploaded with ``wandb login && wandb sync <dir>``.

Usage:
  # offline (default, no login):
  python week4/wandb_backfill.py --run outputs/week4/baseline
  # all runs:
  for r in baseline ablation_trajectory_aware ablation_lora_all_linear baseline_ep9; do
    python week4/wandb_backfill.py --run outputs/week4/$r; done
  # then later:  wandb login && wandb sync wandb/offline-run-*
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir, e.g. outputs/week4/baseline")
    ap.add_argument("--project", default="vlnverse-week4")
    ap.add_argument("--online", action="store_true", help="log online (requires wandb login)")
    args = ap.parse_args()

    if not args.online:
        os.environ.setdefault("WANDB_MODE", "offline")

    import wandb

    name = os.path.basename(os.path.abspath(args.run))
    log_path = os.path.join(args.run, "logs", "train_log.jsonl")
    if not os.path.exists(log_path):
        raise SystemExit(f"no train_log.jsonl at {log_path}")

    # try to attach the run's config for hyperparameters
    cfg = {}
    cfg_path = os.path.join(args.run, "config.json")
    if os.path.exists(cfg_path):
        try:
            cfg = json.load(open(cfg_path))
        except Exception:
            cfg = {}

    run = wandb.init(project=args.project, name=name, config=cfg, reinit=True)
    n_train, n_val = 0, 0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "train_loss" in r and "step" in r:
                wandb.log(
                    {"train/loss": r["train_loss"],
                     "train/learning_rate": r.get("learning_rate"),
                     "epoch": r.get("epoch")},
                    step=int(r["step"]),
                )
                n_train += 1
            if "val_loss" in r:
                payload = {"val/loss": r["val_loss"], "epoch": r.get("epoch")}
                if r.get("val_accuracy") is not None:
                    payload["val/accuracy"] = r["val_accuracy"]
                if r.get("val_invalid_output_rate") is not None:
                    payload["val/invalid_output_rate"] = r["val_invalid_output_rate"]
                step = int(r["step"]) if "step" in r else None
                wandb.log(payload, step=step)
                n_val += 1
    run.finish()
    print(f"[wandb] {name}: logged {n_train} train points, {n_val} val points "
          f"(mode={'online' if args.online else 'offline'})")


if __name__ == "__main__":
    main()
