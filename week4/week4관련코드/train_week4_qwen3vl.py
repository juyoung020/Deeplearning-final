"""Week 4 training: Qwen3-VL-2B + LoRA next-action prediction.

Features:
  * episode-level train/val split (never frame-level)
  * gradient accumulation, cosine schedule + warmup, grad clipping, AMP via bf16
  * per-epoch checkpoint (LoRA adapter only) + resume + best-adapter tracking
  * validation: val loss + generation metrics (accuracy / per-class / invalid rate)
  * W&B logging with JSONL fallback

Usage:
    python train_week4_qwen3vl.py --config configs/week4_qwen3vl_lora.json
    python train_week4_qwen3vl.py --config configs/week4_qwen3vl_lora.json --resume
    # quick smoke train (no full dataset needed is NOT possible here; use
    # smoke_test_week4.py for the synthetic path):
    python train_week4_qwen3vl.py --config ... --max-samples 16 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Logger (W&B with JSONL fallback)
# ---------------------------------------------------------------------------
class RunLogger:
    def __init__(self, cfg, output_dir: str):
        self.use_wandb = bool(cfg.logging.use_wandb)
        self.wandb = None
        self.jsonl_path = os.path.join(output_dir, cfg.logging.jsonl_log)
        os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
        self._jsonl = open(self.jsonl_path, "a", encoding="utf-8")
        if self.use_wandb:
            try:
                import wandb

                wandb.init(
                    project=cfg.logging.wandb_project,
                    name=cfg.logging.wandb_run_name,
                    config=cfg.to_dict(),
                )
                self.wandb = wandb
            except Exception as exc:
                print(f"[week4] W&B unavailable ({exc}); falling back to JSONL only.")
                self.use_wandb = False

    def log(self, data: Dict[str, Any], step: Optional[int] = None):
        rec = dict(data)
        if step is not None:
            rec["step"] = step
        rec["wall_time"] = time.time()
        self._jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._jsonl.flush()
        if self.wandb is not None:
            self.wandb.log(data, step=step)

    def close(self):
        try:
            self._jsonl.close()
        except Exception:
            pass
        if self.wandb is not None:
            self.wandb.finish()


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------
def cosine_warmup_lambda(total_steps: int, warmup_steps: int):
    def fn(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return fn


# ---------------------------------------------------------------------------
# Data building
# ---------------------------------------------------------------------------
def build_train_val_datasets(cfg):
    from vlnverse_dataset import (
        build_episodes_from_split, VLNVerseDataset, episode_level_split, load_split_records,
    )

    episodes = build_episodes_from_split(cfg.dataset)
    if not episodes:
        raise RuntimeError(
            "No episodes resolved. Check dataset_root / split_json and that the "
            "trajectory data has been downloaded (see download_vlnverse.py)."
        )
    forced_ids = None
    if cfg.forced_val_split_json and os.path.exists(cfg.forced_val_split_json):
        recs = load_split_records(cfg.forced_val_split_json)
        forced_ids = [str(r.get("episode_id") or r.get("episode") or r.get("id")) for r in recs]

    train_eps, val_eps = episode_level_split(
        episodes, val_ratio=cfg.training.val_ratio, seed=cfg.training.seed,
        forced_val_episode_ids=forced_ids,
    )
    train_ds = VLNVerseDataset(train_eps, cfg.dataset)
    val_ds = VLNVerseDataset(val_eps, cfg.dataset)
    print(f"[week4] episodes: {len(train_eps)} train / {len(val_eps)} val; "
          f"samples: {len(train_ds)} train / {len(val_ds)} val")
    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------
def compute_val_loss(model, val_loader, max_batches: Optional[int] = None) -> float:
    import torch

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, count = 0.0, 0
    with torch.no_grad():
        for bi, batch in enumerate(val_loader):
            if max_batches is not None and bi >= max_batches:
                break
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            out = model(**batch)
            total += float(out.loss.item())
            count += 1
    if was_training:
        model.train()
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, state: Dict[str, Any], ckpt_dir: str):
    import torch
    from qwen3vl_lora_model import save_adapter

    os.makedirs(ckpt_dir, exist_ok=True)
    save_adapter(model, ckpt_dir)  # LoRA adapter (trainable params only)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": state,
        },
        os.path.join(ckpt_dir, "trainer_state.pt"),
    )


def load_trainer_state(ckpt_dir: str):
    import torch

    p = os.path.join(ckpt_dir, "trainer_state.pt")
    if not os.path.exists(p):
        return None
    return torch.load(p, map_location="cpu")


def find_latest_epoch_ckpt(ckpt_root: str) -> Optional[str]:
    if not os.path.isdir(ckpt_root):
        return None
    epochs = [d for d in os.listdir(ckpt_root) if d.startswith("epoch_")]
    if not epochs:
        return None
    epochs.sort(key=lambda d: int(d.split("_")[1]))
    return os.path.join(ckpt_root, epochs[-1])


def find_resume_ckpt(ckpt_root: str, output_dir: str) -> Optional[str]:
    """Prefer the periodically-updated ``latest/`` (may be mid-epoch), else the
    newest ``epoch_xxx/`` directory."""
    latest = os.path.join(ckpt_root, "latest")
    if os.path.exists(os.path.join(latest, "trainer_state.pt")):
        return latest
    return find_latest_epoch_ckpt(ckpt_root)


def _reload_adapter_weights(model, ckpt_dir: str):
    """Reload trained LoRA weights from a checkpoint dir into ``model``."""
    try:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        wpath = os.path.join(ckpt_dir, "adapter_model.safetensors")
        if os.path.exists(wpath):
            set_peft_model_state_dict(model, load_file(wpath))
    except Exception as exc:
        print(f"[week4] adapter weight reload skipped ({exc})")


def make_epoch_loader(train_ds, cfg, collator, epoch: int, start_batch: int = 0):
    """Deterministic per-epoch DataLoader supporting mid-epoch resume.

    The sample order is a fixed permutation seeded by ``(seed, epoch)`` so that
    skipping the first ``start_batch`` batches on resume reproduces exactly the
    position where training stopped — no sample is dropped or re-trained.
    """
    import random as _random
    from torch.utils.data import DataLoader

    n = len(train_ds)
    order = list(range(n))
    rng = _random.Random(cfg.training.seed * 1000003 + epoch)
    rng.shuffle(order)
    start_index = max(0, int(start_batch) * cfg.training.batch_size)
    order = order[start_index:]
    return DataLoader(
        train_ds, batch_size=cfg.training.batch_size, sampler=order,
        num_workers=cfg.training.num_workers, collate_fn=collator, drop_last=False,
    )


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(cfg, resume: bool = False):
    import torch
    from torch.utils.data import DataLoader

    from qwen3vl_lora_model import load_model_and_processor, save_adapter
    from qwen3vl_prompt import Qwen3VLCollator
    from eval_week4_offline import evaluate_generation, save_metrics

    os.makedirs(cfg.output_dir, exist_ok=True)
    ckpt_root = os.path.join(cfg.output_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)
    # Persist the resolved config for the report.
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)

    torch.manual_seed(cfg.training.seed)

    model, processor = load_model_and_processor(cfg.model)
    collator = Qwen3VLCollator(processor, system_prompt=cfg.system_prompt)

    train_ds, val_ds = build_train_val_datasets(cfg)
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, collate_fn=collator,
    )

    bs = cfg.training.batch_size
    accum = max(1, cfg.training.grad_accum_steps)
    batches_per_epoch = math.ceil(len(train_ds) / bs)
    steps_per_epoch = math.ceil(batches_per_epoch / accum)
    total_steps = steps_per_epoch * cfg.training.epochs
    warmup_steps = int(round(total_steps * cfg.training.warmup_ratio))

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_warmup_lambda(total_steps, warmup_steps)
    )

    logger = RunLogger(cfg, cfg.output_dir)
    start_epoch = 0
    global_step = 0
    best_metric = float("inf")  # lower val loss is better
    resume_batches = 0          # micro-batches already done in start_epoch (mid-epoch resume)

    if resume:
        latest = find_resume_ckpt(ckpt_root, cfg.output_dir)
        if latest:
            ts = load_trainer_state(latest)
            if ts:
                optimizer.load_state_dict(ts["optimizer"])
                scheduler.load_state_dict(ts["scheduler"])
                start_epoch = ts["state"].get("epoch", 0)
                global_step = ts["state"].get("global_step", 0)
                best_metric = ts["state"].get("best_metric", best_metric)
                resume_batches = ts["state"].get("batches_done_in_epoch", 0)
                _reload_adapter_weights(model, latest)
                print(f"[week4] resumed from {latest} "
                      f"(epoch={start_epoch}, step={global_step}, "
                      f"batches_into_epoch={resume_batches})")

    model.train()
    device = next(model.parameters()).device
    save_every = max(int(cfg.training.save_every_steps), 0)

    def _persist(tag_state, ckpt_dir):
        """Save adapter + optimizer/scheduler + resume bookkeeping."""
        save_checkpoint(model, optimizer, scheduler, tag_state, ckpt_dir)

    # The whole loop is wrapped so a crash / Ctrl-C still flushes a "latest"
    # checkpoint with everything trained so far.
    epoch = start_epoch
    batches_done_in_epoch = resume_batches
    try:
        for epoch in range(start_epoch, cfg.training.epochs):
            optimizer.zero_grad(set_to_none=True)
            running = 0.0
            running_count = 0
            t0 = time.time()
            # Deterministic per-epoch order so a mid-epoch resume skips exactly the
            # batches already processed (no sample is silently re-trained or lost).
            start_batch = batches_done_in_epoch if epoch == start_epoch else 0
            batches_done_in_epoch = start_batch  # correct bookkeeping at epoch top
            train_loader = make_epoch_loader(train_ds, cfg, collator, epoch, start_batch)
            if start_batch:
                print(f"[week4] epoch {epoch}: skipping first {start_batch} batches (resume).")

            for local_it, batch in enumerate(train_loader):
                it = start_batch + local_it  # absolute micro-batch index in epoch
                batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
                out = model(**batch)
                loss = out.loss / accum
                loss.backward()
                running += float(out.loss.item())
                running_count += 1

                if (it + 1) % accum == 0 or (it + 1) == batches_per_epoch:
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, cfg.training.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    batches_done_in_epoch = it + 1

                    if global_step % cfg.training.log_every == 0:
                        avg = running / max(running_count, 1)
                        lr = scheduler.get_last_lr()[0]
                        logger.log(
                            {"train_loss": avg, "learning_rate": lr, "epoch": epoch,
                             "grad_norm": float(grad_norm)},
                            step=global_step,
                        )
                        print(f"[week4] epoch {epoch} step {global_step} "
                              f"(batch {batches_done_in_epoch}/{batches_per_epoch}) "
                              f"loss {avg:.4f} lr {lr:.2e}")
                        running = 0.0
                        running_count = 0

                    # ---- mid-epoch periodic checkpoint ----
                    if save_every and global_step % save_every == 0:
                        _persist(
                            {"epoch": epoch, "global_step": global_step,
                             "best_metric": best_metric,
                             "batches_done_in_epoch": batches_done_in_epoch},
                            os.path.join(ckpt_root, "latest"),
                        )
                        print(f"[week4] mid-epoch checkpoint saved at step {global_step} "
                              f"-> {os.path.join(ckpt_root, 'latest')}")

            # ---- epoch end: validation ----
            val_loss = compute_val_loss(model, val_loader, cfg.training.eval_max_batches)
            gen_metrics = evaluate_generation(
                model, processor, val_ds, cfg.system_prompt,
                max_samples=cfg.training.eval_generate_max_samples,
            )
            epoch_metrics = {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": gen_metrics["overall_accuracy"],
                "val_invalid_output_rate": gen_metrics["invalid_output_rate"],
                "epoch_time_s": time.time() - t0,
            }
            for a, acc in gen_metrics["per_class_accuracy"].items():
                epoch_metrics[f"val_acc[{a}]"] = acc
            logger.log(epoch_metrics, step=global_step)
            print(f"[week4] epoch {epoch} val_loss {val_loss:.4f} "
                  f"acc {gen_metrics['overall_accuracy']:.3f} "
                  f"invalid {gen_metrics['invalid_output_rate']:.3f}")

            # save per-epoch checkpoint + predictions (also refresh "latest")
            ckpt_dir = os.path.join(ckpt_root, f"epoch_{epoch+1:03d}")
            epoch_state = {"epoch": epoch + 1, "global_step": global_step,
                           "best_metric": best_metric, "batches_done_in_epoch": 0}
            _persist(epoch_state, ckpt_dir)
            _persist(epoch_state, os.path.join(ckpt_root, "latest"))
            save_metrics(
                gen_metrics,
                os.path.join(cfg.output_dir, "predictions", f"val_epoch_{epoch+1:03d}.json"),
            )

            # best by val loss
            if val_loss < best_metric:
                best_metric = val_loss
                best_dir = os.path.join(cfg.output_dir, "best_adapter")
                save_adapter(model, best_dir)
                with open(os.path.join(best_dir, "best_info.json"), "w", encoding="utf-8") as f:
                    json.dump({"epoch": epoch + 1, "val_loss": val_loss, **epoch_metrics},
                              f, indent=2, ensure_ascii=False)
                print(f"[week4] new best (val_loss={val_loss:.4f}) -> {best_dir}")

        print("[week4] training complete.")
    except BaseException as exc:
        # Crash / Ctrl-C: flush everything trained so far to "latest" so nothing
        # since the last periodic checkpoint is lost, then re-raise.
        try:
            _persist(
                {"epoch": epoch, "global_step": global_step, "best_metric": best_metric,
                 "batches_done_in_epoch": batches_done_in_epoch},
                os.path.join(ckpt_root, "latest"),
            )
            print(f"[week4] interrupted ({type(exc).__name__}); emergency checkpoint saved "
                  f"-> {os.path.join(ckpt_root, 'latest')} "
                  f"(epoch={epoch}, step={global_step}, batch={batches_done_in_epoch}). "
                  f"Resume with --resume.")
        except Exception as save_exc:
            print(f"[week4] emergency checkpoint FAILED: {save_exc}")
        raise
    finally:
        logger.close()


def main():
    parser = argparse.ArgumentParser(description="Week 4 Qwen3-VL LoRA training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None, help="override dataset.max_samples")
    parser.add_argument("--max-episodes", type=int, default=None, help="override dataset.max_episodes")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--dataset-root", default=None, help="override dataset.dataset_root (e.g. a larger drive)")
    parser.add_argument("--split-json", default=None, help="override dataset.split_json")
    parser.add_argument("--batch-size", type=int, default=None, help="override training.batch_size")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="override training.grad_accum_steps")
    parser.add_argument("--output-dir", default=None, help="override output_dir (keep separate runs from overwriting)")
    parser.add_argument("--save-every-steps", type=int, default=None,
                        help="override training.save_every_steps (mid-epoch checkpoint cadence)")
    args = parser.parse_args()

    from week4_config import Week4Config

    cfg = Week4Config.from_json(args.config)
    if args.max_samples is not None:
        cfg.dataset.max_samples = args.max_samples
    if args.max_episodes is not None:
        cfg.dataset.max_episodes = args.max_episodes
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.save_every_steps is not None:
        cfg.training.save_every_steps = args.save_every_steps
    if args.dataset_root is not None:
        cfg.dataset.dataset_root = args.dataset_root
    if args.split_json is not None:
        cfg.dataset.split_json = args.split_json
    if args.batch_size is not None:
        cfg.training.batch_size = args.batch_size
    if args.grad_accum_steps is not None:
        cfg.training.grad_accum_steps = args.grad_accum_steps
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
