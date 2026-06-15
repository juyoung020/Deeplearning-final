"""Train Image+Text -> velocity (Week 3 bonus, PDF §4.2) on VLN-VERSE.

Pipeline (frozen encoders, cache-then-train like Week 3):
  1. Build (image, text, velocity) samples from VLN-VERSE (episode-level split).
  2. Encode text & image ONCE with frozen SigLIP2 (cached tensors).
  3. For each modality in {text, image, image_text}, train a small VelocityMLP
     and evaluate -> MSE + nearest-canonical action accuracy (overall + per class).

This single run produces the ablation table showing what the image branch adds
on top of text alone (and vice-versa).

Usage:
  python week3/train_image_text_velocity.py --config week3/configs/week3_image_text_velocity.json
  python week3/train_image_text_velocity.py --config ... --max-episodes 20 --steps 300   # quick
"""
from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy

import torch
from torch.nn import functional as F

from image_text_velocity_model import (
    VALID_MODALITIES,
    FrozenSigLIP2Encoder,
    ImageTextVelocityModel,
)
from language_velocity_model import DEFAULT_SIGLIP2_MODEL, resolve_device
from vln_image_text_data import VLNImageTextData
from week4_actions import ACTION_TO_COMMAND, CANONICAL_ACTIONS


DEFAULT_CONFIG = {
    "model_name": DEFAULT_SIGLIP2_MODEL,
    "dataset_root": "data/VLNVerse_data",
    "split_json": "data/VLNVerse_data/raw_data/final_splits/fine_train.json.gz",
    "text_source": "instruction",      # "instruction" | "action"
    "max_episodes": 200,
    "per_class_cap": 1500,
    "val_ratio": 0.1,
    "image_size": 256,
    "max_length": 64,
    "normalize_embeddings": True,
    "torch_dtype": "bfloat16",
    "encode_batch_size": 64,
    "modalities": ["text", "image", "image_text"],
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.1,
    "seed": 7,
    "batch_size": 128,
    "training_steps": 2000,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "lr_min": 1e-5,
    "print_every": 200,
    "device": "auto",
    "output_dir": "outputs/week3/image_text_velocity",
}

# Canonical velocity vectors for nearest-action classification.
_CANON_VEL = {a: tuple(float(x) for x in ACTION_TO_COMMAND[a]["velocity"]) for a in CANONICAL_ACTIONS}


def load_config(path):
    cfg = deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, encoding="utf-8") as f:
            cfg.update(json.load(f))
    return cfg


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def nearest_action(vel) -> str:
    best, bd = None, 1e18
    for a, cv in _CANON_VEL.items():
        d = sum((vel[i] - cv[i]) ** 2 for i in range(3))
        if d < bd:
            bd, best = d, a
    return best


def action_metrics(pred: torch.Tensor, gold_actions: list[str]) -> dict:
    pred = pred.detach().cpu().float().tolist()
    correct, total = 0, len(gold_actions)
    per = {a: [0, 0] for a in CANONICAL_ACTIONS}  # [correct, support]
    for p, g in zip(pred, gold_actions):
        pa = nearest_action(p)
        per[g][1] += 1
        if pa == g:
            correct += 1
            per[g][0] += 1
    return {
        "action_accuracy": correct / max(total, 1),
        "per_class_accuracy": {a: (c / s if s else None) for a, (c, s) in per.items()},
        "per_class_support": {a: s for a, (c, s) in per.items()},
    }


@torch.inference_mode()
def encode_samples(encoder, data, samples, batch_size):
    """Return (text_emb[N,Dt], image_emb[N,Di], targets[N,3], gold_actions[N])."""
    texts, imgs, tgts, gold = [], [], [], []
    for s in samples:
        img = data.get_image(s)
        if img is None:
            continue
        texts.append(s.text); imgs.append(img); tgts.append(s.target); gold.append(s.action_text)
    if not texts:
        raise RuntimeError("no decodable image samples; check dataset_root / rgb.npy")
    t_emb, i_emb = [], []
    for b in range(0, len(texts), batch_size):
        t_emb.append(encoder.encode_text(texts[b:b + batch_size]).cpu())
        i_emb.append(encoder.encode_image(imgs[b:b + batch_size]).cpu())
    return (
        torch.cat(t_emb), torch.cat(i_emb),
        torch.tensor(tgts, dtype=torch.float32), gold,
    )


def train_one(modality, tr, va, cfg, device):
    t_tr, i_tr, y_tr, _ = tr
    t_va, i_va, y_va, gold_va = va
    model = ImageTextVelocityModel(
        modalities=modality, text_dim=t_tr.shape[1], image_dim=i_tr.shape[1],
        hidden_dim=int(cfg["hidden_dim"]), dropout=float(cfg["dropout"]),
        num_layers=int(cfg["num_layers"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    steps = int(cfg["training_steps"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1), eta_min=float(cfg["lr_min"]))
    bs = int(cfg["batch_size"]); n = t_tr.shape[0]
    t_tr_d, i_tr_d, y_tr_d = t_tr.to(device), i_tr.to(device), y_tr.to(device)

    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (min(bs, n),), device=device)
        te = t_tr_d[idx] if modality != "image" else None
        ie = i_tr_d[idx] if modality != "text" else None
        loss = F.mse_loss(model(te, ie), y_tr_d[idx])
        opt.zero_grad(set_to_none=True); loss.backward()
        if float(cfg["grad_clip_norm"]) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip_norm"]))
        opt.step(); sched.step()
        if step == 1 or step % int(cfg["print_every"]) == 0 or step == steps:
            print(f"  [{modality}] step {step}/{steps} loss {loss.item():.6f}")

    model.eval()
    with torch.inference_mode():
        te = t_va.to(device) if modality != "image" else None
        ie = i_va.to(device) if modality != "text" else None
        pred = model(te, ie)
        mse = float(F.mse_loss(pred, y_va.to(device)).item())
    am = action_metrics(pred, gold_va)
    return model, {"val_mse": mse, **am}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="week3/configs/week3_image_text_velocity.json")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--dataset-root", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.max_episodes is not None: cfg["max_episodes"] = args.max_episodes
    if args.steps is not None: cfg["training_steps"] = args.steps
    if args.dataset_root is not None: cfg["dataset_root"] = args.dataset_root
    if args.output_dir is not None: cfg["output_dir"] = args.output_dir
    set_seed(int(cfg["seed"]))
    device = resolve_device(cfg["device"])
    os.makedirs(cfg["output_dir"], exist_ok=True)

    print(f"[data] building VLN samples (text_source={cfg['text_source']}, "
          f"max_episodes={cfg['max_episodes']}, per_class_cap={cfg['per_class_cap']})")
    data = VLNImageTextData(
        dataset_root=cfg["dataset_root"], split_json=cfg["split_json"],
        text_source=cfg["text_source"], max_episodes=cfg["max_episodes"],
        per_class_cap=cfg["per_class_cap"], image_size=int(cfg["image_size"]), seed=int(cfg["seed"]),
    )
    print(f"[data] episodes={len(data.episodes)} samples={len(data.samples)} class_counts={data.class_counts()}")
    tr_eps, va_eps = data.episode_split(float(cfg["val_ratio"]))
    tr_set, va_set = set(tr_eps), set(va_eps)
    tr_samples = [s for s in data.samples if s.episode_index in tr_set]
    va_samples = [s for s in data.samples if s.episode_index in va_set]
    print(f"[data] split: {len(tr_eps)} train / {len(va_eps)} val episodes; "
          f"{len(tr_samples)} / {len(va_samples)} samples")

    print(f"[encode] SigLIP2 {cfg['model_name']} (frozen) text+image ...")
    encoder = FrozenSigLIP2Encoder(
        model_name=cfg["model_name"], device=device, max_length=int(cfg["max_length"]),
        normalize=bool(cfg["normalize_embeddings"]), torch_dtype=cfg.get("torch_dtype"),
    )
    bs = int(cfg["encode_batch_size"])
    tr = encode_samples(encoder, data, tr_samples, bs)
    va = encode_samples(encoder, data, va_samples, bs)
    print(f"[encode] text_dim={tr[0].shape[1]} image_dim={tr[1].shape[1]} "
          f"(train {tr[0].shape[0]} / val {va[0].shape[0]} usable)")

    results = {}
    mods = [m for m in cfg["modalities"] if m in VALID_MODALITIES]
    for m in mods:
        print(f"[train] modality = {m}")
        model, metrics = train_one(m, tr, va, cfg, device)
        results[m] = metrics
        torch.save({"model_state_dict": model.state_dict(), "modality": m, "config": cfg},
                   os.path.join(cfg["output_dir"], f"velocity_{m}.pt"))
        pc = " ".join(f"{a.split()[0]}:{(v*100):.0f}" if v is not None else f"{a.split()[0]}:-"
                      for a, v in metrics["per_class_accuracy"].items())
        print(f"  -> val_mse {metrics['val_mse']:.6f} | action_acc {metrics['action_accuracy']*100:.1f}% | {pc}")

    summary = {"config": cfg, "results": results,
               "class_counts": data.class_counts(),
               "n_train": int(tr[0].shape[0]), "n_val": int(va[0].shape[0])}
    out = os.path.join(cfg["output_dir"], "ablation_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[done] saved -> {out}")
    print("\n=== ABLATION (val) ===")
    print(f"{'modality':12s} {'MSE':>10s} {'act_acc':>8s}  per-class(F/R/L/Stop)")
    for m in mods:
        r = results[m]
        pcs = "/".join(f"{(r['per_class_accuracy'][a]*100):.0f}" if r['per_class_accuracy'][a] is not None else "-"
                       for a in CANONICAL_ACTIONS)
        print(f"{m:12s} {r['val_mse']:>10.6f} {r['action_accuracy']*100:>7.1f}%  {pcs}")


if __name__ == "__main__":
    main()
