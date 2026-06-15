"""Generate training plots from a Week 4 run's train_log.jsonl.

Produces:
  - train_loss.png      : train loss vs optimizer step (raw + smoothed)
  - val_curves.png      : val loss & val accuracy vs epoch

Usage:
  python week4/make_plots.py --run outputs/week4/baseline
"""
from __future__ import annotations
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_log(path):
    steps, tloss, lr = [], [], []
    ep, vloss, vacc, vinv = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "train_loss" in r and "step" in r:
                steps.append(r["step"]); tloss.append(r["train_loss"]); lr.append(r.get("learning_rate"))
            if "val_loss" in r:
                ep.append(r.get("epoch", len(ep))); vloss.append(r["val_loss"])
                vacc.append(r.get("val_accuracy")); vinv.append(r.get("val_invalid_output_rate"))
    return dict(steps=steps, tloss=tloss, lr=lr, ep=ep, vloss=vloss, vacc=vacc, vinv=vinv)


def smooth(y, k=21):
    if len(y) < k:
        return y
    out = []
    for i in range(len(y)):
        a = max(0, i - k // 2); b = min(len(y), i + k // 2 + 1)
        out.append(sum(y[a:b]) / (b - a))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="outputs/week4/baseline")
    args = ap.parse_args()
    log = os.path.join(args.run, "logs", "train_log.jsonl")
    d = load_log(log)
    outdir = os.path.join(args.run, "plots"); os.makedirs(outdir, exist_ok=True)

    # 1) train loss
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(d["steps"], d["tloss"], color="#bbb", lw=0.7, label="train loss (raw)")
    ax.plot(d["steps"], smooth(d["tloss"]), color="#1f77b4", lw=2, label="train loss (smoothed)")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("loss"); ax.set_title("Week4 baseline — train loss")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    p1 = os.path.join(outdir, "train_loss.png"); fig.savefig(p1, dpi=130); plt.close(fig)

    # 2) val loss + accuracy vs epoch
    epochs = [e + 1 for e in d["ep"]]  # 1-indexed for display
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    l1 = ax1.plot(epochs, d["vloss"], "o-", color="#d62728", label="val loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("val loss", color="#d62728")
    ax1.set_xticks(epochs)
    ax2 = ax1.twinx()
    l2 = ax2.plot(epochs, [a * 100 for a in d["vacc"]], "s-", color="#2ca02c", label="val accuracy")
    ax2.set_ylabel("val accuracy (%)", color="#2ca02c")
    ax1.set_title("Week4 baseline — validation per epoch")
    ax1.grid(alpha=0.3)
    lns = l1 + l2; ax1.legend(lns, [x.get_label() for x in lns], loc="center right")
    fig.tight_layout()
    p2 = os.path.join(outdir, "val_curves.png"); fig.savefig(p2, dpi=130); plt.close(fig)

    print("saved:", p1, p2)
    print("train points:", len(d["steps"]), "| epochs:", epochs,
          "| val_loss:", [round(x, 4) for x in d["vloss"]],
          "| val_acc:", [round(x, 3) for x in d["vacc"]])


if __name__ == "__main__":
    main()
