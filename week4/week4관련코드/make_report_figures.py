#!/usr/bin/env python3
"""Generate richer report figures (ablation, overfitting, closed-loop, bonus) into 보고서_자료/.
All labels in English to avoid CJK font issues. CPU-only (matplotlib Agg)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/home/ad06/isaacsim/git/Deeplearning-final"
OUT  = "/home/ad06/isaacsim/deeplearning_project/클로드/보고서_자료"
os.makedirs(OUT, exist_ok=True)

CLASSES = ["Move forward 25cm", "Turn right 15 degree", "Turn left 15 degree", "Stop"]
SHORT   = ["Forward", "Right", "Left", "Stop"]

def load(p):
    with open(p) as f: return json.load(f)

evals = {
    "baseline (q,v)\n3.2M": load(f"{REPO}/outputs/week4/baseline/predictions/offline_eval_fullval.json"),
    "trajectory-aware\n3.2M": load(f"{REPO}/outputs/week4/ablation_trajectory_aware/predictions/offline_eval_fullval.json"),
    "all-linear LoRA\n17.4M": load(f"{REPO}/outputs/week4/ablation_lora_all_linear/predictions/offline_eval_fullval.json"),
    "9-epoch best(ep4)\n3.2M": load(f"{REPO}/outputs/week4/baseline_ep9/predictions/offline_eval_fullval.json"),
}
def macro(e): return np.mean([e["per_class_accuracy"][c] for c in CLASSES])

# ---------- FIG 1: per-class accuracy grouped bars (4 models) ----------
fig, ax = plt.subplots(figsize=(11, 6))
models = list(evals.keys())
x = np.arange(len(SHORT) + 2)  # 4 classes + macro + overall
w = 0.2
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
for i, m in enumerate(models):
    e = evals[m]
    vals = [e["per_class_accuracy"][c]*100 for c in CLASSES] + [macro(e)*100, e["overall_accuracy"]*100]
    bars = ax.bar(x + (i-1.5)*w, vals, w, label=m, color=colors[i])
ax.set_xticks(x); ax.set_xticklabels(SHORT + ["MACRO", "Overall"], fontsize=11)
ax.axvline(3.5, color="gray", ls="--", lw=1, alpha=0.6)
ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
ax.set_title("Per-class / Macro / Overall Accuracy across Ablations (full val, n=18,962)", fontsize=12)
ax.legend(fontsize=9, ncol=2, loc="upper right")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/06_ablation_perclass_bars.png", dpi=130); plt.close(fig)

# ---------- FIG 2: capacity vs information insight ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
base = evals["baseline (q,v)\n3.2M"]["per_class_accuracy"]
traj = evals["trajectory-aware\n3.2M"]["per_class_accuracy"]
alll = evals["all-linear LoRA\n17.4M"]["per_class_accuracy"]
d_traj = [(traj[c]-base[c])*100 for c in CLASSES]
d_all  = [(alll[c]-base[c])*100 for c in CLASSES]
x = np.arange(len(SHORT)); w = 0.35
ax.bar(x - w/2, d_traj, w, label="+Trajectory (info)", color="#55A868")
ax.bar(x + w/2, d_all,  w, label="+All-linear LoRA (capacity)", color="#C44E52")
for i,(a,b) in enumerate(zip(d_traj,d_all)):
    ax.text(i-w/2, a+(0.2 if a>=0 else -0.6), f"{a:+.1f}", ha="center", fontsize=9)
    ax.text(i+w/2, b+(0.2 if b>=0 else -0.6), f"{b:+.1f}", ha="center", fontsize=9)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(SHORT)
ax.set_ylabel("Accuracy change vs baseline (%p)")
ax.set_title("Two interventions fix different weaknesses\n(capacity -> turns, information -> Stop)", fontsize=12)
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/07_capacity_vs_information.png", dpi=130); plt.close(fig)

# ---------- FIG 3: 9-epoch U-shape overfitting ----------
ep = list(range(1,10))
val = [0.0803,0.0759,0.0742,0.0736,0.0746,0.0771,0.0827,0.0950,0.1036]
fig, ax = plt.subplots(figsize=(9,5))
ax.plot(ep, val, "o-", color="#C44E52", lw=2, ms=7, label="validation loss")
imin = int(np.argmin(val))
ax.scatter([ep[imin]],[val[imin]], s=200, facecolors="none", edgecolors="green", lw=2.5, zorder=5)
ax.annotate(f"best ep{ep[imin]} = {val[imin]:.4f}", (ep[imin],val[imin]),
            textcoords="offset points", xytext=(12,-22), color="green", fontsize=11,
            arrowprops=dict(arrowstyle="->", color="green"))
ax.axvspan(3, 4, color="green", alpha=0.08)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss"); ax.set_xticks(ep)
ax.set_title("Epoch ablation: U-shaped generalization (overfit after ep4, +40% by ep9)", fontsize=12)
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/04_epoch_overfitting_Ushape.png", dpi=130); plt.close(fig)

# ---------- FIG 4: bonus image-text modality ablation ----------
bonus = load(f"{OUT}/06_bonus_image_text_ablation.json")["results"]
mods = ["text","image","image_text"]; mlabel=["text\n(instruction)","image","image+text"]
acc = [bonus[m]["action_accuracy"]*100 for m in mods]
fig, (a1,a2) = plt.subplots(1,2, figsize=(13,5))
bars=a1.bar(mlabel, acc, color=["#999999","#4C72B0","#8172B3"])
a1.axhline(25, color="red", ls="--", lw=1, label="random (25%)")
for b,v in zip(bars,acc): a1.text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontsize=11)
a1.set_ylabel("Action accuracy (%)"); a1.set_ylim(0,50); a1.legend()
a1.set_title("Bonus: Image+Text->velocity — which modality carries signal?", fontsize=11)
# per-class
x=np.arange(len(SHORT)); w=0.25
for i,m in enumerate(mods):
    pc=bonus[m]["per_class_accuracy"]
    vals=[pc[c]*100 for c in CLASSES]
    a2.bar(x+(i-1)*w, vals, w, label=mlabel[i].replace("\n"," "),
           color=["#999999","#4C72B0","#8172B3"][i])
a2.set_xticks(x); a2.set_xticklabels(SHORT); a2.set_ylabel("Per-class accuracy (%)")
a2.set_title("text-only collapses to Stop; image is balanced", fontsize=11)
a2.legend(fontsize=9); a2.grid(axis="y",alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/10_bonus_modality_ablation.png", dpi=130); plt.close(fig)

# ---------- FIG 5: class imbalance ----------
sup = evals["baseline (q,v)\n3.2M"]["per_class_support"]
fig, ax = plt.subplots(figsize=(8,5))
vals=[sup[c] for c in CLASSES]; total=sum(vals)
bars=ax.bar(SHORT, vals, color=["#4C72B0","#C44E52","#55A868","#DD8452"])
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, v+150, f"{v}\n({v/total*100:.1f}%)", ha="center", fontsize=10)
ax.set_ylabel("Validation samples"); ax.set_title("Severe class imbalance (Forward:Stop ~= 32:1)", fontsize=12)
ax.grid(axis="y",alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/08_class_imbalance.png", dpi=130); plt.close(fig)

# ---------- FIG 6: closed-loop cl20 results ----------
cl = load("/home/ad06/isaacsim/IAmGoodNavigator/myresults_cl20/cl20_metrics.json")
fig,(a1,a2)=plt.subplots(1,2, figsize=(13,5))
mets=["SR","OSR","SPL","nDTW"]; mv=[cl[m] for m in mets]
bars=a1.bar(mets, mv, color=["#4C72B0","#55A868","#C44E52","#8172B3"])
for b,v in zip(bars,mv): a1.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.3f}", ha="center", fontsize=11)
a1.set_ylim(0,1); a1.set_ylabel("score"); a1.set_title(f"Closed-loop 20-set summary (n={cl['n_episodes']})", fontsize=11)
a1.grid(axis="y",alpha=0.3)
# per-episode nDTW sorted, colored by success
pe=cl.get("per_episode",[])
if pe:
    pe2=sorted(pe, key=lambda e: e.get("nDTW",0))
    nd=[e.get("nDTW",0) for e in pe2]
    sr=[e.get("SR",0) for e in pe2]
    cols=["#55A868" if s else "#C44E52" for s in sr]
    a2.bar(range(len(nd)), nd, color=cols)
    a2.set_xlabel("episode (sorted by nDTW)"); a2.set_ylabel("nDTW")
    a2.set_title("Per-episode nDTW (green=success SR=1, red=fail)", fontsize=11)
    a2.axhline(cl["nDTW"], color="k", ls="--", lw=1, label=f"mean {cl['nDTW']:.3f}")
    a2.legend(); a2.grid(axis="y",alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/06_closed_loop_20set.png", dpi=130); plt.close(fig)

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"): print("  ", f)
