"""Week 4 — LoRA fine-tuning of Qwen3-VL-2B for VLN next-action prediction.

Top-score configuration:
  * LoRA on q_proj, v_proj ONLY (spec); vision encoder / base / embeddings frozen.
  * Spec prompt + chat template; loss on answer tokens (incl. <|im_end|>) only.
  * Episode-level train/val split (frame-level split forbidden); the eval-json `val`
    episodes are always validation.
  * **Class-balanced epochs** — the raw data is ~67% "Move forward 25cm" and only
    ~1% "Stop" (1 per episode). Training on that collapses to always-forward and
    never stops -> SR=0. We undersample forward and oversample Stop/turns toward a
    balanced per-class target each epoch (also shrinks the epoch -> faster).
  * **Mini-batching** with right-padding + per-sample answer masking (a single
    processor call batches text+images), so an epoch on the full set is hours not days.
  * Cosine LR + warmup, bf16, gradient checkpointing, per-class val accuracy, best ckpt, W&B.

Usage:
    set HF_HUB_DISABLE_SYMLINKS=1
    python train_week4_lora.py --data data/week4_train.jsonl --image-root data/frames \
        --val-episodes-json data/eval/vlnverse_closed_loop_eval_20episodes.json \
        --output adapters/week4_lora --epochs 3 --batch-size 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from actions import ACTION_SYSTEM_PROMPT, CANONICAL_ACTIONS, build_user_text


def load_samples(path, image_root):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            imgs = s.get("images") or ([s["image"]] if s.get("image") else [])
            s["_image_paths"] = [os.path.join(image_root, p) if image_root else p for p in imgs]
            out.append(s)
    return out


def episode_split(samples, val_episode_ids, val_ratio, seed=0):
    by_ep = {}
    for s in samples:
        by_ep.setdefault(s.get("episode_id", "ep"), []).append(s)
    eps = sorted(by_ep)
    forced_val = set(val_episode_ids or [])
    free = [e for e in eps if e not in forced_val]
    rng = random.Random(seed)
    rng.shuffle(free)
    n_val = int(round(len(free) * val_ratio))
    val_eps = forced_val | set(free[:n_val])
    train, val = [], []
    for e in eps:
        (val if e in val_eps else train).extend(by_ep[e])
    return train, val, sorted(val_eps)


def balanced_indices(samples, alpha, epoch_size, per_class_cap, seed):
    """Sample each action class proportional to count**(1-alpha).

    alpha=1 -> fully balanced (equal per class); alpha=0 -> natural distribution;
    alpha=0.5 -> sqrt re-balance, which keeps "Move forward" dominant (navigation
    is mostly forward) while still boosting the rare Turn/Stop classes. Full
    balancing (alpha=1) made the agent over-turn and stop prematurely, so a
    gentler alpha is used by default."""
    rng = random.Random(seed)
    by_cls = {}
    for i, s in enumerate(samples):
        by_cls.setdefault(s["answer"], []).append(i)
    counts = {cls: len(m) for cls, m in by_cls.items()}
    weights = {cls: c ** (1.0 - alpha) for cls, c in counts.items()}
    Z = sum(weights.values()) or 1.0
    targets = {cls: max(1, min(int(round(epoch_size * w / Z)), per_class_cap)) for cls, w in weights.items()}
    idxs = []
    for cls, members in by_cls.items():
        t = targets[cls]
        idxs.extend(rng.sample(members, t) if t <= len(members) else rng.choices(members, k=t))
    rng.shuffle(idxs)
    return idxs, counts, targets


def build_messages(sample, frames, with_answer):
    user_content = [{"type": "image", "image": im} for im in frames]
    user_content.append({"type": "text", "text": build_user_text(sample["instruction"])})
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": ACTION_SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
    ]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]})
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--output", default="adapters/week4_lora")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--val-episodes-json", default=None)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    ap.add_argument("--per-class-cap", type=int, default=20000, help="max samples per action class per epoch")
    ap.add_argument("--max-val", type=int, default=400, help="cap val samples used for generation accuracy")
    ap.add_argument("--balance-power", type=float, default=0.5, help="0=natural dist, 1=full balance, 0.5=sqrt (keeps forward dominant)")
    ap.add_argument("--epoch-size", type=int, default=40000, help="samples drawn per epoch (across classes)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--select-metric", default="macro", choices=["macro", "overall"],
                    help="best-ckpt selection metric (macro is robust to the 67/16/15/2 imbalance)")
    ap.add_argument("--summary-json", default=None, help="write final metrics JSON here (for the smoke harness)")
    ap.add_argument("--resume", default=None, help="adapter dir to resume from (loads adapter + trainer_state.pt)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        dtype = torch.float32

    samples = load_samples(args.data, args.image_root)

    val_ep_ids = []
    if args.val_episodes_json and os.path.exists(args.val_episodes_json):
        spec = json.load(open(args.val_episodes_json, encoding="utf-8"))
        if isinstance(spec, dict) and "episodes" in spec:
            for e in spec["episodes"]:
                if e.get("split") == "val":
                    val_ep_ids.append(f"{e.get('scene_name')}/{e.get('episode_number')}")
        else:
            val_ep_ids = spec.get("val") or spec.get("val_episodes") or []
    train, val, val_eps = episode_split(samples, val_ep_ids, args.val_ratio, args.seed)
    print(f"[INFO] {len(samples)} samples -> train {len(train)} / val {len(val)} ({len(val_eps)} val episodes)")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"  # training: prompt prefix at a fixed start
    print(f"[INFO] Loading {args.model} dtype={dtype} device={device}")
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    resume_state = None
    if args.resume:
        model = PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        ts_path = os.path.join(args.resume, "trainer_state.pt")
        resume_state = torch.load(ts_path, map_location="cpu", weights_only=False) if os.path.exists(ts_path) else None
        print(f"[RESUME] adapter loaded from {args.resume}; trainer_state={'yes' if resume_state else 'no(adapter only)'}")
    else:
        lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                          bias="none", task_type="CAUSAL_LM", target_modules=list(args.target_modules))
        model = get_peft_model(model, lora)
    model.to(device)
    model.print_trainable_parameters()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    def collate(batch):
        full_texts, frames_list, prompt_lens = [], [], []
        for s in batch:
            frames = [Image.open(p).convert("RGB") for p in s["_image_paths"]]
            frames_list.append(frames)
            full_texts.append(processor.apply_chat_template(
                build_messages(s, frames, True), tokenize=False, add_generation_prompt=False))
            ptext = processor.apply_chat_template(
                build_messages(s, frames, False), tokenize=False, add_generation_prompt=True)
            prompt_lens.append(processor(text=[ptext], images=[frames], return_tensors="pt")["input_ids"].shape[1])
        inputs = processor(text=full_texts, images=frames_list, return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100                      # mask system/user/image/Answer:
        labels[inputs["attention_mask"] == 0] = -100     # mask right-padding
        inputs["labels"] = labels
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    idx0, dist, targets = balanced_indices(train, args.balance_power, args.epoch_size, args.per_class_cap, args.seed)
    print(f"[INFO] raw class dist: {dist}; sampled targets (power={args.balance_power}): {targets}; epoch_size={len(idx0)}")
    total_opt_steps = max(1, (len(idx0) // args.batch_size // args.grad_accum) * args.epochs)
    sched = get_cosine_schedule_with_warmup(optim, int(total_opt_steps * args.warmup_ratio), total_opt_steps)

    if args.wandb:
        import wandb
        wandb.init(project="week4-vln-qwen3vl", config=vars(args))

    os.makedirs(args.output, exist_ok=True)
    best_sel, best_metrics, last_metrics, gstep = -1.0, {}, {}, 0
    start_epoch = 0
    if resume_state:
        try:
            optim.load_state_dict(resume_state["optim"])
            sched.load_state_dict(resume_state["sched"])
        except Exception as e:
            print(f"[RESUME] optim/sched 복원 경고: {e}")
        start_epoch = resume_state.get("next_epoch", 0)
        gstep = resume_state.get("gstep", 0)
        best_sel = resume_state.get("best_sel", best_sel)
        best_metrics = resume_state.get("best_metrics", {})
        if "py_rng" in resume_state:
            random.setstate(resume_state["py_rng"])
        if "torch_rng" in resume_state:
            torch.set_rng_state(resume_state["torch_rng"])
        print(f"[RESUME] epoch {start_epoch}부터 재개 (gstep={gstep}, best_{args.select_metric}={best_sel:.4f})", flush=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        idxs, _, _ = balanced_indices(train, args.balance_power, args.epoch_size, args.per_class_cap, args.seed + epoch)
        optim.zero_grad()
        running, seen = 0.0, 0
        for b in range(0, len(idxs), args.batch_size):
            batch = [train[i] for i in idxs[b:b + args.batch_size]]
            out = model(**collate(batch))
            loss = out.loss / args.grad_accum
            loss.backward()
            running += out.loss.item(); seen += 1
            if (seen % args.grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optim.step(); sched.step(); optim.zero_grad(); gstep += 1
                if args.wandb and gstep % 10 == 0:   # 매 10 스텝만 로깅(동기화 오버헤드↓)
                    import wandb
                    wandb.log({"train_loss": out.loss.item(), "lr": sched.get_last_lr()[0], "epoch": epoch, "step": gstep})
            if seen % 50 == 0:
                print(f"[TRAIN] epoch {epoch} {b+len(batch)}/{len(idxs)} loss={running/seen:.4f} lr={sched.get_last_lr()[0]:.2e}", flush=True)

        val_loss, val_acc, per_class, val_invalid = evaluate(model, processor, val, collate, build_messages, device, args.max_val)
        macro = round(sum(per_class.values()) / len(per_class), 4) if per_class else 0.0
        turn_recall = round((per_class.get("Turn left 15 degree", 0.0) + per_class.get("Turn right 15 degree", 0.0)) / 2, 4)
        stop_recall = per_class.get("Stop", 0.0)
        sel = macro if args.select_metric == "macro" else val_acc
        print(f"[VAL] epoch {epoch} loss={val_loss:.4f} acc={val_acc:.4f} macro={macro} "
              f"turn={turn_recall} stop={stop_recall} invalid={val_invalid} per_class={per_class}", flush=True)
        if args.wandb:
            import wandb
            log = {"val_loss": val_loss, "val_accuracy": val_acc, "val_macro": macro,
                   "val_turn_recall": turn_recall, "val_stop_recall": stop_recall,
                   "val_invalid_rate": val_invalid, "epoch": epoch}
            log.update({f"val_acc/{k}": v for k, v in per_class.items()})
            wandb.log(log)
        ckpt = os.path.join(args.output, f"epoch_{epoch}")
        model.save_pretrained(ckpt); processor.save_pretrained(ckpt)
        last_metrics = {"epoch": epoch, "val_loss": round(val_loss, 4), "overall": round(val_acc, 4),
                        "macro": macro, "turn_recall": turn_recall, "stop_recall": stop_recall,
                        "invalid_rate": val_invalid, "per_class": per_class}
        if sel >= best_sel:
            best_sel = sel
            best_metrics = dict(last_metrics)
            bd = os.path.join(args.output, "best")
            model.save_pretrained(bd); processor.save_pretrained(bd)
            print(f"[BEST] epoch {epoch} {args.select_metric}={sel:.4f} -> {bd}", flush=True)
        # resume용 학습상태 저장 (epoch 체크포인트 폴더 안에 → --resume <epoch_dir>가 adapter+state 동시 로드)
        torch.save({"optim": optim.state_dict(), "sched": sched.state_dict(),
                    "next_epoch": epoch + 1, "gstep": gstep, "best_sel": best_sel,
                    "best_metrics": best_metrics, "py_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state()},
                   os.path.join(ckpt, "trainer_state.pt"))

    summary = {"select_metric": args.select_metric, "best": best_metrics or last_metrics,
               "last": last_metrics, "config": {k: getattr(args, k) for k in
               ("balance_power", "epoch_size", "per_class_cap", "lora_r", "lora_alpha",
                "lora_dropout", "lr", "epochs", "batch_size", "grad_accum", "seed")}}
    if args.summary_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.summary_json)), exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[OK] summary -> {args.summary_json}", flush=True)
    print(f"[OK] done. best {args.select_metric}={best_sel:.4f}; adapter: {os.path.join(args.output,'best')}")


def evaluate(model, processor, val, collate, build_messages, device, max_val):
    import torch
    from PIL import Image
    if not val:
        return 0.0, 0.0, {}
    model.eval()
    sample = val if len(val) <= max_val else random.Random(0).sample(val, max_val)
    losses, correct, invalid = [], 0, 0
    per_total = {a: 0 for a in CANONICAL_ACTIONS}
    per_correct = {a: 0 for a in CANONICAL_ACTIONS}
    gen_pad = processor.tokenizer.padding_side
    with torch.no_grad():
        for s in sample:
            losses.append(model(**collate([s])).loss.item())
            frames = [Image.open(p).convert("RGB") for p in s["_image_paths"]]
            processor.tokenizer.padding_side = "left"
            pin = processor.apply_chat_template(build_messages(s, frames, False), tokenize=True,
                                                add_generation_prompt=True, return_dict=True,
                                                return_tensors="pt").to(device)
            processor.tokenizer.padding_side = gen_pad
            gen = model.generate(**pin, max_new_tokens=8, do_sample=False)
            text = processor.batch_decode(gen[:, pin["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            if text not in CANONICAL_ACTIONS:
                invalid += 1
            tgt = s["answer"]
            per_total[tgt] = per_total.get(tgt, 0) + 1
            if text == tgt:
                correct += 1; per_correct[tgt] = per_correct.get(tgt, 0) + 1
    acc = correct / len(sample)
    per_class = {a: round(per_correct[a] / per_total[a], 3) for a in CANONICAL_ACTIONS if per_total.get(a)}
    invalid_rate = round(invalid / len(sample), 4)
    return sum(losses) / len(losses), acc, per_class, invalid_rate


if __name__ == "__main__":
    main()
