"""Week4 — 본학습 전 폭발방지 프리플라이트 (goodnav env에서 실행).

tiny 데이터로 파이프라인 전체를 사전 점검해, 4시간짜리 본학습이 중간에 터지는 걸 막는다.
train_week4_lora.py의 함수를 그대로 재사용한다(같은 코드 경로를 검증해야 의미가 있음).

PASS/FAIL 요약을 찍고, 하나라도 FAIL이면 종료코드 1.

실행:
    set HF_HUB_DISABLE_SYMLINKS=1
    python preflight.py --data data/week4_train_full.jsonl --image-root data/frames \
        --val-episodes-json data/eval/vlnverse_closed_loop_eval_20episodes.json
"""

from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from actions import CANONICAL_ACTIONS
from train_week4_lora import load_samples, episode_split, build_messages, evaluate

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f"  — {detail}" if detail else ""), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--val-episodes-json", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--tiny", type=int, default=12, help="overfit-tiny 샘플 수(클래스 균등)")
    ap.add_argument("--steps", type=int, default=90, help="overfit-tiny 스텝 수")
    ap.add_argument("--vram-batches", nargs="+", type=int, default=[2, 4, 6, 8])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    random.seed(args.seed); torch.manual_seed(args.seed)
    try:
        import numpy as np; np.random.seed(args.seed)
    except Exception:
        pass
    check("11. 시드 고정(torch/random/numpy)", True, f"seed={args.seed}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check("CUDA 사용 가능", device == "cuda",
          f"{torch.cuda.get_device_name(0)}" if device == "cuda" else "CPU만 — 본학습 불가")

    # ---- 데이터 + episode-level split ----
    samples = load_samples(args.data, args.image_root)
    val_ep_ids = []
    if args.val_episodes_json and os.path.exists(args.val_episodes_json):
        import json
        spec = json.load(open(args.val_episodes_json, encoding="utf-8"))
        if isinstance(spec, dict) and "episodes" in spec:
            val_ep_ids = [f"{e.get('scene_name')}/{e.get('episode_number')}"
                          for e in spec["episodes"] if e.get("split") == "val"]
    train, val, val_eps = episode_split(samples, val_ep_ids, 0.10, args.seed)

    # 9. episode-level 누수 (같은 episode_id가 train·val 양쪽에 없어야)
    tr_eps = {s.get("episode_id") for s in train}
    va_eps = {s.get("episode_id") for s in val}
    check("9. episode-level 누수 없음", tr_eps.isdisjoint(va_eps),
          f"train {len(tr_eps)}ep / val {len(va_eps)}ep, 교집합 {len(tr_eps & va_eps)}")

    # 1. manifest/frame 존재 + 256
    miss, size_ok = [], True
    for s in random.sample(train, min(20, len(train))):
        for p in s["_image_paths"]:
            if not os.path.exists(p):
                miss.append(p)
        if not miss:
            im = Image.open(s["_image_paths"][-1])
            if im.size != (256, 256):
                size_ok = False
    check("1. frame jpg 존재 + 256x256", not miss and size_ok,
          f"누락 {len(miss)}개" + ("" if size_ok else ", 256 아님"))

    # ---- 모델 + LoRA ----
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    print(f"[INFO] loading {args.model} ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, lora); model.to(device)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # 4. LoRA가 q/v에 붙고 trainable%<1%, vision freeze
    lora_mods = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    qv = [n for n in lora_mods if ("q_proj" in n or "v_proj" in n)]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    pct = 100 * n_train / n_all
    vis_frozen = all(not p.requires_grad for n, p in model.named_parameters()
                     if "visual" in n.lower() or "vision" in n.lower())
    check("4. LoRA=q/v only, trainable%<1%, vision frozen",
          len(qv) > 0 and len(qv) == len(lora_mods) and pct < 1.0 and vis_frozen,
          f"lora모듈 {len(lora_mods)}(q/v {len(qv)}), trainable {pct:.3f}%, vision_frozen={vis_frozen}")

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
            labels[i, :plen] = -100
        labels[inputs["attention_mask"] == 0] = -100
        inputs["labels"] = labels
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}, prompt_lens

    # 2 + 3. loss-mask 단위검증 (+ image_pad 정합은 forward 무에러로 검증)
    one, plens = collate([train[0]])
    labels = one["labels"][0]
    unmasked = labels[labels != -100]
    decoded = processor.tokenizer.decode(unmasked, skip_special_tokens=True).strip()
    pre_all_masked = bool((labels[:plens[0]] == -100).all().item())
    check("2. loss-mask: prompt 전부 -100 & 꼬리=answer",
          pre_all_masked and train[0]["answer"] in decoded,
          f"answer='{train[0]['answer']}' vs decoded='{decoded}' (unmasked {len(unmasked)}토큰)")

    # 5. fwd+bwd 1스텝: loss 유한 + grad는 LoRA에만
    model.train()
    out = model(**{k: v for k, v in one.items()})
    loss = out.loss
    loss_ok = torch.isfinite(loss).item()
    loss.backward()
    grad_on_frozen = any((not p.requires_grad) and (p.grad is not None) for p in model.parameters())
    grad_on_lora = any(p.requires_grad and p.grad is not None for p in model.parameters())
    check("3+5. forward/backward OK, grad는 LoRA에만",
          loss_ok and grad_on_lora and not grad_on_frozen,
          f"loss={loss.item():.4f}, lora_grad={grad_on_lora}, frozen_grad={grad_on_frozen}")
    model.zero_grad(set_to_none=True)

    # 8. VRAM 헤드룸 (배치별 peak)
    if device == "cuda":
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        safe = []
        for bs in args.vram_batches:
            try:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                bb, _ = collate([train[i % len(train)] for i in range(bs)])
                o = model(**bb); o.loss.backward(); model.zero_grad(set_to_none=True)
                peak = torch.cuda.max_memory_allocated() / 1e9
                ok = peak < total_gb * 0.92
                if ok: safe.append(bs)
                print(f"      batch {bs}: peak {peak:.1f}/{total_gb:.1f} GB {'OK' if ok else 'RISK'}", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"      batch {bs}: OOM", flush=True)
                torch.cuda.empty_cache(); break
        check("8. VRAM 헤드룸", len(safe) > 0, f"안전 배치={safe} (총 {total_gb:.0f}GB)")

    # 6. overfit-tiny sanity: 클래스 균등 K샘플로 학습 → train 생성정확도↑
    #    (forward 편중 무작위셋이면 'forward만 찍어도 0.7'이라 학습 여부를 구분 못함 → 클래스 균등)
    by_cls = {}
    for s in train:
        by_cls.setdefault(s["answer"], []).append(s)
    per = max(1, args.tiny // max(1, len(by_cls)))
    tiny = []
    for cls in CANONICAL_ACTIONS:
        tiny.extend(by_cls.get(cls, [])[:per])
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4)
    model.train()
    for step in range(args.steps):
        s = tiny[step % len(tiny)]
        bb, _ = collate([s])
        o = model(**bb); o.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    model.eval()
    correct = 0
    with torch.no_grad():
        for s in tiny:
            frames = [Image.open(p).convert("RGB") for p in s["_image_paths"]]
            processor.tokenizer.padding_side = "left"
            pin = processor.apply_chat_template(build_messages(s, frames, False), tokenize=True,
                                                add_generation_prompt=True, return_dict=True,
                                                return_tensors="pt").to(device)
            processor.tokenizer.padding_side = "right"
            gen = model.generate(**pin, max_new_tokens=8, do_sample=False, use_cache=True)
            txt = processor.batch_decode(gen[:, pin["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            correct += (txt == s["answer"])
    tiny_acc = correct / len(tiny)
    check("6. overfit-tiny (파이프라인이 학습은 되는가)", tiny_acc >= 0.8,
          f"{args.tiny}샘플 {args.steps}스텝 후 train acc={tiny_acc:.2f} (≥0.8 기대)")

    # 7. checkpoint 저장 → 재로드 → 동일 출력
    s0 = tiny[0]
    frames = [Image.open(p).convert("RGB") for p in s0["_image_paths"]]
    processor.tokenizer.padding_side = "left"
    pin = processor.apply_chat_template(build_messages(s0, frames, False), tokenize=True,
                                        add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
    processor.tokenizer.padding_side = "right"
    with torch.no_grad():
        before = processor.batch_decode(
            model.generate(**pin, max_new_tokens=8, do_sample=False)[:, pin["input_ids"].shape[1]:],
            skip_special_tokens=True)[0].strip()
    ckdir = os.path.join("adapters", "_preflight_ckpt")
    model.save_pretrained(ckdir)
    only_adapter = os.path.exists(os.path.join(ckdir, "adapter_model.safetensors")) and \
        not os.path.exists(os.path.join(ckdir, "model.safetensors"))
    del opt, model
    torch.cuda.empty_cache()
    base = AutoModelForImageTextToText.from_pretrained(args.model, dtype=dtype, trust_remote_code=True)
    reloaded = PeftModel.from_pretrained(base, ckdir).to(device).eval()
    with torch.no_grad():
        after = processor.batch_decode(
            reloaded.generate(**pin, max_new_tokens=8, do_sample=False)[:, pin["input_ids"].shape[1]:],
            skip_special_tokens=True)[0].strip()
    check("7. checkpoint(trainable만) 저장→재로드 동일출력",
          only_adapter and before == after, f"adapter_only={only_adapter}, '{before}'=='{after}'")

    # ---- 요약 ----
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 60)
    print(f"PREFLIGHT: {n_pass}/{len(RESULTS)} PASS")
    fails = [n for n, ok, _ in RESULTS if not ok]
    if fails:
        print("FAIL:", ", ".join(fails))
    print("=" * 60)
    raise SystemExit(0 if not fails else 1)


if __name__ == "__main__":
    main()
