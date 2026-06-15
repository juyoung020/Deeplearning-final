"""Offline evaluation for Week 4: generation-based action metrics.

Metrics:
  * overall exact-match accuracy (output must EXACTLY equal a canonical string)
  * per-class accuracy for each of the four canonical actions
  * invalid output rate (outputs that don't exactly match any canonical string)

Generation config is fixed: max_new_tokens=8, do_sample=False, use_cache=True.

Can be run standalone:
    python eval_week4_offline.py --config configs/week4_qwen3vl_lora.json \
        --adapter outputs/week4/baseline/best_adapter --max-samples 500
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from week4_actions import CANONICAL_ACTIONS, exact_match, parse_canonical_action
from qwen3vl_prompt import build_inference_inputs

GEN_KWARGS = dict(max_new_tokens=8, do_sample=False, temperature=None, use_cache=True)


def generate_action(model, processor, sample: Dict[str, Any], system_prompt: str) -> str:
    """Generate raw text for one sample and return the decoded continuation."""
    import torch

    inputs = build_inference_inputs(
        processor, sample["instruction"], sample["images"],
        system_prompt=system_prompt, trajectory_text=sample.get("trajectory_text"),
    )
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    gen_kwargs = {k: v for k, v in GEN_KWARGS.items() if v is not None}
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = out[0][prompt_len:]
    tok = getattr(processor, "tokenizer", processor)
    text = tok.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def evaluate_generation(
    model,
    processor,
    dataset,
    system_prompt: str,
    max_samples: Optional[int] = None,
    log_examples: int = 20,
) -> Dict[str, Any]:
    """Run generation over (a subset of) ``dataset`` and compute metrics."""
    was_training = model.training
    model.eval()

    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)

    correct = 0
    invalid = 0
    per_class_total = {a: 0 for a in CANONICAL_ACTIONS}
    per_class_correct = {a: 0 for a in CANONICAL_ACTIONS}
    examples: List[Dict[str, Any]] = []

    for i in range(n):
        sample = dataset[i]
        gold = sample["answer"]
        raw = generate_action(model, processor, sample, system_prompt)
        pred_exact = exact_match(raw)
        pred_norm = parse_canonical_action(raw, allow_normalization=True)

        per_class_total[gold] += 1
        is_correct = pred_exact == gold
        if is_correct:
            correct += 1
            per_class_correct[gold] += 1
        if pred_exact is None:
            invalid += 1

        if len(examples) < log_examples:
            examples.append({
                "episode_id": sample.get("episode_id"),
                "timestep": sample.get("timestep"),
                "gold": gold,
                "raw_output": raw,
                "parsed_exact": pred_exact,
                "parsed_normalized": pred_norm,
                "correct": is_correct,
            })

    metrics = {
        "num_samples": n,
        "overall_accuracy": correct / max(n, 1),
        "invalid_output_rate": invalid / max(n, 1),
        "per_class_accuracy": {
            a: (per_class_correct[a] / per_class_total[a] if per_class_total[a] else None)
            for a in CANONICAL_ACTIONS
        },
        "per_class_support": per_class_total,
        "examples": examples,
    }
    if was_training:
        model.train()
    return metrics


def save_metrics(metrics: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def _build_eval_dataset(cfg):
    """Build the held-out validation dataset (episode-level) from a Week4Config."""
    from vlnverse_dataset import (
        build_episodes_from_split, VLNVerseDataset, episode_level_split, load_split_records,
    )

    episodes = build_episodes_from_split(cfg.dataset)
    forced_ids = None
    if cfg.forced_val_split_json and os.path.exists(cfg.forced_val_split_json):
        recs = load_split_records(cfg.forced_val_split_json)
        forced_ids = [
            str(r.get("episode_id") or r.get("episode") or r.get("id")) for r in recs
        ]
    _train, val = episode_level_split(
        episodes, val_ratio=cfg.training.val_ratio, seed=cfg.training.seed,
        forced_val_episode_ids=forced_ids,
    )
    return VLNVerseDataset(val, cfg.dataset)


def main():
    parser = argparse.ArgumentParser(description="Week 4 offline action evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", required=True, help="LoRA adapter dir (best_adapter/epoch_xxx)")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from week4_config import Week4Config
    from qwen3vl_lora_model import load_adapter

    cfg = Week4Config.from_json(args.config)
    model, processor = load_adapter(cfg.model, args.adapter, for_inference=True)
    dataset = _build_eval_dataset(cfg)
    metrics = evaluate_generation(
        model, processor, dataset, cfg.system_prompt, max_samples=args.max_samples,
    )
    out = args.out or os.path.join(cfg.output_dir, "predictions", "offline_eval.json")
    save_metrics(metrics, out)
    print(json.dumps({k: v for k, v in metrics.items() if k != "examples"}, indent=2))
    print(f"[week4] saved offline eval metrics -> {out}")


if __name__ == "__main__":
    main()
