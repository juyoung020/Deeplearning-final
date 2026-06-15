"""Build the index-map JSON for eval_closed_loop_metrics --index-map.

demo.py saves each closed-loop trajectory CSV as ``<episode_id>.csv``. This maps
each such CSV basename to its index in the demo-json, so the metrics script can
look up the matching goal / reference_path per CSV.

Usage:
  python week4/make_index_map.py \
    --demo-json data/closed_loop_eval/closed_loop_20_demo.json \
    --out data/closed_loop_eval/closed_loop_20_index_map.json
"""
from __future__ import annotations

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eps = json.load(open(args.demo_json, encoding="utf-8"))
    idx_map = {f"{ep['episode_id']}.csv": i for i, ep in enumerate(eps)}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(idx_map, f, ensure_ascii=False, indent=2)
    print(f"[index-map] {len(idx_map)} entries -> {args.out}")


if __name__ == "__main__":
    main()
