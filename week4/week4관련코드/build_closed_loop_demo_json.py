"""Build a demo.py-compatible episode JSON for closed-loop evaluation.

Takes one of the assignment's closed-loop selection files and resolves each
selected episode to its FULL record (start_position, start_rotation,
instruction, reference_path, goals, info, scan, episode_id) by looking it up in
the VLN-VERSE split file(s). Output is a plain list of episode dicts, the same
schema as ``fine_grained_demo.json``, so ``demo.py --demo-json <out>`` can run
each by ``--index``.

Two input modes (auto-detected):
  * 20-episode file  (has ``episodes: [{split, episode_id, ...}]``)
      -> resolve each listed episode_id (preserves ``split``).
  * top53 scans file (has ``scans: [...]``)
      -> select every split-file episode whose ``scan`` is in that set.

Usage:
  python week4/build_closed_loop_demo_json.py \
    --episode-list data/closed_loop_eval/vlnverse_closed_loop_eval_20episodes.json \
    --split-json   data/VLNVerse_data/raw_data/final_splits/fine_train.json.gz \
    --out          data/closed_loop_eval/closed_loop_20_demo.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os


def load_split(path):
    opn = gzip.open if path.endswith(".gz") else open
    with opn(path, "rt", encoding="utf-8") as f:
        d = json.load(f)
    eps = d["episodes"] if isinstance(d, dict) and "episodes" in d else d
    return [r for r in eps if isinstance(r, dict)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-list", required=True, help="20-episode or top53 selection json")
    ap.add_argument("--split-json", action="append", required=True,
                    help="VLN-VERSE split file(s) to resolve records from (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # id -> full record from all provided split files
    by_id, by_scan = {}, {}
    for sp in args.split_json:
        if not os.path.exists(sp):
            print(f"[warn] split file not found, skipping: {sp}")
            continue
        for r in load_split(sp):
            eid = r.get("episode_id")
            if eid:
                by_id.setdefault(eid, r)
                by_scan.setdefault(str(r.get("scan", "")), []).append(r)
    print(f"[build] indexed {len(by_id)} episodes from {len(args.split_json)} split file(s)")

    sel = json.load(open(args.episode_list, encoding="utf-8"))
    out, missing = [], []

    if isinstance(sel, dict) and "episodes" in sel:  # 20-episode mode
        for e in sel["episodes"]:
            eid = e["episode_id"]
            rec = by_id.get(eid)
            if rec is None:
                missing.append(eid); continue
            rec = dict(rec); rec["_split"] = e.get("split")
            out.append(rec)
    elif isinstance(sel, dict) and "scans" in sel:  # top53 mode
        scans = set(sel["scans"])
        for scan in sel["scans"]:
            for rec in by_scan.get(scan, []):
                r = dict(rec); r["_split"] = "val_seen"
                out.append(r)
        # report scans with zero resolved episodes
        for scan in scans:
            if not by_scan.get(scan):
                missing.append(f"scan:{scan}")
    else:
        ap.error("unrecognized selection file (need 'episodes' or 'scans')")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[build] resolved {len(out)} episodes -> {args.out}")
    if missing:
        print(f"[build] MISSING {len(missing)} (not in given split files):")
        for m in missing[:20]:
            print("   ", m)
        if len(missing) > 20:
            print(f"    ... +{len(missing)-20} more")


if __name__ == "__main__":
    main()
