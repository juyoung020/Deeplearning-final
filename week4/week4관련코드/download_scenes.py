"""Download VLN-VERSE closed-loop scenes (Eyz/VLNVerse_scene) to a target dir.

Downloads only the scans needed for the assignment eval sets (20-episode set
first, then the 232/53-scan set), into ``<out>/<scan>/...`` so demo.py can find
them via ``--go2-physics-scene-root <out>``.

Usage:
  HF_HUB_ENABLE_HF_TRANSFER=1 python week4/download_scenes.py \
    --out /media/ad06/Samsung_T5/data \
    --scan-lists data/closed_loop_eval/_scan_lists.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

REPO = "Eyz/VLNVerse_scene"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scan-lists", default="data/closed_loop_eval/_scan_lists.json")
    ap.add_argument("--which", default="union", choices=["s20", "s232", "union"])
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    lists = json.load(open(args.scan_lists))
    s20, s232 = lists["s20"], lists["s232"]
    # phase order: 20-set first (smallest, unblocks the 20-ep table), then the rest
    if args.which == "s20":
        phases = [("20-set", s20)]
    elif args.which == "s232":
        phases = [("232-set", s232)]
    else:
        rest = [s for s in s232 if s not in set(s20)]
        phases = [("20-set", s20), ("232-rest", rest)]

    os.makedirs(args.out, exist_ok=True)
    for name, scans in phases:
        if not scans:
            continue
        print(f"\n===== PHASE {name}: {len(scans)} scans =====", flush=True)
        t0 = time.time()
        patterns = [f"{s}/*" for s in scans]
        # snapshot_download resumes, so retry on transient transport errors
        for attempt in range(1, 21):
            try:
                snapshot_download(
                    repo_id=REPO, repo_type="dataset",
                    local_dir=args.out, allow_patterns=patterns,
                    max_workers=8,
                )
                break
            except Exception as exc:
                print(f"[phase {name}] attempt {attempt} failed: {exc!r}; retrying...", flush=True)
                time.sleep(min(30, 3 * attempt))
        dt = time.time() - t0
        # report downloaded size
        got = [s for s in scans if os.path.isdir(os.path.join(args.out, s))]
        print(f"[phase {name}] {len(got)}/{len(scans)} scan dirs present "
              f"in {dt/60:.1f} min", flush=True)

    print("\n[done] scene download complete ->", args.out, flush=True)


if __name__ == "__main__":
    main()
