"""Resilient per-episode VLN-VERSE downloader (no HF token needed).

snapshot_download stalls on anonymous rate limits and, worse, re-verifies every
cached file (a HEAD request each) on restart — so restarts get slower. This
downloader instead:
  * skips an episode instantly if its files already exist locally (no network), and
  * downloads each missing file individually with retries,
so it grinds forward through rate-limit stalls and resumes for free.

Run (repeatedly / in a watchdog loop is fine):
    python download_resilient.py --root C:/Deeplearning-final/VLNVerse_data --split fine_train
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

DATA_REPO = "Eyz/VLNVerse_data"

# Files we need per episode (depth.npy intentionally skipped).
PER_EPISODE_FILES = [
    "data/chunk-000/episode_000000.parquet",
    "meta/episodes.jsonl",
    "meta/info.json",
    "meta/tasks.jsonl",
    "videos/chunk-000/observation.images.rgb/rgb.npy",
]


def _pairs(split_json):
    opener = gzip.open if split_json.endswith(".gz") else open
    with opener(split_json, "rt", encoding="utf-8") as f:
        data = json.load(f)
    items = data["episodes"] if isinstance(data, dict) and "episodes" in data else data
    out = []
    for e in items:
        scan = e.get("scan") or e.get("scene_name")
        epnum = e.get("episode_number")
        epid = e.get("episode_id") or ""
        if scan and not epnum and epid:
            epnum = epid[len(scan) + 1:] if epid.startswith(f"{scan}_") else epid
        if scan and epnum:
            out.append((str(scan), str(epnum)))
    return out


def _download_episode(scan, ep, root, retries):
    """Returns 'skip' | 'new' | 'fail'. Skips instantly (no network) if present."""
    from huggingface_hub import hf_hub_download

    base = os.path.join(root, "traj_data", "vlnverse", scan, ep)
    rgb = os.path.join(base, "videos", "chunk-000", "observation.images.rgb", "rgb.npy")
    parquet = os.path.join(base, "data", "chunk-000", "episode_000000.parquet")
    if os.path.exists(rgb) and os.path.exists(parquet):
        return "skip"
    ok = True
    for rel in PER_EPISODE_FILES:
        local = os.path.join(base, *rel.split("/"))
        if os.path.exists(local):
            continue
        remote = f"traj_data/vlnverse/{scan}/{ep}/{rel}"
        for attempt in range(retries):
            try:
                hf_hub_download(DATA_REPO, remote, repo_type="dataset", local_dir=root)
                break
            except Exception as exc:
                if attempt == retries - 1:
                    if rel.endswith(("rgb.npy", "episode_000000.parquet", "episodes.jsonl")):
                        ok = False
                    print(f"[WARN] {remote} failed: {repr(exc)[:70]}", flush=True)
                else:
                    time.sleep(1.5 * (attempt + 1))
    return "new" if ok else "fail"


def main():
    import concurrent.futures as cf

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="fine_train")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    split_json = os.path.join(args.root, "raw_data", "final_splits", f"{args.split}.json.gz")
    if not os.path.exists(split_json):
        split_json = hf_hub_download(DATA_REPO, f"raw_data/final_splits/{args.split}.json.gz",
                                     repo_type="dataset", local_dir=args.root)
    pairs = _pairs(split_json)
    print(f"[INFO] {len(pairs)} episodes in split; {args.workers} workers")

    counts = {"skip": 0, "new": 0, "fail": 0}
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_download_episode, s, e, args.root, args.retries): (s, e) for s, e in pairs}
        for i, fut in enumerate(cf.as_completed(futs)):
            try:
                counts[fut.result()] += 1
            except Exception as exc:
                counts["fail"] += 1
                print(f"[WARN] worker error: {repr(exc)[:70]}", flush=True)
            done += 1
            if done % 100 == 0:
                print(f"[INFO] {done}/{len(pairs)} | skip {counts['skip']} new {counts['new']} fail {counts['fail']}", flush=True)

    print(f"[OK] complete pass: skip {counts['skip']} new {counts['new']} fail {counts['fail']}")


if __name__ == "__main__":
    main()
