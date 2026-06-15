"""Week 4 — selective downloader for the VLN-VERSE dataset.

Per the spec, the full traj_data is huge (~hundreds of GB), so we do NOT clone the
whole repo. Instead:
  1. download the split index (fine_train.json.gz / fine_val.json.gz),
  2. parse the (scan, episode) list,
  3. download ONLY the matched traj_data/vlnverse/<scan>/<episode> folders.

Expected footprint: ~65 GiB for the fine-train subset.

Usage (train data):
    set HF_HUB_DISABLE_SYMLINKS=1
    python download_vlnverse.py --root D:/VLNVerse_data --split fine_train --max-episodes 0

Scene data for closed-loop eval comes from a different repo (Eyz/VLNVerse_scene);
pass --scenes to fetch the matched scans for evaluation.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

DATA_REPO = "Eyz/VLNVerse_data"
SCENE_REPO = "Eyz/VLNVerse_scene"


def _split_pairs(split_json):
    """(scan, episode_number) pairs. episode folder = episode_id minus 'scan_' prefix."""
    opener = gzip.open if split_json.endswith(".gz") else open
    with opener(split_json, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "episodes" in data:
        items = data["episodes"]
    elif isinstance(data, list):
        items = data
    else:
        items = data.get("data", [])
    pairs = []
    for it in items:
        if isinstance(it, str) and "/" in it:
            pairs.append(tuple(it.split("/", 1)))
        elif isinstance(it, dict):
            scan = it.get("scan") or it.get("scene_name")
            epnum = it.get("episode_number")
            epid = it.get("episode_id") or ""
            if scan and not epnum and epid:
                epnum = epid[len(scan) + 1:] if epid.startswith(f"{scan}_") else epid
            if scan and epnum:
                pairs.append((str(scan), str(epnum)))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="local dir to download into")
    ap.add_argument("--split", default="fine_train", choices=["fine_train", "fine_val"])
    ap.add_argument("--max-episodes", type=int, default=0, help="0 = all matched episodes")
    ap.add_argument("--scenes", action="store_true", help="also fetch matched scene data for eval")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download, snapshot_download

    os.makedirs(args.root, exist_ok=True)
    rel = f"raw_data/final_splits/{args.split}.json.gz"
    print(f"[1/3] downloading split index {rel}")
    split_path = hf_hub_download(repo_id=DATA_REPO, filename=rel, repo_type="dataset", local_dir=args.root)

    pairs = _split_pairs(split_path)
    if args.max_episodes:
        pairs = pairs[: args.max_episodes]
    scans = sorted({s for s, _ in pairs})
    print(f"[2/3] {len(pairs)} episodes across {len(scans)} scans -> downloading matched traj_data")

    patterns = [f"traj_data/vlnverse/{scan}/{ep}/**" for scan, ep in pairs]
    # rgb.npy + parquet + meta only; depth.npy is large and unused -> skip it.
    ignore = ["*observation.images.depth*"]
    # snapshot_download accepts many allow_patterns; chunk to stay safe.
    for i in range(0, len(patterns), 400):
        chunk = patterns[i:i + 400]
        snapshot_download(repo_id=DATA_REPO, repo_type="dataset", local_dir=args.root,
                          allow_patterns=chunk, ignore_patterns=ignore)
        print(f"   ...{min(i+400, len(patterns))}/{len(patterns)} episode folders")

    if args.scenes:
        print(f"[3/3] downloading scene data for {len(scans)} scans from {SCENE_REPO}")
        scene_patterns = [f"*{scan}*" for scan in scans]
        snapshot_download(repo_id=SCENE_REPO, repo_type="dataset", local_dir=os.path.join(args.root, "scenes"),
                          allow_patterns=scene_patterns)

    print(f"[OK] download complete -> {args.root}")
    print(f"     next: python build_week4_dataset.py --vlnverse-root {args.root} "
          f"--split-json {split_path} --out data/week4_train.jsonl --frames-dir data/frames")


if __name__ == "__main__":
    main()
