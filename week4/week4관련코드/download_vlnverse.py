"""Selective VLN-VERSE downloader.

DEFAULT IS DRY-RUN. Nothing is downloaded unless ``--no-dry-run`` is passed, so
this script never pulls ~65 GiB by accident. It downloads only the split file
plus the trajectory episodes referenced by that split (optionally capped by
``--max-episodes``).

Repo: ``Eyz/VLNVerse_data`` (HF dataset). Scene data (``Eyz/VLNVerse_scene``)
is only needed for closed-loop evaluation and is out of scope here.

Examples:
    # show the split + the first 5 episodes that would be fetched
    python download_vlnverse.py --dest data/VLNVerse_data --max-episodes 5

    # actually download the split file and 5 episodes
    python download_vlnverse.py --dest data/VLNVerse_data --max-episodes 5 --no-dry-run
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

DEFAULT_REPO = "Eyz/VLNVerse_data"
DEFAULT_SPLIT = "raw_data/final_splits/fine_train.json.gz"


def _episode_patterns(scan: str, subdir: str, include_depth: bool = False) -> List[str]:
    """allow_patterns for snapshot_download to fetch one episode's files.

    By default depth.npy is excluded (Week 4 only needs RGB), which roughly
    halves the per-episode download.
    """
    base = f"traj_data/vlnverse/{scan}/{subdir}"
    pats = [
        f"{base}/data/**",
        f"{base}/meta/**",
        f"{base}/videos/**/observation.images.rgb/**",
    ]
    if include_depth:
        pats.append(f"{base}/videos/**/observation.images.depth/**")
    return pats


def main():
    parser = argparse.ArgumentParser(description="Selective VLN-VERSE download (dry-run by default)")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--split-path", default=DEFAULT_SPLIT, help="path within the HF repo")
    parser.add_argument("--dest", default="data/VLNVerse_data")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--include-depth", action="store_true", help="also fetch depth.npy (default: RGB only)")
    parser.add_argument("--workers", type=int, default=16, help="parallel download threads")
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true")
    parser.set_defaults(dry_run=True)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download, HfApi

    os.makedirs(args.dest, exist_ok=True)

    # 1) the split file (small) -- always allowed to download even in dry-run so
    #    we can enumerate episodes. Skip if it already exists locally.
    local_split = os.path.join(args.dest, args.split_path)
    if os.path.exists(local_split):
        print(f"[download] split already present: {local_split}")
    elif args.dry_run:
        print(f"[dry-run] would download split {args.repo_id}:{args.split_path} -> {local_split}")
        print("[dry-run] re-run with --no-dry-run to fetch the split and enumerate episodes.")
        return
    else:
        local_split = hf_hub_download(
            repo_id=args.repo_id, filename=args.split_path, repo_type="dataset",
            local_dir=args.dest,
        )
        print(f"[download] split -> {local_split}")

    # 2) enumerate episodes from the split
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vlnverse_dataset import load_split_records, _record_field, _episode_subdir

    records = load_split_records(local_split)
    print(f"[download] split lists {len(records)} records")

    selected = []
    for rec in records:
        scan = str(_record_field(rec, "scan", "scan_id", "scene", "building") or "")
        ep = _record_field(rec, "episode_id", "episode", "id", "traj_id", "trajectory_id")
        if ep is None:
            continue
        subdir = _episode_subdir(scan, str(ep))
        selected.append((scan, subdir))
        if args.max_episodes and len(selected) >= args.max_episodes:
            break

    print(f"[download] {len(selected)} episodes selected (max_episodes={args.max_episodes})")

    if args.dry_run:
        for scan, subdir in selected[:20]:
            print(f"[dry-run] would fetch traj_data/vlnverse/{scan}/{subdir}/ (rgb"
                  f"{'+depth' if args.include_depth else ''}, no depth by default)")
        if len(selected) > 20:
            print(f"[dry-run] ... and {len(selected) - 20} more")
        print("[dry-run] re-run with --no-dry-run to download these episodes.")
        return

    # Exact-file download (fast + resumable). We compute the predictable per-episode
    # file paths and validate them against the repo file listing once, then fetch
    # each missing file with hf_hub_download (skips files already on disk).
    print("[download] listing repo files (one-time) for validation ...")
    repo_files = set(HfApi().list_repo_files(args.repo_id, repo_type="dataset"))

    def episode_files(scan, subdir):
        base = f"traj_data/vlnverse/{scan}/{subdir}"
        cands = [
            f"{base}/data/chunk-000/episode_000000.parquet",
            f"{base}/meta/episodes.jsonl",
            f"{base}/meta/info.json",
            f"{base}/meta/tasks.jsonl",
            f"{base}/videos/chunk-000/observation.images.rgb/rgb.npy",
        ]
        if args.include_depth:
            cands.append(f"{base}/videos/chunk-000/observation.images.depth/depth.npy")
        # only those that actually exist in the repo
        return [c for c in cands if c in repo_files]

    import time as _time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    def fetch_with_retry(filename, retries=8):
        """hf_hub_download with retry/backoff for flaky HF connections."""
        delay = 3.0
        for attempt in range(1, retries + 1):
            try:
                hf_hub_download(repo_id=args.repo_id, filename=filename,
                                repo_type="dataset", local_dir=args.dest)
                return True
            except Exception as exc:
                if attempt == retries:
                    print(f"[download] GIVE UP {filename} after {retries} tries: {exc}",
                          flush=True)
                    return False
                _time.sleep(delay)
                delay = min(delay * 1.7, 60.0)

    # Build the flat list of files still missing on disk (per-episode files are
    # downloaded concurrently because the bottleneck is per-request latency, not
    # bandwidth -- one-at-a-time is slow even when authenticated).
    total_eps = len(selected)
    needed = []
    skipped = missing_rgb = 0
    for scan, subdir in selected:
        files = episode_files(scan, subdir)
        if not any(f.endswith("rgb.npy") for f in files):
            missing_rgb += 1
        for f in files:
            if os.path.exists(os.path.join(args.dest, f)):
                skipped += 1
            else:
                needed.append(f)

    print(f"[download] {total_eps} episodes, {skipped} files already present, "
          f"{len(needed)} files to fetch, missing rgb={missing_rgb}", flush=True)

    fetched = failed = 0
    lock = threading.Lock()
    done = 0
    n_need = len(needed)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_with_retry, f): f for f in needed}
        for fut in as_completed(futs):
            ok = fut.result()
            with lock:
                done += 1
                if ok:
                    fetched += 1
                else:
                    failed += 1
                if done % 200 == 0 or done == n_need:
                    print(f"[download] files {done}/{n_need} "
                          f"(fetched={fetched}, failed={failed})", flush=True)

    print(f"[download] done -> {args.dest}  "
          f"(episodes={total_eps}, files fetched={fetched}, failed={failed}, "
          f"missing rgb={missing_rgb})")


if __name__ == "__main__":
    main()
