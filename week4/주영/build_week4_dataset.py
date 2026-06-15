"""Week 4 — build LoRA training samples from the real VLN-VERSE dataset.

Reads the LeRobot-style VLN-VERSE trajectory data (spec section 6) and emits one
training sample per timestep:

    {"episode_id": "<scan>/<episode>", "step": t,
     "instruction": "...", "images": ["<rel t-1>.jpg", "<rel t>.jpg"],
     "answer": "Move forward 25cm"}

Per spec:
  * answer = ACTION_ID_TO_STRING[ observation.action[t] ]  (id 0..3; last row -> Stop).
  * images = [I_{t-1}, I_t]; t = 0 -> [I_0, I_0].
  * Only episodes listed in fine_train.json.gz are used (download matched traj_data only).

Dataset layout expected under --vlnverse-root:
    traj_data/vlnverse/<scan>/<episode>/
        data/chunk-000/episode_000000.parquet     # column: observation.action, frame_index
        meta/episodes.jsonl                        # instruction_text
        videos/chunk-000/observation.images.rgb/episode_000000.mp4

Usage:
    python build_week4_dataset.py \
        --vlnverse-root D:/VLNVerse_data \
        --split-json D:/VLNVerse_data/raw_data/final_splits/fine_train.json.gz \
        --out data/week4_train.jsonl --frames-dir data/frames
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os

from actions import ACTION_ID_TO_STRING


def _episode_number(scan, episode_id):
    """traj_data folder name = episode_id with the 'scan_' prefix stripped.

    e.g. scan='kujiale_0254', episode_id='kujiale_0254_23_4' -> '23_4'.
    """
    prefix = f"{scan}_"
    return episode_id[len(prefix):] if episode_id.startswith(prefix) else episode_id


def _load_split_episodes(split_json):
    """Return a list of (scan, episode_number) pairs from fine_train.json(.gz).

    Real format: a dict {"episodes": [ {scan, episode_id, ...}, ... ]}.
    """
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
            scan, ep = it.split("/", 1)
            pairs.append((scan, ep))
            continue
        if not isinstance(it, dict):
            continue
        scan = it.get("scan") or it.get("scene_name")
        epnum = it.get("episode_number")
        epid = it.get("episode_id") or ""
        if scan and not epnum and epid:
            epnum = _episode_number(scan, epid)
        if scan and epnum:
            pairs.append((str(scan), str(epnum)))
    return pairs


def _read_instruction(episode_dir):
    jl = os.path.join(episode_dir, "meta", "episodes.jsonl")
    if os.path.exists(jl):
        with open(jl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    return obj.get("instruction_text") or obj.get("instruction") or ""
    return ""


def _read_actions(episode_dir):
    """Return the list of observation.action ids ordered by frame_index."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(episode_dir, "data", "chunk-*", "episode_*.parquet")))
    if not files:
        return []
    actions = []
    for pf in files:
        table = pq.read_table(pf)
        cols = table.column_names
        acol = "observation.action" if "observation.action" in cols else None
        fcol = "frame_index" if "frame_index" in cols else None
        if acol is None:
            continue
        df = table.to_pydict()
        order = range(len(df[acol]))
        if fcol:
            order = sorted(order, key=lambda i: df[fcol][i])
        for i in order:
            a = df[acol][i]
            actions.append(int(a[0] if isinstance(a, (list, tuple)) else a))
    return actions


def _extract_frames(episode_dir, out_dir, n_frames):
    """Load RGB frames from the episode's rgb.npy into out_dir/frame_XXXXXX.jpg.

    VLN-VERSE stores egocentric RGB as a single numpy array
    (videos/chunk-000/observation.images.rgb/rgb.npy), shape [T, 256, 256, 3]
    (NHWC, uint8). Depth (depth.npy) is not used. Returns written frame paths.
    """
    import numpy as np
    from PIL import Image

    cand = sorted(glob.glob(os.path.join(
        episode_dir, "videos", "chunk-*", "observation.images.rgb", "*.npy")))
    if not cand:
        return []
    arr = np.load(cand[0])
    if arr.ndim == 4 and arr.shape[1] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (0, 2, 3, 1))  # NCHW -> NHWC
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1.5 else (arr * 255).astype(np.uint8)
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in range(arr.shape[0]):
        p = os.path.join(out_dir, f"frame_{i:06d}.jpg")
        Image.fromarray(arr[i]).save(p, quality=95)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlnverse-root", required=True)
    ap.add_argument("--split-json", required=True, help="fine_train.json.gz")
    ap.add_argument("--out", default="data/week4_train.jsonl")
    ap.add_argument("--frames-dir", default="data/frames")
    ap.add_argument("--max-episodes", type=int, default=0)
    ap.add_argument("--no-save-frames", action="store_true",
                    help="skip jpg extraction (paths still recorded; frames must exist already)")
    args = ap.parse_args()

    pairs = _load_split_episodes(args.split_json)
    if args.max_episodes:
        pairs = pairs[: args.max_episodes]
    print(f"[INFO] {len(pairs)} episodes from split")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_samples, n_eps, n_missing = 0, 0, 0
    dist = {v: 0 for v in ACTION_ID_TO_STRING.values()}

    with open(args.out, "w", encoding="utf-8") as out:
        for scan, ep in pairs:
            episode_dir = os.path.join(args.vlnverse_root, "traj_data", "vlnverse", scan, ep)
            if not os.path.isdir(episode_dir):
                n_missing += 1
                continue
            instruction = _read_instruction(episode_dir)
            actions = _read_actions(episode_dir)
            if not actions:
                n_missing += 1
                continue

            frame_out = os.path.join(args.frames_dir, scan, ep)
            if args.no_save_frames:
                frame_paths = [os.path.join(frame_out, f"frame_{i:06d}.jpg") for i in range(len(actions))]
            else:
                frame_paths = _extract_frames(episode_dir, frame_out, len(actions))
                if not frame_paths:
                    n_missing += 1
                    continue

            n = min(len(actions), len(frame_paths))
            for t in range(n):
                prev = frame_paths[t - 1] if t > 0 else frame_paths[0]  # t=0 -> [I_0, I_0]
                cur = frame_paths[t]
                answer = ACTION_ID_TO_STRING.get(actions[t], "Stop")
                dist[answer] = dist.get(answer, 0) + 1
                rel = lambda p: os.path.relpath(p, args.frames_dir).replace("\\", "/")
                out.write(json.dumps({
                    "episode_id": f"{scan}/{ep}",
                    "step": t,
                    "instruction": instruction,
                    "images": [rel(prev), rel(cur)],
                    "answer": answer,
                }, ensure_ascii=False) + "\n")
                n_samples += 1
            n_eps += 1
            if n_eps % 50 == 0:
                print(f"[INFO] {n_eps} episodes, {n_samples} samples...")

    print(f"[OK] wrote {n_samples} samples from {n_eps} episodes "
          f"({n_missing} missing) -> {args.out}")
    print(f"[OK] action distribution: {dist}")
    print(f"[OK] image-root for training: {args.frames_dir}")


if __name__ == "__main__":
    main()
