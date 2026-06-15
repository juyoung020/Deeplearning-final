"""Model-free end-to-end test of the dataset builder on a synthetic VLN-VERSE episode.

Fabricates one LeRobot-style episode (parquet + mp4 + episodes.jsonl), runs
build_week4_dataset.derive logic via subprocess, and checks the JSONL output —
so the parquet/video pipeline is verified without the ~65 GiB download. Run:

    python smoke_test_dataset.py
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def make_episode(root, scan="test_scan", ep="0_0", actions=(1, 1, 2, 3, 0)):
    ep_dir = os.path.join(root, "traj_data", "vlnverse", scan, ep)
    os.makedirs(os.path.join(ep_dir, "meta"), exist_ok=True)
    os.makedirs(os.path.join(ep_dir, "data", "chunk-000"), exist_ok=True)
    rgb_dir = os.path.join(ep_dir, "videos", "chunk-000", "observation.images.rgb")
    os.makedirs(rgb_dir, exist_ok=True)

    # meta/episodes.jsonl
    with open(os.path.join(ep_dir, "meta", "episodes.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"episode_id": ep, "instruction_text": "Walk forward then turn.",
                            "finish_status": "success"}) + "\n")

    # parquet with observation.action + frame_index
    n = len(actions)
    table = pa.table({
        "observation.action": pa.array([[a] for a in actions]),  # list-wrapped like real data
        "frame_index": pa.array(list(range(n))),
        "observation.step": pa.array(list(range(n))),
    })
    pq.write_table(table, os.path.join(ep_dir, "data", "chunk-000", "episode_000000.parquet"))

    # mp4 with n distinct frames
    vw = cv2.VideoWriter(os.path.join(rgb_dir, "episode_000000.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for i in range(n):
        frame = np.full((64, 64, 3), (i * 40 % 255, 0, 0), dtype=np.uint8)
        vw.write(frame)
    vw.release()
    return ep_dir


def main():
    root = tempfile.mkdtemp(prefix="vlnverse_synth_")
    try:
        make_episode(root)
        # split index
        split = os.path.join(root, "fine_train.json.gz")
        with gzip.open(split, "wt", encoding="utf-8") as f:
            json.dump(["test_scan/0_0"], f)

        out_jsonl = os.path.join(root, "out.jsonl")
        frames_dir = os.path.join(root, "frames")
        here = os.path.dirname(os.path.abspath(__file__))
        rc = subprocess.run([sys.executable, os.path.join(here, "build_week4_dataset.py"),
                             "--vlnverse-root", root, "--split-json", split,
                             "--out", out_jsonl, "--frames-dir", frames_dir],
                            capture_output=True, text=True)
        print(rc.stdout)
        if rc.returncode != 0:
            print(rc.stderr)
            raise SystemExit("builder failed")

        rows = [json.loads(l) for l in open(out_jsonl, encoding="utf-8") if l.strip()]
        assert len(rows) == 5, f"expected 5 samples, got {len(rows)}"
        answers = [r["answer"] for r in rows]
        assert answers == ["Move forward 25cm", "Move forward 25cm", "Turn left 15 degree",
                           "Turn right 15 degree", "Stop"], answers
        # t=0 duplicates first frame
        assert rows[0]["images"][0] == rows[0]["images"][1], rows[0]["images"]
        # t=1 uses [I_0, I_1]
        assert rows[1]["images"][0].endswith("frame_000000.jpg")
        assert rows[1]["images"][1].endswith("frame_000001.jpg")
        # frames actually written
        for r in rows:
            for img in r["images"]:
                assert os.path.exists(os.path.join(frames_dir, img)), img
        print(f"[OK] dataset builder: {len(rows)} samples, answers={answers}")
        print("\nWEEK4 DATASET PIPELINE SMOKE TEST PASSED")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
