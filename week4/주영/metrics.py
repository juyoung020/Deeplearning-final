"""Week 4 — closed-loop VLN evaluation metrics (spec section 14).

Given an agent trajectory and the episode's ground-truth reference path + goal,
compute the standard VLN metrics:

    SR        : success — final position within `success_radius` (3 m) of goal
    OSR       : oracle success — any point on the trajectory within radius
    SPL       : success weighted by (GT path length / max(agent, GT) length)
    nDTW      : normalized dynamic time warping vs the GT path (Ilharco et al. 2019)
    GoalDist  : final distance to the goal

All positions are (x, y). Self-contained (numpy only) so it can be unit-tested
without Isaac Sim and reused by the closed-loop runner.
"""

from __future__ import annotations

import math

import numpy as np

SUCCESS_RADIUS = 3.0


def _xy(p):
    return np.asarray([p[0], p[1]], dtype=np.float64)


def path_length(path):
    pts = [_xy(p) for p in path]
    return float(sum(np.linalg.norm(pts[i] - pts[i - 1]) for i in range(1, len(pts)))) if len(pts) > 1 else 0.0


def _dtw(a, b):
    """Classic DTW distance between two 2D polylines."""
    A = [_xy(p) for p in a]
    B = [_xy(p) for p in b]
    n, m = len(A), len(B)
    if n == 0 or m == 0:
        return float("inf")
    inf = float("inf")
    D = [[inf] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(A[i - 1] - B[j - 1]))
            D[i][j] = cost + min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1])
    return D[n][m]


def ndtw(agent_path, gt_path, success_radius=SUCCESS_RADIUS):
    """Normalized DTW in [0, 1]; higher = closer to the GT path."""
    if len(gt_path) == 0:
        return 0.0
    dtw = _dtw(agent_path, gt_path)
    if not math.isfinite(dtw):
        return 0.0
    return float(math.exp(-dtw / (len(gt_path) * success_radius)))


def compute_metrics(agent_path, gt_path, goal, success_radius=SUCCESS_RADIUS):
    """Return a dict of {SR, OSR, SPL, nDTW, GoalDist} for one episode.

    agent_path : list of (x, y[, z]) the robot actually visited (>=1 point)
    gt_path    : list of (x, y[, z]) reference path
    goal       : (x, y[, z]) goal position
    """
    goal_xy = _xy(goal)
    agent = [_xy(p) for p in agent_path] or [goal_xy + 1e3]
    final_dist = float(np.linalg.norm(agent[-1] - goal_xy))
    min_dist = float(min(np.linalg.norm(p - goal_xy) for p in agent))

    sr = 1.0 if final_dist <= success_radius else 0.0
    osr = 1.0 if min_dist <= success_radius else 0.0

    gt_len = path_length(gt_path)
    ag_len = path_length(agent_path)
    spl = sr * (gt_len / max(ag_len, gt_len, 1e-6)) if sr > 0 else 0.0

    return {
        "SR": sr,
        "OSR": osr,
        "SPL": float(spl),
        "nDTW": ndtw(agent_path, gt_path, success_radius),
        "GoalDist": final_dist,
    }


def aggregate(metric_dicts):
    """Mean of each metric over a list of per-episode metric dicts."""
    if not metric_dicts:
        return {}
    keys = metric_dicts[0].keys()
    return {k: float(np.mean([m[k] for m in metric_dicts])) for k in keys}


if __name__ == "__main__":
    # straight 5 m corridor; goal at the end.
    gt = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    goal = (5, 0)

    # perfect follow -> SR=1, OSR=1, SPL~1, nDTW~1, dist=0
    m = compute_metrics(gt, gt, goal)
    assert m["SR"] == 1 and m["OSR"] == 1 and m["GoalDist"] == 0
    assert m["SPL"] > 0.99 and m["nDTW"] > 0.99, m
    print("[OK] perfect:", {k: round(v, 3) for k, v in m.items()})

    # stops 2 m short -> within 3 m radius -> still SR=1
    short = [(0, 0), (1, 0), (2, 0), (3, 0)]
    m2 = compute_metrics(short, gt, goal)
    assert m2["SR"] == 1 and m2["GoalDist"] == 2.0, m2
    print("[OK] short-but-in-radius:", {k: round(v, 3) for k, v in m2.items()})

    # wanders far and ends 6 m away -> SR=0, OSR depends
    far = [(0, 0), (0, 3), (0, 6), (-1, 6)]
    m3 = compute_metrics(far, gt, goal)
    assert m3["SR"] == 0 and m3["GoalDist"] > 3, m3
    print("[OK] failed:", {k: round(v, 3) for k, v in m3.items()})

    print("[OK] aggregate:", {k: round(v, 3) for k, v in aggregate([m, m2, m3]).items()})
    print("\nMETRICS SELF-TEST PASSED")
