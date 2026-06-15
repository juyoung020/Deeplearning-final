"""Closed-loop VLN metrics from a trajectory CSV + episode goal/GT path.

Computes the standard VLN navigation metrics for a Go2 closed-loop run:
  SR   : success (final position within `success_radius` of goal)
  OSR  : oracle success (any point on path within radius of goal)
  SPL  : SR weighted by path efficiency = SR * shortest / max(actual, shortest)
  nDTW : normalized Dynamic Time Warping vs the GT reference_path
  GoalDist : final distance to goal

Inputs:
  * trajectory CSV saved by demo.py (columns: pos_x, pos_y, ... per frame)
  * episode meta (fine_grained_demo.json): goals.position, reference_path,
    info.geodesic_distance.

Usage (single episode):
  python week4/eval_closed_loop_metrics.py \
      --csv /home/ad06/isaacsim/IAmGoodNavigator/myresults/kujiale_0010_4_4.csv \
      --demo-json /home/ad06/isaacsim/IAmGoodNavigator/fine_grained_demo.json --index 4

Usage (batch: a folder of CSVs, matched to demo episodes by --index list):
  python week4/eval_closed_loop_metrics.py --csv-dir .../myresults \
      --demo-json .../fine_grained_demo.json --out metrics.json
"""

from __future__ import annotations

import argparse
import csv as csvmod
import json
import math
import os
from typing import List, Optional, Tuple

SUCCESS_RADIUS = 3.0
Point = Tuple[float, float]


# ---------------------------------------------------------------------------
def load_trajectory(csv_path: str) -> List[Point]:
    pts: List[Point] = []
    with open(csv_path, newline="") as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            pts.append((float(row["pos_x"]), float(row["pos_y"])))
    return pts


def _xy(v) -> Point:
    return (float(v[0]), float(v[1]))


def load_episode(demo_json: str, index: int) -> dict:
    with open(demo_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    ep = data[int(index)]
    goal = None
    goals = ep.get("goals")
    if isinstance(goals, dict) and goals.get("position") is not None:
        goal = _xy(goals["position"])
    elif isinstance(goals, list) and goals:
        g0 = goals[0]
        goal = _xy(g0["position"] if isinstance(g0, dict) else g0)
    ref = ep.get("reference_path") or []
    gt_path = [_xy(p) for p in ref] if ref else []
    if goal is None and gt_path:
        goal = gt_path[-1]
    geodesic = None
    info = ep.get("info")
    if isinstance(info, dict):
        geodesic = info.get("geodesic_distance")
    return {"goal": goal, "gt_path": gt_path, "geodesic": geodesic}


# ---------------------------------------------------------------------------
def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def path_length(p: List[Point]) -> float:
    return sum(dist(p[i], p[i + 1]) for i in range(len(p) - 1))


def dtw_distance(a: List[Point], b: List[Point]) -> float:
    n, m = len(a), len(b)
    inf = float("inf")
    prev = [inf] * (m + 1)
    prev[0] = 0.0
    # DP row by row to keep memory O(m)
    for i in range(1, n + 1):
        cur = [inf] * (m + 1)
        for j in range(1, m + 1):
            c = dist(a[i - 1], b[j - 1])
            cur[j] = c + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[m]


def ndtw(pred: List[Point], gt: List[Point], success_radius: float = SUCCESS_RADIUS) -> Optional[float]:
    if not gt or not pred:
        return None
    d = dtw_distance(pred, gt)
    return math.exp(-d / (len(gt) * success_radius))


def compute_metrics(traj: List[Point], ep: dict, success_radius: float = SUCCESS_RADIUS) -> dict:
    goal = ep["goal"]
    gt = ep["gt_path"]
    last = traj[-1]
    goal_dist = dist(last, goal)
    sr = 1.0 if goal_dist <= success_radius else 0.0
    osr = 1.0 if min(dist(p, goal) for p in traj) <= success_radius else 0.0
    actual = path_length(traj)
    # geodesic_distance may be a sentinel (e.g. -1 = unknown) in VLN-VERSE;
    # only trust it when strictly positive, else fall back to GT path length.
    geo = ep.get("geodesic")
    geo = float(geo) if geo is not None else None
    if geo is not None and geo > 0:
        shortest = geo
    else:
        shortest = path_length(gt) if gt else actual
    spl = sr * shortest / max(actual, shortest, 1e-6)
    return {
        "SR": sr,
        "OSR": osr,
        "SPL": spl,
        "nDTW": ndtw(traj, gt, success_radius),
        "GoalDist": goal_dist,
        "path_length": actual,
        "shortest": shortest,
        "num_points": len(traj),
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Closed-loop VLN metrics from trajectory CSV")
    ap.add_argument("--csv", help="single trajectory CSV")
    ap.add_argument("--csv-dir", help="folder of trajectory CSVs (batch)")
    ap.add_argument("--demo-json", required=True)
    ap.add_argument("--index", type=int, default=0, help="episode index for --csv")
    ap.add_argument("--index-map", default=None,
                    help="JSON {csv_basename: index} for batch mode")
    ap.add_argument("--success-radius", type=float, default=SUCCESS_RADIUS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = []
    if args.csv:
        ep = load_episode(args.demo_json, args.index)
        m = compute_metrics(load_trajectory(args.csv), ep, args.success_radius)
        m["csv"] = args.csv
        m["index"] = args.index
        results.append(m)
    elif args.csv_dir:
        idx_map = {}
        if args.index_map:
            idx_map = json.load(open(args.index_map))
        for fn in sorted(os.listdir(args.csv_dir)):
            if not fn.endswith(".csv"):
                continue
            idx = int(idx_map.get(fn, 0))
            ep = load_episode(args.demo_json, idx)
            traj = load_trajectory(os.path.join(args.csv_dir, fn))
            if not traj:
                # empty trajectory (episode produced no steps, e.g. sim failed to spawn)
                print(f"[skip] empty trajectory: {fn}")
                continue
            m = compute_metrics(traj, ep, args.success_radius)
            m["csv"] = fn
            m["index"] = idx
            results.append(m)
    else:
        ap.error("provide --csv or --csv-dir")

    # aggregate
    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n_episodes": len(results),
        "SR": avg("SR"), "OSR": avg("OSR"), "SPL": avg("SPL"),
        "nDTW": avg("nDTW"), "GoalDist": avg("GoalDist"),
        "per_episode": results,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "per_episode"}, indent=2))
    for r in results:
        nd = f"{r['nDTW']:.3f}" if r["nDTW"] is not None else "n/a"
        print(f"  {os.path.basename(r['csv'])}: SR={r['SR']:.0f} OSR={r['OSR']:.0f} "
              f"SPL={r['SPL']:.3f} nDTW={nd} GoalDist={r['GoalDist']:.2f}m "
              f"path={r['path_length']:.2f}m")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
