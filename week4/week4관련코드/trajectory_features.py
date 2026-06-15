"""Leakage-safe trajectory features for the Week 4 trajectory-aware ablation.

Baseline input = instruction + image history.
Ablation input  = instruction + image history + recent trajectory/action history.

STRICT anti-leakage rules (enforced here):
  * The target ``observation.action[t]`` is NEVER included. Only actions strictly
    before ``t`` (``action[t-k .. t-1]``) are used as "previous executed actions".
  * No future pose/action, no goal distance, no success label.
  * Absolute world coordinates are converted to displacements expressed in the
    CURRENT robot frame (at timestep ``t``), so the model cannot memorize a scan
    from absolute coordinates.

In closed-loop inference the same text must be built from the robot's own
odometry and previously executed/predicted actions, NOT from GT future data
(see ``vln_commander.py``).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from week4_actions import action_id_to_text


def _to_current_frame(dx_w: float, dy_w: float, yaw_now: float) -> Tuple[float, float]:
    """Rotate a world-frame displacement into the current robot frame (yaw_now)."""
    c, s = math.cos(-yaw_now), math.sin(-yaw_now)
    return c * dx_w - s * dy_w, s * dx_w + c * dy_w


def build_relative_path_lines(
    robot_xy: Sequence[Tuple[float, float]],
    robot_yaw: Sequence[float],
    t: int,
    history_len: int,
) -> List[str]:
    """Return human-readable relative-displacement lines for steps before ``t``.

    Each line describes the step from ``t-k-1`` to ``t-k`` as a displacement in
    the current robot frame plus the yaw change. Only indices ``<= t`` are read;
    nothing from the future is used.
    """
    if not robot_xy or not robot_yaw or t <= 0:
        return []
    yaw_now = robot_yaw[t] if t < len(robot_yaw) else robot_yaw[-1]
    lines: List[str] = []
    start = max(1, t - history_len + 1)
    for k in range(start, t + 1):
        if k >= len(robot_xy) or k - 1 < 0:
            continue
        dx_w = robot_xy[k][0] - robot_xy[k - 1][0]
        dy_w = robot_xy[k][1] - robot_xy[k - 1][1]
        dx, dy = _to_current_frame(dx_w, dy_w, yaw_now)
        dyaw = math.degrees(robot_yaw[k] - robot_yaw[k - 1]) if k < len(robot_yaw) else 0.0
        rel = k - 1 - t  # negative step offset
        lines.append(f"  step {rel}: dx={dx:+.2f}m, dy={dy:+.2f}m, dyaw={dyaw:+.0f}deg")
    return lines


def build_previous_actions(action_ids: Sequence[int], t: int, history_len: int) -> List[str]:
    """Canonical strings for actions executed strictly BEFORE ``t`` (no target)."""
    if t <= 0:
        return []
    start = max(0, t - history_len)
    return [action_id_to_text(action_ids[k]) for k in range(start, t)]


def build_trajectory_text(ep, t: int, history_len: int = 4) -> Optional[str]:
    """Build the leakage-safe "Recent history" text block, or ``None``.

    ``ep`` is a ``vlnverse_dataset.EpisodeRef`` (or any object exposing
    ``action_ids``, ``robot_xy``, ``robot_yaw``).
    """
    action_ids = getattr(ep, "action_ids", None) or []
    robot_xy = getattr(ep, "robot_xy", None)
    robot_yaw = getattr(ep, "robot_yaw", None)

    prev_actions = build_previous_actions(action_ids, t, history_len)
    path_lines = (
        build_relative_path_lines(robot_xy, robot_yaw, t, history_len)
        if robot_xy and robot_yaw
        else []
    )
    if not prev_actions and not path_lines:
        return None

    parts = ["Recent history:"]
    if prev_actions:
        parts.append("- previous actions: " + ", ".join(prev_actions))
    if path_lines:
        parts.append("- relative robot path over last steps (current frame):")
        parts.extend(path_lines)
    return "\n".join(parts)


__all__ = [
    "build_trajectory_text",
    "build_relative_path_lines",
    "build_previous_actions",
]
