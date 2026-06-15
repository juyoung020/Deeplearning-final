"""Week 4 IsaacSim commander: QwenVLNActionCommander.

Mirrors the existing Week 2/3 commander interface so it can be dropped into the
``week3/go2_physics_teleop.py`` main loop:

    advance() -> np.ndarray[3]   # [vx, vy, wz], called every control step
    reset()
    status_text() -> str
    .mission_complete : bool
    .quit_requested   : bool

Key behaviors (per Week 4 spec / repo notes):
  * Qwen inference runs only at ACTION-DECISION boundaries, not every physics
    substep. During an action's duration the same velocity command is held
    (``_remaining_steps`` pattern).
  * The image-history buffer is updated at decision time only, kept oldest ->
    newest. ``[I_{t-1}, I_t]`` with ``[I_0, I_0]`` when there is no history.
  * ``Stop`` -> ``mission_complete = True`` (episode terminate).
  * Invalid model output -> ``Move forward 25cm`` fallback in sim, but counted in
    ``invalid_output_count`` / ``invalid_output_rate``.
  * Closed-loop trajectory-aware text uses ONLY the robot's own odometry and the
    PREVIOUSLY executed actions (never GT future / target / goal info).

This module is sim-agnostic: it receives a ``frame_provider`` (returns the
current egocentric RGB as a PIL.Image) and an optional ``odometry_provider``
(returns ``(x, y, yaw)``). The Isaac-specific capture lives in the demo wiring.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Callable, List, Optional, Tuple

import numpy as np

from week4_actions import INVALID_FALLBACK_ACTION
from week4_actions import STOP as STOP_ACTION


class QwenVLNActionCommander:
    def __init__(
        self,
        predictor,                         # Week4ActionPredictor (has .predict)
        instruction: str,
        control_dt: float,
        frame_provider: Callable[[], object],
        odometry_provider: Optional[Callable[[], Tuple[float, float, float]]] = None,
        history_count: int = 1,
        history_stride: int = 1,
        max_image_count: int = 2,
        include_trajectory_text: bool = False,
        trajectory_history_len: int = 4,
        invalid_fallback: str = INVALID_FALLBACK_ACTION,
        max_steps: int = 0,
        image_size: int = 256,
        verbose: bool = True,
    ):
        self.predictor = predictor
        self.instruction = instruction
        self.control_dt = max(float(control_dt or 0.02), 1e-6)
        self.frame_provider = frame_provider
        self.odometry_provider = odometry_provider
        self.history_count = int(history_count)
        self.history_stride = int(history_stride)
        self.max_image_count = int(max_image_count)
        self.include_trajectory_text = bool(include_trajectory_text)
        self.trajectory_history_len = int(trajectory_history_len)
        self.invalid_fallback = invalid_fallback
        self.max_steps = int(max_steps)
        self.image_size = int(image_size)
        self.verbose = verbose

        self.mission_complete = False
        self.quit_requested = False

        self._command = np.zeros(3, dtype=np.float32)
        self._remaining_steps = 0
        self._active_label = "stop"

        # decision-time buffers
        self._frame_buffer: deque = deque(maxlen=64)
        self._executed_actions: List[str] = []
        self._decision_poses: List[Tuple[float, float, float]] = []

        # stats
        self.decision_count = 0
        self.invalid_output_count = 0
        self.total_latency_s = 0.0

    # ------------------------------------------------------------------
    def reset(self):
        self._command[:] = 0.0
        self._remaining_steps = 0
        self._active_label = "stop"
        self.mission_complete = False
        self.quit_requested = False
        self._frame_buffer.clear()
        self._executed_actions.clear()
        self._decision_poses.clear()
        self.decision_count = 0
        self.invalid_output_count = 0
        self.total_latency_s = 0.0

    # ------------------------------------------------------------------
    @property
    def invalid_output_rate(self) -> float:
        return self.invalid_output_count / max(self.decision_count, 1)

    # ------------------------------------------------------------------
    def _capture_frame(self):
        from PIL import Image

        img = self.frame_provider()
        if img is None:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img)).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size))
        return img

    def _select_history(self) -> List[object]:
        """oldest -> newest history images from the decision-frame buffer."""
        buf = list(self._frame_buffer)  # oldest..newest, last == current
        total = min(self.history_count + 1, self.max_image_count)
        cur = len(buf) - 1
        raw = [cur - i * self.history_stride for i in range(total)]
        raw = list(reversed(raw))
        idxs = [i if i >= 0 else 0 for i in raw]
        return [buf[i] for i in idxs]

    def _build_trajectory_text(self) -> Optional[str]:
        if not self.include_trajectory_text:
            return None
        # previous executed actions (already exclude the not-yet-decided action)
        prev_actions = self._executed_actions[-self.trajectory_history_len:]
        lines = []
        poses = self._decision_poses[-(self.trajectory_history_len + 1):]
        if len(poses) >= 2:
            x_now, y_now, yaw_now = poses[-1]
            c, s = math.cos(-yaw_now), math.sin(-yaw_now)
            for k in range(1, len(poses)):
                x0, y0, _ = poses[k - 1]
                x1, y1, yaw1 = poses[k]
                dx_w, dy_w = x1 - x0, y1 - y0
                dx = c * dx_w - s * dy_w
                dy = s * dx_w + c * dy_w
                dyaw = math.degrees(yaw1 - poses[k - 1][2])
                rel = k - len(poses)
                lines.append(f"  step {rel}: dx={dx:+.2f}m, dy={dy:+.2f}m, dyaw={dyaw:+.0f}deg")
        if not prev_actions and not lines:
            return None
        parts = ["Recent history:"]
        if prev_actions:
            parts.append("- previous actions: " + ", ".join(prev_actions))
        if lines:
            parts.append("- relative robot path over last steps (current frame):")
            parts.extend(lines)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    def advance(self) -> np.ndarray:
        # Hold the current command during its duration.
        if self._remaining_steps > 0:
            return self._consume_active_command()

        # ---- new decision ----
        frame = self._capture_frame()
        self._frame_buffer.append(frame)
        if self.odometry_provider is not None:
            try:
                self._decision_poses.append(tuple(self.odometry_provider()))
            except Exception:
                pass

        images = self._select_history()
        traj_text = self._build_trajectory_text()
        raw, converted = self.predictor.predict(self.instruction, images, traj_text)

        self.decision_count += 1
        self.total_latency_s += float(getattr(self.predictor, "last_latency_s", 0.0))
        if not converted.is_valid:
            self.invalid_output_count += 1

        if self.verbose:
            print(
                f"[week4-vln] decision {self.decision_count}: raw={raw!r} -> "
                f"{converted.action_text} (valid={converted.is_valid}) "
                f"latency={getattr(self.predictor,'last_latency_s',0.0)*1000:.0f}ms"
            )

        # record executed action AFTER deciding (so it's never in this step's input)
        self._executed_actions.append(converted.action_text)

        if converted.terminate or converted.action_text == STOP_ACTION:
            self._command[:] = 0.0
            self._remaining_steps = 0
            self._active_label = "stop"
            self.mission_complete = True
            if self.verbose:
                print("[week4-vln] Stop predicted; episode marked complete.")
            return self._command.copy()

        self._command = np.asarray(converted.velocity, dtype=np.float32)
        self._remaining_steps = max(int(math.ceil(converted.duration_s / self.control_dt)), 1)
        self._active_label = converted.action_text

        if self.max_steps > 0 and self.decision_count >= self.max_steps:
            # safety cap: let the loop finish this command then stop
            pass
        return self._consume_active_command()

    def _consume_active_command(self) -> np.ndarray:
        command = self._command.copy()
        self._remaining_steps -= 1
        if self._remaining_steps <= 0:
            self._command[:] = 0.0
            self._active_label = "between-actions"
            if self.max_steps > 0 and self.decision_count >= self.max_steps:
                self.mission_complete = True
                if self.verbose:
                    print(f"[week4-vln] max_steps={self.max_steps} reached; stopping.")
        return command

    def status_text(self) -> str:
        base = (
            f" vln='{self._active_label}' decisions={self.decision_count}"
            f" invalid_rate={self.invalid_output_rate:.2f}"
        )
        if self._remaining_steps > 0:
            base += f" remaining={self._remaining_steps * self.control_dt:.2f}s"
        return base


__all__ = ["QwenVLNActionCommander"]
