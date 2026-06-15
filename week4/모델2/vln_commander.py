"""Week 4 — closed-loop VLN commander.

``QwenVLNActionCommander`` is a drop-in replacement for the Week 2/3 language
commanders in ``go2_physics_teleop.py``. The simulator loop drives it through the
exact same surface:

    commander.quit_requested        # bool — abort the run
    commander.mission_complete      # bool — episode finished successfully
    commander.reset()               # re-arm at episode start
    cmd = commander.advance()       # -> np.float32[3] = [vx, vy, yaw_rate], one per control step
    commander.status_text()         # short HUD string

Closed loop, per *decision*:
    1. capture the current egocentric RGB frame (frame_provider),
    2. assemble history frames + current frame (bounded by max_image_count),
    3. optionally append leakage-safe trajectory text (own odometry + executed
       actions only — never the goal/ground-truth path),
    4. ask the VLM for the next discrete action,
    5. convert that action to a velocity command held for ceil(duration/dt) steps,
    6. when the action is Stop (or max_steps is hit), end the episode.

Between decisions ``advance()`` simply replays the active velocity command for
the remaining hold steps, so VLM inference happens once per action rather than
once per physics step.
"""

from __future__ import annotations

import math

import numpy as np

from actions import parse_action


class QwenVLNActionCommander:
    def __init__(
        self,
        predictor,
        instruction: str,
        control_dt: float,
        frame_provider,
        odometry_provider=None,
        history_count: int = 1,
        history_stride: int = 1,
        max_image_count: int = 2,
        include_trajectory_text: bool = False,
        invalid_fallback: str = "Move forward 25cm",
        max_steps: int = 0,
        image_size: int = 256,
        async_inference: bool = False,
    ):
        self.predictor = predictor
        self.instruction = str(instruction or "")
        self.control_dt = max(float(control_dt or 0.02), 1e-6)
        self.frame_provider = frame_provider
        self.odometry_provider = odometry_provider
        self.history_count = max(int(history_count), 0)
        self.history_stride = max(int(history_stride), 1)
        self.max_image_count = max(int(max_image_count), 1)
        self.include_trajectory_text = bool(include_trajectory_text)
        self.invalid_fallback = invalid_fallback
        self.max_steps = max(int(max_steps), 0)
        self.image_size = int(image_size)
        self.async_inference = bool(async_inference)

        # Runtime state.
        self._command = np.zeros(3, dtype=np.float32)
        self._pending_thread = None       # async: in-flight inference worker thread
        self._pending_result = None       # async: action_text produced by the worker
        self._pending_current_frame = None  # async: frame captured for the in-flight decision
        self._bridge_velocity = np.zeros(3, dtype=np.float32)  # async: keep moving while VLM thinks
        self._remaining_steps = 0
        self._active_label = "stop"
        self.mission_complete = False
        self.quit_requested = False
        self._decisions = 0
        self._invalid_count = 0
        self._frame_history = []   # list of PIL.Image, oldest first
        self._action_log = []      # list of (label, x, y, yaw) at decision time

        print(
            f"[INFO][week4] VLN commander armed. history={self.history_count}x"
            f"(stride {self.history_stride}), max_images={self.max_image_count}, "
            f"trajectory_text={self.include_trajectory_text}, max_steps={self.max_steps or 'inf'}, "
            f"control_dt={self.control_dt:.4f}s"
        )

    # ------------------------------------------------------------------ helpers
    def reset(self):
        self._command[:] = 0.0
        self._remaining_steps = 0
        self._active_label = "stop"
        self.mission_complete = False
        self.quit_requested = False
        self._decisions = 0
        self._invalid_count = 0
        self._frame_history = []
        self._action_log = []
        self._pending_thread = None
        self._pending_result = None
        self._pending_current_frame = None
        self._bridge_velocity[:] = 0.0

    def _capture_frame(self):
        img = self.frame_provider()
        if self.image_size > 0 and hasattr(img, "size") and img.size != (self.image_size, self.image_size):
            try:
                img = img.resize((self.image_size, self.image_size))
            except Exception:
                pass
        return img

    def _select_images(self, current):
        """Build [I_{t-k}, ..., I_{t-1}, I_t], oldest first, current last.

        Per the spec: baseline is [I_{t-1}, I_t]; when there is not enough history
        (e.g. t=0) the earliest available frame is repeated to keep the image
        count and ordering fixed (t=0 -> [I_0, I_0]).
        """
        desired = min(self.history_count + 1, self.max_image_count)
        images = []
        if self.history_count > 0 and self._frame_history:
            sampled = self._frame_history[:: self.history_stride]
            images = list(sampled[-self.history_count:])
        images = images + [current]
        if len(images) > self.max_image_count:
            images = images[-self.max_image_count:]
        # Left-pad with the oldest available frame so the count is always `desired`.
        while len(images) < desired:
            images.insert(0, images[0])
        return images

    def _odometry(self):
        if self.odometry_provider is None:
            return None
        try:
            x, y, yaw = self.odometry_provider()
            return float(x), float(y), float(yaw)
        except Exception:
            return None

    def _trajectory_text(self):
        """Leakage-safe recent history: own odometry + executed actions only."""
        if not self.include_trajectory_text:
            return ""
        lines = []
        odo = self._odometry()
        if odo is not None:
            lines.append(
                f"Current pose: x={odo[0]:.2f} m, y={odo[1]:.2f} m, heading={math.degrees(odo[2]):.0f} deg."
            )
        if self._action_log:
            recent = self._action_log[-5:]
            hist = ", ".join(lbl for (lbl, *_rest) in recent)
            lines.append(f"Recent actions: {hist}.")
        lines.append(f"Decisions so far: {self._decisions}.")
        return "\n".join(lines)

    # -------------------------------------------------------------------- drive
    def advance(self):
        # Input-wait / idle: stand still until a mission is provided.
        if not (self.instruction or "").strip():
            self._remaining_steps = 0
            return np.zeros(3, dtype=np.float32)
        if self._remaining_steps > 0:
            return self._consume_active_command()
        if self.async_inference:
            return self._advance_async()

        # ---- synchronous path (blocks the sim ~1 inference each decision) ----
        if self.max_steps and self._decisions >= self.max_steps:
            if not self.mission_complete:
                print(f"[INFO][week4] max_steps={self.max_steps} reached; forcing Stop.")
            self.mission_complete = True
            return np.zeros(3, dtype=np.float32)

        current = self._capture_frame()
        images = self._select_images(current)
        trajectory_text = self._trajectory_text()

        try:
            action_text = self.predictor.predict_action(images, self.instruction, trajectory_text)
        except Exception as exc:
            print(f"[WARN][week4] VLM inference failed ({exc}); using fallback {self.invalid_fallback!r}.")
            action_text = self.invalid_fallback
            self._invalid_count += 1

        return self._finalize_decision(action_text, current)

    def _advance_async(self):
        """Non-blocking inference: keep moving on the last velocity while the VLM thinks in a
        worker thread, so the GUI never freezes between decisions (smooth motion)."""
        import threading
        # An inference is in flight -> keep the robot moving until it returns.
        if self._pending_thread is not None:
            if self._pending_thread.is_alive():
                return self._bridge_velocity.copy()
            self._pending_thread = None
            action_text = self._pending_result
            self._pending_result = None
            if action_text is None:
                action_text = self.invalid_fallback
            return self._finalize_decision(action_text, self._pending_current_frame)

        if self.max_steps and self._decisions >= self.max_steps:
            if not self.mission_complete:
                print(f"[INFO][week4] max_steps={self.max_steps} reached; forcing Stop.")
            self.mission_complete = True
            return np.zeros(3, dtype=np.float32)

        # Start a new inference worker on the current frame; bridge-move meanwhile.
        current = self._capture_frame()
        images = self._select_images(current)
        trajectory_text = self._trajectory_text()
        self._pending_current_frame = current

        def _worker(imgs=images, instr=self.instruction, traj=trajectory_text):
            try:
                self._pending_result = self.predictor.predict_action(imgs, instr, traj)
            except Exception as exc:
                print(f"[WARN][week4] async VLM inference failed ({exc}); fallback {self.invalid_fallback!r}.")
                self._pending_result = self.invalid_fallback

        self._pending_thread = threading.Thread(target=_worker, daemon=True)
        self._pending_thread.start()
        return self._bridge_velocity.copy()

    def _finalize_decision(self, action_text, current):
        if getattr(self.predictor, "last_was_invalid", False):
            self._invalid_count += 1

        # Record frame AFTER selecting images for this decision (becomes history next time).
        if current is not None:
            self._frame_history.append(current)
            max_hist = max(self.history_count * self.history_stride + 1, 1)
            if len(self._frame_history) > max_hist:
                self._frame_history = self._frame_history[-max_hist:]

        self._decisions += 1
        parsed = parse_action(action_text)
        odo = self._odometry()
        self._action_log.append((parsed.label, *(odo or (float("nan"),) * 3)))

        if parsed.kind == "stop":
            self._active_label = "Stop"
            self.mission_complete = True
            self._bridge_velocity[:] = 0.0
            print(f"[INFO][week4] decision {self._decisions}: Stop - mission complete.")
            return np.zeros(3, dtype=np.float32)

        if parsed.kind == "invalid":
            # Fall back deterministically; still advance so the robot is not frozen.
            parsed = parse_action(self.invalid_fallback)

        self._command = parsed.command.astype(np.float32, copy=True)
        self._remaining_steps = max(int(math.ceil(parsed.duration_s / self.control_dt)), 1)
        self._active_label = parsed.label
        self._bridge_velocity = self._command.copy()  # keep this velocity while next decision computes
        print(
            f"[INFO][week4] decision {self._decisions}: {parsed.label} "
            f"(hold {parsed.duration_s:.2f}s = {self._remaining_steps} steps)"
        )
        return self._consume_active_command()

    def _consume_active_command(self):
        command = self._command.copy()
        self._remaining_steps -= 1
        if self._remaining_steps == 0:
            self._command[:] = 0.0
            self._active_label = "stop"
        return command

    def status_text(self):
        invalid = f" invalid={self._invalid_count}" if self._invalid_count else ""
        if self._remaining_steps <= 0:
            return f" week4 decisions={self._decisions}{invalid}"
        return (
            f" week4='{self._active_label}' "
            f"remaining={self._remaining_steps * self.control_dt:.2f}s "
            f"decisions={self._decisions}{invalid}"
        )
