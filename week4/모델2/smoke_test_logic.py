"""Model-free smoke test for the Week 4 closed-loop logic (spec-aligned).

Validates actions.parse_action and QwenVLNActionCommander against the official
action set / velocity mapping using a mock predictor and mock providers — no VLM
download required. Run:

    python smoke_test_logic.py
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from actions import (
    ACTION_DURATION_S,
    CANONICAL_ACTIONS,
    FORWARD_VEL_MPS,
    YAW_VEL_RAD_S,
    parse_action,
)
from vln_commander import QwenVLNActionCommander

FWD = [FORWARD_VEL_MPS, 0.0, 0.0]
LEFT = [0.0, 0.0, YAW_VEL_RAD_S]
RIGHT = [0.0, 0.0, -YAW_VEL_RAD_S]
ZERO = [0.0, 0.0, 0.0]


def test_parse():
    # (text, kind, command, duration, valid_exact)
    cases = [
        ("Move forward 25cm", "motion", FWD, ACTION_DURATION_S, True),
        ("Turn left 15 degree", "motion", LEFT, ACTION_DURATION_S, True),
        ("Turn right 15 degree", "motion", RIGHT, ACTION_DURATION_S, True),
        ("Turn left 15 degrees", "motion", LEFT, ACTION_DURATION_S, True),   # plural normalized
        ("turn right 15°", "motion", RIGHT, ACTION_DURATION_S, True),         # glyph normalized
        ("Stop", "stop", ZERO, 0.0, True),
        ("Move forward 50cm", "invalid", ZERO, 0.0, False),                  # not in 4-action set
        ("Turn left 30 degree", "invalid", ZERO, 0.0, False),                # not in 4-action set
        ("banana", "invalid", ZERO, 0.0, False),
        ("The next action is: Turn right 15 degree.", "motion", RIGHT, ACTION_DURATION_S, False),  # wrapped
    ]
    for text, kind, cmd, dur, valid in cases:
        p = parse_action(text)
        assert p.kind == kind, f"{text!r}: kind {p.kind} != {kind}"
        assert np.allclose(p.command, cmd, atol=1e-5), f"{text!r}: cmd {p.command} != {cmd}"
        assert abs(p.duration_s - dur) < 1e-5, f"{text!r}: dur {p.duration_s} != {dur}"
        assert p.valid == valid, f"{text!r}: valid {p.valid} != {valid}"
    assert CANONICAL_ACTIONS == ["Move forward 25cm", "Turn left 15 degree",
                                 "Turn right 15 degree", "Stop"]
    print(f"[OK] parse_action: {len(cases)} cases + 4-action vocab")


class MockPredictor:
    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.last_was_invalid = False
        self.seen_images = []

    def predict_action(self, images, instruction, trajectory_text=""):
        self.seen_images.append(len(images))
        action = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        self.last_was_invalid = False
        return action


def test_commander_closed_loop():
    control_dt = 0.05  # 20 Hz -> 0.25 s = 5 control steps per action
    frames = {"n": 0}

    def frame_provider():
        frames["n"] += 1
        return Image.new("RGB", (256, 256), (frames["n"] % 255, 0, 0))

    pose = {"x": 0.0, "y": 0.0, "yaw": 0.0}

    def odometry_provider():
        return pose["x"], pose["y"], pose["yaw"]

    script = ["Move forward 25cm", "Turn left 15 degree", "Move forward 25cm", "Stop"]
    predictor = MockPredictor(script)
    cmd = QwenVLNActionCommander(
        predictor=predictor,
        instruction="Walk to the kitchen and stop.",
        control_dt=control_dt,
        frame_provider=frame_provider,
        odometry_provider=odometry_provider,
        history_count=1,
        history_stride=1,
        max_image_count=2,
        include_trajectory_text=False,
        max_steps=0,
        image_size=256,
    )

    seq = []
    safety = 0
    while not cmd.mission_complete and not cmd.quit_requested and safety < 1000:
        seq.append(cmd.advance().copy())
        safety += 1

    assert cmd.mission_complete, "commander should complete on Stop"
    assert predictor.i == 4, f"decisions {predictor.i} != 4"
    # 5 + 5 + 5 motion steps, then 1 zero-velocity Stop step.
    assert len(seq) == 16, f"control steps {len(seq)} != 16"
    assert np.allclose(seq[0], FWD, atol=1e-5) and np.allclose(seq[4], FWD, atol=1e-5)
    assert np.allclose(seq[5], LEFT, atol=1e-5) and np.allclose(seq[9], LEFT, atol=1e-5)
    assert np.allclose(seq[10], FWD, atol=1e-5) and np.allclose(seq[14], FWD, atol=1e-5)
    assert np.allclose(seq[15], ZERO, atol=1e-5)
    # Spec image rule: every decision sees exactly 2 images; t=0 -> [I_0, I_0].
    assert predictor.seen_images == [2, 2, 2, 2], f"images per decision {predictor.seen_images}"
    print(f"[OK] commander closed loop: {len(seq)} steps, {predictor.i} decisions, "
          f"images_per_decision={predictor.seen_images}")


def test_max_steps_forces_stop():
    def frame_provider():
        return Image.new("RGB", (64, 64), (0, 0, 0))

    predictor = MockPredictor(["Move forward 25cm"])  # never says Stop
    cmd = QwenVLNActionCommander(
        predictor=predictor,
        instruction="go",
        control_dt=0.1,
        frame_provider=frame_provider,
        odometry_provider=None,
        max_steps=3,
        image_size=64,
    )
    safety = 0
    while not cmd.mission_complete and safety < 1000:
        cmd.advance()
        safety += 1
    assert cmd.mission_complete, "max_steps should force mission_complete"
    assert predictor.i <= 3, f"made {predictor.i} decisions, expected <=3"
    print(f"[OK] max_steps forces stop after {predictor.i} decisions")


if __name__ == "__main__":
    test_parse()
    test_commander_closed_loop()
    test_max_steps_forces_stop()
    print("\nALL WEEK4 LOGIC SMOKE TESTS PASSED (spec-aligned)")
