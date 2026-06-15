"""Real-model smoke test: load Qwen3-VL and run the Week 4 predictor + commander.

Validates that the VLM loads on the GPU and that predict_action returns a
canonical action for a synthetic egocentric frame. Run:

    set HF_HUB_DISABLE_SYMLINKS=1
    python smoke_test_vlm.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from PIL import Image, ImageDraw

from actions import CANONICAL_ACTIONS, parse_action
from infer_week4_action import Week4ActionPredictor
from vln_commander import QwenVLNActionCommander


def make_frame(kind: str) -> Image.Image:
    """A crude synthetic egocentric frame: a 'door' or 'wall' the model can see."""
    img = Image.new("RGB", (256, 256), (180, 180, 180))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 180, 256, 256], fill=(120, 90, 60))  # floor
    if kind == "door_ahead":
        d.rectangle([100, 70, 156, 200], fill=(60, 40, 20))   # dark doorway center
    elif kind == "door_right":
        d.rectangle([200, 70, 250, 200], fill=(60, 40, 20))   # doorway on the right
    elif kind == "wall":
        d.rectangle([0, 60, 256, 200], fill=(150, 150, 150))  # blank wall
    return img


def main():
    print("=== Loading Qwen3-VL-2B-Instruct (base, zero-shot) ===")
    predictor = Week4ActionPredictor(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        adapter_dir=None,
        device="auto",
        dtype="bf16",
    )

    print("\n=== Single-frame predictions ===")
    for kind, instruction in [
        ("door_ahead", "Walk through the doorway in front of you."),
        ("door_right", "Go to the door on your right."),
        ("wall", "Find the exit."),
    ]:
        action = predictor.predict_action([make_frame(kind)], instruction)
        valid = parse_action(action).valid
        print(f"  frame={kind:11s} instr={instruction!r}")
        print(f"    -> action={action!r}  valid={valid}  in_vocab={action in CANONICAL_ACTIONS}")
        assert valid, f"predictor returned unparseable action: {action!r}"

    print("\n=== Closed-loop with the real VLM (mock sim frames, 5 decisions max) ===")
    state = {"n": 0}

    def frame_provider():
        state["n"] += 1
        # alternate views so history has signal
        return make_frame("door_ahead" if state["n"] % 2 else "door_right")

    def odometry_provider():
        return 0.1 * state["n"], 0.0, 0.0

    commander = QwenVLNActionCommander(
        predictor=predictor,
        instruction="Walk to the doorway ahead and stop there.",
        control_dt=0.05,
        frame_provider=frame_provider,
        odometry_provider=odometry_provider,
        history_count=1,
        max_image_count=2,
        include_trajectory_text=True,
        max_steps=5,
        image_size=256,
    )

    decisions = 0
    safety = 0
    while not commander.mission_complete and not commander.quit_requested and safety < 2000:
        commander.advance()
        safety += 1
        # count unique decisions via status
        decisions = commander._decisions
    print(f"  closed loop ended: decisions={commander._decisions} "
          f"mission_complete={commander.mission_complete} steps={safety}")
    print("\nVLM SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
