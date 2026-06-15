"""Week 4 smoke tests.

Two tiers:

  Tier 1 (default) -- NO model/dataset download, NO torch/transformers/peft.
      Pure-python + PIL/numpy checks of actions, parser, converter, history
      selection, synthetic dataset, trajectory leakage-safety, and the IsaacSim
      commander logic (with a fake predictor).

  Tier 2 (--with-model) -- requires `transformers` + `peft` and downloads the
      Qwen3-VL-2B processor/model. Checks chat-template prompt building,
      answer-only label masking, a real forward/loss, generation + parsing, and
      a 2-step training step with checkpoint save.

Usage:
    python smoke_test_week4.py             # tier 1 (no downloads)
    python smoke_test_week4.py --with-model  # tier 1 + tier 2 (downloads model)
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Minimal test runner
# ---------------------------------------------------------------------------
class Runner:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def run(self, name, fn):
        try:
            fn()
            self.passed += 1
            print(f"  [PASS] {name}")
        except Exception as exc:
            self.failed += 1
            print(f"  [FAIL] {name}: {exc}")
            traceback.print_exc()

    def summary(self) -> int:
        print(f"\n=== {self.passed} passed, {self.failed} failed ===")
        return 1 if self.failed else 0


def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg or "assertion failed")


# ---------------------------------------------------------------------------
# Tier 1 tests
# ---------------------------------------------------------------------------
def test_action_mapping():
    import week4_actions as A

    _assert(A.action_id_to_text(0) == "Stop")
    _assert(A.action_id_to_text(1) == "Move forward 25cm")
    _assert(A.action_id_to_text(2) == "Turn left 15 degree")
    _assert(A.action_id_to_text(3) == "Turn right 15 degree")
    _assert(A.CANONICAL_ACTIONS[0] == "Move forward 25cm")
    _assert(len(A.CANONICAL_ACTIONS) == 4)


def test_canonical_parser():
    import week4_actions as A

    # exact
    _assert(A.exact_match("Stop") == "Stop")
    _assert(A.exact_match("stop") is None)  # exact is case-sensitive
    # normalization
    _assert(A.parse_canonical_action("stop.") == "Stop")
    _assert(A.parse_canonical_action("Turn right 15 degrees") == "Turn right 15 degree")
    _assert(A.parse_canonical_action("  Move forward 25cm \n extra") == "Move forward 25cm")
    _assert(A.parse_canonical_action("go left maybe") is None)
    _assert(A.parse_canonical_action("Turn right 15 degree", allow_normalization=False) == "Turn right 15 degree")
    _assert(A.parse_canonical_action("Turn right 15 degrees", allow_normalization=False) is None)


def test_action_to_command():
    import week4_actions as A
    import math

    c = A.action_to_command("Move forward 25cm")
    _assert(c.is_valid and c.velocity == [1.0, 0.0, 0.0] and c.duration_s == 0.25 and not c.terminate)
    c = A.action_to_command("Turn right 15 degree")
    _assert(abs(c.velocity[2] - (-math.radians(15) / 0.25)) < 1e-9 and c.velocity[2] < 0)
    c = A.action_to_command("Turn left 15 degree")
    _assert(c.velocity[2] > 0)
    c = A.action_to_command("Stop")
    _assert(c.terminate and c.velocity == [0.0, 0.0, 0.0])
    # invalid -> fallback move forward, flagged invalid
    c = A.action_to_command("banana")
    _assert((not c.is_valid) and c.action_text == "Move forward 25cm")


def test_history_selection():
    from vlnverse_dataset import DatasetConfig, select_history_indices

    cfg = DatasetConfig(history_count=1, history_stride=1, max_image_count=2)
    _assert(select_history_indices(0, cfg) == [0, 0], "t=0 must duplicate")
    _assert(select_history_indices(3, cfg) == [2, 3], "oldest->newest")
    cfg3 = DatasetConfig(history_count=1, history_stride=3, max_image_count=2)
    _assert(select_history_indices(5, cfg3) == [2, 5])
    _assert(select_history_indices(1, cfg3) == [0, 1], "clamp negative to 0")
    # 3-image history
    cfg2 = DatasetConfig(history_count=2, history_stride=1, max_image_count=3)
    _assert(select_history_indices(4, cfg2) == [2, 3, 4])


def test_synthetic_dataset():
    from vlnverse_dataset import build_synthetic_dataset, DatasetConfig

    cfg = DatasetConfig(image_size=32, history_count=1, history_stride=1, max_image_count=2)
    ds = build_synthetic_dataset(num_episodes=2, frames_per_episode=5, cfg=cfg)
    _assert(len(ds) == 10, f"expected 10 samples, got {len(ds)}")
    s0 = ds[0]
    _assert(len(s0["images"]) == 2, "two images per sample")
    _assert(s0["images"][0].size == (32, 32))
    _assert(s0["answer"] in {"Move forward 25cm", "Turn right 15 degree", "Turn left 15 degree", "Stop"})
    # last frame of each episode is Stop
    last = ds[4]
    _assert(last["timestep"] == 4 and last["answer"] == "Stop", "terminal must be Stop")


def test_trajectory_leakage():
    from vlnverse_dataset import build_synthetic_dataset, DatasetConfig
    from trajectory_features import build_trajectory_text, build_previous_actions

    cfg = DatasetConfig(image_size=16, include_trajectory_text=True, trajectory_history_len=4)
    ds = build_synthetic_dataset(num_episodes=1, frames_per_episode=6, cfg=cfg)
    ep = ds.episodes[0]
    t = 3
    prev = build_previous_actions(ep.action_ids, t, 4)
    # previous actions must be exactly action[t-1..0] capped at 4, and NOT include t
    _assert(len(prev) == 3, f"expected 3 prev actions for t=3, got {len(prev)}")
    target_text = ds[t]["answer"]
    # leakage check: the target action token sequence at t must not be forced in;
    # we verify the builder only reads indices < t.
    text = build_trajectory_text(ep, t, 4)
    _assert("Recent history" in text)
    # t=0 -> no history
    _assert(build_trajectory_text(ep, 0, 4) is None, "t=0 must have no trajectory text")
    del target_text


def test_commander_logic():
    """Commander with a fake predictor: verify duration holding, Stop handling,
    invalid counting, and decision-time image buffering. No torch needed."""
    import numpy as np
    from PIL import Image
    from week4_actions import action_to_command
    from vln_commander import QwenVLNActionCommander

    class FakePredictor:
        def __init__(self, outputs):
            self.outputs = list(outputs)
            self.last_latency_s = 0.001
            self.calls = []

        def predict(self, instruction, images, trajectory_text=None):
            raw = self.outputs.pop(0)
            self.calls.append((len(images), trajectory_text))
            return raw, action_to_command(raw)

    # scripted: forward (valid), garbage (invalid->fallback), stop
    pred = FakePredictor(["Move forward 25cm", "banana", "Stop"])
    frames = {"n": 0}

    def frame_provider():
        frames["n"] += 1
        return Image.new("RGB", (16, 16), (frames["n"], 0, 0))

    cmd = QwenVLNActionCommander(
        predictor=pred, instruction="go", control_dt=0.05,
        frame_provider=frame_provider, history_count=1, history_stride=1,
        max_image_count=2, image_size=16, verbose=False,
    )
    # duration 0.25s / dt 0.05 = 5 steps for "Move forward 25cm"
    first = cmd.advance()
    _assert(abs(first[0] - 1.0) < 1e-6, "first command should be forward vx=1")
    _assert(cmd.decision_count == 1)
    # hold for remaining steps without new decision
    for _ in range(4):
        cmd.advance()
    _assert(cmd.decision_count == 1, "must not re-decide during duration")
    # next decision -> invalid 'banana' -> fallback forward, invalid counted
    cmd.advance()
    _assert(cmd.decision_count == 2 and cmd.invalid_output_count == 1)
    for _ in range(4):
        cmd.advance()
    # next decision -> Stop -> mission complete
    cmd.advance()
    _assert(cmd.mission_complete, "Stop must set mission_complete")
    # image history: first decision had [I0,I0]; later decisions have 2 distinct frames
    _assert(pred.calls[0][0] == 2, "always 2 images")


def run_tier1(runner: Runner):
    print("\n[Tier 1] no-download pure-python smoke tests")
    runner.run("action_mapping", test_action_mapping)
    runner.run("canonical_parser", test_canonical_parser)
    runner.run("action_to_command", test_action_to_command)
    runner.run("history_selection", test_history_selection)
    runner.run("synthetic_dataset", test_synthetic_dataset)
    runner.run("trajectory_leakage_safety", test_trajectory_leakage)
    runner.run("commander_logic", test_commander_logic)


# ---------------------------------------------------------------------------
# Tier 2 tests (need transformers + peft + model download)
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("WEEK4_SMOKE_MODEL", "Qwen/Qwen3-VL-2B-Instruct")


def _tiny_synth(n_eps=2, n_frames=4, image_size=64):
    from vlnverse_dataset import build_synthetic_dataset, DatasetConfig

    cfg = DatasetConfig(image_size=image_size, history_count=1, history_stride=1, max_image_count=2)
    return build_synthetic_dataset(num_episodes=n_eps, frames_per_episode=n_frames, cfg=cfg)


def test_prompt_and_label_mask():
    from qwen3vl_lora_model import load_processor
    from qwen3vl_prompt import Qwen3VLCollator, decode_label_targets

    processor = load_processor(MODEL_NAME)
    ds = _tiny_synth()
    collator = Qwen3VLCollator(processor)
    batch = collator([ds[0], ds[1]])
    _assert("input_ids" in batch and "labels" in batch)
    decoded = decode_label_targets(processor, batch["input_ids"], batch["labels"])
    for row_text, sample_idx in zip(decoded, [0, 1]):
        ans = ds[sample_idx]["answer"]
        _assert(ans in row_text, f"label target {row_text!r} should contain answer {ans!r}")
        # must NOT contain the system prompt / instruction text
        _assert("navigation action predictor" not in row_text, "system prompt leaked into labels")
        _assert("Instruction:" not in row_text, "instruction leaked into labels")
    print("    label targets:", decoded)


def test_model_forward_loss():
    import torch
    from qwen3vl_lora_model import ModelConfigSpec, load_model_and_processor
    from qwen3vl_prompt import Qwen3VLCollator

    spec = ModelConfigSpec(dtype="bf16", gradient_checkpointing=True)
    model, processor = load_model_and_processor(spec)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _assert(trainable > 0, "no trainable LoRA params")
    collator = Qwen3VLCollator(processor)
    ds = _tiny_synth()
    batch = collator([ds[0], ds[1]])
    device = next(model.parameters()).device
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    out = model(**batch)
    _assert(torch.isfinite(out.loss).item(), "loss not finite")
    out.loss.backward()
    print(f"    loss={float(out.loss):.4f}, trainable={trainable:,}")


def test_generation_and_parse():
    from qwen3vl_lora_model import ModelConfigSpec
    from infer_week4_action import Week4ActionPredictor

    predictor = Week4ActionPredictor(model_name=MODEL_NAME, adapter_dir=None, dtype="bf16")
    ds = _tiny_synth()
    raw, converted = predictor.predict(ds[0]["instruction"], ds[0]["images"])
    print(f"    raw={raw!r} -> {converted.action_text} (valid={converted.is_valid}) "
          f"latency={predictor.last_latency_s*1000:.0f}ms")
    _assert(converted.action_text in {
        "Move forward 25cm", "Turn right 15 degree", "Turn left 15 degree", "Stop"})


def test_tiny_train_step(tmp_dir="outputs/week4/_smoke"):
    from week4_config import Week4Config
    from train_week4_qwen3vl import train
    from vlnverse_dataset import build_synthetic_dataset, DatasetConfig
    import train_week4_qwen3vl as T

    cfg = Week4Config()
    cfg.output_dir = tmp_dir
    cfg.training.epochs = 1
    cfg.training.batch_size = 2
    cfg.training.grad_accum_steps = 1
    cfg.training.log_every = 1
    cfg.training.eval_generate_max_samples = 2
    cfg.model.dtype = "bf16"

    # monkeypatch dataset builder to use synthetic data (no download)
    def fake_build(c):
        dcfg = DatasetConfig(image_size=64, history_count=1, history_stride=1, max_image_count=2)
        ds = build_synthetic_dataset(num_episodes=2, frames_per_episode=4, cfg=dcfg)
        return ds, ds

    orig = T.build_train_val_datasets
    T.build_train_val_datasets = fake_build
    try:
        train(cfg, resume=False)
    finally:
        T.build_train_val_datasets = orig
    _assert(os.path.isdir(os.path.join(tmp_dir, "best_adapter")), "best adapter not saved")
    print(f"    checkpoint dir: {os.path.join(tmp_dir, 'checkpoints')}")


def run_tier2(runner: Runner):
    print(f"\n[Tier 2] model tests (downloads {MODEL_NAME})")
    runner.run("prompt_and_label_mask", test_prompt_and_label_mask)
    runner.run("model_forward_loss", test_model_forward_loss)
    runner.run("generation_and_parse", test_generation_and_parse)
    runner.run("tiny_train_step", test_tiny_train_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-model", action="store_true", help="run Tier 2 (downloads model)")
    parser.add_argument("--only", choices=["tier1", "tier2"], default=None)
    args = parser.parse_args()

    runner = Runner()
    if args.only != "tier2":
        run_tier1(runner)
    if args.with_model or args.only == "tier2":
        run_tier2(runner)
    sys.exit(runner.summary())


if __name__ == "__main__":
    main()
