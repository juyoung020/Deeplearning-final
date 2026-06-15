"""Week 4 — out-of-process VLM inference server.

Loading Qwen3-VL inside the Isaac Sim Kit process crashes with a native DLL
conflict (Windows fatal exception 0xc0000139). The model loads and runs fine in
a clean Python process, so we host it here as a subprocess and talk to it over
stdin/stdout. `Week4SubprocessPredictor` (in infer_week4_action.py) drives this.

Protocol (one JSON object per line):
  stdout: "READY"                         once the model is loaded
  stdin : {"images": [png_path, ...], "instruction": str, "trajectory": str}
  stdout: "RESP " + {"action": str, "invalid": bool}

All model/loader chatter goes to stderr so stdout carries only READY / RESP lines.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--invalid-fallback", default="Move forward 25cm")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    # Route any predictor stdout prints to stderr so stdout stays protocol-only.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    from PIL import Image
    from infer_week4_action import Week4ActionPredictor

    predictor = Week4ActionPredictor(
        model_name=args.model, adapter_dir=args.adapter or None,
        device=args.device, dtype=args.dtype, invalid_fallback=args.invalid_fallback,
    )

    real_stdout.write("READY\n")
    real_stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            images = [Image.open(p).convert("RGB") for p in req["images"]]
            action = predictor.predict_action(images, req.get("instruction", ""), req.get("trajectory", ""))
            resp = {"action": action, "invalid": bool(getattr(predictor, "last_was_invalid", False))}
        except Exception as exc:  # never kill the server on a bad request
            sys.stderr.write(f"[vlm_server] request failed: {exc}\n")
            resp = {"action": args.invalid_fallback, "invalid": True}
        real_stdout.write("RESP " + json.dumps(resp) + "\n")
        real_stdout.flush()


if __name__ == "__main__":
    main()
