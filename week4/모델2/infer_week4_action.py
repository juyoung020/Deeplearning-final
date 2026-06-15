"""Week 4 — Qwen3-VL next-action predictor (multimodal: image + text).

``Week4ActionPredictor`` wraps a Qwen3-VL vision-language model and turns
(egocentric frames + instruction [+ optional trajectory text]) into ONE discrete
navigation action string from ``week4.actions.CANONICAL_ACTIONS``.

Design goals
------------
* Multimodal by construction — every decision conditions on the robot's camera
  frame(s). This is the Week 4 "image input" extension (graded bonus) and the
  whole point of moving from Week 3's text-only MLP to a VLM.
* Optional LoRA adapter (PEFT). Zero-shot with the base model when no adapter is
  given; fine-tuned behaviour when an adapter dir is supplied.
* Robust output handling — the raw generation is parsed to a valid action; if it
  matches nothing, ``invalid_fallback`` is substituted and the call is flagged
  invalid (so closed-loop metrics can count invalid rate).

The class is import-light at module load: heavy deps (torch/transformers) are
imported inside ``__init__`` so the file can be linted / unit-imported without a
full ML environment.
"""

from __future__ import annotations

import os

from actions import (
    ACTION_SYSTEM_PROMPT,
    build_user_text,
    parse_action,
)

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
}


class Week4ActionPredictor:
    """Qwen3-VL → discrete navigation action."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        adapter_dir: str | None = None,
        device: str = "auto",
        dtype: str = "bf16",
        invalid_fallback: str = "Move forward 25cm",
        max_new_tokens: int = 8,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name = model_name
        self.adapter_dir = adapter_dir or None
        self.invalid_fallback = invalid_fallback
        self.max_new_tokens = int(max_new_tokens)
        self.last_was_invalid = False

        self.torch = torch
        self.device = self._resolve_device(device)
        self.torch_dtype = getattr(torch, _DTYPE_ALIASES.get(str(dtype).lower(), "bfloat16"))
        # fp16/bf16 are GPU features; fall back to fp32 on CPU.
        if self.device == "cpu" and self.torch_dtype in (torch.float16, torch.bfloat16):
            self.torch_dtype = torch.float32

        print(f"[INFO][week4] Loading VLM '{model_name}' dtype={self.torch_dtype} device={self.device}")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            dtype=self.torch_dtype,
            trust_remote_code=True,
        )

        if self.adapter_dir:
            from peft import PeftModel

            print(f"[INFO][week4] Attaching LoRA adapter: {self.adapter_dir}")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_dir)

        self.model.to(self.device)
        self.model.eval()
        print("[INFO][week4] VLM ready.")

    def _resolve_device(self, device: str) -> str:
        device = (device or "auto").lower()
        if device == "auto":
            return "cuda" if self.torch.cuda.is_available() else "cpu"
        return device

    def _build_messages(self, images, instruction: str, trajectory_text: str = ""):
        """Build the spec's Qwen3-VL chat: images first, then 'Instruction:.. Answer:'.

        The baseline text is exactly the spec format. ``trajectory_text`` is an
        optional ablation hook: when supplied it is inserted before the 'Answer:'
        cue (the model must have been trained the same way to benefit).
        """
        user_content = [{"type": "image", "image": img} for img in images]

        if trajectory_text:
            text = f"Instruction:\n{instruction.strip()}\n\n{trajectory_text.strip()}\n\nAnswer:"
        else:
            text = build_user_text(instruction)
        user_content.append({"type": "text", "text": text})

        return [
            {"role": "system", "content": [{"type": "text", "text": ACTION_SYSTEM_PROMPT}]},
            {"role": "user", "content": user_content},
        ]

    def predict_action(self, images, instruction: str, trajectory_text: str = "") -> str:
        """Return a canonical action string for the given observation.

        ``images`` is a list of PIL.Image (history first, current last). Always
        returns a non-empty action; on unparseable output, returns
        ``invalid_fallback`` and sets ``self.last_was_invalid = True``.
        """
        if not images:
            raise ValueError("predict_action requires at least one image (the current frame).")

        messages = self._build_messages(images, instruction, trajectory_text)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        with self.torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = generated[:, inputs["input_ids"].shape[1]:]
        raw = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        parsed = parse_action(raw)
        if parsed.valid:
            self.last_was_invalid = False
            return parsed.label
        self.last_was_invalid = True
        print(f"[WARN][week4] Unparseable action {raw!r}; using fallback {self.invalid_fallback!r}.")
        return self.invalid_fallback


class Week4SubprocessPredictor:
    """Same interface as Week4ActionPredictor, but runs the VLM in a separate
    process (vlm_server.py). Use this from inside Isaac Sim, where loading
    Qwen3-VL in-process crashes with a native DLL conflict (0xc0000139)."""

    def __init__(self, model_name="Qwen/Qwen3-VL-2B-Instruct", adapter_dir=None,
                 device="auto", dtype="bf16", invalid_fallback="Move forward 25cm"):
        import subprocess
        import sys
        import tempfile

        self.invalid_fallback = invalid_fallback
        self.last_was_invalid = False
        self._tmpdir = tempfile.mkdtemp(prefix="week4vlm_")
        self._n = 0

        self._dbg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "week4_parent_debug.log")
        self._log("=== Week4SubprocessPredictor init; python=%s" % sys.executable)
        server = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vlm_server.py")
        cmd = [sys.executable, server, "--model", model_name, "--device", device,
               "--dtype", dtype, "--invalid-fallback", invalid_fallback]
        if adapter_dir:
            cmd += ["--adapter", adapter_dir]
        env = dict(os.environ)
        env["HF_HUB_DISABLE_SYMLINKS"] = "1"
        print(f"[INFO][week4] Spawning out-of-process VLM server (isolates DLLs from Isaac Sim)...")
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, bufsize=1, env=env, cwd=os.path.dirname(server))
        # Wait for READY (server logs go to stderr, which streams to our console).
        self._log("server spawned, waiting for READY...")
        while True:
            line = self.proc.stdout.readline()
            if not line:
                self._log("ERROR: VLM server exited before READY")
                raise RuntimeError("VLM server exited before becoming ready")
            if line.strip() == "READY":
                break
        self._log("VLM server READY received.")
        print("[INFO][week4] VLM server ready.")

    def _log(self, msg):
        try:
            with open(self._dbg, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    def predict_action(self, images, instruction, trajectory_text=""):
        import json
        paths = []
        for i, im in enumerate(images):
            p = os.path.join(self._tmpdir, f"f_{self._n}_{i}.png")
            im.save(p)
            paths.append(p)
        self._n += 1
        req = json.dumps({"images": paths, "instruction": instruction, "trajectory": trajectory_text})
        try:
            self.proc.stdin.write(req + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("VLM server died")
                if line.startswith("RESP "):
                    resp = json.loads(line[5:])
                    self.last_was_invalid = bool(resp.get("invalid", False))
                    act = resp.get("action", self.invalid_fallback)
                    self._log(f"predict #{self._n}: {act} (invalid={self.last_was_invalid})")
                    return act
        except Exception as exc:
            self._log(f"predict #{self._n} FAILED: {exc}")
            print(f"[WARN][week4] VLM server call failed ({exc}); using fallback.")
            self.last_was_invalid = True
            return self.invalid_fallback

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass
