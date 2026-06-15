"""Week 4 inference: predict the next navigation action from an instruction and
an egocentric image history.

Provides:
  * ``Week4ActionPredictor`` — reusable class (loads adapter, holds model +
    processor), used by both the CLI and the IsaacSim commander.
  * a CLI for offline single-sample inference / quick checks.

Generation is greedy and fixed: max_new_tokens=8, do_sample=False, use_cache=True.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, List, Optional, Tuple

from week4_actions import action_to_command, ConvertedAction, INVALID_FALLBACK_ACTION
from qwen3vl_prompt import DEFAULT_SYSTEM_PROMPT, build_inference_inputs

GEN_KWARGS = dict(max_new_tokens=8, do_sample=False, use_cache=True)


class Week4ActionPredictor:
    """Loads a LoRA adapter (or base model) and predicts canonical actions."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        adapter_dir: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bf16",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        invalid_fallback: str = INVALID_FALLBACK_ACTION,
    ):
        from qwen3vl_lora_model import ModelConfigSpec, load_adapter, load_model_and_processor

        self.system_prompt = system_prompt
        self.invalid_fallback = invalid_fallback
        spec = ModelConfigSpec(model_name=model_name, dtype=dtype, gradient_checkpointing=False)

        if adapter_dir and os.path.isdir(adapter_dir):
            self.model, self.processor = load_adapter(spec, adapter_dir, for_inference=True)
        else:
            # No adapter -> base model (zero-shot baseline). Skip LoRA wrapping.
            from qwen3vl_lora_model import load_processor, load_base_model
            import torch

            self.processor = load_processor(model_name)
            self.model = load_base_model(spec)
            self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
            self.model.eval()

        self.device = next(self.model.parameters()).device
        self.last_latency_s: float = 0.0

    def predict_raw(
        self, instruction: str, images: List[Any], trajectory_text: Optional[str] = None
    ) -> str:
        import torch

        inputs = build_inference_inputs(
            self.processor, instruction, images,
            system_prompt=self.system_prompt, trajectory_text=trajectory_text,
        )
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        t0 = time.time()
        with torch.no_grad():
            out = self.model.generate(**inputs, **GEN_KWARGS)
        self.last_latency_s = time.time() - t0
        prompt_len = inputs["input_ids"].shape[1]
        tok = getattr(self.processor, "tokenizer", self.processor)
        return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    def predict(
        self, instruction: str, images: List[Any], trajectory_text: Optional[str] = None
    ) -> Tuple[str, ConvertedAction]:
        """Return (raw_text, ConvertedAction). ConvertedAction.is_valid flags
        whether the raw output parsed to a canonical action."""
        raw = self.predict_raw(instruction, images, trajectory_text)
        converted = action_to_command(raw, invalid_fallback=self.invalid_fallback)
        return raw, converted


def main():
    parser = argparse.ArgumentParser(description="Week 4 single-sample inference")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir (optional)")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--images", nargs="+", required=True, help="image paths oldest->newest")
    parser.add_argument("--dtype", default="bf16")
    args = parser.parse_args()

    from PIL import Image

    images = [Image.open(p).convert("RGB") for p in args.images]
    predictor = Week4ActionPredictor(
        model_name=args.model_name, adapter_dir=args.adapter, dtype=args.dtype,
    )
    raw, converted = predictor.predict(args.instruction, images)
    print(f"raw_output : {raw!r}")
    print(f"action     : {converted.action_text}  (valid={converted.is_valid})")
    print(f"command    : velocity={converted.velocity} duration={converted.duration_s}s "
          f"terminate={converted.terminate}")
    print(f"latency    : {predictor.last_latency_s*1000:.1f} ms")


if __name__ == "__main__":
    main()
