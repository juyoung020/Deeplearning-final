"""Qwen3-VL-2B loading + LoRA (q_proj, v_proj only) for Week 4.

* Loads ``Qwen/Qwen3-VL-2B-Instruct`` and its processor.
* Applies LoRA to the LANGUAGE-MODEL ``q_proj`` / ``v_proj`` linear layers only.
  The vision encoder, base LM weights, tokenizer and token embeddings stay
  frozen. Target module discovery explicitly excludes ``visual``/``vision``
  submodules so LoRA never touches the (frozen) vision tower.
* Logs discovered target module names and the trainable-parameter count.

Heavy imports (torch / transformers / peft) are deferred to call time so the
rest of the package can be imported without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

DEFAULT_MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"


@dataclass
class LoRAConfigSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])


@dataclass
class ModelConfigSpec:
    model_name: str = DEFAULT_MODEL_NAME
    dtype: str = "bf16"           # "bf16" | "fp16" | "fp32"
    device_map: Optional[str] = None   # e.g. "auto"; None -> single device .to()
    gradient_checkpointing: bool = True
    attn_implementation: Optional[str] = None  # e.g. "flash_attention_2"; None -> default
    lora: LoRAConfigSpec = field(default_factory=LoRAConfigSpec)


# ---------------------------------------------------------------------------
def _torch_dtype(name: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        name, torch.bfloat16
    )


def load_processor(model_name: str = DEFAULT_MODEL_NAME):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return processor


def load_base_model(spec: ModelConfigSpec):
    """Load the base Qwen3-VL model (no LoRA yet)."""
    import torch  # noqa: F401
    from transformers import AutoModelForImageTextToText

    kwargs = dict(torch_dtype=_torch_dtype(spec.dtype), trust_remote_code=True)
    if spec.device_map:
        kwargs["device_map"] = spec.device_map
    if spec.attn_implementation:
        kwargs["attn_implementation"] = spec.attn_implementation

    try:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(spec.model_name, **kwargs)
    except Exception:
        model = AutoModelForImageTextToText.from_pretrained(spec.model_name, **kwargs)
    return model


def discover_lora_targets(
    model,
    include: Tuple[str, ...] = ("q_proj", "v_proj"),
    exclude_substrings: Tuple[str, ...] = ("visual", "vision", "vit"),
) -> List[str]:
    """Full module names of LM q_proj/v_proj linears (vision tower excluded)."""
    import torch.nn as nn

    targets: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in include:
            if any(x in name.lower() for x in exclude_substrings):
                continue
            targets.append(name)
    return targets


def count_parameters(model) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def apply_lora(model, spec: ModelConfigSpec, verbose: bool = True):
    from peft import LoraConfig, get_peft_model

    targets = discover_lora_targets(model, include=tuple(spec.lora.target_modules))
    if not targets:
        raise RuntimeError(
            "No q_proj/v_proj linear layers found outside the vision tower. "
            "Inspect model.named_modules() to find the correct projection names."
        )
    if verbose:
        # Print a compact summary: count + a few sample names.
        print(f"[week4] LoRA target modules discovered: {len(targets)} "
              f"(e.g. {targets[:4]})")

    lora_config = LoraConfig(
        r=spec.lora.r,
        lora_alpha=spec.lora.alpha,
        lora_dropout=spec.lora.dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora_config)

    if spec.gradient_checkpointing:
        if hasattr(peft_model, "enable_input_require_grads"):
            peft_model.enable_input_require_grads()
        base = getattr(peft_model, "base_model", peft_model)
        inner = getattr(base, "model", base)
        if hasattr(inner, "gradient_checkpointing_enable"):
            try:
                inner.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                inner.gradient_checkpointing_enable()

    if verbose:
        trainable, total = count_parameters(peft_model)
        pct = 100.0 * trainable / max(total, 1)
        print(f"[week4] trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")
        _assert_vision_frozen(peft_model, verbose=verbose)
    return peft_model


def _assert_vision_frozen(model, verbose: bool = True):
    """Sanity check: no trainable parameter lives inside the vision tower."""
    bad = [
        n for n, p in model.named_parameters()
        if p.requires_grad and any(x in n.lower() for x in ("visual", "vision", "vit"))
    ]
    if bad:
        raise RuntimeError(f"Vision tower has trainable params (should be frozen): {bad[:4]}")
    if verbose:
        print("[week4] vision encoder confirmed frozen (no trainable vision params).")


def load_model_and_processor(spec: ModelConfigSpec):
    """Load processor + LoRA-wrapped model. Returns (model, processor)."""
    processor = load_processor(spec.model_name)
    model = load_base_model(spec)
    if not spec.device_map:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    model = apply_lora(model, spec)
    return model, processor


def save_adapter(model, output_dir: str):
    """Save only the trainable LoRA adapter (PEFT ``save_pretrained``)."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)


def load_adapter(spec: ModelConfigSpec, adapter_dir: str, for_inference: bool = True):
    """Reload base model + LoRA adapter from ``adapter_dir`` for inference."""
    from peft import PeftModel

    processor = load_processor(spec.model_name)
    base = load_base_model(spec)
    if not spec.device_map:
        import torch

        base = base.to("cuda" if torch.cuda.is_available() else "cpu")
    model = PeftModel.from_pretrained(base, adapter_dir)
    if for_inference:
        model.eval()
    return model, processor


__all__ = [
    "DEFAULT_MODEL_NAME",
    "LoRAConfigSpec",
    "ModelConfigSpec",
    "load_processor",
    "load_base_model",
    "discover_lora_targets",
    "count_parameters",
    "apply_lora",
    "load_model_and_processor",
    "save_adapter",
    "load_adapter",
]
