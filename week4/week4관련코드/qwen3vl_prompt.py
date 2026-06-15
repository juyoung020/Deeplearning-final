"""Qwen3-VL chat-template prompt building and answer-only label masking.

Rules enforced here (per Week 4 spec):
  * Always go through ``processor.apply_chat_template`` — never inject Qwen
    special tokens (``<|im_start|>`` etc.) as raw strings, never add BOS/EOS by
    hand.
  * Training messages contain the assistant answer and use
    ``add_generation_prompt=False``.
  * Inference messages omit the assistant answer and use
    ``add_generation_prompt=True``.
  * Loss is applied ONLY to the assistant answer tokens (+ its end token). Every
    prompt/image/instruction/padding token is set to ``-100``. Labels are NOT
    shifted manually (HF causal LM shifts internally).

The label mask is derived by measuring the tokenized length of the
inference-style prompt (same images), then masking everything before the answer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DEFAULT_SYSTEM_PROMPT = (
    "You are a navigation action predictor for a mobile robot.\n\n"
    "Given the full navigation instruction and the egocentric image history, "
    "predict the next action.\n\n"
    "You must answer with exactly one of the following four actions:\n"
    "- Move forward 25cm\n"
    "- Turn right 15 degree\n"
    "- Turn left 15 degree\n"
    "- Stop"
)


def build_user_text(instruction: str, trajectory_text: Optional[str] = None) -> str:
    """Build the textual portion of the user turn."""
    if trajectory_text:
        return (
            "Instruction:\n"
            f"{instruction}\n\n"
            f"{trajectory_text}\n\n"
            "Answer:"
        )
    return "Instruction:\n" f"{instruction}\n\n" "Answer:"


def build_messages(
    instruction: str,
    images: List[Any],
    answer: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    trajectory_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build a Qwen3-VL chat ``messages`` list.

    Images are inserted in the given order (oldest -> newest). When ``answer`` is
    provided an assistant turn is appended (training); otherwise it is omitted
    (inference).
    """
    user_content: List[Dict[str, Any]] = [{"type": "image", "image": img} for img in images]
    user_content.append({"type": "text", "text": build_user_text(instruction, trajectory_text)})

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def _render_text(processor, messages, add_generation_prompt: bool) -> str:
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _prompt_token_len(processor, instruction, images, system_prompt, trajectory_text) -> int:
    """Tokenized length of the inference-style prompt (with the same images)."""
    prompt_messages = build_messages(
        instruction, images, answer=None, system_prompt=system_prompt,
        trajectory_text=trajectory_text,
    )
    prompt_text = _render_text(processor, prompt_messages, add_generation_prompt=True)
    enc = processor(text=[prompt_text], images=list(images), return_tensors="pt")
    return int(enc["input_ids"].shape[1])


class Qwen3VLCollator:
    """Collate dataset samples into a model batch with answer-only labels."""

    def __init__(self, processor, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.processor = processor
        self.system_prompt = system_prompt
        # Per-sample label masking below assumes RIGHT padding (sample i occupies
        # input_ids[i, :len_i]); force it so the answer span is masked correctly
        # regardless of the tokenizer's default padding side.
        tok = getattr(processor, "tokenizer", None)
        if tok is not None:
            tok.padding_side = "right"
            if tok.pad_token_id is None and tok.eos_token_id is not None:
                tok.pad_token = tok.eos_token
        self.pad_token_id = getattr(tok, "pad_token_id", None)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        full_texts: List[str] = []
        flat_images: List[Any] = []
        prompt_lens: List[int] = []

        for s in samples:
            images = s["images"]
            messages_full = build_messages(
                s["instruction"], images, answer=s["answer"],
                system_prompt=self.system_prompt,
                trajectory_text=s.get("trajectory_text"),
            )
            full_texts.append(_render_text(self.processor, messages_full, add_generation_prompt=False))
            flat_images.extend(images)
            prompt_lens.append(
                _prompt_token_len(
                    self.processor, s["instruction"], images, self.system_prompt,
                    s.get("trajectory_text"),
                )
            )

        batch = self.processor(
            text=full_texts,
            images=flat_images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = input_ids.clone()

        # Mask everything before the answer for each sample, plus padding.
        # NOTE: assumes right padding (default for these processors at train time).
        for i, plen in enumerate(prompt_lens):
            labels[i, :plen] = -100
        if attention_mask is not None:
            labels[attention_mask == 0] = -100
        if self.pad_token_id is not None:
            labels[input_ids == self.pad_token_id] = -100

        batch["labels"] = labels
        return dict(batch)


def decode_label_targets(processor, input_ids, labels) -> List[str]:
    """Return, per batch row, the decoded text of the non-masked label tokens.

    Used by smoke tests to verify that loss is applied to the answer (+ end
    token) only.
    """
    tok = getattr(processor, "tokenizer", processor)
    out = []
    for ids_row, lab_row in zip(input_ids.tolist(), labels.tolist()):
        kept = [tid for tid, lid in zip(ids_row, lab_row) if lid != -100]
        out.append(tok.decode(kept, skip_special_tokens=False))
    return out


def build_inference_inputs(
    processor,
    instruction: str,
    images: List[Any],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    trajectory_text: Optional[str] = None,
):
    """Tokenized inference inputs (add_generation_prompt=True, no answer)."""
    messages = build_messages(
        instruction, images, answer=None, system_prompt=system_prompt,
        trajectory_text=trajectory_text,
    )
    text = _render_text(processor, messages, add_generation_prompt=True)
    return processor(text=[text], images=list(images), return_tensors="pt")


__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "build_user_text",
    "build_messages",
    "Qwen3VLCollator",
    "decode_label_targets",
    "build_inference_inputs",
]
