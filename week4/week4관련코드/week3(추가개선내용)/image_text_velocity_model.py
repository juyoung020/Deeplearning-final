"""Week 3 bonus (PDF §4.2): Image + Text -> velocity.

Extends the Week 3 text-only SigLIP2 -> VelocityMLP pipeline by adding a frozen
SigLIP2 *vision* branch and fusing it (concat) with the text branch before the
MLP head. The base SigLIP2 model is loaded once and exposes BOTH a text encoder
and an image encoder, so the two modalities share weights/memory.

Design:
    text  -> SigLIP2 text_model   (frozen) -> 1024-d ┐
    image -> SigLIP2 vision_model (frozen) -> 1024-d ┤ fuse -> MLP -> [vx, vy, wz]

`modalities` selects which branches feed the MLP, enabling the ablation with a
single codebase:
    "text"        : text only (== original Week 3)
    "image"       : image only
    "image_text"  : both (concat)

Encoders are frozen; only the small MLP head is trained. The MLP is reused from
``language_velocity_model.VelocityMLP``.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from language_velocity_model import (
    DEFAULT_SIGLIP2_MODEL,
    VelocityMLP,
    resolve_device,
    resolve_torch_dtype,
)

VALID_MODALITIES = ("text", "image", "image_text")


def _as_feature_tensor(out) -> torch.Tensor:
    """Coerce a model output to a pooled feature tensor.

    Depending on the transformers version, ``get_text_features`` /
    ``get_image_features`` may return a tensor directly or a model-output object
    (e.g. ``BaseModelOutputWithPooling``). Handle both.
    """
    if isinstance(out, torch.Tensor):
        return out
    pooled = getattr(out, "pooler_output", None)
    if pooled is not None:
        return pooled
    hidden = getattr(out, "last_hidden_state", None)
    if hidden is not None:
        return hidden.mean(dim=1)
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    raise TypeError(f"cannot extract feature tensor from {type(out)}")


def modality_input_dim(modalities: str, text_dim: int, image_dim: int) -> int:
    if modalities == "text":
        return text_dim
    if modalities == "image":
        return image_dim
    if modalities == "image_text":
        return text_dim + image_dim
    raise ValueError(f"modalities must be one of {VALID_MODALITIES}, got {modalities!r}")


class FrozenSigLIP2Encoder:
    """Frozen SigLIP2 encoder exposing both text and image feature extraction.

    Loads the full ``AutoModel`` once (text_model + vision_model) plus the
    matching ``AutoProcessor``. All parameters are frozen.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_SIGLIP2_MODEL,
        device: str | torch.device | None = "auto",
        max_length: int = 64,
        normalize: bool = False,
        torch_dtype: str | None = None,
    ):
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:  # pragma: no cover - env guard
            raise ImportError("pip install transformers to use SigLIP2") from exc

        self.model_name = model_name
        self.device = resolve_device(str(device) if device is not None else "auto")
        self.max_length = int(max_length) if max_length else None
        self.normalize = bool(normalize)

        model_kwargs = {}
        dtype = resolve_torch_dtype(torch_dtype)
        if dtype is not None and dtype != "auto":
            model_kwargs["torch_dtype"] = dtype

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

        cfg = getattr(self.model, "config", None)
        tcfg = getattr(cfg, "text_config", None)
        vcfg = getattr(cfg, "vision_config", None)
        self.text_dim = int(getattr(tcfg, "hidden_size", 0) or 0)
        self.image_dim = int(getattr(vcfg, "hidden_size", 0) or 0)

    # -- text ---------------------------------------------------------------
    @torch.inference_mode()
    def encode_text(self, texts: str | Sequence[str]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        padding = "max_length" if self.max_length else True
        tokens = self.processor(
            text=list(texts),
            padding=padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items() if k in {"input_ids", "attention_mask"}}
        feats = _as_feature_tensor(self.model.get_text_features(**tokens))
        if self.text_dim <= 0:
            self.text_dim = int(feats.shape[-1])
        if self.normalize:
            feats = F.normalize(feats, dim=-1)
        return feats.float()

    # -- image --------------------------------------------------------------
    @torch.inference_mode()
    def encode_image(self, images) -> torch.Tensor:
        """images: a single PIL.Image or a list of PIL.Image."""
        if not isinstance(images, (list, tuple)):
            images = [images]
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        feats = _as_feature_tensor(self.model.get_image_features(pixel_values=pixel_values))
        if self.image_dim <= 0:
            self.image_dim = int(feats.shape[-1])
        if self.normalize:
            feats = F.normalize(feats, dim=-1)
        return feats.float()


class ImageTextVelocityModel(nn.Module):
    """Frozen SigLIP2 (text + image) -> fusion -> VelocityMLP -> [vx, vy, wz].

    Forward consumes *pre-computed* frozen embeddings (text_emb / image_emb),
    matching the Week 3 cache-then-train recipe. Pass only the embeddings the
    selected ``modalities`` needs.
    """

    def __init__(
        self,
        modalities: str = "image_text",
        text_dim: int = 1024,
        image_dim: int = 1024,
        hidden_dim: int = 128,
        output_dim: int = 3,
        dropout: float = 0.0,
        num_layers: int = 1,
    ):
        super().__init__()
        if modalities not in VALID_MODALITIES:
            raise ValueError(f"modalities must be one of {VALID_MODALITIES}")
        self.modalities = modalities
        self.text_dim = int(text_dim)
        self.image_dim = int(image_dim)
        in_dim = modality_input_dim(modalities, self.text_dim, self.image_dim)
        self.mlp = VelocityMLP(
            input_dim=in_dim, hidden_dim=hidden_dim, output_dim=output_dim,
            dropout=dropout, num_layers=num_layers,
        )

    def fuse(self, text_emb: torch.Tensor | None, image_emb: torch.Tensor | None) -> torch.Tensor:
        if self.modalities == "text":
            assert text_emb is not None, "text modality needs text_emb"
            return text_emb
        if self.modalities == "image":
            assert image_emb is not None, "image modality needs image_emb"
            return image_emb
        assert text_emb is not None and image_emb is not None, "image_text needs both"
        return torch.cat([text_emb, image_emb], dim=-1)

    def forward(self, text_emb: torch.Tensor | None = None, image_emb: torch.Tensor | None = None) -> torch.Tensor:
        return self.mlp(self.fuse(text_emb, image_emb))
