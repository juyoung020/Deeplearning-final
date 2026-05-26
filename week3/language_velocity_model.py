from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_SIGLIP2_MODEL = "google/siglip2-large-patch16-256"


@dataclass(frozen=True)
class VelocityExample:
    text: str
    target: tuple[float, float, float]


def default_velocity_examples() -> list[VelocityExample]:
    return [
        VelocityExample("move forward 25cm", (0.25, 0.0, 0.0)),
        VelocityExample("move forward 50cm", (0.50, 0.0, 0.0)),
        VelocityExample("move forward 75cm", (0.75, 0.0, 0.0)),
        VelocityExample("move forward 1m", (1.00, 0.0, 0.0)),
        VelocityExample("turn left 15 degree", (0.0, 0.0, math.pi / 12.0)),
        VelocityExample("turn left 30 degree", (0.0, 0.0, math.pi / 6.0)),
        VelocityExample("turn left 45 degree", (0.0, 0.0, math.pi / 4.0)),
        VelocityExample("turn right 15 degree", (0.0, 0.0, -math.pi / 12.0)),
        VelocityExample("turn right 30 degree", (0.0, 0.0, -math.pi / 6.0)),
        VelocityExample("turn right 45 degree", (0.0, 0.0, -math.pi / 4.0)),
        VelocityExample("stop", (0.0, 0.0, 0.0)),
    ]


def examples_from_config(config: dict) -> list[VelocityExample]:
    rows = config.get("dataset")
    if not rows:
        return default_velocity_examples()

    examples = []
    for row in rows:
        text = str(row["text"])
        target = tuple(float(value) for value in row["target"])
        if len(target) != 3:
            raise ValueError(f"target for {text!r} must have 3 values")
        examples.append(VelocityExample(text, target))
    return examples


def resolve_device(device_name: str | None) -> torch.device:
    if not device_name or device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_torch_dtype(dtype_name: str | None):
    if not dtype_name:
        return None
    if dtype_name == "auto":
        return "auto"
    if not hasattr(torch, dtype_name):
        raise ValueError(f"unknown torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


class FrozenSigLIPTextEncoder:
    def __init__(
        self,
        model_name: str = DEFAULT_SIGLIP2_MODEL,
        device: str | torch.device | None = "auto",
        max_length: int = 64,
        normalize: bool = False,
        torch_dtype: str | None = None,
    ):
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install transformers to use the SigLIP2 text encoder: pip install transformers") from exc

        self.model_name = model_name
        self.device = resolve_device(str(device) if device is not None else "auto")
        self.max_length = int(max_length) if max_length else None
        self.normalize = bool(normalize)

        model_kwargs = {}
        dtype = resolve_torch_dtype(torch_dtype)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, **model_kwargs)
        self.model.eval()
        self.model.to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        text_config = getattr(getattr(self.model, "config", None), "text_config", None)
        self.embedding_dim = int(getattr(text_config, "hidden_size", 0) or getattr(text_config, "projection_size", 0) or 0)
        if self.embedding_dim <= 0:
            self.embedding_dim = int(getattr(getattr(self.model, "config", None), "projection_dim", 0) or 0)

    def tokenize(self, texts: str | Sequence[str]) -> dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        padding = "max_length" if self.max_length else True
        tokens = self.tokenizer(
            list(texts),
            padding=padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in tokens.items()}

    @torch.inference_mode()
    def encode(self, texts: str | Sequence[str]) -> torch.Tensor:
        tokens = self.tokenize(texts)
        embeddings = self._forward_text(tokens)
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def _forward_text(self, tokens: dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.model, "text_model"):
            text_inputs = {
                key: value
                for key, value in tokens.items()
                if key in {"input_ids", "attention_mask", "position_ids"}
            }
            try:
                outputs = self.model.text_model(**text_inputs, return_dict=True)
            except TypeError:
                outputs = self.model.text_model(**text_inputs)
            pooled = getattr(outputs, "pooler_output", None)
            if pooled is not None:
                return pooled

            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden = outputs[0]
            mask = tokens.get("attention_mask")
            if mask is None:
                return hidden[:, 0]
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        if hasattr(self.model, "get_text_features"):
            features = self.model.get_text_features(**tokens)
            if self.embedding_dim <= 0:
                self.embedding_dim = int(features.shape[-1])
            return features

        raise RuntimeError(f"{self.model_name} does not expose text_model or get_text_features")


class VelocityMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 128,
        output_dim: int = 3,
        dropout: float = 0.0,
        num_layers: int = 1,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(embeddings)


def target_tensor(examples: Iterable[VelocityExample], device: torch.device | str | None = None) -> torch.Tensor:
    return torch.tensor([example.target for example in examples], dtype=torch.float32, device=device)


def checkpoint_load(path: str, map_location: torch.device | str | None = None) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class LanguageVelocityPredictor:
    def __init__(self, checkpoint_path: str, device: str | torch.device | None = "auto"):
        self.device = resolve_device(str(device) if device is not None else "auto")
        payload = checkpoint_load(checkpoint_path, map_location=self.device)
        config = dict(payload.get("config", {}))
        self.config = config

        model_name = config.get("model_name", DEFAULT_SIGLIP2_MODEL)
        max_length = int(config.get("max_length", 64))
        normalize_embeddings = bool(config.get("normalize_embeddings", False))
        torch_dtype = config.get("torch_dtype")
        self.encoder = FrozenSigLIPTextEncoder(
            model_name=model_name,
            device=self.device,
            max_length=max_length,
            normalize=normalize_embeddings,
            torch_dtype=torch_dtype,
        )

        input_dim = int(config.get("embedding_dim") or self.encoder.embedding_dim or 1024)
        hidden_dim = int(config.get("hidden_dim", 128))
        dropout = float(config.get("dropout", 0.0))
        num_layers = int(config.get("num_layers", 1))
        self.model = VelocityMLP(
            input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers
        ).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def predict_tensor(self, texts: str | Sequence[str]) -> torch.Tensor:
        embeddings = self.encoder.encode(texts)
        return self.model(embeddings)

    def predict(self, text: str) -> tuple[float, float, float]:
        tensor = self.predict_tensor(text)[0].detach().cpu().float()
        return tuple(float(value) for value in tensor.tolist())

    def predict_batch(self, texts: list[str]) -> list[tuple[float, float, float]]:
        tensors = self.predict_tensor(texts).detach().cpu().float()
        return [tuple(float(v) for v in row.tolist()) for row in tensors]


def format_velocity(velocity: Sequence[float]) -> str:
    return f"[{float(velocity[0]):+.4f}, {float(velocity[1]):+.4f}, {float(velocity[2]):+.4f}]"
