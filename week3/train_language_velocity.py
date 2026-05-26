from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy

import torch
from torch.nn import functional as F

from language_velocity_model import (
    DEFAULT_SIGLIP2_MODEL,
    FrozenSigLIPTextEncoder,
    VelocityMLP,
    examples_from_config,
    format_velocity,
    resolve_device,
    target_tensor,
)


DEFAULT_CONFIG = {
    "model_name": DEFAULT_SIGLIP2_MODEL,
    "embedding_dim": 1024,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.0,
    "max_length": 64,
    "normalize_embeddings": False,
    "torch_dtype": None,
    "device": "auto",
    "seed": 7,
    "batch_size": 11,
    "training_steps": 1500,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "grad_clip_norm": 1.0,
    "lr_scheduler": "cosine",
    "lr_min": 1e-5,
    "cache_embeddings": True,
    "print_every": 100,
    "checkpoint_path": "checkpoints/language_velocity_mlp.pt",
    "eval_path": "runs/week3_language_velocity_eval.json",
}


def load_config(path: str | None) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        config.update(user_config)
    return config


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    overrides = {
        "device": args.device,
        "training_steps": args.steps,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "checkpoint_path": args.checkpoint_out,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_all(encoder: FrozenSigLIPTextEncoder, texts: list[str]) -> torch.Tensor:
    return encoder.encode(texts).detach()


def evaluate(model: VelocityMLP, embeddings: torch.Tensor, targets: torch.Tensor) -> tuple[dict, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        predictions = model(embeddings)
        error = predictions - targets
        abs_error = error.abs()
        metrics = {
            "mse": float(F.mse_loss(predictions, targets).item()),
            "mae": float(abs_error.mean().item()),
            "max_abs_error": float(abs_error.max().item()),
            "mae_vx": float(abs_error[:, 0].mean().item()),
            "mae_vy": float(abs_error[:, 1].mean().item()),
            "mae_yaw": float(abs_error[:, 2].mean().item()),
        }
    return metrics, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SigLIP2 text embedding -> velocity MLP for Week 3.")
    parser.add_argument("--config", default="configs/week3_siglip_velocity.json", help="JSON hyperparameter config.")
    parser.add_argument("--device", default=None, help="Override device: auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--steps", default=None, type=int, help="Override number of training steps.")
    parser.add_argument("--lr", default=None, type=float, help="Override learning rate.")
    parser.add_argument("--batch-size", default=None, type=int, help="Override batch size.")
    parser.add_argument("--checkpoint-out", default=None, help="Override checkpoint output path.")
    args = parser.parse_args()

    config = apply_overrides(load_config(args.config), args)
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device"))
    examples = examples_from_config(config)
    texts = [example.text for example in examples]
    targets = target_tensor(examples, device=device)

    print(f"[INFO] Training examples: {len(examples)}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] SigLIP2 text encoder: {config['model_name']} (frozen)")
    encoder = FrozenSigLIPTextEncoder(
        model_name=config["model_name"],
        device=device,
        max_length=int(config["max_length"]),
        normalize=bool(config["normalize_embeddings"]),
        torch_dtype=config.get("torch_dtype"),
    )
    if encoder.embedding_dim > 0 and int(config["embedding_dim"]) != encoder.embedding_dim:
        print(f"[INFO] Encoder embedding dim is {encoder.embedding_dim}; updating config embedding_dim.")
        config["embedding_dim"] = encoder.embedding_dim

    model = VelocityMLP(
        input_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        num_layers=int(config.get("num_layers", 1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )

    scheduler = None
    if config.get("lr_scheduler") == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(config["training_steps"]), 1),
            eta_min=float(config.get("lr_min", 1e-5)),
        )
        print(f"[INFO] LR scheduler: CosineAnnealingLR (lr {config['learning_rate']} → {config.get('lr_min', 1e-5)})")

    cache_embeddings = bool(config["cache_embeddings"])
    all_embeddings = encode_all(encoder, texts) if cache_embeddings else None
    print(f"[INFO] Cache embeddings: {cache_embeddings}")

    batch_size = max(int(config["batch_size"]), 1)
    training_steps = max(int(config["training_steps"]), 1)
    print_every = max(int(config["print_every"]), 1)
    grad_clip_norm = float(config["grad_clip_norm"])

    model.train()
    for step in range(1, training_steps + 1):
        indices = torch.randint(0, len(examples), (batch_size,), device=device)
        if all_embeddings is not None:
            embeddings = all_embeddings[indices]
        else:
            batch_texts = [texts[index] for index in indices.detach().cpu().tolist()]
            embeddings = encoder.encode(batch_texts).detach()
        batch_targets = targets[indices]

        predictions = model(embeddings)
        loss = F.mse_loss(predictions, batch_targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if step == 1 or step % print_every == 0 or step == training_steps:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"[TRAIN] step={step:04d}/{training_steps} loss={loss.item():.8f} lr={lr_now:.2e}")

    if all_embeddings is None:
        all_embeddings = encode_all(encoder, texts)
    metrics, predictions = evaluate(model, all_embeddings, targets)

    eval_rows = []
    for example, prediction in zip(examples, predictions.detach().cpu(), strict=True):
        predicted = tuple(float(value) for value in prediction.tolist())
        eval_rows.append(
            {
                "text": example.text,
                "target": list(example.target),
                "prediction": list(predicted),
            }
        )
        print(f"[EVAL] {example.text:24s} target={format_velocity(example.target)} pred={format_velocity(predicted)}")
    print(
        "[RESULT] "
        f"mse={metrics['mse']:.8f}, mae={metrics['mae']:.8f}, max_abs_error={metrics['max_abs_error']:.8f}"
    )
    print(
        "[RESULT] per-axis MAE "
        f"vx={metrics['mae_vx']:.6f}, vy={metrics['mae_vy']:.6f}, yaw={metrics['mae_yaw']:.6f}"
    )

    checkpoint_path = os.path.abspath(os.path.expanduser(config["checkpoint_path"]))
    eval_path = os.path.abspath(os.path.expanduser(config["eval_path"]))
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(eval_path), exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "examples": [{"text": example.text, "target": list(example.target)} for example in examples],
            "metrics": metrics,
        },
        checkpoint_path,
    )
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "examples": eval_rows}, f, indent=2)

    print(f"[INFO] Saved checkpoint: {checkpoint_path}")
    print(f"[INFO] Saved eval predictions: {eval_path}")


if __name__ == "__main__":
    main()
