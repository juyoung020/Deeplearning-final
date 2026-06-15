"""Week 4 config: aggregates dataset / model / LoRA / training settings.

Loaded from a JSON file (see ``week4/configs/``). Unknown keys are ignored with
a warning so configs can carry documentation fields.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from vlnverse_dataset import DatasetConfig
from qwen3vl_lora_model import LoRAConfigSpec, ModelConfigSpec
from qwen3vl_prompt import DEFAULT_SYSTEM_PROMPT


@dataclass
class TrainingConfig:
    batch_size: int = 2
    grad_accum_steps: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    epochs: int = 3
    max_grad_norm: float = 1.0
    log_every: int = 10
    save_every_steps: int = 200     # mid-epoch checkpoint cadence (optimizer steps); 0 = epoch-end only
    seed: int = 7
    num_workers: int = 2
    # validation
    val_ratio: float = 0.1
    eval_max_batches: Optional[int] = None       # cap val batches (speed)
    eval_generate_max_samples: Optional[int] = 200  # cap generation-based metrics


@dataclass
class LoggingConfig:
    use_wandb: bool = False
    wandb_project: str = "vlnverse-week4"
    wandb_run_name: Optional[str] = None
    jsonl_log: str = "logs/train_log.jsonl"


@dataclass
class Week4Config:
    output_dir: str = "outputs/week4"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    forced_val_split_json: Optional[str] = None  # vlnverse_closed_loop_eval_20episodes.json (val ids)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfigSpec = field(default_factory=ModelConfigSpec)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def from_json(path: str) -> "Week4Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return Week4Config.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict) -> "Week4Config":
        cfg = Week4Config()
        for top in ("output_dir", "system_prompt", "forced_val_split_json"):
            if top in raw:
                setattr(cfg, top, raw[top])

        if "dataset" in raw:
            cfg.dataset = _fill(DatasetConfig(), raw["dataset"], "dataset")
        if "model" in raw:
            mdict = dict(raw["model"])
            lora = mdict.pop("lora", None)
            cfg.model = _fill(ModelConfigSpec(), mdict, "model")
            if lora is not None:
                cfg.model.lora = _fill(LoRAConfigSpec(), lora, "model.lora")
        if "training" in raw:
            cfg.training = _fill(TrainingConfig(), raw["training"], "training")
        if "logging" in raw:
            cfg.logging = _fill(LoggingConfig(), raw["logging"], "logging")
        return cfg

    def to_dict(self) -> dict:
        return {
            "output_dir": self.output_dir,
            "system_prompt": self.system_prompt,
            "forced_val_split_json": self.forced_val_split_json,
            "dataset": asdict(self.dataset),
            "model": {**asdict(self.model)},
            "training": asdict(self.training),
            "logging": asdict(self.logging),
        }


def _fill(obj, data: dict, ctx: str):
    valid = set(vars(obj).keys())
    for k, v in data.items():
        if k in valid:
            setattr(obj, k, v)
        else:
            warnings.warn(f"[week4_config] ignoring unknown key {ctx}.{k}")
    return obj


__all__ = ["Week4Config", "TrainingConfig", "LoggingConfig"]
