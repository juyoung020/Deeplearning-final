"""Build (image, text, velocity) samples from VLN-VERSE for the Week 3 bonus.

Reuses the Week 4 loaders (``week4/vlnverse_dataset.py`` + ``week4_actions.py``)
to read each episode's egocentric RGB (``rgb.npy``) and per-frame action ids,
then turns every timestep into a training sample:

    image  = egocentric RGB frame I_t  (PIL.Image)
    text   = instruction (default) OR the canonical action string (config)
    target = action_to_command(action[t]).velocity  ->  [vx, vy, wz]

Episode-level train/val split (seed 7) mirrors Week 4 so there is no frame-level
leakage. Optional per-class capping fights the heavy "move forward" imbalance so
the image signal for the minority (turn / stop) actions is learnable.
"""
from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Make the Week 4 package importable (sibling dir to week3/).
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WEEK4 = os.path.join(_REPO, "week4")
if _WEEK4 not in sys.path:
    sys.path.insert(0, _WEEK4)

from vlnverse_dataset import (  # noqa: E402  (path set above)
    DatasetConfig,
    build_frame_reader,
    load_split_records,
    populate_episode_from_parquet,
    resolve_episode,
)
from week4_actions import ACTION_TO_COMMAND, action_id_to_text  # noqa: E402


@dataclass
class Sample:
    episode_index: int
    timestep: int
    text: str
    target: Tuple[float, float, float]
    action_text: str


class VLNImageTextData:
    """Holds resolved episodes and a flat list of samples; fetches frames lazily."""

    def __init__(
        self,
        dataset_root: str,
        split_json: str,
        text_source: str = "instruction",   # "instruction" | "action"
        max_episodes: Optional[int] = None,
        per_class_cap: Optional[int] = None,
        image_size: int = 256,
        seed: int = 7,
    ):
        if text_source not in ("instruction", "action"):
            raise ValueError("text_source must be 'instruction' or 'action'")
        self.text_source = text_source
        self.image_size = int(image_size)
        self.seed = int(seed)
        self.cfg = DatasetConfig(
            dataset_root=dataset_root, split_json=split_json,
            image_size=image_size, include_terminal_stop=True, seed=seed,
        )

        records = load_split_records(split_json)
        self.episodes = []
        for rec in records:
            ep = resolve_episode(dataset_root, rec)
            if ep is None:
                continue
            if not populate_episode_from_parquet(ep, self.cfg):
                continue
            if not ep.action_ids:
                continue
            self.episodes.append(ep)
            if max_episodes is not None and len(self.episodes) >= max_episodes:
                break

        self.samples: List[Sample] = []
        for ei, ep in enumerate(self.episodes):
            instr = ep.instruction or ""
            for t, aid in enumerate(ep.action_ids):
                action_text = action_id_to_text(aid)
                vel = ACTION_TO_COMMAND[action_text]["velocity"]
                text = instr if text_source == "instruction" else action_text
                self.samples.append(
                    Sample(ei, t, text, (float(vel[0]), float(vel[1]), float(vel[2])), action_text)
                )

        if per_class_cap:
            self.samples = self._cap_per_class(self.samples, per_class_cap, seed)

    @staticmethod
    def _cap_per_class(samples: List[Sample], cap: int, seed: int) -> List[Sample]:
        by_cls = {}
        for s in samples:
            by_cls.setdefault(s.action_text, []).append(s)
        rng = random.Random(seed)
        out: List[Sample] = []
        for cls, items in by_cls.items():
            rng.shuffle(items)
            out.extend(items[:cap])
        rng.shuffle(out)
        return out

    # -- frame access -------------------------------------------------------
    def get_image(self, sample: Sample):
        ep = self.episodes[sample.episode_index]
        if ep._frame_reader is None:
            ep._frame_reader = build_frame_reader(ep.episode_dir)
        reader = ep._frame_reader
        if reader is None:
            return None
        img = reader.get(sample.timestep)
        if img is not None and self.image_size:
            img = img.resize((self.image_size, self.image_size))
        return img

    def class_counts(self) -> dict:
        counts: dict = {}
        for s in self.samples:
            counts[s.action_text] = counts.get(s.action_text, 0) + 1
        return counts

    def episode_split(self, val_ratio: float = 0.1) -> Tuple[List[int], List[int]]:
        """Episode-level split (indices into self.episodes), seed-stable."""
        idx = list(range(len(self.episodes)))
        rng = random.Random(self.seed)
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio)) if idx else 0
        val = set(idx[:n_val])
        return [i for i in idx if i not in val], sorted(val)
