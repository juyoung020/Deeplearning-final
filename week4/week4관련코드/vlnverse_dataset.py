"""VLN-VERSE dataset loader for Week 4 next-action prediction.

Responsibilities
----------------
* Parse a split file (``fine_train.json.gz`` etc.) into a list of episode refs.
* Resolve each episode to its on-disk trajectory directory and parquet file.
* Read per-frame ``observation.action`` ids and egocentric RGB frames.
* Build one training sample per timestep ``t`` with an image history
  ``[I_{t-1}, I_t]`` (oldest -> newest), duplicating the first frame when there
  is no history (``t == 0`` -> ``[I_0, I_0]``).
* Map ``observation.action[t]`` to its canonical action string (the target).
* Provide an EPISODE-LEVEL train/validation split (never frame-level).

Design notes
------------
* No torch / pyarrow / decord import at module load time. Heavy deps are imported
  lazily inside the methods that need them, so unit tests and the action module
  stay importable on a minimal environment.
* The real VLN-VERSE trajectory format (LeRobot-style: parquet + per-episode RGB,
  either as a video or a folder of frames) is resolved defensively at runtime.
  Where the layout is ambiguous the resolver logs a warning and tries the next
  candidate, so the loader degrades gracefully instead of crashing.
* ``build_synthetic_dataset`` produces an in-memory dataset (random PIL images)
  so the full pipeline (collator, model forward, training step) can be smoke
  tested WITHOUT any download.

``timestamp`` in the parquet is treated as a frame/step index, not seconds.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from week4_actions import action_id_to_text

# Lazy PIL import marker; PIL is light and present in the env, but we still keep
# the import local to avoid hard-failing module import in odd environments.


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    dataset_root: str = ""
    split_json: str = ""           # path to fine_train.json.gz (or .json)
    image_size: int = 256          # frames resized to (image_size, image_size)
    history_count: int = 1         # number of history frames before I_t
    history_stride: int = 1        # gap between sampled history frames
    max_image_count: int = 2       # total images per sample (history + current)
    duplicate_first_when_missing_history: bool = True
    max_episodes: Optional[int] = None   # cap episodes (selective / dry runs)
    max_samples: Optional[int] = None    # cap total samples
    include_terminal_stop: bool = True   # ensure final row maps to Stop target
    # Trajectory-aware ablation (leakage-safe; see trajectory_features.py):
    include_trajectory_text: bool = False
    trajectory_history_len: int = 4
    seed: int = 7


# ---------------------------------------------------------------------------
# Episode reference + flat sample index
# ---------------------------------------------------------------------------
@dataclass
class EpisodeRef:
    scan: str
    episode_id: str
    episode_dir: str
    parquet_path: Optional[str]
    instruction: str = ""
    # Filled lazily after the parquet is read:
    num_frames: int = 0
    action_ids: List[int] = field(default_factory=list)
    # Optional per-frame pose for trajectory-aware ablation (leakage-safe use only)
    robot_xy: Optional[List[Tuple[float, float]]] = None
    robot_yaw: Optional[List[float]] = None
    _frame_reader: Optional["FrameReader"] = None


@dataclass
class FrameSampleRef:
    episode_index: int   # index into the dataset's episode list
    timestep: int        # t


# ---------------------------------------------------------------------------
# Split-file parsing
# ---------------------------------------------------------------------------
def load_split_records(split_json: str) -> List[Dict[str, Any]]:
    """Load a split file (``.json`` or ``.json.gz``) into a list of records.

    Accepts several shapes seen across VLN-VERSE dumps:
      * a top-level list of dicts;
      * a dict with an ``episodes`` / ``data`` / ``samples`` list;
      * a dict keyed by episode id.
    Each record is normalized to a dict; downstream resolution figures out the
    scan/episode fields heuristically.
    """
    opener = gzip.open if split_json.endswith(".gz") else open
    with opener(split_json, "rt", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for key in ("episodes", "data", "samples", "items"):
            if isinstance(obj.get(key), list):
                return [r for r in obj[key] if isinstance(r, dict)]
        # dict keyed by id -> list of records, injecting the key as episode id
        records = []
        for k, v in obj.items():
            if isinstance(v, dict):
                rec = dict(v)
                rec.setdefault("episode_id", k)
                records.append(rec)
        if records:
            return records
    raise ValueError(f"Unrecognized split-file structure in {split_json!r}")


def _record_field(record: Dict[str, Any], *names: str) -> Optional[Any]:
    for n in names:
        if n in record and record[n] not in (None, ""):
            return record[n]
    return None


# ---------------------------------------------------------------------------
# Episode resolution on disk
# ---------------------------------------------------------------------------
_TRAJ_SUBDIRS = (
    os.path.join("traj_data", "vlnverse"),
    "traj_data",
    "vlnverse",
    "",
)


def _candidate_episode_dirs(dataset_root: str, scan: str, episode_id: str) -> List[str]:
    cands = []
    for sub in _TRAJ_SUBDIRS:
        base = os.path.join(dataset_root, sub) if sub else dataset_root
        if scan:
            cands.append(os.path.join(base, scan, episode_id))
            cands.append(os.path.join(base, scan, str(episode_id)))
        cands.append(os.path.join(base, episode_id))
    # de-dup, keep order
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _find_parquet(episode_dir: str) -> Optional[str]:
    data_dir = os.path.join(episode_dir, "data")
    search_roots = [data_dir, episode_dir]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in sorted(filenames):
                if fn.endswith(".parquet"):
                    return os.path.join(dirpath, fn)
    return None


def _extract_instruction(record: Dict[str, Any]) -> str:
    instr = _record_field(record, "instruction", "instruction_text", "instructions",
                          "language", "text")
    if isinstance(instr, dict):  # {"instruction_text": ..., "instruction_tokens": ...}
        return str(instr.get("instruction_text") or instr.get("text") or "")
    if isinstance(instr, list):
        return str(instr[0]) if instr else ""
    return str(instr or "")


def _episode_subdir(scan: str, episode_id: str) -> str:
    """Local trajectory subdir for an episode.

    VLN-VERSE split ``episode_id`` is scan-prefixed (e.g. ``kujiale_0254_23_4``)
    while the directory is ``<scan>/23_4``. Strip the ``<scan>_`` prefix.
    """
    if scan and episode_id.startswith(scan + "_"):
        return episode_id[len(scan) + 1:]
    return episode_id


def resolve_episode(dataset_root: str, record: Dict[str, Any]) -> Optional[EpisodeRef]:
    """Resolve a split record to an on-disk :class:`EpisodeRef`, or ``None``."""
    scan = _record_field(record, "scan", "scan_id", "scene", "building") or ""
    episode_id = _record_field(
        record, "episode_id", "episode", "id", "traj_id", "trajectory_id", "ep_id"
    )
    instruction = _extract_instruction(record)
    if episode_id is None:
        return None
    scan = str(scan)
    episode_id = str(episode_id)
    subdir = _episode_subdir(scan, episode_id)

    # Try the scan-stripped subdir first, then the raw episode_id as fallback.
    candidates = []
    for eid in (subdir, episode_id):
        candidates.extend(_candidate_episode_dirs(dataset_root, scan, eid))
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.isdir(cand):
            parquet = _find_parquet(cand)
            return EpisodeRef(
                scan=scan,
                episode_id=episode_id,
                episode_dir=cand,
                parquet_path=parquet,
                instruction=instruction,
            )
    return None


# ---------------------------------------------------------------------------
# Frame readers
# ---------------------------------------------------------------------------
class FrameReader:
    """Abstract per-episode RGB frame reader returning PIL.Image (RGB)."""

    def get(self, frame_index: int):  # pragma: no cover - interface
        raise NotImplementedError

    def __len__(self):  # pragma: no cover - interface
        raise NotImplementedError


class FolderFrameReader(FrameReader):
    """Reads ``frame_000000.jpg``/``.png`` style frames from a directory."""

    def __init__(self, frame_paths: Sequence[str]):
        self.frame_paths = list(frame_paths)

    def get(self, frame_index: int):
        from PIL import Image

        idx = max(0, min(frame_index, len(self.frame_paths) - 1))
        return Image.open(self.frame_paths[idx]).convert("RGB")

    def __len__(self):
        return len(self.frame_paths)


class VideoFrameReader(FrameReader):
    """Reads frames from a single mp4 (LeRobot ``observation.images.rgb``).

    Uses ``decord`` if available, otherwise ``imageio``/``av`` via imageio. The
    decoder is created lazily and cached.
    """

    def __init__(self, video_path: str, num_frames_hint: int = 0):
        self.video_path = video_path
        self._backend = None
        self._reader = None
        self._len = num_frames_hint

    def _ensure_reader(self):
        if self._reader is not None:
            return
        try:
            import decord  # type: ignore

            self._reader = decord.VideoReader(self.video_path)
            self._backend = "decord"
            self._len = len(self._reader)
            return
        except Exception:
            pass
        import imageio.v3 as iio  # type: ignore

        # Read lazily frame-by-frame to avoid loading the whole clip into RAM.
        self._reader = iio
        self._backend = "imageio"
        if self._len == 0:
            try:
                meta = iio.immeta(self.video_path, plugin="pyav")
                fps = meta.get("fps")
                dur = meta.get("duration")
                if fps and dur:
                    self._len = int(round(fps * dur))
            except Exception:
                self._len = 0

    def get(self, frame_index: int):
        from PIL import Image

        self._ensure_reader()
        if self._backend == "decord":
            idx = max(0, min(frame_index, len(self._reader) - 1))
            arr = self._reader[idx].asnumpy()
            return Image.fromarray(arr).convert("RGB")
        # imageio: index-based read
        import imageio.v3 as iio

        try:
            arr = iio.imread(self.video_path, index=frame_index, plugin="pyav")
        except Exception:
            arr = iio.imread(self.video_path, index=0, plugin="pyav")
        return Image.fromarray(arr).convert("RGB")

    def __len__(self):
        self._ensure_reader()
        return self._len


class NpyFrameReader(FrameReader):
    """Reads frames from a stacked ``rgb.npy`` array of shape (T, H, W, 3) uint8.

    This is the actual VLN-VERSE on-disk format
    (``videos/chunk-000/observation.images.rgb/rgb.npy``). The array is memory
    mapped so only the requested frame is materialized.
    """

    def __init__(self, npy_path: str):
        self.npy_path = npy_path
        self._arr = None

    def _ensure(self):
        if self._arr is None:
            import numpy as np

            self._arr = np.load(self.npy_path, mmap_mode="r")

    def get(self, frame_index: int):
        from PIL import Image
        import numpy as np

        self._ensure()
        n = self._arr.shape[0]
        idx = max(0, min(frame_index, n - 1))
        frame = np.ascontiguousarray(self._arr[idx])
        if frame.ndim == 3 and frame.shape[2] >= 3:
            frame = frame[:, :, :3]
        return Image.fromarray(frame.astype("uint8"), "RGB")

    def __len__(self):
        self._ensure()
        return int(self._arr.shape[0])


def build_frame_reader(episode_dir: str) -> Optional[FrameReader]:
    """Locate the egocentric RGB source for an episode and return a reader."""
    # 0) stacked rgb.npy (the actual VLN-VERSE format)
    for dirpath, _d, filenames in os.walk(episode_dir):
        if "observation.images.rgb" in dirpath or dirpath.endswith("rgb"):
            for fn in sorted(filenames):
                if fn.lower().endswith(".npy"):
                    return NpyFrameReader(os.path.join(dirpath, fn))
    # 1) folder of frames under videos/.../observation.images.rgb/<ep>/
    rgb_keys = ("observation.images.rgb", "observation.image.rgb", "rgb", "images.rgb")
    for dirpath, dirnames, filenames in os.walk(episode_dir):
        base = os.path.basename(dirpath)
        # folder of image frames
        frames = sorted(
            os.path.join(dirpath, fn)
            for fn in filenames
            if fn.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if frames and any(k in dirpath for k in rgb_keys):
            return FolderFrameReader(frames)
        # single video file
        for fn in sorted(filenames):
            if fn.lower().endswith((".mp4", ".mkv", ".avi")) and any(k in dirpath for k in rgb_keys):
                return VideoFrameReader(os.path.join(dirpath, fn))
        del base, dirnames
    # 2) fallback: any video named rgb anywhere in the episode dir
    for dirpath, _d, filenames in os.walk(episode_dir):
        for fn in sorted(filenames):
            if fn.lower().endswith(".mp4"):
                return VideoFrameReader(os.path.join(dirpath, fn))
    return None


# ---------------------------------------------------------------------------
# Parquet reading
# ---------------------------------------------------------------------------
_ACTION_COLS = ("observation.action", "action", "observation.actions", "actions")
_STEP_COLS = ("timestamp", "observation.step", "step", "frame_index", "index")
_POS_COLS = ("observation.robot_position", "observation.camera_position", "robot_position")
_YAW_COLS = ("observation.robot_yaw", "observation.camera_yaw", "robot_yaw")


def _first_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cset = set(columns)
    for c in candidates:
        if c in cset:
            return c
    return None


def read_episode_table(parquet_path: str):
    """Read a parquet episode into a list-of-rows dict. Lazily imports pandas."""
    import pandas as pd  # requires pyarrow or fastparquet engine

    df = pd.read_parquet(parquet_path)
    return df


def _parse_vec(value: Any) -> List[float]:
    """Parse a vector cell that may be a list/ndarray or a string repr "[x,y,z]"."""
    if isinstance(value, str):
        import ast

        try:
            parsed = ast.literal_eval(value)
            return [float(x) for x in parsed]
        except Exception:
            return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return [float(x) for x in value.reshape(-1)]
    except Exception:
        pass
    try:
        return [float(value)]
    except Exception:
        return []


def _scalar_action(value: Any) -> int:
    """Coerce an action cell to an int id (handles scalars / length-1 arrays)."""
    if isinstance(value, (list, tuple)):
        return int(value[0])
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return int(value.reshape(-1)[0])
    except Exception:
        pass
    return int(value)


def populate_episode_from_parquet(ep: EpisodeRef, cfg: DatasetConfig) -> bool:
    """Fill ``ep.action_ids`` / frame reader / optional pose from parquet.

    Returns True on success. The final row is forced to ``Stop`` when
    ``cfg.include_terminal_stop`` and the data does not already end in 0.
    """
    if not ep.parquet_path or not os.path.exists(ep.parquet_path):
        warnings.warn(f"No parquet for episode {ep.scan}/{ep.episode_id}")
        return False
    df = read_episode_table(ep.parquet_path)
    cols = list(df.columns)
    action_col = _first_col(cols, _ACTION_COLS)
    if action_col is None:
        warnings.warn(f"No action column in {ep.parquet_path}; columns={cols}")
        return False

    action_ids = [_scalar_action(v) for v in df[action_col].tolist()]
    if cfg.include_terminal_stop and action_ids and action_ids[-1] != 0:
        action_ids.append(0)  # terminal Stop target
    ep.action_ids = action_ids
    ep.num_frames = len(action_ids)

    # Optional pose columns for trajectory-aware ablation. VLN-VERSE stores
    # position/orientation as STRING reprs like "[x,y,z]"; parse defensively.
    pos_col = _first_col(cols, _POS_COLS)
    yaw_col = _first_col(cols, _YAW_COLS)
    if pos_col is not None:
        xy = []
        for v in df[pos_col].tolist():
            vec = _parse_vec(v)
            xy.append((float(vec[0]), float(vec[1])) if len(vec) >= 2 else (0.0, 0.0))
        ep.robot_xy = xy
    if yaw_col is not None:
        ep.robot_yaw = [float(v) for v in df[yaw_col].tolist()]

    ep._frame_reader = build_frame_reader(ep.episode_dir)
    if ep._frame_reader is None:
        warnings.warn(f"No RGB frame source found under {ep.episode_dir}")
    return True


# ---------------------------------------------------------------------------
# History selection
# ---------------------------------------------------------------------------
def select_history_indices(t: int, cfg: DatasetConfig) -> List[int]:
    """Return frame indices (oldest -> newest) ending at ``t``.

    Baseline (history_count=1, stride=1, max_image_count=2): ``[t-1, t]`` and for
    ``t == 0`` -> ``[0, 0]`` (first frame duplicated). The list always has length
    ``min(history_count+1, max_image_count)`` and is clamped to >= 0.
    """
    total = min(cfg.history_count + 1, cfg.max_image_count)
    raw = [t - i * cfg.history_stride for i in range(total)]  # newest..oldest
    raw = list(reversed(raw))  # oldest..newest, last == t
    if cfg.duplicate_first_when_missing_history:
        clamped = [idx if idx >= 0 else 0 for idx in raw]
    else:
        clamped = [max(idx, 0) for idx in raw]
    return clamped


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class VLNVerseDataset:
    """Flat per-timestep dataset. ``__getitem__`` returns a dict:

        {
          "instruction": str,
          "images": [PIL.Image, ...]  (oldest -> newest, resized),
          "answer": str  (canonical action),
          "action_id": int,
          "episode_id": str,
          "scan": str,
          "timestep": int,
          "trajectory_text": Optional[str],
        }

    Works as a ``torch.utils.data.Dataset`` (implements __len__/__getitem__)
    without importing torch.
    """

    def __init__(self, episodes: List[EpisodeRef], cfg: DatasetConfig):
        self.episodes = episodes
        self.cfg = cfg
        self.index: List[FrameSampleRef] = []
        self._build_index()

    def _build_index(self):
        self.index = []
        for ei, ep in enumerate(self.episodes):
            for t in range(ep.num_frames):
                self.index.append(FrameSampleRef(episode_index=ei, timestep=t))
                if self.cfg.max_samples and len(self.index) >= self.cfg.max_samples:
                    return

    def __len__(self):
        return len(self.index)

    def _load_image(self, ep: EpisodeRef, frame_index: int):
        from PIL import Image

        size = (self.cfg.image_size, self.cfg.image_size)
        if ep._frame_reader is not None:
            img = ep._frame_reader.get(frame_index)
        else:
            img = Image.new("RGB", size, color=(0, 0, 0))
        if img.size != size:
            img = img.resize(size)
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ref = self.index[idx]
        ep = self.episodes[ref.episode_index]
        t = ref.timestep
        hist_idx = select_history_indices(t, self.cfg)
        images = [self._load_image(ep, fi) for fi in hist_idx]
        action_id = ep.action_ids[t]
        answer = action_id_to_text(action_id)

        traj_text = None
        if self.cfg.include_trajectory_text:
            from trajectory_features import build_trajectory_text

            traj_text = build_trajectory_text(ep, t, self.cfg.trajectory_history_len)

        return {
            "instruction": ep.instruction,
            "images": images,
            "answer": answer,
            "action_id": int(action_id),
            "episode_id": ep.episode_id,
            "scan": ep.scan,
            "timestep": t,
            "trajectory_text": traj_text,
        }


# ---------------------------------------------------------------------------
# Build from disk
# ---------------------------------------------------------------------------
def build_episodes_from_split(cfg: DatasetConfig) -> List[EpisodeRef]:
    records = load_split_records(cfg.split_json)
    episodes: List[EpisodeRef] = []
    skipped = 0
    for rec in records:
        if cfg.max_episodes is not None and len(episodes) >= cfg.max_episodes:
            break
        ep = resolve_episode(cfg.dataset_root, rec)
        if ep is None:
            skipped += 1
            continue
        ok = populate_episode_from_parquet(ep, cfg)
        if not ok or ep.num_frames == 0:
            skipped += 1
            continue
        episodes.append(ep)
    if skipped:
        warnings.warn(f"build_episodes_from_split: skipped {skipped} unresolved episodes")
    return episodes


def build_dataset_from_split(cfg: DatasetConfig) -> VLNVerseDataset:
    episodes = build_episodes_from_split(cfg)
    return VLNVerseDataset(episodes, cfg)


# ---------------------------------------------------------------------------
# Episode-level split (NEVER frame-level)
# ---------------------------------------------------------------------------
def episode_level_split(
    episodes: List[EpisodeRef],
    val_ratio: float = 0.1,
    seed: int = 7,
    forced_val_episode_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[EpisodeRef], List[EpisodeRef]]:
    """Split episodes (not frames) into train/val.

    ``forced_val_episode_ids`` (e.g. the val split of
    ``vlnverse_closed_loop_eval_20episodes.json``) are always placed in val.
    """
    forced = set(str(x) for x in (forced_val_episode_ids or []))
    forced_eps = [e for e in episodes if e.episode_id in forced]
    rest = [e for e in episodes if e.episode_id not in forced]

    rng = random.Random(seed)
    rest_shuffled = rest[:]
    rng.shuffle(rest_shuffled)
    n_val = max(0, int(round(len(rest_shuffled) * val_ratio)))
    val = forced_eps + rest_shuffled[:n_val]
    train = rest_shuffled[n_val:]
    return train, val


# ---------------------------------------------------------------------------
# Synthetic dataset (NO download) for smoke tests
# ---------------------------------------------------------------------------
class _SyntheticFrameReader(FrameReader):
    def __init__(self, num_frames: int, size: int, seed: int):
        self.num_frames = num_frames
        self.size = size
        self.seed = seed

    def get(self, frame_index: int):
        from PIL import Image
        import numpy as np

        rng = np.random.RandomState((self.seed * 1000003 + frame_index) % (2**31))
        arr = rng.randint(0, 256, size=(self.size, self.size, 3), dtype=np.uint8)
        return Image.fromarray(arr, "RGB")

    def __len__(self):
        return self.num_frames


def build_synthetic_dataset(
    num_episodes: int = 2,
    frames_per_episode: int = 6,
    cfg: Optional[DatasetConfig] = None,
) -> VLNVerseDataset:
    """In-memory dataset with random images. Used by smoke tests only."""
    cfg = cfg or DatasetConfig(image_size=64)
    rng = random.Random(cfg.seed)
    instructions = [
        "Turn right and proceed toward the doorway beside the wall-mounted TV.",
        "Walk forward through the hallway and stop at the kitchen entrance.",
        "Go straight, then turn left at the sofa and continue to the window.",
    ]
    episodes: List[EpisodeRef] = []
    for e in range(num_episodes):
        n = frames_per_episode
        # random action ids 0..3, force last to 0 (Stop)
        action_ids = [rng.randint(1, 3) for _ in range(n - 1)] + [0]
        xy = [(0.25 * i, 0.0) for i in range(n)]
        yaw = [0.0 for _ in range(n)]
        ep = EpisodeRef(
            scan=f"kujiale_{e:04d}",
            episode_id=f"{e}_0",
            episode_dir="<synthetic>",
            parquet_path=None,
            instruction=instructions[e % len(instructions)],
            num_frames=n,
            action_ids=action_ids,
            robot_xy=xy,
            robot_yaw=yaw,
        )
        ep._frame_reader = _SyntheticFrameReader(n, cfg.image_size, seed=cfg.seed + e)
        episodes.append(ep)
    return VLNVerseDataset(episodes, cfg)


__all__ = [
    "DatasetConfig",
    "EpisodeRef",
    "VLNVerseDataset",
    "load_split_records",
    "resolve_episode",
    "build_frame_reader",
    "populate_episode_from_parquet",
    "select_history_indices",
    "build_episodes_from_split",
    "build_dataset_from_split",
    "episode_level_split",
    "build_synthetic_dataset",
]
