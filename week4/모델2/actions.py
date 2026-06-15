"""Week 4 — discrete VLN action space (per the official assignment spec).

VLN next-action prediction: a Qwen3-VL model sees egocentric RGB frames + a
navigation instruction and emits EXACTLY ONE of four action strings. Each action
maps deterministically to an IsaacSim velocity command held for a fixed duration.

Authoritative values come from the assignment ("Deep Learning Term Project
Week 4"). Do NOT change these — the grader matches the four strings exactly and
the simulator expects the exact velocity/duration below.

Action set (exactly four, "degree" singular):
    Move forward 25cm
    Turn left 15 degree
    Turn right 15 degree
    Stop

Action Execution Rule in IsaacSim:
    Move forward 25cm    -> [vx, vy, wz] = [1.0, 0.0,  0.000], 0.25 s   (1.0 m/s * 0.25 = 0.25 m)
    Turn left 15 degree  -> [0.0, 0.0,  1.047], 0.25 s                  (1.047 rad/s * 0.25 ~= 15 deg)
    Turn right 15 degree -> [0.0, 0.0, -1.047], 0.25 s
    Stop                 -> [0.0, 0.0,  0.000], terminate episode
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import numpy as np

# IsaacSim execution units (from the spec's Action Execution table).
FORWARD_VEL_MPS = 1.0
YAW_VEL_RAD_S = 1.047        # ~= 60 deg/s; * 0.25 s ~= 15 degree
ACTION_DURATION_S = 0.25

# The four canonical action strings — exact, order-stable.
ACTION_MOVE_FORWARD = "Move forward 25cm"
ACTION_TURN_LEFT = "Turn left 15 degree"
ACTION_TURN_RIGHT = "Turn right 15 degree"
ACTION_STOP = "Stop"

CANONICAL_ACTIONS = [
    ACTION_MOVE_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    ACTION_STOP,
]

# parquet observation.action id -> training target string (spec section 3).
ACTION_ID_TO_STRING = {
    0: ACTION_STOP,
    1: ACTION_MOVE_FORWARD,
    2: ACTION_TURN_LEFT,
    3: ACTION_TURN_RIGHT,
}

# Velocity command + duration for each canonical action.
_ACTION_TO_COMMAND = {
    ACTION_MOVE_FORWARD: (np.asarray([FORWARD_VEL_MPS, 0.0, 0.0], dtype=np.float32), ACTION_DURATION_S),
    ACTION_TURN_LEFT:    (np.asarray([0.0, 0.0,  YAW_VEL_RAD_S], dtype=np.float32), ACTION_DURATION_S),
    ACTION_TURN_RIGHT:   (np.asarray([0.0, 0.0, -YAW_VEL_RAD_S], dtype=np.float32), ACTION_DURATION_S),
    ACTION_STOP:         (np.zeros(3, dtype=np.float32), 0.0),
}

# Exact system prompt mandated by the spec (section 8).
ACTION_SYSTEM_PROMPT = (
    "You are a navigation action predictor for a mobile robot. "
    "Given the full navigation instruction and the egocentric image history, "
    "predict the next action. You must answer with exactly one of the following "
    "four actions:\n"
    "- Move forward 25cm\n"
    "- Turn right 15 degree\n"
    "- Turn left 15 degree\n"
    "- Stop"
)

# Default action substituted for unparseable model output (spec section 12).
INVALID_FALLBACK = ACTION_MOVE_FORWARD


def build_user_text(instruction: str) -> str:
    """The fixed user text block: instruction followed by the 'Answer:' cue."""
    return f"Instruction:\n{instruction.strip()}\n\nAnswer:"


def command_for(action: str):
    """Return (command np.float32[3], duration_s) for a canonical action string."""
    return _ACTION_TO_COMMAND[action]


def _normalize(text) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("°", " degree")
    text = re.sub(r"\bdegrees\b", "degree", text)
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .\t\r\n")


# Pre-normalized lookups for exact canonical matching.
_NORMALIZED_TO_CANONICAL = {_normalize(a): a for a in CANONICAL_ACTIONS}


def parse_action(raw_text):
    """Map raw model text to a canonical action.

    Returns a SimpleNamespace:
        kind      : "motion" | "stop" | "invalid"
        label     : canonical action string (or "invalid")
        command   : np.float32[3] = [vx, vy, yaw_rate]
        duration_s: hold seconds (0 for stop/invalid)
        valid     : whether the text matched one of the four canonical actions

    Exact offline matching per spec: anything not matching the four canonical
    strings is invalid (callers may substitute INVALID_FALLBACK for the sim).
    Never raises.
    """
    norm = _normalize(raw_text)

    label = None
    if norm in _NORMALIZED_TO_CANONICAL:
        label = _NORMALIZED_TO_CANONICAL[norm]
    else:
        # Lenient fallback: find a canonical action as a substring of the output
        # (covers models that wrap the answer in extra words). Stop is matched
        # last so "do not stop and move forward" still resolves to forward.
        for canon in (ACTION_MOVE_FORWARD, ACTION_TURN_LEFT, ACTION_TURN_RIGHT, ACTION_STOP):
            if _normalize(canon) in norm:
                label = canon
                break

    if label is None:
        return SimpleNamespace(kind="invalid", label="invalid", raw=raw_text,
                               command=np.zeros(3, dtype=np.float32), duration_s=0.0, valid=False)

    command, duration = _ACTION_TO_COMMAND[label]
    kind = "stop" if label == ACTION_STOP else "motion"
    # valid=True only for an exact canonical match (offline accuracy is exact-match).
    exact = norm in _NORMALIZED_TO_CANONICAL
    return SimpleNamespace(kind=kind, label=label, raw=raw_text,
                           command=command.astype(np.float32, copy=True),
                           duration_s=duration, valid=exact)
