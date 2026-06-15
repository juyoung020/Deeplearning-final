"""Week 4 VLN-VERSE canonical actions, label mapping, parser and IsaacSim converter.

This module is intentionally dependency-free (standard library only) so that the
core action semantics can be unit-tested without downloading any model or dataset
and without importing torch / transformers / Isaac Sim.

Canonical actions (the model must emit EXACTLY one of these four strings):

    Move forward 25cm
    Turn right 15 degree
    Turn left 15 degree
    Stop

Label mapping for VLN-VERSE ``observation.action[t]`` (action that moves
``pose[t] -> pose[t+1]``; the training target at timestep ``t``):

    0 -> Stop
    1 -> Move forward 25cm
    2 -> Turn left 15 degree
    3 -> Turn right 15 degree

IsaacSim execution (linear m/s, angular rad/s):

    Move forward 25cm    -> duration 0.25s, [vx, vy, wz] = [1.0, 0.0,  0.0]
    Turn right 15 degree -> duration 0.25s, [vx, vy, wz] = [0.0, 0.0, -1.047]
    Turn left 15 degree  -> duration 0.25s, [vx, vy, wz] = [0.0, 0.0,  1.047]
    Stop                 -> terminate episode,            [0.0, 0.0,  0.0]

Sanity:  1.0 m/s * 0.25 s = 0.25 m;  1.047 rad/s * 0.25 s ~= 0.2618 rad ~= 15 deg.

IMPORTANT (per repo notes): Week 4 canonical actions must NOT be routed through the
Week 2 ``parse_language_velocity_command()``. Use ``action_to_command()`` here.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Canonical action strings (order is the canonical class order used for metrics)
# ---------------------------------------------------------------------------
MOVE_FORWARD = "Move forward 25cm"
TURN_RIGHT = "Turn right 15 degree"
TURN_LEFT = "Turn left 15 degree"
STOP = "Stop"

CANONICAL_ACTIONS: List[str] = [
    MOVE_FORWARD,
    TURN_RIGHT,
    TURN_LEFT,
    STOP,
]

# VLN-VERSE observation.action id -> canonical text (target at timestep t).
ACTION_ID_TO_TEXT: Dict[int, str] = {
    0: STOP,
    1: MOVE_FORWARD,
    2: TURN_LEFT,
    3: TURN_RIGHT,
}

TEXT_TO_ACTION_ID: Dict[str, int] = {v: k for k, v in ACTION_ID_TO_TEXT.items()}

# Default fallback used ONLY inside the simulator when the model emits an
# uninterpretable string. In offline accuracy an invalid output is always wrong.
INVALID_FALLBACK_ACTION = MOVE_FORWARD

# Turn rate magnitude: 15 degree over 0.25 s -> 1.0471975511965976 rad/s.
_TURN_RATE = math.radians(15.0) / 0.25  # ~= 1.0472 rad/s

# IsaacSim execution table. velocity = [vx, vy, wz].
ACTION_TO_COMMAND: Dict[str, Dict[str, object]] = {
    MOVE_FORWARD: {"duration": 0.25, "velocity": [1.0, 0.0, 0.0], "terminate": False},
    TURN_RIGHT: {"duration": 0.25, "velocity": [0.0, 0.0, -_TURN_RATE], "terminate": False},
    TURN_LEFT: {"duration": 0.25, "velocity": [0.0, 0.0, _TURN_RATE], "terminate": False},
    STOP: {"duration": 0.0, "velocity": [0.0, 0.0, 0.0], "terminate": True},
}


# ---------------------------------------------------------------------------
# action id -> canonical text
# ---------------------------------------------------------------------------
def action_id_to_text(action_id: int) -> str:
    """Map a VLN-VERSE ``observation.action`` id to its canonical action string.

    Raises KeyError on an unknown id so dataset bugs surface loudly rather than
    being silently mapped to a default.
    """
    return ACTION_ID_TO_TEXT[int(action_id)]


def is_canonical(text: str) -> bool:
    return text in ACTION_TO_COMMAND


# ---------------------------------------------------------------------------
# Canonical output parser
# ---------------------------------------------------------------------------
# Pre-compiled tolerant matchers. We first try an EXACT match (used for offline
# accuracy). ``parse_canonical_action`` additionally allows light normalization
# (trailing period, surrounding whitespace, case) so that closed-loop control is
# robust, while ``exact_match`` stays strict for reported accuracy numbers.
_NORMALIZE_PATTERNS = [
    (re.compile(r"^\s*move\s+forward\s+25\s*cm\s*\.?\s*$", re.IGNORECASE), MOVE_FORWARD),
    (re.compile(r"^\s*turn\s+right\s+15\s*degree[s]?\s*\.?\s*$", re.IGNORECASE), TURN_RIGHT),
    (re.compile(r"^\s*turn\s+left\s+15\s*degree[s]?\s*\.?\s*$", re.IGNORECASE), TURN_LEFT),
    (re.compile(r"^\s*stop\s*\.?\s*$", re.IGNORECASE), STOP),
]


def exact_match(text: str) -> Optional[str]:
    """Return the canonical action iff ``text`` is EXACTLY one of the four
    canonical strings (no normalization). Used for offline accuracy."""
    text = "" if text is None else str(text)
    return text if text in ACTION_TO_COMMAND else None


def parse_canonical_action(text: str, allow_normalization: bool = True) -> Optional[str]:
    """Parse generated text into a canonical action string.

    Returns one of the four canonical strings, or ``None`` if the text cannot be
    interpreted. With ``allow_normalization=True`` (the closed-loop default) a
    tolerant regex handles trailing periods, plural "degrees", extra whitespace
    and casing. With ``allow_normalization=False`` only an exact match is allowed.
    """
    if text is None:
        return None
    text = str(text)
    direct = exact_match(text)
    if direct is not None:
        return direct
    if not allow_normalization:
        return None
    # Only consider the first line; generation may append stray tokens.
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    for pattern, canonical in _NORMALIZE_PATTERNS:
        if pattern.match(first_line):
            return canonical
    return None


# ---------------------------------------------------------------------------
# IsaacSim action converter
# ---------------------------------------------------------------------------
class ConvertedAction:
    """Result of converting a (possibly invalid) model output to a command.

    Attributes:
        action_text: canonical action actually executed (after fallback).
        raw_text: the raw model output that was parsed.
        velocity: [vx, vy, wz] list of floats.
        duration_s: how long to hold the velocity (0.0 for Stop).
        terminate: True for Stop (episode should end).
        is_valid: whether ``raw_text`` parsed to a canonical action.
    """

    __slots__ = ("action_text", "raw_text", "velocity", "duration_s", "terminate", "is_valid")

    def __init__(self, action_text, raw_text, velocity, duration_s, terminate, is_valid):
        self.action_text = action_text
        self.raw_text = raw_text
        self.velocity = velocity
        self.duration_s = duration_s
        self.terminate = terminate
        self.is_valid = is_valid

    def __repr__(self):
        return (
            f"ConvertedAction(action_text={self.action_text!r}, valid={self.is_valid}, "
            f"velocity={self.velocity}, duration_s={self.duration_s}, terminate={self.terminate})"
        )


def action_to_command(
    raw_text: str,
    invalid_fallback: str = INVALID_FALLBACK_ACTION,
    allow_normalization: bool = True,
) -> ConvertedAction:
    """Convert a model output string into an IsaacSim command.

    Invalid outputs are mapped to ``invalid_fallback`` (simulation only) and
    flagged via ``is_valid=False`` so the caller can record the invalid-output
    rate. This deliberately bypasses the Week 2 rule-based parser.
    """
    parsed = parse_canonical_action(raw_text, allow_normalization=allow_normalization)
    is_valid = parsed is not None
    action_text = parsed if is_valid else invalid_fallback
    spec = ACTION_TO_COMMAND[action_text]
    return ConvertedAction(
        action_text=action_text,
        raw_text=raw_text,
        velocity=list(spec["velocity"]),
        duration_s=float(spec["duration"]),
        terminate=bool(spec["terminate"]),
        is_valid=is_valid,
    )


__all__ = [
    "MOVE_FORWARD",
    "TURN_RIGHT",
    "TURN_LEFT",
    "STOP",
    "CANONICAL_ACTIONS",
    "ACTION_ID_TO_TEXT",
    "TEXT_TO_ACTION_ID",
    "ACTION_TO_COMMAND",
    "INVALID_FALLBACK_ACTION",
    "action_id_to_text",
    "is_canonical",
    "exact_match",
    "parse_canonical_action",
    "action_to_command",
    "ConvertedAction",
]
