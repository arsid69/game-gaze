"""Landmark geometry: normalisation and hand-pose features.

MediaPipe gives 21 landmarks per hand. Raw coordinates depend on where the
hand is on screen, how far away it is and how it is rotated -- useless for
classification. Everything here converts those raw points into a
*view-independent* description of the pose.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

# --------------------------------------------------------------------------
# MediaPipe Hands landmark indices
# --------------------------------------------------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

NUM_LANDMARKS = 21

#: (mcp, pip, dip, tip) for thumb, index, middle, ring, pinky.
FINGER_CHAINS: tuple[tuple[int, int, int, int], ...] = (
    (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

#: Bone connections, used for drawing the skeleton overlay.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

_EPS = 1e-9


def as_landmark_array(landmarks: Sequence) -> np.ndarray:
    """Coerce input into a validated ``(21, 3)`` float array."""
    arr = np.asarray(landmarks, dtype=np.float64)
    if arr.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected landmarks of shape (21, 3), got {arr.shape}")
    return arr


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > _EPS else np.zeros_like(v)


def hand_scale(lm: np.ndarray) -> float:
    """A rotation-invariant size estimate: wrist -> middle knuckle length.

    Used to turn absolute distances into ratios so that a hand held close to
    the camera and one held far away produce the same features.
    """
    span = float(np.linalg.norm(lm[MIDDLE_MCP] - lm[WRIST]))
    palm = float(np.linalg.norm(lm[INDEX_MCP] - lm[PINKY_MCP]))
    return max(span, palm, _EPS)


def palm_center(lm: np.ndarray) -> np.ndarray:
    """Centroid of the wrist and the four knuckles."""
    return lm[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]].mean(axis=0)


def canonical_basis(lm: np.ndarray) -> np.ndarray:
    """Orthonormal ``(3, 3)`` basis attached to the palm (rows = x, y, z).

    ``y`` points from the wrist towards the middle knuckle, ``x`` runs across
    the knuckles and ``z`` is the palm normal. Gram-Schmidt keeps it
    orthonormal even when the raw vectors are not perpendicular.
    """
    y = _unit(lm[MIDDLE_MCP] - lm[WRIST])
    across = lm[INDEX_MCP] - lm[PINKY_MCP]
    x = _unit(across - np.dot(across, y) * y)
    if not x.any():                      # degenerate: pick any perpendicular
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, y)) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0])
        x = _unit(fallback - np.dot(fallback, y) * y)
    z = np.cross(x, y)
    return np.stack([x, y, z])


def normalize_landmarks(lm: np.ndarray) -> np.ndarray:
    """Translate to the wrist, rotate into the palm frame, scale to unit size.

    The result is invariant to where the hand is, how big it looks and how it
    is rotated -- only the *pose* survives, which is exactly what a gesture is.
    """
    lm = as_landmark_array(lm)
    basis = canonical_basis(lm)
    centred = lm - lm[WRIST]
    return (centred @ basis.T) / hand_scale(lm)


def feature_vector(lm: np.ndarray) -> np.ndarray:
    """Flattened ``(63,)`` descriptor used by the custom-gesture classifier."""
    return normalize_landmarks(lm).reshape(-1)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Bend angle at ``b`` in degrees; 0 = perfectly straight."""
    v1, v2 = _unit(b - a), _unit(c - b)
    if not v1.any() or not v2.any():
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))))


def finger_curls(lm: np.ndarray) -> np.ndarray:
    """Bend angle in degrees for each of the five fingers (0 = straight)."""
    lm = as_landmark_array(lm)
    out = np.empty(5)
    for i, (mcp, pip, dip, tip) in enumerate(FINGER_CHAINS):
        # Average the two joint bends: more stable than either alone.
        out[i] = 0.5 * (joint_angle(lm[mcp], lm[pip], lm[dip])
                        + joint_angle(lm[pip], lm[dip], lm[tip]))
    return out


def fingers_extended(lm: np.ndarray, max_curl_deg: float = 48.0) -> np.ndarray:
    """Boolean ``(5,)`` array: is each finger straight? Order = FINGER_NAMES."""
    lm = as_landmark_array(lm)
    curls = finger_curls(lm)
    ext = curls < max_curl_deg

    # The thumb bends very little even when tucked across the palm, so add a
    # spatial test: an extended thumb sits far from the index knuckle.
    scale = hand_scale(lm)
    thumb_reach = float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_MCP])) / scale
    ext[0] = bool(ext[0] and thumb_reach > 0.45)
    return ext


def pinch_ratio(lm: np.ndarray) -> float:
    """Thumb-tip to index-tip distance in hand-size units.

    Roughly < 0.35 when pinched shut, > 0.6 when open.
    """
    lm = as_landmark_array(lm)
    return float(np.linalg.norm(lm[THUMB_TIP] - lm[INDEX_TIP])) / hand_scale(lm)


def spread_ratio(lm: np.ndarray) -> float:
    """Index-tip to pinky-tip distance in hand-size units (fingers splayed?)."""
    lm = as_landmark_array(lm)
    return float(np.linalg.norm(lm[INDEX_TIP] - lm[PINKY_TIP])) / hand_scale(lm)


def pinch_point(lm: np.ndarray) -> np.ndarray:
    """Midpoint between thumb and index tips -- where a pinch 'grabs'."""
    lm = as_landmark_array(lm)
    return (lm[THUMB_TIP] + lm[INDEX_TIP]) * 0.5


def describe(lm: np.ndarray) -> dict:
    """Human-readable feature dump; handy for debugging and the recorder UI."""
    ext = fingers_extended(lm)
    return {
        "extended": {n: bool(e) for n, e in zip(FINGER_NAMES, ext)},
        "curls_deg": {n: round(c, 1) for n, c in zip(FINGER_NAMES, finger_curls(lm))},
        "pinch_ratio": round(pinch_ratio(lm), 3),
        "spread_ratio": round(spread_ratio(lm), 3),
        "hand_scale": round(hand_scale(lm), 4),
    }
