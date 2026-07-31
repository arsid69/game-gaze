"""Synthetic hand generator.

Builds plausible 21-landmark hands from a description of which fingers are
straight. Lets the classifier, engine and geometry be tested deterministically
with no webcam and no MediaPipe install.
"""

from __future__ import annotations

import numpy as np

from . import features as F

# Rest positions in canonical palm space (y up the hand, x across the
# knuckles, z out of the palm). Roughly matches real hand proportions.
_MCP = {
    F.INDEX_MCP: (0.26, 0.78, 0.00),
    F.MIDDLE_MCP: (0.02, 1.00, 0.00),
    F.RING_MCP: (-0.20, 0.94, -0.02),
    F.PINKY_MCP: (-0.40, 0.80, -0.04),
}
_SEGMENTS = {                     # proximal, middle, distal phalanx lengths
    F.INDEX_MCP: (0.34, 0.22, 0.16),
    F.MIDDLE_MCP: (0.38, 0.24, 0.17),
    F.RING_MCP: (0.34, 0.22, 0.16),
    F.PINKY_MCP: (0.26, 0.17, 0.13),
}
_CHAIN = {
    F.INDEX_MCP: (F.INDEX_PIP, F.INDEX_DIP, F.INDEX_TIP),
    F.MIDDLE_MCP: (F.MIDDLE_PIP, F.MIDDLE_DIP, F.MIDDLE_TIP),
    F.RING_MCP: (F.RING_PIP, F.RING_DIP, F.RING_TIP),
    F.PINKY_MCP: (F.PINKY_PIP, F.PINKY_DIP, F.PINKY_TIP),
}


def _rot_x(deg: float) -> np.ndarray:
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def make_hand(
    index: bool = True,
    middle: bool = True,
    ring: bool = True,
    pinky: bool = True,
    thumb: bool = True,
    pinch: bool = False,
    scale: float = 1.0,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Return a ``(21, 3)`` landmark array for the requested pose.

    ``pinch`` overrides the thumb and index tips so they touch.
    ``rotation`` is an optional ``(3, 3)`` matrix applied about the wrist,
    used to check that features really are rotation invariant.
    """
    lm = np.zeros((21, 3))
    lm[F.WRIST] = (0.0, 0.0, 0.0)

    straight = {
        F.INDEX_MCP: index,
        F.MIDDLE_MCP: middle,
        F.RING_MCP: ring,
        F.PINKY_MCP: pinky,
    }

    for mcp_idx, pos in _MCP.items():
        base = np.array(pos)
        lm[mcp_idx] = base
        pip_i, dip_i, tip_i = _CHAIN[mcp_idx]
        l1, l2, l3 = _SEGMENTS[mcp_idx]

        direction = _unit(base - lm[F.WRIST])
        pip = base + direction * l1
        lm[pip_i] = pip

        if straight[mcp_idx]:
            lm[dip_i] = pip + direction * l2
            lm[tip_i] = lm[dip_i] + direction * l3
        else:
            # Curl: bend ~85 deg at PIP and again at DIP, folding into the palm.
            d2 = _unit(_rot_x(-85.0) @ direction)
            lm[dip_i] = pip + d2 * l2
            d3 = _unit(_rot_x(-85.0) @ d2)
            lm[tip_i] = lm[dip_i] + d3 * l3

    # Thumb: splays out to the +x side of the palm.
    lm[F.THUMB_CMC] = (0.24, 0.14, 0.03)
    lm[F.THUMB_MCP] = (0.46, 0.34, 0.07)
    if thumb:
        tdir = _unit(lm[F.THUMB_MCP] - lm[F.THUMB_CMC])
        lm[F.THUMB_IP] = lm[F.THUMB_MCP] + tdir * 0.26
        lm[F.THUMB_TIP] = lm[F.THUMB_IP] + tdir * 0.21
    else:
        # Tucked across the palm, towards the middle knuckle.
        tdir = _unit(np.array(_MCP[F.MIDDLE_MCP]) - lm[F.THUMB_MCP])
        lm[F.THUMB_IP] = lm[F.THUMB_MCP] + tdir * 0.26
        lm[F.THUMB_TIP] = lm[F.THUMB_IP] + tdir * 0.21

    if pinch:
        # Bring thumb and index tips together just in front of the knuckles.
        meet = np.array([0.30, 0.95, 0.10])
        lm[F.INDEX_TIP] = meet + np.array([-0.015, 0.0, 0.0])
        lm[F.INDEX_DIP] = lm[F.INDEX_PIP] + _unit(meet - lm[F.INDEX_PIP]) * 0.20
        lm[F.THUMB_TIP] = meet + np.array([0.015, 0.0, 0.0])
        lm[F.THUMB_IP] = lm[F.THUMB_MCP] + _unit(meet - lm[F.THUMB_MCP]) * 0.22

    if rotation is not None:
        lm = lm @ np.asarray(rotation, dtype=np.float64).T

    return lm * scale + np.array(origin)


# Named poses, mirroring the labels in gestures.Gesture.
POSES = {
    "open_palm": dict(index=True, middle=True, ring=True, pinky=True, thumb=True),
    "fist": dict(index=False, middle=False, ring=False, pinky=False, thumb=False),
    "point": dict(index=True, middle=False, ring=False, pinky=False, thumb=False),
    "peace": dict(index=True, middle=True, ring=False, pinky=False, thumb=False),
    "thumbs_up": dict(index=False, middle=False, ring=False, pinky=False, thumb=True),
    "pinch": dict(index=True, middle=False, ring=False, pinky=False, thumb=True,
                  pinch=True),
}


def make_pose(name: str, **overrides) -> np.ndarray:
    """Build one of the named poses in :data:`POSES`."""
    if name not in POSES:
        raise KeyError(f"unknown pose {name!r}; options: {sorted(POSES)}")
    kwargs = dict(POSES[name])
    kwargs.update(overrides)
    return make_hand(**kwargs)


def to_image_space(
    lm: np.ndarray,
    center: tuple[float, float] = (0.5, 0.5),
    size: float = 0.25,
    anchor: int | None = None,
) -> np.ndarray:
    """Map a canonical hand into MediaPipe-style [0, 1] image coordinates.

    Image y grows downward, so the hand is flipped vertically on the way.
    ``anchor`` is a landmark index that should land exactly on ``center``
    (e.g. ``features.INDEX_TIP`` to aim the fingertip); by default the wrist
    is placed there.
    """
    out = lm.copy() * size
    out[:, 1] *= -1.0
    out[:, 2] *= size

    origin = out[anchor].copy() if anchor is not None else np.zeros(3)
    out[:, 0] += center[0] - origin[0]
    out[:, 1] += center[1] - origin[1]
    return out
