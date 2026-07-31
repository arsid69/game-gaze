"""Movement-based (temporal) gestures.

Static classification answers "what shape is the hand in". This answers
"what did the hand just *do*" -- currently swipes, which the demo uses to
cycle through selectable objects.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from .config import MotionConfig


@dataclass
class Swipe:
    #: "left", "right", "up" or "down".
    direction: str
    #: Peak speed in NDC units per second.
    speed: float
    #: Total displacement along the dominant axis, in NDC units.
    travel: float


class SwipeDetector:
    """Detects a fast, straight, dominant-axis flick of the pointer.

    Works on smoothed NDC coordinates, so it is resolution independent. Three
    conditions must all hold: the hand moved far enough, fast enough, and much
    more along one axis than the other. A cooldown stops one physical swipe
    from firing several times.
    """

    def __init__(self, config: MotionConfig | None = None):
        self.config = config or MotionConfig()
        self._history: deque[tuple[float, float, float]] = deque()
        self._last_fired = -1e9

    def reset(self) -> None:
        self._history.clear()
        self._last_fired = -1e9

    def update(self, timestamp: float, x: float, y: float) -> Optional[Swipe]:
        self._history.append((timestamp, x, y))
        cutoff = timestamp - self.config.history_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        if len(self._history) < 3:
            return None
        if timestamp - self._last_fired < self.config.swipe_cooldown:
            return None

        t0, x0, y0 = self._history[0]
        dt = timestamp - t0
        if dt <= 1e-3:
            return None

        dx, dy = x - x0, y - y0
        adx, ady = abs(dx), abs(dy)

        if adx >= ady:
            travel, other, axis = adx, ady, "x"
        else:
            travel, other, axis = ady, adx, "y"

        if travel < self.config.swipe_min_travel:
            return None
        if travel / dt < self.config.swipe_min_speed:
            return None
        if travel < other * self.config.swipe_dominance:
            return None      # too diagonal to call it a clean swipe

        if axis == "x":
            direction = "right" if dx > 0 else "left"
        else:
            direction = "up" if dy > 0 else "down"

        self._last_fired = timestamp
        self._history.clear()
        return Swipe(direction=direction, speed=travel / dt, travel=travel)
