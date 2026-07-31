"""Event types and the bus that delivers them to the game.

Two delivery styles, both fed by the same emit call:

* **callbacks** - ``bus.subscribe(EventType.SELECT, handler)``; fired
  immediately, good for reactive code.
* **polling**   - ``for e in bus.poll(): ...`` once per game tick; good for
  engines with a fixed update loop, and thread safe by design.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterator, Optional

import numpy as np


class EventType(str, Enum):
    # Tracking lifecycle
    HAND_DETECTED = "hand_detected"
    HAND_LOST = "hand_lost"

    # Static gestures
    GESTURE_BEGIN = "gesture_begin"
    GESTURE_END = "gesture_end"

    # Pointing / hovering
    POINTER_MOVE = "pointer_move"
    HOVER_ENTER = "hover_enter"
    HOVER_EXIT = "hover_exit"

    # Selection
    SELECT = "select"
    DESELECT = "deselect"
    CLEAR_SELECTION = "clear_selection"

    # Manipulation
    GRAB_BEGIN = "grab_begin"
    GRAB_MOVE = "grab_move"
    GRAB_END = "grab_end"

    # Movement gestures
    SWIPE = "swipe"


@dataclass
class GestureEvent:
    type: EventType
    timestamp: float
    #: "Right" / "Left" - which hand caused this.
    hand: str = ""
    #: Gesture label involved, when relevant.
    gesture: str = ""
    #: Scene object id involved, when relevant.
    object_id: Optional[str] = None
    #: World-space position, when relevant.
    position: Optional[np.ndarray] = None
    #: Pointer position in normalised device coords, x/y in [-1, 1].
    ndc: Optional[tuple[float, float]] = None
    data: dict = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - display only
        bits = [self.type.value]
        if self.hand:
            bits.append(f"hand={self.hand}")
        if self.gesture:
            bits.append(f"gesture={self.gesture}")
        if self.object_id:
            bits.append(f"object={self.object_id}")
        return f"<{' '.join(bits)}>"


Handler = Callable[[GestureEvent], None]


class EventBus:
    """Thread-safe fan-out of gesture events."""

    def __init__(self, max_queue: int = 512):
        self._handlers: dict[Optional[EventType], list[Handler]] = {}
        self._queue: deque[GestureEvent] = deque(maxlen=max_queue)
        self._lock = threading.Lock()

    def subscribe(self, event_type: Optional[EventType], handler: Handler) -> Handler:
        """Register ``handler``; pass ``event_type=None`` to receive everything.

        Returns the handler so it can be used as a decorator.
        """
        with self._lock:
            self._handlers.setdefault(event_type, []).append(handler)
        return handler

    def on(self, event_type: Optional[EventType] = None) -> Callable[[Handler], Handler]:
        """Decorator form of :meth:`subscribe`."""
        def wrap(handler: Handler) -> Handler:
            return self.subscribe(event_type, handler)
        return wrap

    def unsubscribe(self, event_type: Optional[EventType], handler: Handler) -> None:
        with self._lock:
            handlers = self._handlers.get(event_type)
            if handlers and handler in handlers:
                handlers.remove(handler)

    def emit(self, event: GestureEvent) -> None:
        with self._lock:
            self._queue.append(event)
            targets = list(self._handlers.get(event.type, ()))
            targets += list(self._handlers.get(None, ()))
        for handler in targets:
            handler(event)          # called outside the lock: handlers may emit

    def poll(self) -> Iterator[GestureEvent]:
        """Drain everything queued since the last call. Safe from any thread."""
        with self._lock:
            pending, self._queue = list(self._queue), deque(maxlen=self._queue.maxlen)
        return iter(pending)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
