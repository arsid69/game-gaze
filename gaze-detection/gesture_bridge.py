"""gesture_bridge.py — adapts the gesture-control module to the gaze server's camera.

Why this file exists
--------------------
Both input systems want the webcam, and Windows will not hand index 0 to two
owners at once. ``gaze_server.py`` already runs a capture thread that keeps the
newest frame in a shared slot, so the fix is not to open a second camera but to
let the gesture engine read the frames the gaze server has already captured.

``GestureEngine.update(frame)`` is pure and synchronous, and ``HandTracker`` is
a public ABC that the module documents as its extension point for alternate
backends. So no part of the gesture module is modified or duplicated here — this
is only the adapter the module was designed to accept.

    gaze_server capture thread ──► _latest_frame ──► SharedFrameTracker
                                                          │
                                          MediaPipe Hands ▼
                                              GestureEngine (no scene)
                                                          │
                                                    POINTER_MOVE / GESTURE_* / SWIPE

Scene-less by design
--------------------
The engine ships its own Scene, raycasting, hover and selection. The game
already does all of that in three.js. Mirroring the scene over the wire every
frame would put selection logic in two places and let them disagree, so the
engine runs with an empty scene and we use it purely as an input device:
a smoothed pointer plus gesture labels. The game stays authoritative over what
is hovered and selected.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Callable, Optional

import numpy as np

# The gesture module lives outside this package, beside it in the repo root.
_MODULE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gesture-control-module",
)
if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)

from gesture_control import (EngineConfig, EventType, GestureEngine,  # noqa: E402
                             HandTracker, TrackerFrame)
# _select_backend is module-private, but it is the only way to run MediaPipe
# hand detection on a frame we already hold: MediaPipeHandTracker opens a camera
# in its constructor. The alternative was patching the gesture module to accept
# an external frame source; leaving it untouched is worth this one coupling.
# If it ever disappears upstream, this import is the single point to fix.
from gesture_control.tracker import _select_backend  # noqa: E402


class SharedFrameTracker(HandTracker):
    """A HandTracker fed by someone else's camera.

    ``read_frame`` returns ``(bgr_image, sequence)``. The sequence number lets us
    tell a genuinely new frame from the same one polled twice; we block until it
    changes rather than re-running detection on a stale image.

    Mirroring
    ---------
    The shared slot holds the camera frame **exactly as captured** — unmirrored.
    The gaze loop flips its own local copy, which never reaches the shared slot.
    So we mirror here, matching what ``MediaPipeHandTracker`` does standalone
    (``config.mirror`` defaults to True). Skipping this makes the cursor travel
    the opposite way to the hand and inverts the Left/Right labels.
    """

    def __init__(
        self,
        read_frame: Callable[[], tuple[Optional[np.ndarray], int]],
        config: Optional[EngineConfig] = None,
        detect_every: int = 1,
    ):
        import cv2

        self._cv2 = cv2
        self._read_frame = read_frame
        self._config = (config or EngineConfig()).tracker
        self._backend = _select_backend(self._config)
        self._detect_every = max(1, int(detect_every))
        self._counter = 0
        self._last_seq = -1
        self._closed = False

    def read(self) -> Optional[TrackerFrame]:
        """Next frame, or None once closed.

        Never returns None just because no frame is ready — the engine's loop
        treats None as "source exhausted" and shuts down. It waits instead.
        """
        while not self._closed:
            image, seq = self._read_frame()
            if image is None or seq == self._last_seq:
                time.sleep(0.002)          # nothing new yet; don't busy-spin
                continue
            self._last_seq = seq

            self._counter += 1
            if self._counter % self._detect_every:
                # Decimated tick: skip this frame outright. An earlier version
                # re-emitted the previous landmarks with a fresh timestamp, which
                # looked harmless but corrupted three consumers at once — the 1€
                # filter saw zero motion over elapsing time and over-smoothed,
                # the stabiliser counted a duplicate as new evidence, and the
                # swipe detector's speed was computed from repeated points.
                # Skipping simply lowers the rate; it never feeds a lie.
                continue

            # Mirror to match the standalone tracker: the shared slot is raw.
            mirrored = self._cv2.flip(image, 1) if self._config.mirror else image
            rgb = self._cv2.cvtColor(mirrored, self._cv2.COLOR_BGR2RGB)
            hands = self._backend.detect(rgb, self._config.mirror)
            return TrackerFrame(timestamp=time.monotonic(), hands=hands, image=None)
        return None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._backend.close()          # the camera belongs to the gaze server


def ndc_to_screen(ndc_x: float, ndc_y: float) -> tuple[float, float]:
    """Gesture NDC ([-1,1], y up) -> the gaze server's contract ([0,1], y down).

    This is the whole reason the two input systems can share one wire format:
    both end up describing "a point on the screen" in the same units, so the
    browser never has to know which device produced it.

    Clamped, because MediaPipe reports landmarks slightly outside [0,1] when the
    hand is partly out of frame. Unclamped, that puts the cursor beyond the
    viewport and every consumer has to defend itself. The gaze side already
    guarantees [0,1], so the gesture side guarantees it too and the contract
    holds no matter which device is driving.
    """
    x = min(max((ndc_x + 1.0) * 0.5, 0.0), 1.0)
    y = min(max((1.0 - ndc_y) * 0.5, 0.0), 1.0)
    return x, y


class GesturePointer:
    """Runs the gesture engine over shared frames and keeps the latest state.

    Mirrors how the gaze loop publishes into ``LATEST``: a background thread
    updates plain attributes, and the broadcaster reads whatever is current. The
    reader never blocks on the tracker.
    """

    #: Gestures the game reacts to. Anything else (including recorded custom
    #: gestures) still arrives in ``last_gesture`` for the browser to use.
    CLICK_GESTURE = "pinch"
    CLEAR_GESTURE = "open_palm"

    def __init__(self, read_frame=None, config: Optional[EngineConfig] = None,
                 detect_every: int = 1, frame_size: tuple[int, int] = (1280, 720),
                 own_camera: bool = False, camera_index: int = 0):
        """
        ``own_camera=True`` is the accurate path and the one the server uses.
        The engine builds its own ``MediaPipeHandTracker`` and runs the same
        sequential read->detect loop as the standalone module — no shared frames,
        no second thread competing for the GIL. It requires the camera to be free.

        ``own_camera=False`` feeds from someone else's capture thread via
        :class:`SharedFrameTracker`. Kept because it is the only way two
        pipelines can see the camera at once, but measured ~20% slower at
        detection, so the server no longer uses it.
        """
        self.config = config or EngineConfig()
        # Match the standalone demo, which also overrides these to the true
        # camera resolution rather than the 640x480 default.
        self.config.tracker.frame_width, self.config.tracker.frame_height = frame_size
        self.config.tracker.camera_index = camera_index
        self.engine = GestureEngine(config=self.config)     # no scene: see module docstring
        # None lets GestureEngine.start() build the stock MediaPipeHandTracker.
        self._tracker = (None if own_camera
                         else SharedFrameTracker(read_frame, self.config, detect_every))
        self._lock = threading.Lock()

        self.hand_ok = False
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.last_gesture = ""
        #: Monotonically increasing; the broadcaster sends a click when it moves.
        self.click_seq = 0
        self.clear_seq = 0
        self.swipe_seq = 0
        self.swipe_direction = ""
        self.updated_at = 0.0

        self.engine.subscribe(None, self._on_event)

    # -- event handling ---------------------------------------------------
    def _on_event(self, event) -> None:
        with self._lock:
            if event.ndc is not None:
                self.x, self.y = ndc_to_screen(*event.ndc)

            if event.type is EventType.HAND_DETECTED:
                self.hand_ok = True
            elif event.type is EventType.HAND_LOST:
                self.hand_ok = False
                self.x = self.y = None
                self.last_gesture = ""
            elif event.type is EventType.GESTURE_BEGIN:
                self.last_gesture = event.gesture
                if event.gesture == self.CLICK_GESTURE:
                    self.click_seq += 1
                elif event.gesture == self.CLEAR_GESTURE:
                    self.clear_seq += 1
            elif event.type is EventType.GESTURE_END:
                if event.gesture == self.last_gesture:
                    self.last_gesture = ""
            elif event.type is EventType.SWIPE:
                self.swipe_seq += 1
                self.swipe_direction = event.data.get("direction", "")

            self.updated_at = time.time()

    def snapshot(self) -> dict:
        """A self-consistent copy of the current pointer state."""
        with self._lock:
            return {
                "ok": self.hand_ok and self.x is not None,
                "x": round(self.x, 4) if self.x is not None else None,
                "y": round(self.y, 4) if self.y is not None else None,
                "gesture": self.last_gesture,
                "click_seq": self.click_seq,
                "clear_seq": self.clear_seq,
                "swipe_seq": self.swipe_seq,
                "swipe": self.swipe_direction,
            }

    # -- lifecycle --------------------------------------------------------
    def start(self) -> "GesturePointer":
        self.engine.start(self._tracker)
        return self

    def stop(self) -> None:
        self.engine.stop()
