"""GestureEngine - turns tracked hands into game-level interaction events.

Interaction model
-----------------
=================  ==========================================================
Gesture            Meaning
=================  ==========================================================
Point (index out)  Move the pointer; the object under the ray is *hovered*
Pinch              Select the hovered object and start dragging it
Pinch + move       Drag along the ray, keeping its distance from the camera
Release pinch      Drop the object
Open palm          Clear the selection
Swipe left/right   Cycle the selection through the scene
Custom gesture     Emitted as GESTURE_BEGIN / GESTURE_END for the game to use
=================  ==========================================================

``update()`` is pure and synchronous: feed it a :class:`TrackerFrame` from
your own loop. ``start()`` runs the camera on a background thread instead and
you drain ``engine.poll()`` once per game tick.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import features as F
from .config import EngineConfig
from .events import EventBus, EventType, GestureEvent
from .gestures import (Classifier, Gesture, GestureResult, GestureStabilizer,
                       build_classifier)
from .motion import SwipeDetector
from .pointer import (Camera, OneEuroFilter2D, Ray, Scene, image_to_ndc)
from .tracker import HandObservation, HandTracker, TrackerFrame


@dataclass
class HandState:
    """Per-hand interaction state; one of these per tracked hand."""

    hand_id: str
    stabilizer: GestureStabilizer
    filter: OneEuroFilter2D
    swipe: SwipeDetector
    gesture: str = Gesture.UNKNOWN
    ndc: tuple[float, float] = (0.0, 0.0)
    ray: Optional[Ray] = None
    hover_id: Optional[str] = None
    hover_distance: float = 0.0
    grab_id: Optional[str] = None
    grab_depth: float = 0.0
    grab_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    last_seen: float = 0.0
    raw: GestureResult = field(default_factory=GestureResult)


class GestureEngine:
    """Drives the whole pipeline and publishes :class:`GestureEvent` objects."""

    def __init__(
        self,
        scene: Optional[Scene] = None,
        camera: Optional[Camera] = None,
        config: Optional[EngineConfig] = None,
        classifier: Optional[Classifier] = None,
        tracker: Optional[HandTracker] = None,
    ):
        self.config = config or EngineConfig()
        self.scene = scene if scene is not None else Scene()
        self.camera = camera if camera is not None else Camera(
            position=np.array([0.0, 0.0, -6.0]),
            forward=np.array([0.0, 0.0, 1.0]),
            aspect=self.config.tracker.frame_width / self.config.tracker.frame_height,
        )
        self.classifier = classifier or build_classifier(self.config.gestures)
        self.bus = EventBus()
        self.tracker = tracker

        self._hands: dict[str, HandState] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.last_frame: Optional[TrackerFrame] = None

    # -- convenience passthroughs -----------------------------------------
    def subscribe(self, event_type, handler):
        return self.bus.subscribe(event_type, handler)

    def on(self, event_type=None):
        return self.bus.on(event_type)

    def poll(self):
        return self.bus.poll()

    def hand_state(self, hand_id: str) -> Optional[HandState]:
        return self._hands.get(hand_id)

    @property
    def hands(self) -> dict[str, HandState]:
        """Currently tracked hands, keyed by "Right" / "Left"."""
        return dict(self._hands)

    @property
    def primary(self) -> Optional[HandState]:
        """The hand driving the pointer, or None when nothing is tracked."""
        return next(iter(self._hands.values()), None)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def update(self, frame: TrackerFrame) -> None:
        """Process one tracker frame and emit whatever events it implies."""
        self.last_frame = frame
        observations = self._choose_hands(frame.hands)

        seen: set[str] = set()
        for obs in observations:
            hand_id = obs.handedness or "Right"
            seen.add(hand_id)
            state = self._hands.get(hand_id)
            if state is None:
                state = self._new_hand_state(hand_id)
                self._hands[hand_id] = state
                self._emit(EventType.HAND_DETECTED, frame.timestamp, state)
            state.last_seen = frame.timestamp
            self._update_hand(state, obs, frame.timestamp)

        self._expire_hands(seen, frame.timestamp)

    def _choose_hands(self, hands: list[HandObservation]) -> list[HandObservation]:
        """Apply the primary-hand / multi-hand policy."""
        if not hands:
            return []
        wanted = self.config.primary_hand
        if wanted in ("Right", "Left"):
            hands = [h for h in hands if h.handedness == wanted] or []
        if not self.config.multi_hand:
            # Highest detection score wins when several hands are visible.
            hands = sorted(hands, key=lambda h: h.score, reverse=True)[:1]
        return hands

    def _new_hand_state(self, hand_id: str) -> HandState:
        cfg = self.config
        return HandState(
            hand_id=hand_id,
            stabilizer=GestureStabilizer(cfg.gestures.stable_frames),
            filter=OneEuroFilter2D(cfg.pointer.min_cutoff, cfg.pointer.beta,
                                   cfg.pointer.d_cutoff),
            swipe=SwipeDetector(cfg.motion),
        )

    # ------------------------------------------------------------------
    # Per-hand pipeline
    # ------------------------------------------------------------------
    def _update_hand(self, state: HandState, obs: HandObservation, t: float) -> None:
        lm = obs.landmarks

        # 1. Classify the pose, then require it to hold for a few frames.
        raw = self.classifier.classify(lm, state.hand_id)
        state.raw = raw
        stable = state.stabilizer.push(raw)

        # 2. Pointer position: smoothed NDC from the index fingertip.
        px, py = self._pointer_source(lm)
        ndc_x, ndc_y = image_to_ndc(px, py)
        state.ndc = state.filter(ndc_x, ndc_y, t)
        state.ray = self.camera.ray_from_ndc(*state.ndc)
        self._emit(EventType.POINTER_MOVE, t, state, gesture=stable)

        # 3. Gesture transitions (fires select/grab logic).
        if stable != state.gesture:
            previous, state.gesture = state.gesture, stable
            if previous != Gesture.UNKNOWN:
                self._emit(EventType.GESTURE_END, t, state, gesture=previous)
                self._on_gesture_end(state, previous, t)
            if stable != Gesture.UNKNOWN:
                self._emit(EventType.GESTURE_BEGIN, t, state, gesture=stable,
                           data={"confidence": raw.confidence, "source": raw.source})
                self._on_gesture_begin(state, stable, t)

        # 4. Hover (skipped while dragging - the ray is busy carrying an object).
        if state.grab_id is None:
            self._update_hover(state, t)
        else:
            self._update_drag(state, t)

        # 5. Swipes, but not mid-drag (a drag looks a lot like a swipe).
        if state.grab_id is None:
            swipe = state.swipe.update(t, *state.ndc)
            if swipe is not None:
                self._emit(EventType.SWIPE, t, state,
                           data={"direction": swipe.direction,
                                 "speed": swipe.speed,
                                 "travel": swipe.travel})
                self._cycle_selection(state, swipe.direction, t)

    def _pointer_source(self, lm: np.ndarray) -> tuple[float, float]:
        """Which landmark the cursor follows: always the index fingertip.

        This used to switch landmark depending on the gesture - fingertip while
        pointing, thumb/index midpoint while pinching. That was a mistake. Every
        gesture change teleported the cursor by the gap between the two
        landmarks, and if it happened mid-drag the held object lurched with it.
        A single landmark for every pose means the cursor path stays continuous
        no matter what the classifier decides, which matters far more than
        picking the theoretically ideal point for each pose.

        The fingertip also happens to be the right choice for pinching: when
        thumb and index meet, the fingertip *is* the pinch point.
        """
        p = lm[F.INDEX_TIP]
        return float(p[0]), float(p[1])

    # ------------------------------------------------------------------
    # Interaction rules
    # ------------------------------------------------------------------
    def _on_gesture_begin(self, state: HandState, gesture: str, t: float) -> None:
        if gesture == Gesture.PINCH:
            self._begin_grab(state, t)
        elif gesture == Gesture.OPEN_PALM and self.scene.selected:
            self.scene.clear_selection()
            self._emit(EventType.CLEAR_SELECTION, t, state, gesture=gesture)

    def _on_gesture_end(self, state: HandState, gesture: str, t: float) -> None:
        if gesture == Gesture.PINCH:
            self._end_grab(state, t)

    def _update_hover(self, state: HandState, t: float) -> None:
        hit = (self.scene.pick(state.ray, self.config.pointer.grab_assist)
               if state.ray is not None else None)
        new_id = hit.object.id if hit else None
        state.hover_distance = hit.distance if hit else 0.0
        if new_id == state.hover_id:
            return
        if state.hover_id is not None:
            self._emit(EventType.HOVER_EXIT, t, state, object_id=state.hover_id)
        state.hover_id = new_id
        if hit is not None:
            self._emit(EventType.HOVER_ENTER, t, state, object_id=new_id,
                       position=hit.point,
                       data={"distance": hit.distance})

    def _begin_grab(self, state: HandState, t: float) -> None:
        """Pinch closed: select whatever the player was aiming at and pick it up."""
        if state.ray is None:
            return

        # Prefer the object that was already hovered. Closing the fingers takes
        # a moment and moves the hand slightly, so re-testing the ray at this
        # instant can miss a target the player was clearly aiming at.
        obj = self.scene.get(state.hover_id) if state.hover_id else None
        if obj is None:
            hit = self.scene.pick(state.ray, self.config.pointer.grab_assist)
            if hit is None:
                return
            obj = hit.object

        self.scene.select(obj.id, exclusive=True)
        self._emit(EventType.SELECT, t, state, object_id=obj.id,
                   position=obj.position.copy(), gesture=Gesture.PINCH)

        if not obj.grabbable:
            return

        state.grab_id = obj.id

        # Anchor on the object's centre, not on the surface point the ray hit.
        # Measuring depth to the surface leaves a lever arm between the anchor
        # and the centre, which swings the object around as the ray rotates.
        state.grab_depth = float(np.clip(
            np.linalg.norm(obj.position - self.camera.position),
            self.config.pointer.drag_depth_min,
            self.config.pointer.drag_depth_max))

        # Whatever is left over keeps the object exactly where it is right now,
        # so picking something up never makes it snap to the cursor.
        state.grab_offset = obj.position - state.ray.at(state.grab_depth)

        self._emit(EventType.GRAB_BEGIN, t, state, object_id=obj.id,
                   position=obj.position.copy(), gesture=Gesture.PINCH,
                   data={"depth": state.grab_depth})

    def _update_drag(self, state: HandState, t: float) -> None:
        obj = self.scene.get(state.grab_id) if state.grab_id else None
        if obj is None or state.ray is None:
            state.grab_id = None
            return

        target = state.ray.at(state.grab_depth) + state.grab_offset

        # Ease towards the target instead of snapping. The pointer is already
        # filtered, but a held object magnifies whatever noise is left, because
        # a fraction of a degree at the fingertip is centimetres out at depth.
        smoothing = float(np.clip(self.config.pointer.drag_smoothing, 0.0, 0.95))
        obj.position = (obj.position * smoothing + target * (1.0 - smoothing)
                        if smoothing > 0.0 else target)

        self._emit(EventType.GRAB_MOVE, t, state, object_id=obj.id,
                   position=obj.position.copy(), gesture=Gesture.PINCH)

    def _end_grab(self, state: HandState, t: float) -> None:
        if state.grab_id is None:
            return
        obj = self.scene.get(state.grab_id)
        self._emit(EventType.GRAB_END, t, state, object_id=state.grab_id,
                   position=obj.position.copy() if obj else None)
        state.grab_id = None
        state.grab_offset = np.zeros(3)

    def _cycle_selection(self, state: HandState, direction: str, t: float) -> None:
        """Swipe left/right steps the selection through the scene."""
        if direction not in ("left", "right"):
            return
        ids = self.scene.ids()
        if not ids:
            return
        step = 1 if direction == "right" else -1
        current = next(iter(self.scene.selected), None)
        index = (ids.index(current) + step) % len(ids) if current in ids else 0

        if current is not None:
            self._emit(EventType.DESELECT, t, state, object_id=current)
        new_id = ids[index]
        self.scene.select(new_id, exclusive=True)
        obj = self.scene.get(new_id)
        self._emit(EventType.SELECT, t, state, object_id=new_id,
                   position=obj.position.copy() if obj else None,
                   data={"via": "swipe"})

    # ------------------------------------------------------------------
    # Hand lifecycle
    # ------------------------------------------------------------------
    def _expire_hands(self, seen: set[str], t: float) -> None:
        timeout = self.config.pointer.hand_lost_timeout
        for hand_id in [h for h in self._hands if h not in seen]:
            state = self._hands[hand_id]
            if t - state.last_seen < timeout:
                continue                       # brief dropout, keep the state
            self._end_grab(state, t)
            if state.hover_id is not None:
                self._emit(EventType.HOVER_EXIT, t, state, object_id=state.hover_id)
            self._emit(EventType.HAND_LOST, t, state)
            if hasattr(self.classifier, "reset"):
                self.classifier.reset(hand_id)
            del self._hands[hand_id]

    def _emit(self, event_type: EventType, t: float, state: HandState,
              gesture: str = "", object_id: Optional[str] = None,
              position: Optional[np.ndarray] = None,
              data: Optional[dict] = None) -> None:
        self.bus.emit(GestureEvent(
            type=event_type,
            timestamp=t,
            hand=state.hand_id,
            gesture=gesture or state.gesture,
            object_id=object_id,
            position=position,
            ndc=state.ndc,
            data=data or {},
        ))

    # ------------------------------------------------------------------
    # Threaded camera loop
    # ------------------------------------------------------------------
    def start(self, tracker: Optional[HandTracker] = None) -> "GestureEngine":
        """Run the camera on a background thread. Drain events with ``poll()``."""
        if self._thread is not None:
            raise RuntimeError("engine already started")
        if tracker is not None:
            self.tracker = tracker
        if self.tracker is None:
            from .tracker import build_tracker
            self.tracker = build_tracker(self.config.tracker)

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="gesture-engine",
                                        daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            frame = self.tracker.read()
            if frame is None:
                break
            with self._lock:
                self.update(frame)
        self._stop.set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.tracker is not None:
            self.tracker.close()

    def __enter__(self) -> "GestureEngine":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def run(self, tracker: Optional[HandTracker] = None) -> None:
        """Blocking single-threaded loop, for scripts that have nothing else to do."""
        if tracker is not None:
            self.tracker = tracker
        if self.tracker is None:
            from .tracker import build_tracker
            self.tracker = build_tracker(self.config.tracker)
        try:
            for frame in self.tracker.frames():
                self.update(frame)
        finally:
            self.tracker.close()
