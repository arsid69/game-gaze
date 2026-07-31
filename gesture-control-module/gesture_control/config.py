"""Configuration objects for the gesture control module.

Every tunable number in the pipeline lives here so that behaviour can be
adjusted without touching algorithm code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackerConfig:
    """Camera + MediaPipe Hands settings."""

    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    #: Flip the camera image horizontally so the feed behaves like a mirror.
    mirror: bool = True
    max_hands: int = 2
    #: 0 = fast/lite, 1 = accurate. Use 0 on low-end laptops.
    #: Legacy MediaPipe backend only; the Tasks API picks this per model file.
    model_complexity: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    #: "auto" picks the Tasks API when present and falls back to the legacy
    #: mp.solutions API. Force one with "tasks" or "legacy".
    backend: str = "auto"
    #: Path to the .task model file. None downloads and caches it under models/.
    model_path: Optional[str] = None


@dataclass
class GestureConfig:
    """Static gesture recognition settings."""

    #: Pinch closes below this thumb-tip/index-tip distance (hand-size units).
    pinch_close: float = 0.34
    #: ...and only re-opens above this one (hysteresis stops flickering).
    pinch_open: float = 0.48
    #: A finger counts as extended when its PIP joint bends less than this.
    extended_angle_deg: float = 48.0
    #: Consecutive identical frames required before a gesture is committed.
    stable_frames: int = 3
    #: k for the custom-gesture k-NN classifier.
    knn_k: int = 5
    #: Reject custom-gesture matches further away than this (feature units).
    knn_max_distance: float = 0.55
    #: Where recorded custom gestures are stored.
    dataset_path: str = "gestures_dataset.json"
    #: Custom gestures only override rules above this confidence.
    custom_min_confidence: float = 0.6


@dataclass
class MotionConfig:
    """Temporal (movement based) gesture settings, in NDC units."""

    swipe_min_speed: float = 1.4          # NDC units per second
    swipe_min_travel: float = 0.35        # total NDC displacement
    swipe_dominance: float = 1.8          # main axis must beat the other by this
    swipe_cooldown: float = 0.60          # seconds between swipes
    history_seconds: float = 0.35


@dataclass
class PointerConfig:
    """Pointer smoothing and drag behaviour."""

    # One Euro filter parameters.
    min_cutoff: float = 1.2
    beta: float = 0.02
    d_cutoff: float = 1.0
    #: Clamp for how near/far a dragged object may be pushed.
    drag_depth_min: float = 1.0
    drag_depth_max: float = 60.0
    #: Seconds without a detection before the hand is declared lost.
    hand_lost_timeout: float = 0.35
    #: 0 = the dragged object snaps straight to the cursor, 0.9 = very floaty.
    #: A little easing hides the last of the tracking noise.
    drag_smoothing: float = 0.45
    #: Grab assist. At the moment of the pinch, an object counts as "under the
    #: ray" if the ray passes within this multiple of its bounding radius, so a
    #: near miss still picks it up. 1.0 disables the assist.
    grab_assist: float = 1.7


@dataclass
class EngineConfig:
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    gestures: GestureConfig = field(default_factory=GestureConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    pointer: PointerConfig = field(default_factory=PointerConfig)
    #: "Right", "Left" or "any" - which hand drives the pointer.
    primary_hand: str = "any"
    #: Allow more than one hand to produce events at the same time.
    multi_hand: bool = False
