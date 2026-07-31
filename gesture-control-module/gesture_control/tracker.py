"""Hand tracking backends.

``HandTracker`` is the interface the rest of the module depends on;
``MediaPipeHandTracker`` is the concrete webcam implementation. Keeping them
apart means a Leap Motion / depth-camera / replay backend can be dropped in
later without touching the gesture or interaction code.

Two MediaPipe APIs are supported, because Google changed theirs:

* **Tasks API** (``mediapipe.tasks``) - the current, supported one. Needs a
  model file, which is downloaded automatically the first time you run.
* **Legacy solutions API** (``mp.solutions.hands``) - removed in MediaPipe
  releases after 0.10.9, kept here so older installs still work.

The right one is picked automatically. MediaPipe and OpenCV are imported
lazily so the rest of the package (and the test suite) works on a machine
with neither installed.
"""

from __future__ import annotations

import os
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from .config import TrackerConfig
from .features import NUM_LANDMARKS

#: Google's pre-trained hand landmark model for the Tasks API (~7 MB).
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass
class HandObservation:
    """One hand seen in one frame."""

    #: (21, 3) landmarks in normalised image space: x, y in [0, 1], z is a
    #: relative depth where smaller means closer to the camera.
    landmarks: np.ndarray
    #: (21, 3) metric landmarks in metres, origin at the hand centre. May be None.
    world_landmarks: Optional[np.ndarray] = None
    handedness: str = "Right"
    score: float = 1.0

    def __post_init__(self) -> None:
        self.landmarks = np.asarray(self.landmarks, dtype=np.float64)
        if self.landmarks.shape != (NUM_LANDMARKS, 3):
            raise ValueError("landmarks must be (21, 3)")


@dataclass
class TrackerFrame:
    """Everything the tracker produced for a single camera frame."""

    timestamp: float
    hands: list[HandObservation] = field(default_factory=list)
    #: BGR image, already mirrored if mirroring is on. May be None (headless).
    image: Optional[np.ndarray] = None

    def hand(self, handedness: str) -> Optional[HandObservation]:
        for h in self.hands:
            if h.handedness == handedness:
                return h
        return None


class HandTracker(ABC):
    """Minimal interface a tracking backend has to satisfy."""

    @abstractmethod
    def read(self) -> Optional[TrackerFrame]:
        """Return the next frame, or None when the source is exhausted."""

    def close(self) -> None:  # pragma: no cover - trivial
        pass

    def frames(self) -> Iterator[TrackerFrame]:
        """Iterate until the source runs dry."""
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# --------------------------------------------------------------------------
# Model file management (Tasks API only)
# --------------------------------------------------------------------------
def default_model_path() -> str:
    """Where the landmark model is cached: ``<project>/models/``."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "models", "hand_landmarker.task")


def ensure_model(path: Optional[str] = None) -> str:
    """Return a path to the landmark model, downloading it if necessary."""
    path = path or default_model_path()
    if os.path.exists(path):
        return path

    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading the hand landmark model (~7 MB) to {path} ...")
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, path)
    except Exception as exc:
        if os.path.exists(path):
            os.remove(path)          # never leave a truncated file behind
        raise RuntimeError(
            "Could not download the hand landmark model.\n"
            f"  from: {HAND_LANDMARKER_URL}\n"
            f"  to:   {path}\n"
            "Download it manually in a browser, save it to that path, and run again."
        ) from exc
    print("Model downloaded.")
    return path


# --------------------------------------------------------------------------
# MediaPipe backends
# --------------------------------------------------------------------------
class _TasksBackend:
    """MediaPipe Tasks API (``mediapipe.tasks.python.vision.HandLandmarker``)."""

    def __init__(self, config: TrackerConfig):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model_path = ensure_model(config.model_path)

        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.max_hands,
            min_hand_detection_confidence=config.min_detection_confidence,
            min_hand_presence_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, rgb: np.ndarray, mirrored: bool) -> list[HandObservation]:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode demands strictly increasing millisecond timestamps.
        ms = max(int(time.monotonic() * 1000), self._last_ms + 1)
        self._last_ms = ms

        result = self._landmarker.detect_for_video(image, ms)

        hands: list[HandObservation] = []
        for i, points in enumerate(result.hand_landmarks or []):
            lm = np.array([[p.x, p.y, p.z] for p in points])

            world = None
            if result.hand_world_landmarks and i < len(result.hand_world_landmarks):
                world = np.array([[p.x, p.y, p.z]
                                  for p in result.hand_world_landmarks[i]])

            label, score = "Right", 1.0
            if result.handedness and i < len(result.handedness):
                category = result.handedness[i][0]
                label = category.category_name
                score = float(category.score)
                if not mirrored:
                    label = "Left" if label == "Right" else "Right"

            hands.append(HandObservation(lm, world, label, score))
        return hands

    def close(self) -> None:
        self._landmarker.close()


class _LegacyBackend:
    """The old ``mp.solutions.hands`` API, for MediaPipe 0.10.9 and earlier."""

    def __init__(self, config: TrackerConfig):
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=config.max_hands,
            model_complexity=config.model_complexity,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        )

    def detect(self, rgb: np.ndarray, mirrored: bool) -> list[HandObservation]:
        rgb.flags.writeable = False
        result = self._hands.process(rgb)

        hands: list[HandObservation] = []
        if not result.multi_hand_landmarks:
            return hands

        world_sets = result.multi_hand_world_landmarks or []
        for i, hand_lms in enumerate(result.multi_hand_landmarks):
            lm = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark])

            world = None
            if i < len(world_sets):
                world = np.array([[p.x, p.y, p.z] for p in world_sets[i].landmark])

            label, score = "Right", 1.0
            if result.multi_handedness and i < len(result.multi_handedness):
                cls = result.multi_handedness[i].classification[0]
                label, score = cls.label, float(cls.score)
                if not mirrored:
                    label = "Left" if label == "Right" else "Right"

            hands.append(HandObservation(lm, world, label, score))
        return hands

    def close(self) -> None:
        self._hands.close()


def _select_backend(config: TrackerConfig):
    """Build whichever MediaPipe backend this installation actually supports."""
    try:
        import mediapipe as mp
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "MediaPipeHandTracker needs 'mediapipe' and 'opencv-python'.\n"
            "Install them with:  python -m pip install -r requirements.txt"
        ) from exc

    has_tasks = hasattr(mp, "tasks")
    has_legacy = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")

    wanted = config.backend
    if wanted == "tasks" or (wanted == "auto" and has_tasks):
        if not has_tasks:
            raise RuntimeError("This MediaPipe build has no Tasks API.")
        return _TasksBackend(config)

    if wanted == "legacy" or (wanted == "auto" and has_legacy):
        if not has_legacy:
            raise RuntimeError(
                "This MediaPipe version removed the legacy solutions API. "
                "Use backend='tasks' instead."
            )
        return _LegacyBackend(config)

    raise RuntimeError(
        f"MediaPipe {getattr(mp, '__version__', '?')} exposes neither the Tasks "
        "API nor the legacy solutions API. Try reinstalling it:\n"
        "  python -m pip install --force-reinstall mediapipe"
    )


class MediaPipeHandTracker(HandTracker):
    """Webcam hand tracking via MediaPipe, on whichever API is available."""

    def __init__(self, config: TrackerConfig | None = None):
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "MediaPipeHandTracker needs 'opencv-python'.\n"
                "Install it with:  python -m pip install -r requirements.txt"
            ) from exc

        self.config = config or TrackerConfig()
        self._cv2 = cv2

        self._capture = cv2.VideoCapture(self.config.camera_index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.config.camera_index}. "
                "Is another application using the webcam?"
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)

        try:
            self._backend = _select_backend(self.config)
        except Exception:
            self._capture.release()      # do not leave the camera held open
            raise

        self._closed = False

    @property
    def backend_name(self) -> str:
        return type(self._backend).__name__.strip("_").replace("Backend", "").lower()

    def read(self) -> Optional[TrackerFrame]:
        if self._closed:
            return None
        ok, image = self._capture.read()
        if not ok:
            return None

        if self.config.mirror:
            image = self._cv2.flip(image, 1)

        rgb = self._cv2.cvtColor(image, self._cv2.COLOR_BGR2RGB)
        hands = self._backend.detect(rgb, self.config.mirror)
        return TrackerFrame(timestamp=time.monotonic(), hands=hands, image=image)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._backend.close()
        finally:
            self._capture.release()


class ReplayTracker(HandTracker):
    """Plays back a list of pre-built frames. Used by tests and offline demos."""

    def __init__(self, frames: list[TrackerFrame], fps: float = 30.0):
        self._frames = list(frames)
        self._i = 0
        self._dt = 1.0 / fps if fps > 0 else 0.0

    def read(self) -> Optional[TrackerFrame]:
        if self._i >= len(self._frames):
            return None
        frame = self._frames[self._i]
        self._i += 1
        return frame


def build_tracker(config: TrackerConfig | None = None) -> HandTracker:
    """Factory so callers never have to name a concrete backend."""
    return MediaPipeHandTracker(config)
