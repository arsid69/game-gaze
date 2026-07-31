"""Static gesture classification.

Two classifiers, combined:

* :class:`RuleClassifier`   - hand-written geometric rules. No training data,
  predictable, covers the gestures the interaction layer needs.
* :class:`KNNGestureClassifier` - k-nearest-neighbour over recorded samples,
  so a user can add their own gestures with ``python -m gesture_control.recorder``.

:class:`CompositeClassifier` prefers a confident custom match and otherwise
falls back to the rules. :class:`GestureStabilizer` then requires N identical
frames before a label is committed, which removes almost all flicker.
"""

from __future__ import annotations

import json
import os
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol

import numpy as np

from . import features as F
from .config import GestureConfig


class Gesture(str):
    """Built-in gesture labels (plain strings, so custom names work too)."""

    UNKNOWN = "unknown"
    OPEN_PALM = "open_palm"
    FIST = "fist"
    POINT = "point"
    PINCH = "pinch"
    PEACE = "peace"
    THUMBS_UP = "thumbs_up"
    GUN = "gun"

    BUILTIN = (UNKNOWN, OPEN_PALM, FIST, POINT, PINCH, PEACE, THUMBS_UP, GUN)


@dataclass
class GestureResult:
    label: str = Gesture.UNKNOWN
    confidence: float = 0.0
    #: "rule" or "custom" - which classifier produced the label.
    source: str = "rule"
    details: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.label != Gesture.UNKNOWN


class Classifier(Protocol):
    def classify(self, landmarks: np.ndarray, hand_id: str = "default") -> GestureResult:
        ...


# --------------------------------------------------------------------------
# Rule based
# --------------------------------------------------------------------------
class RuleClassifier:
    """Geometric rules over finger extension and the pinch distance.

    Pinch uses hysteresis (``pinch_close`` / ``pinch_open``) because a single
    threshold makes the grab flicker whenever the fingers hover near it. State
    is kept per hand id so left and right hands never interfere.
    """

    def __init__(self, config: GestureConfig | None = None):
        self.config = config or GestureConfig()
        self._pinching: dict[str, bool] = {}

    def reset(self, hand_id: str | None = None) -> None:
        if hand_id is None:
            self._pinching.clear()
        else:
            self._pinching.pop(hand_id, None)

    def _update_pinch(self, hand_id: str, ratio: float) -> bool:
        was = self._pinching.get(hand_id, False)
        if was:
            now = ratio < self.config.pinch_open        # stays pinched longer
        else:
            now = ratio < self.config.pinch_close       # needs a firm pinch
        self._pinching[hand_id] = now
        return now

    def classify(self, landmarks: np.ndarray, hand_id: str = "default") -> GestureResult:
        lm = F.as_landmark_array(landmarks)
        ext = F.fingers_extended(lm, self.config.extended_angle_deg)
        thumb, index, middle, ring, pinky = (bool(e) for e in ext)
        ratio = F.pinch_ratio(lm)
        folded = not middle and not ring and not pinky

        details = {
            "extended": [thumb, index, middle, ring, pinky],
            "pinch_ratio": round(ratio, 3),
        }

        # Pinch first: it is the grab gesture and must win over "point".
        #
        # Deliberately NOT requiring the other fingers to be curled. Most people
        # pinch with them sticking out - the natural "OK sign" shape - and
        # demanding a curled fist made the grab fail for them. The thumb/index
        # distance alone is a good enough test: a real fist keeps the index tip
        # tucked into the palm, well beyond the pinch threshold.
        if self._update_pinch(hand_id, ratio):
            # Confidence rises the tighter the pinch is.
            conf = float(np.clip(1.0 - ratio / max(self.config.pinch_open, 1e-6),
                                 0.0, 1.0))
            if folded:
                conf = min(1.0, conf + 0.1)   # curled fingers are a clearer pinch
            return GestureResult(Gesture.PINCH, max(conf, 0.5), "rule", details)

        if thumb and index and middle and ring and pinky:
            return GestureResult(Gesture.OPEN_PALM, 0.95, "rule", details)

        if not any((thumb, index, middle, ring, pinky)):
            return GestureResult(Gesture.FIST, 0.9, "rule", details)

        if index and middle and not ring and not pinky:
            return GestureResult(Gesture.PEACE, 0.9, "rule", details)

        if index and folded:
            label = Gesture.GUN if thumb else Gesture.POINT
            return GestureResult(label, 0.9, "rule", details)

        if thumb and not index and folded:
            return GestureResult(Gesture.THUMBS_UP, 0.85, "rule", details)

        return GestureResult(Gesture.UNKNOWN, 0.0, "rule", details)


# --------------------------------------------------------------------------
# Learned / custom
# --------------------------------------------------------------------------
@dataclass
class GestureSample:
    label: str
    features: np.ndarray

    def to_json(self) -> dict:
        return {"label": self.label, "features": [float(v) for v in self.features]}

    @staticmethod
    def from_json(obj: dict) -> "GestureSample":
        return GestureSample(str(obj["label"]),
                             np.asarray(obj["features"], dtype=np.float64))


class GestureDataset:
    """Recorded custom-gesture samples, persisted as JSON."""

    def __init__(self, samples: Iterable[GestureSample] = ()):
        self.samples: list[GestureSample] = list(samples)

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "GestureDataset":
        if not os.path.exists(path):
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(GestureSample.from_json(s) for s in data.get("samples", []))

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        payload = {"version": 1, "samples": [s.to_json() for s in self.samples]}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)

    # -- editing -----------------------------------------------------------
    def add(self, label: str, landmarks: np.ndarray) -> None:
        self.samples.append(GestureSample(label, F.feature_vector(landmarks)))

    def remove_label(self, label: str) -> int:
        before = len(self.samples)
        self.samples = [s for s in self.samples if s.label != label]
        return before - len(self.samples)

    def labels(self) -> dict[str, int]:
        return dict(Counter(s.label for s in self.samples))

    def matrix(self) -> tuple[np.ndarray, list[str]]:
        if not self.samples:
            return np.zeros((0, 63)), []
        return (np.stack([s.features for s in self.samples]),
                [s.label for s in self.samples])

    def __len__(self) -> int:
        return len(self.samples)


class KNNGestureClassifier:
    """k-NN over normalised landmark vectors.

    Distance-weighted voting; matches further than ``knn_max_distance`` are
    thrown away, so an unseen pose reports ``unknown`` instead of guessing.
    """

    def __init__(self, dataset: GestureDataset, config: GestureConfig | None = None):
        self.config = config or GestureConfig()
        self.dataset = dataset
        self._X, self._y = dataset.matrix()

    def refresh(self) -> None:
        """Re-read the dataset after new samples were recorded."""
        self._X, self._y = self.dataset.matrix()

    def classify(self, landmarks: np.ndarray, hand_id: str = "default") -> GestureResult:
        if len(self._y) == 0:
            return GestureResult(Gesture.UNKNOWN, 0.0, "custom")

        query = F.feature_vector(landmarks)
        distances = np.linalg.norm(self._X - query, axis=1)

        k = min(self.config.knn_k, len(distances))
        nearest = np.argpartition(distances, k - 1)[:k]
        nearest = nearest[np.argsort(distances[nearest])]

        keep = [i for i in nearest if distances[i] <= self.config.knn_max_distance]
        if not keep:
            return GestureResult(Gesture.UNKNOWN, 0.0, "custom",
                                 {"nearest_distance": float(distances[nearest[0]])})

        votes: dict[str, float] = {}
        for i in keep:
            votes[self._y[i]] = votes.get(self._y[i], 0.0) + 1.0 / (distances[i] + 1e-6)

        label = max(votes, key=votes.get)
        confidence = votes[label] / sum(votes.values())
        best = float(min(distances[i] for i in keep if self._y[i] == label))
        return GestureResult(label, float(confidence), "custom",
                             {"nearest_distance": best, "neighbours": len(keep)})


class CompositeClassifier:
    """Custom gestures win when confident; otherwise the rules decide."""

    def __init__(
        self,
        rules: RuleClassifier,
        custom: Optional[KNNGestureClassifier] = None,
        config: GestureConfig | None = None,
    ):
        self.rules = rules
        self.custom = custom
        self.config = config or GestureConfig()

    def classify(self, landmarks: np.ndarray, hand_id: str = "default") -> GestureResult:
        rule_result = self.rules.classify(landmarks, hand_id)

        # Pinch drives grabbing; never let a custom pose steal it.
        if rule_result.label == Gesture.PINCH:
            return rule_result

        if self.custom is not None:
            custom_result = self.custom.classify(landmarks, hand_id)
            if (custom_result.label != Gesture.UNKNOWN
                    and custom_result.confidence >= self.config.custom_min_confidence):
                return custom_result

        return rule_result

    def reset(self, hand_id: str | None = None) -> None:
        self.rules.reset(hand_id)


def build_classifier(config: GestureConfig | None = None,
                     dataset_path: str | None = None) -> CompositeClassifier:
    """Rules + whatever custom gestures have been recorded so far."""
    config = config or GestureConfig()
    path = dataset_path or config.dataset_path
    dataset = GestureDataset.load(path)
    custom = KNNGestureClassifier(dataset, config) if len(dataset) else None
    return CompositeClassifier(RuleClassifier(config), custom, config)


# --------------------------------------------------------------------------
# Temporal smoothing
# --------------------------------------------------------------------------
class GestureStabilizer:
    """Commit a label only after it has held for N consecutive frames.

    Raw per-frame predictions flicker on the boundary between poses; without
    this, a single bad frame would fire a spurious select or drop an object.
    """

    def __init__(self, stable_frames: int = 3):
        self.stable_frames = max(1, stable_frames)
        self._window: deque[str] = deque(maxlen=self.stable_frames)
        self._committed = Gesture.UNKNOWN
        self._last_result = GestureResult()

    @property
    def current(self) -> str:
        return self._committed

    @property
    def result(self) -> GestureResult:
        return self._last_result

    def push(self, result: GestureResult) -> str:
        """Feed one raw prediction, get the currently committed label back."""
        self._window.append(result.label)
        if (len(self._window) == self.stable_frames
                and len(set(self._window)) == 1
                and self._window[0] != self._committed):
            self._committed = self._window[0]
            self._last_result = result
        elif self._window and self._window[-1] == self._committed:
            self._last_result = result
        return self._committed

    def reset(self) -> None:
        self._window.clear()
        self._committed = Gesture.UNKNOWN
        self._last_result = GestureResult()
