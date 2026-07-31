"""Gesture control module for 3D games.

Webcam hand tracking -> gesture recognition -> ray-cast selection and
manipulation of objects in a 3D scene.

Quick start::

    from gesture_control import GestureEngine, Scene, SceneObject, Camera, EventType

    scene = Scene([SceneObject("crate", position=(0, 0, 5), radius=1.0)])
    engine = GestureEngine(scene=scene, camera=Camera(position=(0, 0, -6)))

    engine.start()                       # camera runs on its own thread
    while running:                       # your game loop
        for event in engine.poll():
            if event.type is EventType.SELECT:
                print("selected", event.object_id)
    engine.stop()

See ``demo/demo_scene.py`` for a complete runnable example.
"""

from .config import (EngineConfig, GestureConfig, MotionConfig, PointerConfig,
                     TrackerConfig)
from .engine import GestureEngine, HandState
from .events import EventBus, EventType, GestureEvent
from .gestures import (CompositeClassifier, Gesture, GestureDataset,
                       GestureResult, GestureStabilizer, KNNGestureClassifier,
                       RuleClassifier, build_classifier)
from .motion import Swipe, SwipeDetector
from .pointer import (Camera, Hit, OneEuroFilter, OneEuroFilter2D, Ray, Scene,
                      SceneObject, image_to_ndc, intersect_aabb,
                      intersect_sphere, ndc_to_pixel)
from .tracker import (HandObservation, HandTracker, MediaPipeHandTracker,
                      ReplayTracker, TrackerFrame, build_tracker)

__version__ = "1.0.0"

__all__ = [
    "EngineConfig", "GestureConfig", "MotionConfig", "PointerConfig",
    "TrackerConfig",
    "GestureEngine", "HandState",
    "EventBus", "EventType", "GestureEvent",
    "Gesture", "GestureResult", "GestureDataset", "RuleClassifier",
    "KNNGestureClassifier", "CompositeClassifier", "GestureStabilizer",
    "build_classifier",
    "Swipe", "SwipeDetector",
    "Camera", "Ray", "Scene", "SceneObject", "Hit", "OneEuroFilter",
    "OneEuroFilter2D", "intersect_sphere", "intersect_aabb", "image_to_ndc",
    "ndc_to_pixel",
    "HandTracker", "MediaPipeHandTracker", "ReplayTracker", "HandObservation",
    "TrackerFrame", "build_tracker",
    "__version__",
]
