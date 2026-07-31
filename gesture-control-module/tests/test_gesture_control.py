"""Test suite for the gesture control module.

Everything runs against synthetically generated hands, so no camera and no
MediaPipe install are needed.

Run with pytest::      pytest tests/
or standalone::        python tests/test_gesture_control.py
"""

from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_control import features as F
from gesture_control import synthetic as S
from gesture_control.config import EngineConfig, GestureConfig
from gesture_control.events import EventType
from gesture_control.engine import GestureEngine
from gesture_control.gestures import (CompositeClassifier, Gesture,
                                      GestureDataset, GestureStabilizer,
                                      KNNGestureClassifier, RuleClassifier)
from gesture_control.motion import SwipeDetector
from gesture_control.pointer import (Camera, OneEuroFilter, Ray, Scene,
                                     SceneObject, image_to_ndc, intersect_aabb,
                                     intersect_sphere)
from gesture_control.tracker import HandObservation, TrackerFrame


# ==========================================================================
# Features
# ==========================================================================
def test_finger_extension_matches_pose():
    for name, expected in [
        ("open_palm", [True, True, True, True, True]),
        ("fist", [False, False, False, False, False]),
        ("point", [False, True, False, False, False]),
        ("peace", [False, True, True, False, False]),
        ("thumbs_up", [True, False, False, False, False]),
    ]:
        got = list(F.fingers_extended(S.make_pose(name)))
        assert got == expected, f"{name}: expected {expected}, got {got}"


def test_normalisation_is_invariant():
    """Same pose, different place/size/orientation -> same feature vector."""
    base = S.make_pose("point")
    ref = F.feature_vector(base)

    moved = S.make_hand(**S.POSES["point"], origin=(3.0, -1.0, 2.0))
    assert np.allclose(F.feature_vector(moved), ref, atol=1e-8), "not translation invariant"

    bigger = S.make_hand(**S.POSES["point"], scale=2.7)
    assert np.allclose(F.feature_vector(bigger), ref, atol=1e-8), "not scale invariant"

    angle = math.radians(37)
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    turned = S.make_hand(**S.POSES["point"], rotation=rot)
    assert np.allclose(F.feature_vector(turned), ref, atol=1e-8), "not rotation invariant"


def test_pinch_ratio_ordering():
    pinched = F.pinch_ratio(S.make_pose("pinch"))
    open_hand = F.pinch_ratio(S.make_pose("open_palm"))
    assert pinched < 0.34, f"pinched ratio too large: {pinched}"
    assert open_hand > pinched * 2, "pinch and open hand are not separable"


def test_canonical_basis_is_orthonormal():
    basis = F.canonical_basis(S.make_pose("open_palm"))
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(basis), 1.0, atol=1e-9)


# ==========================================================================
# Classification
# ==========================================================================
def test_rule_classifier_labels():
    clf = RuleClassifier()
    for pose, expected in [
        ("open_palm", Gesture.OPEN_PALM),
        ("fist", Gesture.FIST),
        ("point", Gesture.POINT),
        ("peace", Gesture.PEACE),
        ("thumbs_up", Gesture.THUMBS_UP),
        ("pinch", Gesture.PINCH),
    ]:
        clf.reset()
        result = clf.classify(S.make_pose(pose))
        assert result.label == expected, \
            f"{pose}: expected {expected}, got {result.label} ({result.details})"


def test_pinch_hysteresis_holds_the_grab():
    """Once pinched, a slightly loosening hand must stay pinched."""
    clf = RuleClassifier(GestureConfig(pinch_close=0.34, pinch_open=0.48))
    lm = S.make_pose("pinch")
    assert clf.classify(lm, "Right").label == Gesture.PINCH

    # Widen the gap past 'close' but not past 'open'.
    loose = lm.copy()
    direction = loose[F.THUMB_TIP] - loose[F.INDEX_TIP]
    direction /= np.linalg.norm(direction)
    loose[F.THUMB_TIP] = loose[F.INDEX_TIP] + direction * 0.40 * F.hand_scale(lm)
    ratio = F.pinch_ratio(loose)
    assert 0.34 < ratio < 0.48, f"test fixture ratio {ratio} outside hysteresis band"
    assert clf.classify(loose, "Right").label == Gesture.PINCH, "grab dropped too early"


def test_pinch_works_with_the_other_fingers_extended():
    """The natural 'OK sign' pinch must register, not just a curled fist."""
    clf = RuleClassifier()
    open_fingered = S.make_hand(index=True, middle=True, ring=True, pinky=True,
                                thumb=True, pinch=True)
    assert F.pinch_ratio(open_fingered) < 0.34, "fixture is not actually pinched"
    assert clf.classify(open_fingered).label == Gesture.PINCH


def test_a_fist_is_not_mistaken_for_a_pinch():
    """Dropping the curled-fingers requirement must not break the fist."""
    clf = RuleClassifier()
    assert clf.classify(S.make_pose("fist")).label == Gesture.FIST


def test_hand_states_are_independent():
    clf = RuleClassifier()
    clf.classify(S.make_pose("pinch"), "Right")
    assert clf.classify(S.make_pose("open_palm"), "Left").label == Gesture.OPEN_PALM


def test_stabilizer_needs_consecutive_frames():
    from gesture_control.gestures import GestureResult
    stab = GestureStabilizer(stable_frames=3)
    assert stab.push(GestureResult(Gesture.POINT)) == Gesture.UNKNOWN
    assert stab.push(GestureResult(Gesture.POINT)) == Gesture.UNKNOWN
    assert stab.push(GestureResult(Gesture.POINT)) == Gesture.POINT
    # A single bad frame must not change the committed label.
    assert stab.push(GestureResult(Gesture.FIST)) == Gesture.POINT


def test_knn_learns_custom_gestures_and_rejects_unknowns():
    dataset = GestureDataset()
    rng = np.random.default_rng(0)
    for pose in ("thumbs_up", "peace"):
        for _ in range(12):
            lm = S.make_pose(pose) + rng.normal(0, 0.004, (21, 3))
            dataset.add(f"custom_{pose}", lm)

    knn = KNNGestureClassifier(dataset, GestureConfig(knn_k=3, knn_max_distance=0.4))
    assert knn.classify(S.make_pose("thumbs_up")).label == "custom_thumbs_up"
    assert knn.classify(S.make_pose("peace")).label == "custom_peace"
    # An unseen pose must be rejected, not forced into a known class.
    assert knn.classify(S.make_pose("open_palm")).label == Gesture.UNKNOWN


def test_dataset_roundtrip():
    dataset = GestureDataset()
    dataset.add("wave", S.make_pose("open_palm"))
    dataset.add("wave", S.make_pose("open_palm"))
    dataset.add("gun", S.make_pose("point"))

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "nested", "gestures.json")
        dataset.save(path)
        loaded = GestureDataset.load(path)

    assert loaded.labels() == {"wave": 2, "gun": 1}
    assert np.allclose(loaded.samples[0].features, dataset.samples[0].features)
    assert loaded.remove_label("wave") == 2


def test_composite_prefers_pinch_over_custom():
    dataset = GestureDataset()
    for _ in range(6):
        dataset.add("my_pinch", S.make_pose("pinch"))
    composite = CompositeClassifier(RuleClassifier(),
                                    KNNGestureClassifier(dataset))
    assert composite.classify(S.make_pose("pinch")).label == Gesture.PINCH


# ==========================================================================
# Geometry
# ==========================================================================
def test_ray_sphere_intersection():
    ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert np.isclose(intersect_sphere(ray, np.array([0.0, 0.0, 5.0]), 1.0), 4.0)
    assert intersect_sphere(ray, np.array([0.0, 5.0, 5.0]), 1.0) is None   # miss
    assert intersect_sphere(ray, np.array([0.0, 0.0, -5.0]), 1.0) is None  # behind


def test_ray_aabb_intersection():
    ray = Ray(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    half = np.array([1.0, 1.0, 1.0])
    assert np.isclose(intersect_aabb(ray, np.array([0.0, 0.0, 6.0]), half), 5.0)
    assert intersect_aabb(ray, np.array([4.0, 0.0, 6.0]), half) is None
    # Parallel ray outside the slab must miss rather than divide by zero.
    side = Ray(np.array([0.0, 9.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    assert intersect_aabb(side, np.array([0.0, 0.0, 6.0]), half) is None


def test_camera_ray_and_projection_round_trip():
    cam = Camera(position=np.array([1.0, 2.0, -3.0]),
                 forward=np.array([0.0, 0.0, 1.0]), fov_y_deg=60.0, aspect=16 / 9)
    for ndc in [(0.0, 0.0), (0.5, -0.3), (-0.9, 0.75)]:
        ray = cam.ray_from_ndc(*ndc)
        back = cam.world_to_ndc(ray.at(7.5))
        assert back is not None
        assert np.allclose(back[:2], ndc, atol=1e-9), f"round trip failed for {ndc}"


def test_camera_centre_ray_points_forward():
    cam = Camera(position=np.zeros(3), forward=np.array([0.0, 0.0, 1.0]))
    assert np.allclose(cam.ray_from_ndc(0.0, 0.0).direction, [0, 0, 1], atol=1e-9)


def test_scene_raycast_picks_nearest():
    scene = Scene([
        SceneObject("far", position=np.array([0.0, 0.0, 12.0]), radius=1.0),
        SceneObject("near", position=np.array([0.0, 0.0, 5.0]), radius=1.0),
        SceneObject("ignored", position=np.array([0.0, 0.0, 3.0]), radius=1.0,
                    selectable=False),
    ])
    ray = Ray(np.zeros(3), np.array([0.0, 0.0, 1.0]))
    hit = scene.raycast(ray)
    assert hit is not None and hit.object.id == "near"
    assert np.isclose(hit.distance, 4.0)


def test_image_to_ndc_mapping():
    assert image_to_ndc(0.5, 0.5) == (0.0, 0.0)
    assert image_to_ndc(1.0, 0.0) == (1.0, 1.0)     # top-right of the image
    assert image_to_ndc(0.0, 1.0) == (-1.0, -1.0)   # bottom-left


def test_one_euro_filter_smooths_but_tracks():
    filt = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    noisy = [0.0, 0.4, -0.4, 0.4, -0.4, 0.4]
    out = [filt(v, i / 30.0) for i, v in enumerate(noisy)]
    assert max(abs(v) for v in out[1:]) < 0.4, "jitter was not attenuated"

    filt.reset()
    steady = [filt(1.0, i / 30.0) for i in range(120)]
    assert abs(steady[-1] - 1.0) < 1e-3, "filter failed to converge"


# ==========================================================================
# Engine / interaction
# ==========================================================================
def _frame(pose: str, t: float, center=(0.5, 0.5), hand="Right") -> TrackerFrame:
    # Anchor on the fingertip: that is the landmark the pointer follows, so
    # `center` is literally where the player is aiming.
    lm = S.to_image_space(S.make_pose(pose), center=center, anchor=F.INDEX_TIP)
    return TrackerFrame(timestamp=t, hands=[HandObservation(lm, handedness=hand)])


def _engine() -> GestureEngine:
    scene = Scene([SceneObject("target", position=np.array([0.0, 0.0, 8.0]),
                               radius=2.0)])
    camera = Camera(position=np.zeros(3), forward=np.array([0.0, 0.0, 1.0]),
                    fov_y_deg=60.0, aspect=1.0)
    config = EngineConfig()
    config.gestures.stable_frames = 2
    config.pointer.min_cutoff = 60.0      # near-instant response for the test
    config.pointer.beta = 0.0
    return GestureEngine(scene=scene, camera=camera, config=config,
                         classifier=RuleClassifier(config.gestures))


def _feed(engine: GestureEngine, pose: str, frames: int, t0: float,
          center=(0.5, 0.5), dt: float = 1 / 30) -> float:
    t = t0
    for _ in range(frames):
        engine.update(_frame(pose, t, center))
        t += dt
    return t


def test_hand_detected_and_lost():
    engine = _engine()
    t = _feed(engine, "point", 3, 0.0)
    types = [e.type for e in engine.poll()]
    assert EventType.HAND_DETECTED in types

    # No hands for longer than the timeout -> HAND_LOST.
    engine.update(TrackerFrame(timestamp=t + 1.0, hands=[]))
    assert EventType.HAND_LOST in [e.type for e in engine.poll()]


def test_pointing_at_object_hovers_it():
    engine = _engine()
    _feed(engine, "point", 5, 0.0)
    hover = [e for e in engine.poll() if e.type is EventType.HOVER_ENTER]
    assert hover and hover[0].object_id == "target"


def test_pinch_selects_and_grabs_then_release_drops():
    engine = _engine()
    t = _feed(engine, "point", 4, 0.0)
    engine.poll()

    t = _feed(engine, "pinch", 4, t)
    types = [e.type for e in engine.poll()]
    assert EventType.SELECT in types and EventType.GRAB_BEGIN in types
    assert engine.scene.is_selected("target")

    _feed(engine, "point", 4, t)
    assert EventType.GRAB_END in [e.type for e in engine.poll()]


def test_dragging_moves_the_object():
    engine = _engine()
    engine.config.pointer.drag_smoothing = 0.0   # so the invariant is exact
    start = engine.scene.get("target").position.copy()

    t = _feed(engine, "point", 4, 0.0)
    t = _feed(engine, "pinch", 4, t)
    _feed(engine, "pinch", 12, t, center=(0.72, 0.34))   # move the hand

    moved = engine.scene.get("target").position
    assert not np.allclose(moved, start, atol=1e-3), "object did not follow the hand"

    # Dragging must preserve the distance to the camera: the object swings
    # around on the ray instead of flying towards or away from the player.
    state = engine.hand_state("Right")
    on_ray = moved - state.grab_offset
    distance = float(np.linalg.norm(on_ray - engine.camera.position))
    assert np.isclose(distance, state.grab_depth, atol=1e-6), "drag changed the depth"


def test_grabbing_does_not_move_the_object():
    """Picking something up must not make it snap or jump anywhere."""
    engine = _engine()
    engine.config.pointer.drag_smoothing = 0.0
    start = engine.scene.get("target").position.copy()

    t = _feed(engine, "point", 4, 0.0)
    _feed(engine, "pinch", 4, t)          # grab, but hold the hand still

    moved = engine.scene.get("target").position
    assert np.allclose(moved, start, atol=1e-6), \
        f"object jumped by {np.linalg.norm(moved - start):.3f} on grab"


def test_pointer_does_not_teleport_when_the_gesture_changes():
    """The cursor must follow one landmark, whatever the classifier says.

    It used to switch landmark per gesture, which lurched a held object across
    the scene the moment the label changed.
    """
    engine = _engine()
    t = _feed(engine, "point", 5, 0.0)
    before = engine.hand_state("Right").ndc

    _feed(engine, "pinch", 5, t)          # same hand position, different label
    after = engine.hand_state("Right").ndc

    shift = math.dist(before, after)
    assert shift < 0.15, f"cursor jumped {shift:.3f} in NDC on a gesture change"


def test_grab_assist_forgives_a_near_miss():
    scene = Scene([SceneObject("ball", position=np.array([0.0, 0.0, 10.0]),
                               radius=1.0)])
    # Aimed past the edge of the sphere: a strict raycast finds nothing.
    ray = Ray(np.zeros(3), np.array([0.13, 0.0, 1.0]))
    assert scene.raycast(ray) is None
    assisted = scene.pick(ray, assist=1.7)
    assert assisted is not None and assisted.object.id == "ball"
    # ...but a wild miss is still a miss.
    assert scene.pick(Ray(np.zeros(3), np.array([1.0, 0.0, 1.0])), 1.7) is None


def test_drag_smoothing_lags_then_catches_up():
    """Easing must trail the cursor briefly and then settle on it exactly."""
    engine = _engine()
    engine.config.pointer.drag_smoothing = 0.6

    t = _feed(engine, "point", 4, 0.0)
    t = _feed(engine, "pinch", 4, t)
    t = _feed(engine, "pinch", 2, t, center=(0.75, 0.30))
    early = engine.scene.get("target").position.copy()

    _feed(engine, "pinch", 40, t, center=(0.75, 0.30))
    settled = engine.scene.get("target").position

    state = engine.hand_state("Right")
    target = state.ray.at(state.grab_depth) + state.grab_offset
    assert np.linalg.norm(early - target) > np.linalg.norm(settled - target)
    assert np.allclose(settled, target, atol=1e-4), "easing never converged"


def test_open_palm_clears_selection():
    engine = _engine()
    t = _feed(engine, "point", 4, 0.0)
    t = _feed(engine, "pinch", 4, t)
    assert engine.scene.selected
    engine.poll()

    _feed(engine, "open_palm", 5, t)
    assert not engine.scene.selected
    assert EventType.CLEAR_SELECTION in [e.type for e in engine.poll()]


def test_swipe_detector_direction_and_cooldown():
    det = SwipeDetector()
    fired = None
    for i in range(10):
        fired = det.update(i * 0.02, -0.5 + i * 0.12, 0.0) or fired
    assert fired is not None and fired.direction == "right"

    # Slow drift must not register.
    det.reset()
    slow = [det.update(i * 0.2, -0.5 + i * 0.05, 0.0) for i in range(10)]
    assert all(s is None for s in slow)


def test_event_bus_callbacks_and_polling():
    engine = _engine()
    seen = []
    engine.subscribe(EventType.HAND_DETECTED, lambda e: seen.append(e))
    _feed(engine, "point", 3, 0.0)
    assert len(seen) == 1
    assert list(engine.poll()), "queue should also hold the events"
    assert not list(engine.poll()), "poll should drain the queue"


# ==========================================================================
# Runner
# ==========================================================================
def _run_all() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
