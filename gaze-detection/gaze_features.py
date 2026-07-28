"""
gaze_features.py — full per-frame feature extraction for model training.

Everything measurable about the user's head and eyes in one frame, so the
dataset can support richer models than the current (gaze_pitch, gaze_yaw)
polynomial.

Feature groups
--------------
gaze_*   L2CS-Net gaze angles (radians). The single strongest signal — a
         dedicated deep model trained on Gaze360, not a geometric heuristic.
head_*   Head pose from MediaPipe's facial transformation matrix:
         rotation (pitch/yaw/roll, degrees) + translation (tx/ty/tz).
iris_*   Iris centers, both in normalized frame coordinates and — more
         usefully — as a ratio inside the eye-corner box, which is the
         actual eye-direction signal independent of where the head sits.
eye_*    Eye corner landmarks and EAR (eye aspect ratio = openness/blink).
face_*   Face width/height/IPD — scale cues that co-vary with distance.
pos_*    Distance and framing from the positioning gate.

Landmark indices are MediaPipe FaceLandmarker (478-point, refined irises).
"""

import math

import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError:  # scipy is optional; head rotation is skipped without it
    Rotation = None

# --- Landmark indices ---
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# Eye corners: (outer/lateral, inner/medial) as seen in the image
LEFT_EYE_OUTER, LEFT_EYE_INNER = 33, 133
RIGHT_EYE_INNER, RIGHT_EYE_OUTER = 362, 263

# Standard 6-point EAR landmarks: p1..p6 (p1/p4 horizontal, rest vertical)
LEFT_EAR_PTS = (33, 160, 158, 133, 153, 144)
RIGHT_EAR_PTS = (362, 385, 387, 263, 373, 380)

# Face extent reference points
FACE_LEFT, FACE_RIGHT = 234, 454
FACE_TOP, FACE_BOTTOM = 10, 152


def _px(landmark, w, h):
    return landmark.x * w, landmark.y * h


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(landmarks, pts, w, h):
    """EAR = (|p2-p6| + |p3-p5|) / (2*|p1-p4|). ~0.3 open, ~0.1 closed."""
    p1, p2, p3, p4, p5, p6 = [_px(landmarks[i], w, h) for i in pts]
    horizontal = _dist(p1, p4)
    if horizontal < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def iris_ratio(landmarks, iris_idx, outer_idx, inner_idx, w, h):
    """Iris position inside its eye, as (x_ratio, y_ratio).

    x_ratio: 0 at the outer corner, 1 at the inner corner.
    y_ratio: iris offset from the eye-corner midline, scaled by eye width.

    This is head-position invariant in a way raw iris pixels are not — the
    same eyeball rotation gives the same ratio wherever the face sits in
    frame, which is exactly what a gaze model wants.
    """
    iris = _px(landmarks[iris_idx], w, h)
    outer = _px(landmarks[outer_idx], w, h)
    inner = _px(landmarks[inner_idx], w, h)
    eye_w = _dist(outer, inner)
    if eye_w < 1e-6:
        return 0.0, 0.0
    x_ratio = ((iris[0] - outer[0]) * (inner[0] - outer[0])
               + (iris[1] - outer[1]) * (inner[1] - outer[1])) / (eye_w ** 2)
    mid_y = (outer[1] + inner[1]) / 2.0
    y_ratio = (iris[1] - mid_y) / eye_w
    return x_ratio, y_ratio


def head_pose_from_matrix(matrix):
    """(pitch, yaw, roll) degrees and (tx, ty, tz) translation from the
    4x4 facial transformation matrix. Returns Nones if unavailable."""
    if matrix is None or Rotation is None:
        return (None,) * 6
    m = np.array(matrix)
    if m.shape != (4, 4):
        return (None,) * 6
    pitch, yaw, roll = Rotation.from_matrix(m[:3, :3]).as_euler('xyz', degrees=True)
    tx, ty, tz = m[:3, 3]
    return float(pitch), float(yaw), float(roll), float(tx), float(ty), float(tz)


def extract_features(landmarks, frame_shape, gaze_pitch=None, gaze_yaw=None,
                     transform_matrix=None, gate_status=None):
    """Build the full feature dict for one frame.

    landmarks: MediaPipe face landmark list (478 points)
    frame_shape: (h, w, ...) of the source frame
    gaze_pitch/gaze_yaw: L2CS model output in radians (may be None)
    transform_matrix: 4x4 facial transformation matrix (may be None)
    gate_status: PositioningGate.evaluate() dict (may be None)

    Returns a flat dict; missing values are None so the CSV writer can blank
    them without breaking the column layout.
    """
    h, w = frame_shape[:2]
    f = {}

    # --- Gaze model (strongest single signal) ---
    f["gaze_pitch"] = gaze_pitch
    f["gaze_yaw"] = gaze_yaw

    # --- Head pose ---
    (f["head_pitch"], f["head_yaw"], f["head_roll"],
     f["head_tx"], f["head_ty"], f["head_tz"]) = head_pose_from_matrix(transform_matrix)

    # --- Iris centers (normalized frame coords) ---
    li, ri = landmarks[LEFT_IRIS_CENTER], landmarks[RIGHT_IRIS_CENTER]
    f["iris_left_x"], f["iris_left_y"] = li.x, li.y
    f["iris_right_x"], f["iris_right_y"] = ri.x, ri.y

    # --- Iris position within the eye (head-position invariant) ---
    f["iris_left_ratio_x"], f["iris_left_ratio_y"] = iris_ratio(
        landmarks, LEFT_IRIS_CENTER, LEFT_EYE_OUTER, LEFT_EYE_INNER, w, h)
    f["iris_right_ratio_x"], f["iris_right_ratio_y"] = iris_ratio(
        landmarks, RIGHT_IRIS_CENTER, RIGHT_EYE_OUTER, RIGHT_EYE_INNER, w, h)

    # --- Eye corners (normalized frame coords) ---
    for name, idx in [("eye_left_outer", LEFT_EYE_OUTER),
                      ("eye_left_inner", LEFT_EYE_INNER),
                      ("eye_right_inner", RIGHT_EYE_INNER),
                      ("eye_right_outer", RIGHT_EYE_OUTER)]:
        f[f"{name}_x"] = landmarks[idx].x
        f[f"{name}_y"] = landmarks[idx].y

    # --- Eye openness / blink ---
    f["ear_left"] = eye_aspect_ratio(landmarks, LEFT_EAR_PTS, w, h)
    f["ear_right"] = eye_aspect_ratio(landmarks, RIGHT_EAR_PTS, w, h)

    # --- Face geometry (scale cues) ---
    f["face_width"] = abs(landmarks[FACE_RIGHT].x - landmarks[FACE_LEFT].x)
    f["face_height"] = abs(landmarks[FACE_BOTTOM].y - landmarks[FACE_TOP].y)
    f["ipd_px"] = _dist(_px(li, w, h), _px(ri, w, h))

    # --- Position / framing (from the gate) ---
    if gate_status is not None:
        f["pos_distance_cm"] = gate_status["distance_cm"]
        f["pos_face_dx"] = gate_status["dx"]
        f["pos_face_dy"] = gate_status["dy"]
    else:
        f["pos_distance_cm"] = f["pos_face_dx"] = f["pos_face_dy"] = None

    return f


# Canonical feature column order (labels and metadata are added by the
# dataset writer). Keep in sync with extract_features().
FEATURE_COLUMNS = [
    "gaze_pitch", "gaze_yaw",
    "head_pitch", "head_yaw", "head_roll", "head_tx", "head_ty", "head_tz",
    "iris_left_x", "iris_left_y", "iris_right_x", "iris_right_y",
    "iris_left_ratio_x", "iris_left_ratio_y",
    "iris_right_ratio_x", "iris_right_ratio_y",
    "eye_left_outer_x", "eye_left_outer_y", "eye_left_inner_x", "eye_left_inner_y",
    "eye_right_inner_x", "eye_right_inner_y", "eye_right_outer_x", "eye_right_outer_y",
    "ear_left", "ear_right",
    "face_width", "face_height", "ipd_px",
    "pos_distance_cm", "pos_face_dx", "pos_face_dy",
]
