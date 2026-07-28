"""
gaze_pipeline.py — shared gaze prediction pipeline used by Milestone 4.

Wraps the MediaPipe face landmarker + ONNX gaze model into one call
(get_gaze_reading), with a square, frame-smoothed face crop and the
correct (yaw, pitch) model output order.
"""

import math
import pathlib
import cv2
import numpy as np
import onnxruntime as ort

CWD = pathlib.Path.cwd()
ONNX_MODEL_PATH = CWD / "models" / "l2cs_gaze360.onnx"
BINS = 90
BBOX_SMOOTHING = 0.3

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
idx_array = np.arange(BINS, dtype=np.float32)

session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4
# Prefer the GPU via DirectML (DirectX 12) when available — the L2CS model is
# ~10x faster there than on CPU — and fall back to CPU so every machine still runs.
_preferred = ["DmlExecutionProvider", "CPUExecutionProvider"]
_providers = [p for p in _preferred if p in ort.get_available_providers()] or ["CPUExecutionProvider"]
ort_session = ort.InferenceSession(
    str(ONNX_MODEL_PATH), sess_options=session_options, providers=_providers
)
print(f"[gaze_pipeline] ONNX running on: {ort_session.get_providers()[0]}")

_smoothed_bbox = None  # module-level state: (center_x, center_y, half_size)


def reset_bbox_smoothing():
    """Call this when starting a new session/script run to clear stale state."""
    global _smoothed_bbox
    _smoothed_bbox = None


def _softmax_np(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def _preprocess(face_crop_bgr):
    img_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (448, 448)).astype(np.float32) / 255.0
    img_norm = (img_resized - IMAGENET_MEAN) / IMAGENET_STD
    img_chw = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(img_chw, axis=0).astype(np.float32)


def _predict_gaze_radians(face_crop_bgr):
    input_tensor = _preprocess(face_crop_bgr)
    # CONFIRMED FIX: model outputs are (yaw, pitch) order, not (pitch, yaw).
    yaw_bins, pitch_bins = ort_session.run(None, {"input": input_tensor})
    pitch_probs = _softmax_np(pitch_bins)
    yaw_probs = _softmax_np(yaw_bins)
    pitch_deg = np.sum(pitch_probs * idx_array, axis=1) * 4 - 180
    yaw_deg = np.sum(yaw_probs * idx_array, axis=1) * 4 - 180
    pitch_rad = float(pitch_deg[0]) * math.pi / 180
    yaw_rad = float(yaw_deg[0]) * math.pi / 180
    return pitch_rad, yaw_rad


def get_smoothed_square_crop(frame, landmarks, padding_ratio=0.3):
    """Square, frame-smoothed face crop. Returns (crop, bbox) or (None, None)."""
    global _smoothed_bbox
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_size = max(x_max - x_min, y_max - y_min) / 2 * (1 + padding_ratio)

    if _smoothed_bbox is None:
        _smoothed_bbox = [cx, cy, half_size]
    else:
        _smoothed_bbox[0] = BBOX_SMOOTHING * cx + (1 - BBOX_SMOOTHING) * _smoothed_bbox[0]
        _smoothed_bbox[1] = BBOX_SMOOTHING * cy + (1 - BBOX_SMOOTHING) * _smoothed_bbox[1]
        _smoothed_bbox[2] = BBOX_SMOOTHING * half_size + (1 - BBOX_SMOOTHING) * _smoothed_bbox[2]

    scx, scy, shalf = _smoothed_bbox
    x_min_s = max(int(scx - shalf), 0)
    x_max_s = min(int(scx + shalf), w)
    y_min_s = max(int(scy - shalf), 0)
    y_max_s = min(int(scy + shalf), h)

    if x_max_s <= x_min_s or y_max_s <= y_min_s:
        return None, None

    crop = frame[y_min_s:y_max_s, x_min_s:x_max_s]
    return crop, (x_min_s, y_min_s, x_max_s, y_max_s)


def detect_face(frame, face_landmarker, frame_idx):
    """
    MediaPipe detection + square smoothed crop only (no ONNX).
    Returns (landmarks, bbox, transform_matrix) or (None, None, None) if no
    usable face. transform_matrix is the 4x4 facial transformation matrix
    when the landmarker was created with output_facial_transformation_matrixes
    enabled, otherwise None.

    Used by the pre-calibration setup check so distance/position feedback
    stays responsive without running the gaze model every frame.
    """
    import mediapipe as mp

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int(frame_idx * (1000 / 30))
    result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.face_landmarks:
        return None, None, None

    landmarks = result.face_landmarks[0]
    matrix = None
    if getattr(result, "facial_transformation_matrixes", None):
        matrix = result.facial_transformation_matrixes[0]

    crop, bbox = get_smoothed_square_crop(frame, landmarks)
    if crop is None or crop.size == 0 or bbox is None:
        return None, None, None
    return landmarks, bbox, matrix


def predict_gaze_from_crop(face_crop_bgr):
    """Public wrapper around the canonical ONNX pitch/yaw decode."""
    return _predict_gaze_radians(face_crop_bgr)


def get_gaze_reading(frame, face_landmarker, frame_idx):
    """
    High-level function: takes a BGR frame + a MediaPipe FaceLandmarker
    instance + a running frame index (for video timestamps).
    Returns (pitch_rad, yaw_rad, bbox) or (None, None, None) if no face found.
    """
    landmarks, bbox, _matrix = detect_face(frame, face_landmarker, frame_idx)
    if landmarks is None:
        return None, None, None

    # Re-crop from the already-smoothed bbox (detect_face updated smoothing).
    x_min, y_min, x_max, y_max = bbox
    crop = frame[y_min:y_max, x_min:x_max]
    if crop is None or crop.size == 0:
        return None, None, None

    pitch, yaw = predict_gaze_from_crop(crop)
    return pitch, yaw, bbox
