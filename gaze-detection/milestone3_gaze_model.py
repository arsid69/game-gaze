"""
Milestone 3 (v5) — Pretrained Gaze Estimation, ONNX Runtime

Same architecture as v4 (MediaPipe face crop -> gaze model directly, no
internal detector), but the gaze model now runs via ONNX Runtime instead of
raw PyTorch eager mode. ONNX Runtime applies graph optimizations and better
CPU threading, typically 2-5x faster for the same model/weights/accuracy.

Requires:
  pip install onnx onnxruntime
  models/l2cs_gaze360.onnx  (created by running export_to_onnx.py once)
  face_landmarker.task (from Milestone 1)

Run: python milestone3_gaze_model.py
Press 'q' to quit.
"""

import time
import math
import pathlib
import cv2
import numpy as np
import onnxruntime as ort

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

CWD = pathlib.Path.cwd()
ONNX_MODEL_PATH = CWD / "models" / "l2cs_gaze360.onnx"
FACE_MODEL_PATH = "face_landmarker.task"
BINS = 90

if not ONNX_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"{ONNX_MODEL_PATH} not found. Run export_to_onnx.py first."
    )

# ---- Face landmarker (MediaPipe) ----
base_options = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)

# ---- ONNX Runtime session for the gaze model ----
session_options = ort.SessionOptions()
session_options.intra_op_num_threads = 4  # tune based on your CPU core count
ort_session = ort.InferenceSession(
    str(ONNX_MODEL_PATH), sess_options=session_options, providers=["CPUExecutionProvider"]
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
idx_array = np.arange(BINS, dtype=np.float32)


def softmax_np(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def preprocess(face_crop_bgr):
    img_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (448, 448)).astype(np.float32) / 255.0
    img_norm = (img_resized - IMAGENET_MEAN) / IMAGENET_STD
    img_chw = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img_chw, axis=0).astype(np.float32)


def predict_gaze(face_crop_bgr):
    input_tensor = preprocess(face_crop_bgr)
    # Model output order is (yaw, pitch), not (pitch, yaw) — verified by
    # directional testing (look left/right vs up/down and compare signs).
    yaw_bins, pitch_bins = ort_session.run(None, {"input": input_tensor})

    pitch_probs = softmax_np(pitch_bins)
    yaw_probs = softmax_np(yaw_bins)

    pitch_deg = np.sum(pitch_probs * idx_array, axis=1) * 4 - 180
    yaw_deg = np.sum(yaw_probs * idx_array, axis=1) * 4 - 180

    pitch_rad = float(pitch_deg[0]) * math.pi / 180
    yaw_rad = float(yaw_deg[0]) * math.pi / 180
    return pitch_rad, yaw_rad


def get_face_bbox(landmarks, frame_shape, padding=20):
    h, w = frame_shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x_min, x_max = max(int(min(xs)) - padding, 0), min(int(max(xs)) + padding, w)
    y_min, y_max = max(int(min(ys)) - padding, 0), min(int(max(ys)) + padding, h)
    return x_min, y_min, x_max, y_max


def draw_gaze_arrow(frame, bbox, pitch, yaw, length=100):
    x_min, y_min, x_max, y_max = bbox
    cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
    dx = -length * math.sin(pitch) * math.cos(yaw)
    dy = -length * math.sin(yaw)
    cv2.arrowedLine(frame, (cx, cy), (int(cx + dx), int(cy + dy)), (0, 0, 255), 3)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


def main():
    window_name = "Milestone 3 (v5) - ONNX Runtime Gaze Model"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # camera's native 16:9 resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    prev_time = 0
    frame_idx = 0
    gaze_ms = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(frame_idx * (1000 / 30))
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_idx += 1

        status = "NO_FACE"

        if result.face_landmarks:
            status = "TRACKING"
            landmarks = result.face_landmarks[0]
            bbox = get_face_bbox(landmarks, frame.shape)
            x_min, y_min, x_max, y_max = bbox
            face_crop = frame[y_min:y_max, x_min:x_max]

            if face_crop.size > 0:
                t0 = time.time()
                pitch, yaw = predict_gaze(face_crop)
                gaze_ms = (time.time() - t0) * 1000
                draw_gaze_arrow(frame, bbox, pitch, yaw)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gaze inference: {gaze_ms:.0f} ms", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Status: {status}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if status == "TRACKING" else (0, 0, 255), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_landmarker.close()


if __name__ == "__main__":
    main()
