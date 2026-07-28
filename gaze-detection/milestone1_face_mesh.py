"""
Milestone 1 — Face & Iris Landmark Detection (Tasks API version)
Uses MediaPipe's current FaceLandmarker API (replaces deprecated mp.solutions).

Requires: face_landmarker.task in the same folder.
Run: python milestone1_face_mesh.py
Press 'q' to quit.
"""

import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --- Landmark indices you'll reuse in later milestones ---
LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

MODEL_PATH = "face_landmarker.task"

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_facial_transformation_matrixes=True,  # useful for head pose later
)

landmarker = mp_vision.FaceLandmarker.create_from_options(options)

WINDOW_NAME = "Milestone 1 - Face & Iris Landmarks (Tasks API)"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # camera's native 16:9 resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera index / permissions.")

prev_time = 0
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not received, skipping.")
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(frame_idx * (1000 / 30))  # approximate video timestamp
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    frame_idx += 1

    h, w = frame.shape[:2]
    tracking_status = "NO_FACE"

    if result.face_landmarks:
        tracking_status = "TRACKING"
        landmarks = result.face_landmarks[0]  # first detected face

        # Draw all landmarks as small dots
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Highlight iris points distinctly
        for idx in LEFT_IRIS + RIGHT_IRIS:
            lm = landmarks[idx]
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

    # FPS overlay
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Status: {tracking_status}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 0) if tracking_status == "TRACKING" else (0, 0, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
