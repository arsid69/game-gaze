"""
Milestone 5 — Positioning Gate demo (distance + centering, webcam only).

Shows a car-reverse-style display: stacked distance level segments (stay in
the green levels), a center target box, guidance arrows/text, and a border
pulse that flashes faster the further you are out of the allowed zone.

Requires: face_landmarker.task in the same folder.
Run: python milestone5_positioning.py
Keys:
  q — quit
  c — one-time focal calibration: sit exactly 50 cm from the camera, press c
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from positioning_gate import PositioningGate, LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER

MODEL_PATH = "face_landmarker.task"
FOCAL_CALIB_DIST_CM = 50.0


def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    window_name = "Milestone 5 - Positioning Gate"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # camera's native 16:9 resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    gate = PositioningGate()
    prev_time = 0
    frame_idx = 0
    flash_msg, flash_until = "", 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect_for_video(mp_image, int(frame_idx * (1000 / 30)))
        frame_idx += 1

        status = None
        landmarks = None
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            status = gate.evaluate(landmarks, frame.shape)

        gate.draw_overlay(frame, status)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if time.time() < flash_until:
            cv2.putText(frame, flash_msg, (10, frame.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('c') and landmarks is not None:
            h, w = frame.shape[:2]
            li, ri = landmarks[LEFT_IRIS_CENTER], landmarks[RIGHT_IRIS_CENTER]
            ipd_px = (((ri.x - li.x) * w) ** 2 + ((ri.y - li.y) * h) ** 2) ** 0.5
            focal = gate.calibrate_focal(ipd_px, w, FOCAL_CALIB_DIST_CM)
            flash_msg = f"Focal calibrated: {focal:.0f} px @ {FOCAL_CALIB_DIST_CM:.0f} cm"
            flash_until = time.time() + 3

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
