"""
Milestone 2 (v3) — Live feature inspector.

Shows every one of the 32 features that get written to the dataset, updating
in real time, next to the video. The landmarks each feature is derived from
are highlighted on your face in matching colours, so you can see exactly
where every number comes from.

Requires: face_landmarker.task, models/l2cs_gaze360.onnx
Run: python milestone2_eye_headpose.py
Keys:
  q  quit
  g  toggle the gaze neural network (off = much higher FPS, gaze_* blank)
  l  toggle the landmark overlay on the face
"""

import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gaze_features import (
    extract_features, LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER,
    LEFT_EYE_OUTER, LEFT_EYE_INNER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
    LEFT_EAR_PTS, RIGHT_EAR_PTS, FACE_LEFT, FACE_RIGHT, FACE_TOP, FACE_BOTTOM,
)
from gaze_pipeline import (
    get_smoothed_square_crop, predict_gaze_from_crop, reset_bbox_smoothing,
)
from positioning_gate import PositioningGate, NOSE_TIP

MODEL_PATH = "face_landmarker.task"
VIDEO_W, VIDEO_H = 1280, 720
PANEL_W = 560
DEG = 57.2957795  # radians -> degrees

# Colour key (BGR) shared by the face overlay and the panel labels
C_GAZE = (80, 200, 255)     # amber
C_HEAD = (120, 255, 180)    # mint
C_IRIS = (0, 165, 255)      # orange
C_EYE = (255, 190, 90)      # light blue
C_EAR = (230, 120, 255)     # magenta
C_FACE = (90, 230, 230)     # yellow
C_POS = (140, 255, 140)     # green
C_DIM = (150, 150, 150)
C_TXT = (235, 235, 235)


def fmt(value, digits=3):
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def draw_landmarks(frame, landmarks):
    """Highlight only the points that actually feed a feature."""
    h, w = frame.shape[:2]

    def px(idx):
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)

    # EAR points (drawn first, smallest)
    for idx in LEFT_EAR_PTS + RIGHT_EAR_PTS:
        cv2.circle(frame, px(idx), 2, C_EAR, -1)

    # Eye corners
    for idx in (LEFT_EYE_OUTER, LEFT_EYE_INNER, RIGHT_EYE_INNER, RIGHT_EYE_OUTER):
        cv2.circle(frame, px(idx), 4, C_EYE, -1)

    # Eye-corner axis the iris ratio is measured along
    cv2.line(frame, px(LEFT_EYE_OUTER), px(LEFT_EYE_INNER), C_EYE, 1)
    cv2.line(frame, px(RIGHT_EYE_INNER), px(RIGHT_EYE_OUTER), C_EYE, 1)

    # Iris centres + the IPD line between them
    li, ri = px(LEFT_IRIS_CENTER), px(RIGHT_IRIS_CENTER)
    cv2.line(frame, li, ri, C_IRIS, 1)
    for p in (li, ri):
        cv2.circle(frame, p, 5, C_IRIS, -1)
        cv2.circle(frame, p, 7, (255, 255, 255), 1)

    # Face extent box
    x0, _ = px(FACE_LEFT)
    x1, _ = px(FACE_RIGHT)
    _, y0 = px(FACE_TOP)
    _, y1 = px(FACE_BOTTOM)
    cv2.rectangle(frame, (x0, y0), (x1, y1), C_FACE, 1)

    # Nose tip (centring)
    cv2.circle(frame, px(NOSE_TIP), 5, C_POS, -1)


def draw_panel(f, gaze_on, fps, zone):
    """Build the readings panel. f is the feature dict (or None)."""
    panel = np.zeros((VIDEO_H, PANEL_W, 3), dtype=np.uint8)
    panel[:] = (26, 22, 20)

    cv2.putText(panel, "LIVE FEATURE READINGS", (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_TXT, 2)
    cv2.putText(panel, "the 32 values written to the dataset", (14, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_DIM, 1)
    cv2.line(panel, (14, 58), (PANEL_W - 14, 58), (70, 65, 62), 1)

    if f is None:
        cv2.putText(panel, "NO FACE DETECTED", (14, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        return panel

    # (label, value string, colour) laid out in two columns
    col1, col2 = [], []

    col1.append(("GAZE  (neural network)", None, C_GAZE))
    if gaze_on and f["gaze_pitch"] is not None:
        col1.append(("gaze_pitch", f"{f['gaze_pitch']:+.3f} rad  {f['gaze_pitch']*DEG:+6.1f}d", C_TXT))
        col1.append(("gaze_yaw", f"{f['gaze_yaw']:+.3f} rad  {f['gaze_yaw']*DEG:+6.1f}d", C_TXT))
    else:
        col1.append(("gaze_pitch", "-- (press g)", C_DIM))
        col1.append(("gaze_yaw", "-- (press g)", C_DIM))

    col1.append(("", None, None))
    col1.append(("HEAD  (3D face matrix)", None, C_HEAD))
    col1.append(("head_pitch", f"{fmt(f['head_pitch'], 2)} deg", C_TXT))
    col1.append(("head_yaw", f"{fmt(f['head_yaw'], 2)} deg", C_TXT))
    col1.append(("head_roll", f"{fmt(f['head_roll'], 2)} deg", C_TXT))
    col1.append(("head_tx", fmt(f["head_tx"], 2), C_TXT))
    col1.append(("head_ty", fmt(f["head_ty"], 2), C_TXT))
    col1.append(("head_tz", fmt(f["head_tz"], 2), C_TXT))

    col1.append(("", None, None))
    col1.append(("IRIS IN EYE  (0=outer 1=inner)", None, C_IRIS))
    col1.append(("iris_left_ratio_x", fmt(f["iris_left_ratio_x"]), C_TXT))
    col1.append(("iris_left_ratio_y", fmt(f["iris_left_ratio_y"]), C_TXT))
    col1.append(("iris_right_ratio_x", fmt(f["iris_right_ratio_x"]), C_TXT))
    col1.append(("iris_right_ratio_y", fmt(f["iris_right_ratio_y"]), C_TXT))

    col1.append(("", None, None))
    col1.append(("EYE OPENNESS", None, C_EAR))
    col1.append(("ear_left", fmt(f["ear_left"]), C_TXT))
    col1.append(("ear_right", fmt(f["ear_right"]), C_TXT))

    col2.append(("IRIS POSITION  (in image)", None, C_IRIS))
    col2.append(("iris_left_x", fmt(f["iris_left_x"]), C_TXT))
    col2.append(("iris_left_y", fmt(f["iris_left_y"]), C_TXT))
    col2.append(("iris_right_x", fmt(f["iris_right_x"]), C_TXT))
    col2.append(("iris_right_y", fmt(f["iris_right_y"]), C_TXT))

    col2.append(("", None, None))
    col2.append(("EYE CORNERS", None, C_EYE))
    for name in ("eye_left_outer", "eye_left_inner",
                 "eye_right_inner", "eye_right_outer"):
        col2.append((name, f"{fmt(f[name + '_x'])}, {fmt(f[name + '_y'])}", C_TXT))

    col2.append(("", None, None))
    col2.append(("FACE SIZE", None, C_FACE))
    col2.append(("face_width", fmt(f["face_width"]), C_TXT))
    col2.append(("face_height", fmt(f["face_height"]), C_TXT))
    col2.append(("ipd_px", fmt(f["ipd_px"], 1), C_TXT))

    col2.append(("", None, None))
    col2.append(("POSITION  (gate)", None, C_POS))
    col2.append(("pos_distance_cm", fmt(f["pos_distance_cm"], 1), C_TXT))
    col2.append(("pos_face_dx", fmt(f["pos_face_dx"]), C_TXT))
    col2.append(("pos_face_dy", fmt(f["pos_face_dy"]), C_TXT))

    def render(items, x, y0):
        y = y0
        for label, value, colour in items:
            if not label:
                y += 9
                continue
            if value is None:                      # group heading
                cv2.putText(panel, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, colour, 1)
                y += 18
            else:
                cv2.putText(panel, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.36, C_DIM, 1)
                cv2.putText(panel, value, (x + 118, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.36, colour, 1)
                y += 17
        return y

    render(col1, 14, 82)
    render(col2, 300, 82)

    # Footer: FPS, zone, keys
    cv2.line(panel, (14, VIDEO_H - 62), (PANEL_W - 14, VIDEO_H - 62), (70, 65, 62), 1)
    zone_colour = {"GREEN": (0, 220, 0), "YELLOW": (0, 220, 220),
                   "RED": (0, 0, 255)}.get(zone, C_DIM)
    cv2.putText(panel, f"FPS {fps:.1f}", (14, VIDEO_H - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(panel, f"zone {zone}", (110, VIDEO_H - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone_colour, 1)
    cv2.putText(panel, f"gaze model {'ON' if gaze_on else 'OFF'}",
                (230, VIDEO_H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0) if gaze_on else C_DIM, 1)
    cv2.putText(panel, "q quit    g gaze model    l landmarks",
                (14, VIDEO_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_DIM, 1)
    return panel


def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    window_name = "Milestone 2 (v3) - Live Feature Readings"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_W)   # camera's native 16:9 resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_H)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    reset_bbox_smoothing()
    gate = PositioningGate()
    gaze_on = True
    show_landmarks = True
    prev_time = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, int(frame_idx * (1000 / 30)))
        frame_idx += 1

        features, zone = None, "-"
        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            matrix = (result.facial_transformation_matrixes[0]
                      if result.facial_transformation_matrixes else None)

            status = gate.evaluate(landmarks, frame.shape)
            zone = status["zone"] if status else "-"

            pitch = yaw = None
            if gaze_on:
                crop, _bbox = get_smoothed_square_crop(frame, landmarks)
                if crop is not None and crop.size > 0:
                    pitch, yaw = predict_gaze_from_crop(crop)

            features = extract_features(landmarks, frame.shape,
                                        gaze_pitch=pitch, gaze_yaw=yaw,
                                        transform_matrix=matrix,
                                        gate_status=status)
            if show_landmarks:
                draw_landmarks(frame, landmarks)

        now = time.time()
        fps = 1 / (now - prev_time) if prev_time else 0
        prev_time = now

        panel = draw_panel(features, gaze_on, fps, zone)
        cv2.imshow(window_name, np.hstack([frame, panel]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('g'):
            gaze_on = not gaze_on
        elif key == ord('l'):
            show_landmarks = not show_landmarks

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
