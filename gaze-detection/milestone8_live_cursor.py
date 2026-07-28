"""
Milestone 8 — Live gaze cursor (free-movement test).

A cursor follows your gaze in real time so you can test the calibration by
just looking around: sweep the corners, trace the edges, jump between spots.
A fading trail shows the path your gaze took.

Modes (toggle with 'm'):
  WINDOW mode (default) — cursor drawn on the fullscreen canvas.
  MOUSE mode            — your gaze moves the REAL Windows mouse pointer.

Requires: models/calibration_model.pkl (run milestone4_calibration.py first).
Run: python milestone8_live_cursor.py
Keys: q quit, m toggle mouse control, t toggle trail.
"""

import ctypes
import time

import cv2
import numpy as np

from calibration_utils import load_calibration
from gaze_pipeline import reset_bbox_smoothing
from positioning_gate import PositioningGate

from milestone4_calibration import (
    FrameReader, phase_a_positioning, measure_drift, aborted, CANVAS_W, CANVAS_H,
)
from milestone7_gaze_game import GazeCursor

TRAIL_SECONDS = 1.5          # how long the trail stays visible


def screen_resolution():
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def main():
    try:
        model = load_calibration()
    except FileNotFoundError:
        print("No calibration found - run milestone4_calibration.py first.")
        return

    scr_w, scr_h = screen_resolution()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # camera's native 16:9 resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    window_name = "Live Gaze Cursor"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    reset_bbox_smoothing()
    reader = FrameReader(cap, PositioningGate())

    if not phase_a_positioning(reader, window_name):
        print("Aborted during positioning.")
        cap.release()
        cv2.destroyAllWindows()
        return

    drift = measure_drift(reader, window_name, model)
    if drift is None:
        print("Aborted during drift check.")
        cap.release()
        cv2.destroyAllWindows()
        return

    cursor = GazeCursor(model, drift)
    trail = []               # list of (x_px, y_px, timestamp)
    mouse_mode = False
    show_trail = True
    prev_time = 0

    try:
        while True:
            reading = reader.read(with_gaze=True)
            now = time.time()

            gaze_px = None
            if reading["pitch"] is not None:
                gx, gy = cursor.update(reading["pitch"], reading["yaw"])
                gaze_px = (int(gx * CANVAS_W), int(gy * CANVAS_H))
                trail.append((gaze_px[0], gaze_px[1], now))
                if mouse_mode:
                    ctypes.windll.user32.SetCursorPos(
                        int(gx * scr_w), int(gy * scr_h))

            trail = [p for p in trail if now - p[2] < TRAIL_SECONDS]

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            mode_txt = "MOUSE mode - gaze moves the real pointer" if mouse_mode \
                else "WINDOW mode"
            cv2.putText(canvas, f"{mode_txt}   [m] toggle mouse  [t] trail  [q] quit",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if show_trail and len(trail) > 1:
                for i in range(1, len(trail)):
                    age = (now - trail[i][2]) / TRAIL_SECONDS
                    c = int(200 * (1 - age))
                    cv2.line(canvas, trail[i - 1][:2], trail[i][:2],
                             (c, c // 2, 0), 2)

            if gaze_px is not None:
                cv2.circle(canvas, gaze_px, 18, (255, 200, 0), 2)
                cv2.circle(canvas, gaze_px, 4, (255, 200, 0), -1)

            status = reading["status"]
            if status is not None and not status["in_zone"]:
                cv2.putText(canvas, "HOLD POSITION - " + " / ".join(status["messages"]),
                            (30, CANVAS_H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)
            elif status is None:
                cv2.putText(canvas, "NO FACE", (30, CANVAS_H - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            fps = 1 / (now - prev_time) if prev_time else 0
            prev_time = now
            cv2.putText(canvas, f"FPS: {fps:.1f}", (30, CANVAS_H - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                mouse_mode = not mouse_mode
            elif key == ord('t'):
                show_trail = not show_trail
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
