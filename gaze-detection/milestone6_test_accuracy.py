"""
Milestone 6 — Accuracy test with interpreted results.

Loads the saved calibration (run milestone4_calibration.py first), enforces
the same positioning gate it was calibrated under, then shows a 3x3 grid of
test dots the model has never seen. For each dot it compares where you were
told to look vs. where the model says you looked.

Errors are reported in real centimeters on YOUR screen (physical size read
from Windows), drawn on a results map, and interpreted in plain language.

Run: python milestone6_test_accuracy.py
Press 'q' to abort.
"""

import ctypes
import math
import time

import cv2
import numpy as np

from calibration_utils import load_calibration, apply_calibration
from gaze_pipeline import reset_bbox_smoothing
from positioning_gate import PositioningGate

# Reuse milestone 4's capture/gate machinery and canvas size.
from milestone4_calibration import (
    FrameReader, phase_a_positioning, measure_drift, aborted,
    CANVAS_W, CANVAS_H, DWELL_TIME, SAMPLE_TIME,
)

# 3x3 grid including the TRUE screen corners (5% in, same as the pursuit
# margin) — tests the model right where extrapolation is hardest.
TEST_POINTS = [(x, y) for y in (0.05, 0.5, 0.95) for x in (0.05, 0.5, 0.95)]


def screen_size_cm():
    """Physical screen size reported by Windows (falls back to 15.6\" 16:9)."""
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        dc = user32.GetDC(0)
        gdi32 = ctypes.windll.gdi32
        w_mm = gdi32.GetDeviceCaps(dc, 4)   # HORZSIZE
        h_mm = gdi32.GetDeviceCaps(dc, 6)   # VERTSIZE
        user32.ReleaseDC(0, dc)
        if w_mm > 0 and h_mm > 0:
            return w_mm / 10.0, h_mm / 10.0
    except Exception:
        pass
    return 34.4, 19.4


def measure_points(reader, window_name, model, drift=(0.0, 0.0)):
    """Dwell + sample each test dot. Returns list of result dicts (or None)."""
    results = []
    for i, (fx, fy) in enumerate(TEST_POINTS):
        px, py = int(fx * CANVAS_W), int(fy * CANVAS_H)
        readings = []

        for label, duration, color, collect in [
                ("look at the dot", DWELL_TIME, (0, 165, 255), False),
                ("measuring...", SAMPLE_TIME, (0, 255, 0), True)]:
            phase_start = time.time()
            while time.time() - phase_start < duration:
                canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
                cv2.putText(canvas, f"Accuracy test {i + 1}/{len(TEST_POINTS)} - {label}",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.circle(canvas, (px, py), 18, color, -1)
                cv2.imshow(window_name, canvas)

                reading = reader.read(with_gaze=True)
                if collect and reading["pitch"] is not None:
                    readings.append((reading["pitch"], reading["yaw"]))
                if aborted():
                    return None

        if readings:
            med_pitch = float(np.median([r[0] for r in readings]))
            med_yaw = float(np.median([r[1] for r in readings]))
            pred_x, pred_y = apply_calibration(model, med_pitch, med_yaw)
            pred_x = min(max(pred_x - drift[0], 0.0), 1.0)
            pred_y = min(max(pred_y - drift[1], 0.0), 1.0)
            results.append({"target": (fx, fy), "pred": (pred_x, pred_y),
                            "n": len(readings)})
        else:
            results.append({"target": (fx, fy), "pred": None, "n": 0})
    return results


def error_cm(result, scr_w_cm, scr_h_cm):
    fx, fy = result["target"]
    px, py = result["pred"]
    return math.hypot((px - fx) * scr_w_cm, (py - fy) * scr_h_cm)


def interpret(results, scr_w_cm, scr_h_cm):
    """Build plain-language conclusions from the per-point errors."""
    valid = [r for r in results if r["pred"] is not None]
    lines = []
    if not valid:
        return ["No valid measurements - check lighting and stay in the green zone."], None

    errs = [error_cm(r, scr_w_cm, scr_h_cm) for r in valid]
    avg, worst = float(np.mean(errs)), float(np.max(errs))
    worst_pt = valid[int(np.argmax(errs))]["target"]

    if avg < 1.5:
        rating = "EXCELLENT - near commercial eye-tracker territory."
    elif avg < 3.0:
        rating = "GOOD - reliable for button-sized targets (~3 cm)."
    elif avg < 5.5:
        rating = "FAIR - trust it for screen regions/quadrants, not buttons."
    else:
        rating = "POOR - recalibrate (see advice below)."
    lines.append(f"Average error: {avg:.1f} cm  |  worst: {worst:.1f} cm  ->  {rating}")

    # Center vs edge comparison
    center_errs = [error_cm(r, scr_w_cm, scr_h_cm) for r in valid
                   if r["target"] == (0.5, 0.5)]
    edge_errs = [error_cm(r, scr_w_cm, scr_h_cm) for r in valid
                 if r["target"] != (0.5, 0.5)]
    if center_errs and edge_errs:
        c, e = np.mean(center_errs), np.mean(edge_errs)
        if e > 2 * c:
            lines.append(f"Center is strong ({c:.1f} cm) but edges drift ({e:.1f} cm): "
                         "raise POLY_DEGREE to 3 in milestone4 and recalibrate.")
        elif c > 2 * e:
            lines.append(f"Edges are fine ({e:.1f} cm) but center drifts ({c:.1f} cm): "
                         "you likely shifted position since calibrating - recalibrate.")
        else:
            lines.append(f"Even accuracy across the screen "
                         f"(center {c:.1f} cm vs edges {e:.1f} cm).")

    lines.append(f"Weakest spot: ({worst_pt[0]:.0%}, {worst_pt[1]:.0%}) of screen.")

    missing = len(results) - len(valid)
    if missing:
        lines.append(f"{missing} point(s) got no samples - lighting or positioning issue.")
    if avg >= 5.5:
        lines.append("Advice: redo focal calibration ('c' in milestone5 at 50 cm), "
                     "then recalibrate milestone4 sitting exactly where you'll test.")
    return lines, avg


def show_results_screen(window_name, results, lines, scr_w_cm, scr_h_cm):
    """Map of target dots vs predicted gaze + the interpretation text.

    The map is drawn inside its own bordered area (not at literal screen
    coordinates) so corner dots and their labels never collide with the
    title or the summary text below.
    """
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    cv2.putText(canvas, "Results - o target, x where the model thinks you looked",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    text_h = 30 * len(lines) + 30
    map_x0, map_y0 = 80, 70
    map_x1, map_y1 = CANVAS_W - 80, CANVAS_H - text_h - 30
    map_w, map_h = map_x1 - map_x0, map_y1 - map_y0
    cv2.rectangle(canvas, (map_x0, map_y0), (map_x1, map_y1), (80, 80, 80), 1)

    def to_map(nx, ny):
        return int(map_x0 + nx * map_w), int(map_y0 + ny * map_h)

    for r in results:
        tx, ty = to_map(*r["target"])
        cv2.circle(canvas, (tx, ty), 14, (0, 165, 255), 2)
        # Put the label on whichever side has room
        label_dx = -95 if r["target"][0] > 0.8 else 18
        if r["pred"] is None:
            cv2.putText(canvas, "n/a", (tx + label_dx, ty + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
            continue
        px_, py_ = to_map(*r["pred"])
        err = error_cm(r, scr_w_cm, scr_h_cm)
        color = (0, 220, 0) if err < 3.0 else (0, 0, 255)
        cv2.line(canvas, (tx, ty), (px_, py_), color, 1)
        cv2.drawMarker(canvas, (px_, py_), color, cv2.MARKER_TILTED_CROSS, 16, 2)
        cv2.putText(canvas, f"{err:.1f}cm", (tx + label_dx, ty + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    y = map_y1 + 40
    for line in lines:
        cv2.putText(canvas, line, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 1)
        y += 30

    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)


def main():
    try:
        model = load_calibration()
    except FileNotFoundError:
        print("No calibration found - run milestone4_calibration.py first.")
        return

    scr_w_cm, scr_h_cm = screen_size_cm()
    print(f"Screen physical size: {scr_w_cm:.1f} x {scr_h_cm:.1f} cm")
    if model.get("validation_error") is not None:
        print(f"Calibration's own validation error was "
              f"{model['validation_error'] * 100:.1f}% of screen.")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # camera's native 16:9 resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    window_name = "Accuracy Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    reset_bbox_smoothing()
    reader = FrameReader(cap, PositioningGate())

    try:
        if not phase_a_positioning(reader, window_name):
            print("Aborted during positioning.")
            return

        drift = measure_drift(reader, window_name, model)
        if drift is None:
            print("Aborted during drift check.")
            return
        print(f"Session drift offset: ({drift[0] * 100:+.1f}%, {drift[1] * 100:+.1f}%) "
              "of screen - subtracted from all predictions.")

        results = measure_points(reader, window_name, model, drift)
        if results is None:
            print("Aborted during measurement.")
            return

        lines, avg = interpret(results, scr_w_cm, scr_h_cm)
        print("\n--- Accuracy report ---")
        for r in results:
            if r["pred"] is None:
                print(f"  ({r['target'][0]:.0%}, {r['target'][1]:.0%}): no samples")
            else:
                print(f"  ({r['target'][0]:.0%}, {r['target'][1]:.0%}): "
                      f"{error_cm(r, scr_w_cm, scr_h_cm):.1f} cm off "
                      f"({r['n']} samples)")
        for line in lines:
            print(line)

        show_results_screen(window_name, results, lines, scr_w_cm, scr_h_cm)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
