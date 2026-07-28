"""
Milestone 4 (v3) — Gated smooth-pursuit calibration with held-out validation.

Three phases:
  A. Positioning gate — the user must sit inside the restricted area
     (distance band + centered face, see positioning_gate.py) and hold a
     GREEN status for 2 consecutive seconds before calibration starts.
  B. Smooth-pursuit training sweep — a dot glides in a snake path across
     the whole screen (~1 min). Gaze readings are recorded continuously
     (~hundreds of samples instead of 9 medians). Samples are rejected when
     the eye is still catching up after a turn, when the user drifts out of
     the positioning zone, or as robust-fit outliers (blinks/glances).
  C. Validation — 5 held-out static dots measure the real on-screen error
     of the fitted model, displayed before saving.

Run: python milestone4_calibration.py
Press 'q' to abort.
"""

import math
import time
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gaze_pipeline import detect_face, predict_gaze_from_crop, reset_bbox_smoothing
from gaze_features import extract_features
from calibration_utils import (
    save_calibration, apply_calibration, append_dataset, robust_fit_samples,
    SOURCE_PURSUIT,
)
from positioning_gate import PositioningGate, MIN_DIST_CM, MAX_DIST_CM, CENTER_TOL

FACE_MODEL_PATH = "face_landmarker.task"
CANVAS_W, CANVAS_H = 1280, 720

HOLD_GREEN_SECS = 2.0        # phase A: required stable positioning time
PURSUIT_SPEED = 0.12         # phase B: dot speed in screen-fractions per second
PURSUIT_MARGIN = 0.05        # fraction of screen kept clear at the edges
PURSUIT_ROWS = 5             # horizontal sweeps, top to bottom
SETTLE_TIME = 0.3            # seconds to skip after each direction change
OUTLIER_MAD_FACTOR = 2.5     # robust-refit rejection threshold
POLY_DEGREE = 3              # cubic: captures the bottom-of-screen distortion
                             # (looking down = drooping eyelids = weaker signal);
                             # safe with the ~400 samples the full sweep collects
MIN_TRAIN_SAMPLES = 60

DWELL_TIME = 1.2             # phase C dwell before sampling a dot
SAMPLE_TIME = 1.5            # phase C sampling window per dot

# Held-out validation points (fractions of screen). Chosen off the pursuit
# rows so they genuinely test interpolation.
VALIDATION_POINTS = [
    (0.25, 0.25), (0.75, 0.25),
    (0.5, 0.5),
    (0.25, 0.75), (0.75, 0.75),
]

base_options = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_facial_transformation_matrixes=True,  # head pose for the feature set
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(face_options)


def build_pursuit_path():
    """Snake rows + a full perimeter loop + a diagonal.

    The snake alone only touches the screen edges for an instant at each
    turn (and the settle-time filter drops the samples right after a turn),
    so the fit used to extrapolate at the corners — measured as systematic
    right-edge / corner errors. The perimeter loop gives every edge and
    corner a dedicated slow pass, approached from two directions.
    Returns list of (x_norm, y_norm) polyline corners.
    """
    m = PURSUIT_MARGIN
    ys = np.linspace(m, 1 - m, PURSUIT_ROWS)
    pts = []
    for i, y in enumerate(ys):
        if i % 2 == 0:
            pts += [(m, y), (1 - m, y)]
        else:
            pts += [(1 - m, y), (m, y)]
    # PURSUIT_ROWS is odd, so the snake ends bottom-right at (1-m, 1-m).
    # Perimeter: up the right edge, across the top, down the left, along
    # the bottom — then a diagonal through the center back to the top-left.
    pts += [(1 - m, m), (m, m), (m, 1 - m), (1 - m, 1 - m), (m, m)]
    return pts


def path_length(pts):
    return sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
               for i in range(len(pts) - 1))


def path_position(pts, t_frac):
    """Point on the polyline at constant speed. Returns (x, y, segment_idx)."""
    seg_lens = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                for i in range(len(pts) - 1)]
    total = sum(seg_lens)
    target = min(max(t_frac, 0.0), 1.0) * total
    acc = 0.0
    for i, seg_len in enumerate(seg_lens):
        if acc + seg_len >= target or i == len(seg_lens) - 1:
            f = (target - acc) / seg_len if seg_len > 0 else 0.0
            x = pts[i][0] + f * (pts[i + 1][0] - pts[i][0])
            y = pts[i][1] + f * (pts[i + 1][1] - pts[i][1])
            return x, y, i
        acc += seg_len
    return pts[-1][0], pts[-1][1], len(seg_lens) - 1


class FrameReader:
    """Wraps webcam grab + face detect + gate evaluation + gaze prediction."""

    def __init__(self, cap, gate):
        self.cap = cap
        self.gate = gate
        self.frame_idx = 0

    def read(self, with_gaze=True, with_features=False):
        """
        Returns dict {status, pitch, yaw, features}:
          status   — gate status (None if no frame/face)
          pitch/yaw— gaze angles, only when with_gaze and in zone
          features — full per-frame feature dict when with_features, else None
        """
        empty = {"status": None, "pitch": None, "yaw": None, "features": None}
        ret, frame = self.cap.read()
        if not ret:
            return empty
        frame = cv2.flip(frame, 1)
        landmarks, bbox, matrix = detect_face(frame, face_landmarker, self.frame_idx)
        self.frame_idx += 1
        if landmarks is None:
            return empty

        status = self.gate.evaluate(landmarks, frame.shape)
        pitch = yaw = None
        if with_gaze and status is not None and status["in_zone"]:
            x_min, y_min, x_max, y_max = bbox
            crop = frame[y_min:y_max, x_min:x_max]
            if crop.size > 0:
                pitch, yaw = predict_gaze_from_crop(crop)

        features = None
        if with_features:
            features = extract_features(landmarks, frame.shape,
                                        gaze_pitch=pitch, gaze_yaw=yaw,
                                        transform_matrix=matrix,
                                        gate_status=status)
        return {"status": status, "pitch": pitch, "yaw": yaw, "features": features}


def aborted():
    return cv2.waitKey(1) & 0xFF == ord('q')


def phase_a_positioning(reader, window_name):
    """Block until the user holds a GREEN positioning status for HOLD_GREEN_SECS."""
    green_since = None
    while True:
        reading = reader.read(with_gaze=False)
        status = reading["status"]

        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        cv2.putText(canvas, "STEP 1/3 - Get into position",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas,
                    f"Sit {MIN_DIST_CM:.0f}-{MAX_DIST_CM:.0f} cm away, face centered.",
                    (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if status is None:
            cv2.putText(canvas, "NO FACE DETECTED", (CANVAS_W // 2 - 160, CANVAS_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            green_since = None
        else:
            zone_color = {"GREEN": (0, 220, 0), "YELLOW": (0, 220, 220),
                          "RED": (0, 0, 255)}[status["zone"]]
            cv2.putText(canvas, f"Distance: {status['distance_cm']:.0f} cm",
                        (CANVAS_W // 2 - 130, CANVAS_H // 2 - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, zone_color, 2)
            for i, msg in enumerate(status["messages"]):
                cv2.putText(canvas, msg, (CANVAS_W // 2 - 130, CANVAS_H // 2 + i * 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            if status["zone"] == "GREEN":
                if green_since is None:
                    green_since = time.time()
                held = time.time() - green_since
                if held >= HOLD_GREEN_SECS:
                    return True
                frac = held / HOLD_GREEN_SECS
                cv2.putText(canvas, "HOLD STILL...",
                            (CANVAS_W // 2 - 110, CANVAS_H // 2 + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 0), 2)
                cv2.rectangle(canvas, (CANVAS_W // 4, CANVAS_H - 80),
                              (CANVAS_W // 4 + int(frac * CANVAS_W // 2), CANVAS_H - 50),
                              (0, 220, 0), -1)
                cv2.rectangle(canvas, (CANVAS_W // 4, CANVAS_H - 80),
                              (3 * CANVAS_W // 4, CANVAS_H - 50), (255, 255, 255), 2)
            else:
                green_since = None

        cv2.imshow(window_name, canvas)
        if aborted():
            return False


def phase_b_pursuit(reader, window_name):
    """Moving-dot sweep. Returns list of (pitch, yaw, x_norm, y_norm) samples,
    or None if aborted."""
    pts = build_pursuit_path()
    duration = path_length(pts) / PURSUIT_SPEED  # constant slow speed everywhere
    samples = []
    start = time.time()
    last_segment = -1
    segment_entered = start

    while True:
        elapsed = time.time() - start
        if elapsed >= duration:
            break
        x, y, seg = path_position(pts, elapsed / duration)
        if seg != last_segment:
            last_segment = seg
            segment_entered = time.time()

        px, py = int(x * CANVAS_W), int(y * CANVAS_H)
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        cv2.putText(canvas, "STEP 2/3 - Follow the dot with your eyes",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.circle(canvas, (px, py), 16, (0, 165, 255), -1)
        cv2.circle(canvas, (px, py), 5, (255, 255, 255), -1)

        reading = reader.read(with_gaze=True, with_features=True)
        status = reading["status"]
        settled = time.time() - segment_entered >= SETTLE_TIME

        if status is not None and not status["in_zone"]:
            cv2.putText(canvas, "HOLD POSITION - " + " / ".join(status["messages"]),
                        (30, CANVAS_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif reading["pitch"] is not None and settled:
            # Full feature row; the current polynomial fit uses only the two
            # gaze angles, the rest is banked for richer models.
            row = dict(reading["features"])
            row["target_x"], row["target_y"] = x, y
            samples.append(row)

        cv2.putText(canvas, f"samples: {len(samples)}", (CANVAS_W - 220, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        cv2.putText(canvas, f"{duration - elapsed:.0f}s left", (CANVAS_W - 220, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        cv2.imshow(window_name, canvas)
        if aborted():
            return None
    return samples


def robust_fit(samples):
    """Fit, reject residual outliers (blinks/glances) via MAD, refit."""
    return robust_fit_samples(samples, degree=POLY_DEGREE,
                              mad_factor=OUTLIER_MAD_FACTOR,
                              min_keep=MIN_TRAIN_SAMPLES)


def measure_drift(reader, window_name, model):
    """One-dot drift correction, done at the START of any session that uses
    a saved calibration. You never sit in exactly the calibration posture
    again; a 1-2 degree head-pitch difference shifts ALL predictions by a
    constant offset. Measure it on the center dot and subtract it.

    Returns (dx, dy) normalized offset to subtract, or None if aborted.
    """
    cx, cy = CANVAS_W // 2, CANVAS_H // 2
    readings = []
    for label, duration, color, collect in [
            ("look at the center dot", DWELL_TIME, (0, 165, 255), False),
            ("measuring drift...", SAMPLE_TIME, (0, 255, 0), True)]:
        phase_start = time.time()
        while time.time() - phase_start < duration:
            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            cv2.putText(canvas, f"Drift check - {label}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.circle(canvas, (cx, cy), 18, color, -1)
            cv2.imshow(window_name, canvas)
            reading = reader.read(with_gaze=True)
            if collect and reading["pitch"] is not None:
                readings.append((reading["pitch"], reading["yaw"]))
            if aborted():
                return None
    if not readings:
        return 0.0, 0.0
    med_pitch = float(np.median([r[0] for r in readings]))
    med_yaw = float(np.median([r[1] for r in readings]))
    pred_x, pred_y = apply_calibration(model, med_pitch, med_yaw)
    return pred_x - 0.5, pred_y - 0.5


def rest_screen(window_name, seconds=5.0):
    """Short break after the pursuit sweep — tired eyes ruin validation."""
    start = time.time()
    while time.time() - start < seconds:
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        cv2.putText(canvas, "Sweep done - close your eyes and rest a moment.",
                    (CANVAS_W // 2 - 340, CANVAS_H // 2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(canvas, f"Validation starts in {seconds - (time.time() - start):.0f}s",
                    (CANVAS_W // 2 - 180, CANVAS_H // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        cv2.imshow(window_name, canvas)
        if aborted():
            return False
    return True


def phase_c_validation(reader, window_name, model):
    """Measure held-out error on static dots. Returns mean normalized error,
    or None if aborted / no valid points."""
    errors = []
    for i, (fx, fy) in enumerate(VALIDATION_POINTS):
        px, py = int(fx * CANVAS_W), int(fy * CANVAS_H)
        point_readings = []

        for phase_label, duration, color, collect in [
                ("look at the dot", DWELL_TIME, (0, 165, 255), False),
                ("measuring...", SAMPLE_TIME, (0, 255, 0), True)]:
            phase_start = time.time()
            while time.time() - phase_start < duration:
                canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
                cv2.putText(canvas,
                            f"STEP 3/3 - Validation {i + 1}/{len(VALIDATION_POINTS)} - {phase_label}",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.circle(canvas, (px, py), 18, color, -1)
                cv2.imshow(window_name, canvas)

                reading = reader.read(with_gaze=True)
                if collect and reading["pitch"] is not None:
                    point_readings.append((reading["pitch"], reading["yaw"]))
                if aborted():
                    return None

        if point_readings:
            med_pitch = float(np.median([r[0] for r in point_readings]))
            med_yaw = float(np.median([r[1] for r in point_readings]))
            pred_x, pred_y = apply_calibration(model, med_pitch, med_yaw)
            err = math.hypot(pred_x - fx, pred_y - fy)
            errors.append(err)
            print(f"Validation point {i + 1}: error = {err * 100:.1f}% of screen")
        else:
            print(f"Validation point {i + 1}: no valid samples")

    return float(np.mean(errors)) if errors else None


def show_result_screen(window_name, val_error, n_samples):
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    cv2.putText(canvas, "Calibration complete", (CANVAS_W // 2 - 200, CANVAS_H // 2 - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 0), 2)
    if val_error is not None:
        cv2.putText(canvas, f"Held-out avg error: {val_error * 100:.1f}% of screen",
                    (CANVAS_W // 2 - 280, CANVAS_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(canvas, f"Training samples used: {n_samples}",
                (CANVAS_W // 2 - 280, CANVAS_H // 2 + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(canvas, "Press any key to finish", (CANVAS_W // 2 - 180, CANVAS_H - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)   # MSMF gives 720p @ 31fps (DSHOW capped it at 10fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # MUST match gaze_server.py runtime (1280x720),
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)   # so the model is trained on the same face-crop scale
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    window_name = "Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    reset_bbox_smoothing()
    gate = PositioningGate()
    reader = FrameReader(cap, gate)

    try:
        if not phase_a_positioning(reader, window_name):
            print("Aborted during positioning.")
            return

        samples = phase_b_pursuit(reader, window_name)
        if samples is None:
            print("Aborted during pursuit sweep.")
            return
        print(f"Pursuit sweep collected {len(samples)} in-zone samples.")
        append_dataset(samples, SOURCE_PURSUIT)
        if len(samples) < MIN_TRAIN_SAMPLES:
            print(f"Too few samples (<{MIN_TRAIN_SAMPLES}). Improve lighting / stay in "
                  "the green zone and re-run.")
            return

        # The polynomial fit consumes only the two gaze angles; the remaining
        # features are stored in the dataset for future models.
        fit_rows = [(s["gaze_pitch"], s["gaze_yaw"], s["target_x"], s["target_y"])
                    for s in samples]
        model, n_dropped = robust_fit(fit_rows)
        print(f"Fitted degree-{POLY_DEGREE} model "
              f"({n_dropped} outlier samples rejected).")

        if not rest_screen(window_name):
            print("Aborted during rest.")
            return

        val_error = phase_c_validation(reader, window_name, model)

        model["validation_error"] = val_error
        model["position_constraints"] = {
            "min_dist_cm": MIN_DIST_CM,
            "max_dist_cm": MAX_DIST_CM,
            "center_tol": CENTER_TOL,
        }
        save_calibration(model)
        print("Calibration saved to models/calibration_model.pkl")
        if val_error is not None:
            print(f"Held-out validation error: {val_error * 100:.1f}% of screen")

        show_result_screen(window_name, val_error, len(fit_rows))
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_landmarker.close()


if __name__ == "__main__":
    main()
