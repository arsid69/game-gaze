"""
positioning_gate.py — webcam-only distance + centering constraints.

Estimates the user's distance from the screen with the pinhole camera model
applied to the interpupillary distance (IPD): closer face -> bigger IPD in
pixels, exactly like a car reverse sensor's "closer = bigger" logic.

    distance_cm = (focal_px * REAL_IPD_CM) / ipd_px

The focal length defaults to an estimate from a typical webcam field of view,
and can optionally be refined once by sitting at a known distance and calling
calibrate_focal() (press 'c' in milestone5_positioning.py).

Also checks that the face (nose tip) stays near the frame center. Both
constraints together define the "restricted area" the user must sit in for
accurate gaze calibration.

No extra sensors, no extra dependencies — cv2 + numpy + math only.
"""

import json
import math
import pathlib
import time

import cv2
import numpy as np

# --- Tunable constraints (the "restricted area") ---
MIN_DIST_CM = 45.0
MAX_DIST_CM = 65.0
WARN_MARGIN_CM = 5.0        # yellow band inside the green zone edges
CENTER_TOL = 0.12           # nose may drift this fraction of frame w/h off center

REAL_IPD_CM = 6.3           # adult average interpupillary distance
# Typical laptop webcam horizontal FOV at the sensor's NATIVE 16:9 aspect.
# (Lenovo Legion R7000 ARP8 / Realtek 720p module: exact FOV unpublished —
# press 'c' in milestone5 at a measured 50 cm for an exact focal length.)
# NOTE: capture at native 1280x720; at 640x480 the sensor is center-cropped
# to 4:3, which changes the effective FOV and skews this default.
ASSUMED_HFOV_DEG = 60.0

FOCAL_PATH = pathlib.Path("models") / "camera_focal.json"

# MediaPipe FaceLandmarker indices
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
NOSE_TIP = 1

# Parking-display level segments: (lower_cm, upper_cm) from near to far.
LEVEL_STEP_CM = 5.0
LEVELS_MIN_CM = 30.0
LEVELS_MAX_CM = 80.0


def _default_focal(frame_w):
    return frame_w / (2.0 * math.tan(math.radians(ASSUMED_HFOV_DEG) / 2.0))


class PositioningGate:
    def __init__(self):
        self._focal_override = self._load_focal()

    # ---------- focal length ----------

    def _load_focal(self):
        try:
            with open(FOCAL_PATH) as f:
                data = json.load(f)
            return float(data["focal_px"]), int(data["frame_w"])
        except (OSError, KeyError, ValueError):
            return None

    def focal_px(self, frame_w):
        if self._focal_override is not None:
            focal, calib_w = self._focal_override
            return focal * (frame_w / calib_w)  # rescale if resolution changed
        return _default_focal(frame_w)

    def calibrate_focal(self, ipd_px, frame_w, known_dist_cm=50.0):
        """One-time refinement: call while sitting at known_dist_cm."""
        focal = ipd_px * known_dist_cm / REAL_IPD_CM
        FOCAL_PATH.parent.mkdir(exist_ok=True)
        with open(FOCAL_PATH, "w") as f:
            json.dump({"focal_px": focal, "frame_w": frame_w}, f)
        self._focal_override = (focal, frame_w)
        return focal

    # ---------- evaluation ----------

    def evaluate(self, landmarks, frame_shape):
        """
        landmarks: MediaPipe face landmark list (must include iris points).
        Returns a status dict:
          distance_cm, dx, dy (nose offset as fraction of frame, signed),
          distance_ok, centered, in_zone, zone ('GREEN'|'YELLOW'|'RED'),
          messages (list of guidance strings).
        """
        h, w = frame_shape[:2]

        li = landmarks[LEFT_IRIS_CENTER]
        ri = landmarks[RIGHT_IRIS_CENTER]
        ipd_px = math.hypot((ri.x - li.x) * w, (ri.y - li.y) * h)
        if ipd_px < 1:
            return None
        distance_cm = self.focal_px(w) * REAL_IPD_CM / ipd_px

        nose = landmarks[NOSE_TIP]
        dx = nose.x - 0.5
        dy = nose.y - 0.5

        distance_ok = MIN_DIST_CM <= distance_cm <= MAX_DIST_CM
        centered = abs(dx) <= CENTER_TOL and abs(dy) <= CENTER_TOL
        in_zone = distance_ok and centered

        messages = []
        if distance_cm < MIN_DIST_CM:
            messages.append("MOVE BACK")
        elif distance_cm > MAX_DIST_CM:
            messages.append("COME CLOSER")
        if dx > CENTER_TOL:
            messages.append("MOVE LEFT")
        elif dx < -CENTER_TOL:
            messages.append("MOVE RIGHT")
        if dy > CENTER_TOL:
            messages.append("SIT UP / RAISE CAMERA")
        elif dy < -CENTER_TOL:
            messages.append("LOWER YOURSELF / CAMERA")

        if not in_zone:
            zone = "RED"
        elif (distance_cm < MIN_DIST_CM + WARN_MARGIN_CM
              or distance_cm > MAX_DIST_CM - WARN_MARGIN_CM
              or abs(dx) > CENTER_TOL * 0.7 or abs(dy) > CENTER_TOL * 0.7):
            zone = "YELLOW"
        else:
            zone = "GREEN"

        return {
            "distance_cm": distance_cm,
            "dx": dx, "dy": dy,
            "distance_ok": distance_ok,
            "centered": centered,
            "in_zone": in_zone,
            "zone": zone,
            "messages": messages,
        }

    # ---------- drawing ----------

    def draw_overlay(self, frame, status):
        """Car-reverse-style UI: level bars, center target, guidance, border pulse."""
        h, w = frame.shape[:2]

        if status is None:
            cv2.putText(frame, "NO FACE", (w // 2 - 70, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return frame

        self._draw_distance_levels(frame, status)
        self._draw_center_target(frame, status)

        # Guidance text (top center)
        for i, msg in enumerate(status["messages"]):
            (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(frame, msg, ((w - tw) // 2, 40 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Border pulse: flashes faster the further out of zone (the "beep")
        if not status["in_zone"]:
            over = 0.0
            d = status["distance_cm"]
            if d < MIN_DIST_CM:
                over = max(over, (MIN_DIST_CM - d) / 15.0)
            elif d > MAX_DIST_CM:
                over = max(over, (d - MAX_DIST_CM) / 15.0)
            over = max(over,
                       (abs(status["dx"]) - CENTER_TOL) / CENTER_TOL,
                       (abs(status["dy"]) - CENTER_TOL) / CENTER_TOL)
            freq = 2.0 + 6.0 * min(max(over, 0.0), 1.0)
            if math.sin(time.time() * freq * 2 * math.pi) > 0:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)
        return frame

    def _draw_distance_levels(self, frame, status):
        """Stacked parking-sensor level segments on the right edge."""
        h, w = frame.shape[:2]
        d = status["distance_cm"]

        n_levels = int(round((LEVELS_MAX_CM - LEVELS_MIN_CM) / LEVEL_STEP_CM))
        bar_w = 78
        margin = 10
        x0 = w - bar_w - margin
        seg_gap = 4
        top, bottom = 60, h - 40
        seg_h = (bottom - top - seg_gap * (n_levels - 1)) // n_levels

        # Big cm readout above the stack
        label = f"{d:.0f} cm"
        color = {"GREEN": (0, 220, 0), "YELLOW": (0, 220, 220), "RED": (0, 0, 255)}[status["zone"]]
        cv2.putText(frame, label, (x0 - 8, top - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Segments: nearest range at the BOTTOM (like a car display), far at top
        for i in range(n_levels):
            lo = LEVELS_MIN_CM + i * LEVEL_STEP_CM
            hi = lo + LEVEL_STEP_CM
            y1 = bottom - i * (seg_h + seg_gap)
            y0 = y1 - seg_h

            in_green = lo >= MIN_DIST_CM and hi <= MAX_DIST_CM
            seg_color = (0, 200, 0) if in_green else (0, 0, 220)
            # yellow warn segments just inside the green edges
            if in_green and (lo < MIN_DIST_CM + WARN_MARGIN_CM or hi > MAX_DIST_CM - WARN_MARGIN_CM):
                seg_color = (0, 200, 200)

            is_current = lo <= d < hi
            if is_current:
                cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y1), seg_color, -1)
                cv2.rectangle(frame, (x0 - 2, y0 - 2), (x0 + bar_w + 2, y1 + 2), (255, 255, 255), 2)
                txt_color = (0, 0, 0)
            else:
                cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y1), seg_color, 1)
                txt_color = seg_color
            cv2.putText(frame, f"{lo:.0f}-{hi:.0f}", (x0 + 5, y1 - seg_h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, txt_color, 1)

    def _draw_center_target(self, frame, status):
        """Tolerance box + crosshair at center, dot at nose position."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        tol_x, tol_y = int(CENTER_TOL * w), int(CENTER_TOL * h)

        box_color = (0, 220, 0) if status["centered"] else (0, 0, 255)
        cv2.rectangle(frame, (cx - tol_x, cy - tol_y), (cx + tol_x, cy + tol_y), box_color, 1)
        cv2.line(frame, (cx - 12, cy), (cx + 12, cy), box_color, 1)
        cv2.line(frame, (cx, cy - 12), (cx, cy + 12), box_color, 1)

        nx = int((0.5 + status["dx"]) * w)
        ny = int((0.5 + status["dy"]) * h)
        cv2.circle(frame, (nx, ny), 6, box_color, -1)
