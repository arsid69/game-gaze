"""
Milestone 7 — Gaze target game (gamified accuracy testing).

Same mechanic as calibration data collection, flipped into a game: dots
appear one at a time and you "pop" them by LOOKING at them. Your live gaze
(calibrated model) is drawn as a cursor; hold it inside a target's ring to
charge it up and score. Faster hits score more. Miss = target times out.

The final score is itself a calibration test: a well-calibrated model makes
targets easy everywhere; if bottom-screen targets keep timing out, that is
exactly where your calibration is weak (compare milestone6's report).

The game is also a DATA COLLECTOR: while your gaze is settled on a target we
know exactly where you are looking, so those (pitch, yaw) -> (x, y) pairs are
banked to data/gaze_samples.csv, and after each game the calibration model is
retrained on the full accumulated dataset — accuracy improves as you play.

AIM ASSIST: once your gaze cursor gets near a target it is magnetically
pulled onto it (stronger the closer you are), like console-shooter aim assist.
The faint dot shows raw gaze; the bright ring is the assisted cursor.

Requires: models/calibration_model.pkl (run milestone4_calibration.py first).
Run: python milestone7_gaze_game.py
Keys: q quit, a toggle aim assist, d reveal the hidden assist zones.
"""

import math
import random
import time
from collections import deque

import cv2
import numpy as np

from calibration_utils import (
    load_calibration, save_calibration, apply_calibration,
    append_dataset, load_dataset, robust_fit_samples, SOURCE_GAME,
)
from gaze_pipeline import reset_bbox_smoothing
from positioning_gate import PositioningGate

from milestone4_calibration import (
    FrameReader, phase_a_positioning, measure_drift, aborted, CANVAS_W, CANVAS_H,
)

NUM_TARGETS = 10
TARGET_RADIUS = 70           # px: gaze must land within this ring
HOLD_TO_POP = 0.8            # seconds gaze must stay on target
TARGET_TIMEOUT = 10.0        # seconds before a target counts as a miss
MARGIN = 0.12                # keep targets away from the very edge
GAZE_SMOOTHING = 0.25        # EMA factor for the gaze cursor (lower = smoother)
BASE_POINTS = 100
SPEED_BONUS_PER_SEC = 20     # extra points per second left on the clock

# --- Aim assist: two-zone magnet (all sizes in canvas px; canvas is 1280x720) ---
# The visible ring is TARGET_RADIUS. The magnet field is much larger and is
# NOT drawn ("hidden rigging"): the cursor is dragged toward the target from
# anywhere inside it, and LOCKS onto the target center inside the snap zone.
# ASSIST_RADIUS must exceed the model's typical error (~4-6 cm ~= 150-220 px)
# or the assist never engages — that was the bug.
AIM_ASSIST = True
ASSIST_RADIUS = 330          # hidden magnet field (~9 cm on a 34 cm screen)
SNAP_RADIUS = 130            # inside this the cursor locks fully to the target
SHOW_ASSIST_DEBUG = False    # 'd' at runtime reveals the hidden zones

# --- Data collection ---
# A sample is only trustworthy when the user is REALLY looking at the target.
# We never use the assisted cursor for this decision (the assist would drag
# the cursor onto the target and label garbage). Instead: the RAW gaze must
# be stable (a fixation) and within a generous bound of the target.
FIXATION_MIN_HOLD = 0.25     # seconds the raw gaze must stay settled
FIXATION_STABILITY_PX = 120  # max spread of raw gaze to count as a fixation
FIXATION_MAX_RAW_DIST = 420  # raw gaze must be at least this close to the target
RETRAIN_MIN_SAMPLES = 200    # retrain the model once the dataset reaches this


def apply_aim_assist(raw_px, tx, ty):
    """Drag the cursor toward the target: full lock inside SNAP_RADIUS,
    fading pull out to ASSIST_RADIUS, nothing beyond. Returns assisted point."""
    d = math.hypot(raw_px[0] - tx, raw_px[1] - ty)
    if d >= ASSIST_RADIUS:
        return raw_px, 0.0
    if d <= SNAP_RADIUS:
        pull = 1.0
    else:
        pull = (ASSIST_RADIUS - d) / float(ASSIST_RADIUS - SNAP_RADIUS)
    return (int(raw_px[0] + (tx - raw_px[0]) * pull),
            int(raw_px[1] + (ty - raw_px[1]) * pull)), pull


def is_fixating(history, now, tx, ty):
    """True when the RAW gaze has been settled near the target long enough.

    history: deque of (x_px, y_px, timestamp) raw gaze points.
    Uses stability (a real fixation) rather than closeness to the target, so
    samples where the model is currently WRONG still get collected — those
    are exactly the ones that improve it — while wild glances are rejected.
    """
    # Gather over a window wider than the required hold: gaze inference runs
    # at only ~8-15 FPS, so demanding points exactly at the window edge made
    # fixations fire only by luck.
    recent = [p for p in history if now - p[2] <= FIXATION_MIN_HOLD * 1.8]
    if len(recent) < 3 or now - recent[0][2] < FIXATION_MIN_HOLD:
        return False
    xs = [p[0] for p in recent]
    ys = [p[1] for p in recent]
    spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    if spread > FIXATION_STABILITY_PX:
        return False
    cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
    return math.hypot(cx - tx, cy - ty) <= FIXATION_MAX_RAW_DIST


def random_target(prev):
    """Random spot, far enough from the previous target to force a real move."""
    while True:
        fx = random.uniform(MARGIN, 1 - MARGIN)
        fy = random.uniform(MARGIN, 1 - MARGIN)
        if prev is None or math.hypot(fx - prev[0], fy - prev[1]) > 0.25:
            return fx, fy


class GazeCursor:
    """EMA-smoothed on-screen gaze point, with per-session drift offset.

    `smoothing` trades stability for responsiveness: lower = steadier but
    laggier. Callers that need quick reactions (steering) pass a higher value.
    """

    def __init__(self, model, drift=(0.0, 0.0), smoothing=None):
        self.model = model
        self.drift = drift
        self.smoothing = GAZE_SMOOTHING if smoothing is None else smoothing
        self.pos = None

    def update(self, pitch, yaw):
        x, y = apply_calibration(self.model, pitch, yaw)
        x = min(max(x - self.drift[0], 0.0), 1.0)
        y = min(max(y - self.drift[1], 0.0), 1.0)
        a = self.smoothing
        if self.pos is None:
            self.pos = [x, y]
        else:
            self.pos[0] = a * x + (1 - a) * self.pos[0]
            self.pos[1] = a * y + (1 - a) * self.pos[1]
        return self.pos


def play(reader, window_name, model, drift=(0.0, 0.0)):
    """Run the rounds. Returns (results, fixation_samples) or (None, None).

    fixation_samples: (pitch, yaw, target_x, target_y) rows logged while the
    gaze was settled on a target — ground-truth labeled data, because in
    those moments we KNOW where the user is looking.
    """
    global AIM_ASSIST, SHOW_ASSIST_DEBUG
    cursor = GazeCursor(model, drift)
    results = []
    fixation_samples = []
    raw_history = deque(maxlen=60)
    prev_target = None

    for round_i in range(NUM_TARGETS):
        target = random_target(prev_target)
        prev_target = target
        tx, ty = int(target[0] * CANVAS_W), int(target[1] * CANVAS_H)
        round_start = time.time()
        hold_start = None
        hit = False
        raw_history.clear()

        while True:
            now = time.time()
            elapsed = now - round_start
            if elapsed >= TARGET_TIMEOUT:
                break

            reading = reader.read(with_gaze=True, with_features=True)
            raw_px = None
            gaze_px = None
            pull = 0.0
            if reading["pitch"] is not None:
                gx, gy = cursor.update(reading["pitch"], reading["yaw"])
                raw_px = (int(gx * CANVAS_W), int(gy * CANVAS_H))
                raw_history.append((raw_px[0], raw_px[1], now))
                gaze_px = raw_px
                if AIM_ASSIST:
                    gaze_px, pull = apply_aim_assist(raw_px, tx, ty)

                # Collect training data from the RAW gaze only — never from
                # the assisted cursor, or the assist would label its own pull.
                if is_fixating(raw_history, now, tx, ty) and reading["features"]:
                    row = dict(reading["features"])
                    row["target_x"], row["target_y"] = target[0], target[1]
                    fixation_samples.append(row)

            on_target = (gaze_px is not None and
                         math.hypot(gaze_px[0] - tx, gaze_px[1] - ty) <= TARGET_RADIUS)
            if on_target:
                if hold_start is None:
                    hold_start = now
                elif now - hold_start >= HOLD_TO_POP:
                    hit = True
            else:
                hold_start = None

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            score = sum(r["points"] for r in results)
            cv2.putText(canvas, f"Target {round_i + 1}/{NUM_TARGETS}   Score: {score}",
                        (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(canvas, f"{TARGET_TIMEOUT - elapsed:.0f}s",
                        (CANVAS_W - 90, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (150, 150, 150), 2)

            # Hidden magnet zones — invisible unless debug is toggled on
            if SHOW_ASSIST_DEBUG and AIM_ASSIST:
                cv2.circle(canvas, (tx, ty), ASSIST_RADIUS, (60, 60, 60), 1)
                cv2.circle(canvas, (tx, ty), SNAP_RADIUS, (0, 90, 90), 1)

            # Target ring + charge-up arc while gaze holds on it
            cv2.circle(canvas, (tx, ty), TARGET_RADIUS, (0, 165, 255), 2)
            cv2.circle(canvas, (tx, ty), 10, (0, 165, 255), -1)
            if hold_start is not None:
                frac = min((time.time() - hold_start) / HOLD_TO_POP, 1.0)
                cv2.ellipse(canvas, (tx, ty), (TARGET_RADIUS, TARGET_RADIUS),
                            -90, 0, int(360 * frac), (0, 255, 0), 6)

            # Gaze cursor (assisted). Faint dot = raw gaze, so the pull is visible.
            if raw_px is not None and raw_px != gaze_px:
                cv2.circle(canvas, raw_px, 4, (120, 90, 0), -1)
            if gaze_px is not None:
                cv2.circle(canvas, gaze_px, 12, (255, 200, 0), 2)
                cv2.circle(canvas, gaze_px, 3, (255, 200, 0), -1)

            hud = [f"assist {'ON' if AIM_ASSIST else 'OFF'} [a]",
                   f"data: {len(fixation_samples)}"]
            if AIM_ASSIST and pull > 0:
                hud.append(f"pull {pull * 100:.0f}%")
            cv2.putText(canvas, "  |  ".join(hud), (30, CANVAS_H - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)

            status = reading["status"]
            if status is not None and not status["in_zone"]:
                cv2.putText(canvas, "HOLD POSITION - " + " / ".join(status["messages"]),
                            (30, CANVAS_H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return None, None
            elif key == ord('a'):
                AIM_ASSIST = not AIM_ASSIST
            elif key == ord('d'):
                SHOW_ASSIST_DEBUG = not SHOW_ASSIST_DEBUG
            if hit:
                break

        time_taken = time.time() - round_start
        points = 0
        if hit:
            points = BASE_POINTS + int((TARGET_TIMEOUT - time_taken) * SPEED_BONUS_PER_SEC)
        results.append({"target": target, "hit": hit,
                        "time": time_taken, "points": points})

        # Brief pop/miss flash
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        if hit:
            cv2.circle(canvas, (tx, ty), TARGET_RADIUS + 20, (0, 255, 0), 4)
            cv2.putText(canvas, f"+{points}", (tx - 40, ty + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(canvas, "MISS", (tx - 50, ty + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.imshow(window_name, canvas)
        cv2.waitKey(400)

    return results, fixation_samples


def show_final_screen(window_name, results):
    hits = [r for r in results if r["hit"]]
    score = sum(r["points"] for r in results)
    avg_time = np.mean([r["time"] for r in hits]) if hits else None

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    cv2.putText(canvas, "GAME OVER", (CANVAS_W // 2 - 160, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 165, 255), 3)
    cv2.putText(canvas, f"Score: {score}", (CANVAS_W // 2 - 110, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(canvas, f"Hits: {len(hits)}/{len(results)}"
                + (f"   avg time to hit: {avg_time:.1f}s" if avg_time else ""),
                (CANVAS_W // 2 - 250, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (200, 200, 200), 2)

    # Mini-map of where hits and misses happened — misses cluster where the
    # calibration is weak, mirroring milestone6's error map.
    map_w, map_h = 500, 280
    mx, my = (CANVAS_W - map_w) // 2, 300
    cv2.rectangle(canvas, (mx, my), (mx + map_w, my + map_h), (100, 100, 100), 1)
    cv2.putText(canvas, "hit/miss map (misses = weak calibration spots)",
                (mx, my - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
    for r in results:
        px = mx + int(r["target"][0] * map_w)
        py = my + int(r["target"][1] * map_h)
        if r["hit"]:
            cv2.circle(canvas, (px, py), 7, (0, 255, 0), -1)
        else:
            cv2.drawMarker(canvas, (px, py), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 16, 3)

    verdicts = [
        (10, "PERFECT ROUND - your calibration is dialed in."),
        (8, "Great - calibration is solid almost everywhere."),
        (6, "Decent - note where the misses cluster and recalibrate."),
        (0, "Rough - recalibrate (milestone4) and check milestone6's report."),
    ]
    verdict = next(v for n, v in verdicts if len(hits) >= n)
    cv2.putText(canvas, verdict, (CANVAS_W // 2 - 340, my + map_h + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, "Press any key to exit", (CANVAS_W // 2 - 160, CANVAS_H - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 2)

    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)

    print("\n--- Game report ---")
    print(f"Score: {score}   Hits: {len(hits)}/{len(results)}")
    for i, r in enumerate(results):
        state = f"HIT in {r['time']:.1f}s (+{r['points']})" if r["hit"] else "MISS"
        print(f"  Target {i + 1} at ({r['target'][0]:.0%}, {r['target'][1]:.0%}): {state}")
    print(verdict)


def retrain_from_game(old_model, fixation_samples):
    """Bank this game's fixation data and retrain on the full dataset.

    Every game adds ground-truth samples, so the model keeps improving the
    more you play. Constraints/metadata from the old model are carried over.
    """
    if not fixation_samples:
        print("No fixation data collected this game - model unchanged.")
        return
    append_dataset(fixation_samples, SOURCE_GAME)
    all_samples, counts = load_dataset()
    print(f"Banked {len(fixation_samples)} fixation samples "
          f"(dataset now {len(all_samples)}: {counts}).")

    if len(all_samples) < RETRAIN_MIN_SAMPLES:
        print(f"Need {RETRAIN_MIN_SAMPLES}+ samples before retraining.")
        return

    degree = old_model.get("degree", 3)
    new_model, n_dropped = robust_fit_samples(all_samples, degree=degree)
    for key in ("validation_error", "position_constraints"):
        if key in old_model:
            new_model[key] = old_model[key]
    new_model["trained_on"] = counts
    save_calibration(new_model)
    print(f"Model retrained on {len(all_samples)} samples "
          f"({n_dropped} outliers dropped) and saved. "
          "Run milestone6_test_accuracy.py to re-measure.")


def main():
    try:
        model = load_calibration()
    except FileNotFoundError:
        print("No calibration found - run milestone4_calibration.py first.")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # camera's native 16:9 resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    window_name = "Gaze Game"
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
        results, fixation_samples = play(reader, window_name, model, drift)
        if results is None:
            print("Game aborted.")
            return
        show_final_screen(window_name, results)
        retrain_from_game(model, fixation_samples)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
