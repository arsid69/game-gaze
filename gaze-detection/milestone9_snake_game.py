"""
Milestone 9 — Gaze-controlled Snake.

Steer by looking. Blink to press a button. Two buttons only: RESTART, QUIT.

Tracking, positioning gate, drift correction and the smoothed gaze cursor are
all reused from milestone 4/7 — this file only adds the game.

  Look inside the play area  -> the snake turns toward where you look
  Look at a button + BLINK   -> presses it
  Out of position            -> game pauses until you are back in the zone

Requires: models/calibration_model.pkl (run milestone4_calibration.py first).
Run: python milestone9_snake_game.py
Keys: q quit (escape hatch — the game itself needs no keyboard).
"""

import math
import random
import time
from collections import deque

import cv2
import numpy as np

from calibration_utils import load_calibration
from gaze_pipeline import reset_bbox_smoothing
from positioning_gate import PositioningGate

from milestone4_calibration import (
    FrameReader, phase_a_positioning, measure_drift, aborted, CANVAS_W, CANVAS_H,
)
from milestone7_gaze_game import GazeCursor

# --- Board ---
CELL = 40
PLAY_H = 600                      # play area height; buttons live below it
COLS, ROWS = CANVAS_W // CELL, PLAY_H // CELL      # 32 x 15
TICK = 0.35                       # seconds per snake step (~3 moves/sec)
START_GRACE = 2.0                 # seconds before it starts moving, so you can
                                  # aim first — at start and after every restart
CURSOR_SMOOTHING = 0.5            # higher than milestone 7's 0.25: steering needs
                                  # to react quickly, and a jittery cursor matters
                                  # less when it only picks one of four directions
SHOW_INTENT = True                # draw head -> gaze line so the aim is visible

# --- Blink = click ---
BLINK_EAR = 0.18                  # eyes counted shut below this (open ~0.30)
BLINK_COOLDOWN = 0.6              # seconds before another blink can register
EYES_CLOSED_GRACE = 0.45          # a normal blink lasts ~0.1-0.4 s and must be
                                  # ignored by the game; only a closure LONGER
                                  # than this pauses play
BUTTON_DWELL = 0.25               # gaze must rest on a button before it locks
LOCK_WINDOW = 2.0                 # seconds the button stays locked, waiting for a blink
LOCK_REFRACTORY = 0.8             # pause before the same button can lock again

# --- Aim assist: buttons ---
# The radius must comfortably EXCEED the button's own half-width (130 px), or
# a gaze that lands on the button's edge gets no help at all.
ASSIST_RADIUS = 260               # px (~7 cm): pull starts here
ASSIST_SNAP = 95                  # px (~2.5 cm): inside this, lock to the centre
ASSIST_MIN_Y = PLAY_H - 80        # only active near the button strip

# --- Aim assist: food ---
# Pulls the steering point toward the food so you can home in on it without
# pixel-perfect gaze. Weaker than the buttons: you must still be able to steer
# away from the food deliberately.
FOOD_ASSIST_RADIUS = 190          # px (~5 cm)
FOOD_ASSIST_SNAP = 55             # px (~1.5 cm): inside this, lock onto the food
SHOW_FOOD_ZONE = False            # hidden by default — the help should be felt,
                                  # not seen. Set True to debug the radius.

# This is a demo: it CONSUMES the calibration model but writes nothing to the
# dataset. Snake has no "look at this dot" moment, so any label would be a
# guess. Training data comes from milestone 4 and milestone 7 only.

# --- Colours (BGR) ---
BG = (24, 20, 18)
GRID = (38, 33, 30)
SNAKE_HEAD = (120, 255, 180)
SNAKE_BODY = (60, 190, 120)
FOOD = (0, 165, 255)
CURSOR = (255, 200, 0)
TEXT = (235, 235, 235)
DIM = (140, 140, 140)


def eyes_are_open(ear_left, ear_right):
    """Both eyes open? Uses the WORSE eye, not the average.

    Averaging hid a wink: (0.30 + 0.10)/2 = 0.20 sits above the 0.18
    threshold, so one closed eye read as 'open' and the gaze model kept
    emitting readings from a half-shut face.
    """
    if ear_left is None or ear_right is None:
        return False
    return min(ear_left, ear_right) >= BLINK_EAR


class BlinkDetector:
    """Fires once per blink: needs an open -> shut transition, then a cooldown.

    Requires BOTH eyes shut (the better eye must be below threshold), so a
    squint or a wink cannot trigger a button press by accident.
    """

    def __init__(self):
        self.was_shut = False
        self.last_blink = 0.0

    def update(self, ear_left, ear_right, now):
        if ear_left is None or ear_right is None:
            return False
        shut = max(ear_left, ear_right) < BLINK_EAR
        fired = False
        if shut and not self.was_shut and now - self.last_blink > BLINK_COOLDOWN:
            fired = True
            self.last_blink = now
        self.was_shut = shut
        return fired


class Snake:
    def __init__(self, now=None):
        self.reset(now)

    def reset(self, now=None):
        mid = (COLS // 2, ROWS // 2)
        self.body = deque([(mid[0] - 2, mid[1]), (mid[0] - 1, mid[1]), mid])
        self.direction = (1, 0)
        # The direction actually executed on the last tick. Reversals must be
        # judged against THIS, not against self.direction — otherwise two
        # steers between ticks (right -> up -> left) sneak a 180 through and
        # the snake eats its own neck on the next step.
        self.last_dir = self.direction
        # Rendering state: the grid is the truth, but drawing interpolates
        # between ticks so the snake glides instead of teleporting.
        self.prev_head = self.body[-1]
        self.removed_tail = None
        self.score = 0
        self.dead = False
        self.start_at = (time.time() if now is None else now) + START_GRACE
        self.place_food()

    def place_food(self):
        free = [(x, y) for x in range(COLS) for y in range(ROWS)
                if (x, y) not in self.body]
        self.food = random.choice(free) if free else None

    def head_px(self):
        hx, hy = self.body[-1]
        return hx * CELL + CELL // 2, hy * CELL + CELL // 2

    def is_safe(self, direction):
        """Would stepping this way survive? The tail vacates its cell on the
        same tick, so it is only an obstacle when we are about to eat."""
        hx, hy = self.body[-1]
        nx, ny = hx + direction[0], hy + direction[1]
        if not (0 <= nx < COLS and 0 <= ny < ROWS):
            return False
        occupied = self.body if (nx, ny) == self.food else list(self.body)[1:]
        return (nx, ny) not in occupied

    def is_reversal(self, direction):
        return (direction[0] + self.last_dir[0],
                direction[1] + self.last_dir[1]) == (0, 0)

    def steer(self, gaze_px):
        """Turn toward the gaze point, never into an immediate death.

        Candidate directions are ranked by how much of the gap each closes,
        then filtered: no 180-degree reversals, and no move that would hit a
        wall or the body next tick. Gaze tracking is only accurate to ~2 cm,
        so obeying a stray glance into a wall made the game end by itself.
        """
        head = self.head_px()
        dx, dy = gaze_px[0] - head[0], gaze_px[1] - head[1]
        if abs(dx) < CELL * 0.6 and abs(dy) < CELL * 0.6:
            return                                   # looking at the head itself

        candidates = []
        if abs(dx) >= CELL * 0.5:
            candidates.append(((1 if dx > 0 else -1, 0), abs(dx)))
        if abs(dy) >= CELL * 0.5:
            candidates.append(((0, 1 if dy > 0 else -1), abs(dy)))
        candidates.sort(key=lambda c: -c[1])          # biggest gap first

        # 1. Best direction you asked for that is legal and survivable.
        for direction, _gap in candidates:
            if not self.is_reversal(direction) and self.is_safe(direction):
                self.direction = direction
                return

        # 2. You asked for something fatal or impossible (a reversal, or a
        #    wall). Turn perpendicular instead — preferring the side you are
        #    looking toward — so the snake keeps going rather than crashing.
        if self.last_dir[0] != 0:                     # travelling horizontally
            first = (0, 1 if dy > 0 else -1)
            options = [first, (0, -first[1])]
        else:                                          # travelling vertically
            first = (1 if dx > 0 else -1, 0)
            options = [first, (-first[0], 0)]
        for direction in options:
            if self.is_safe(direction):
                self.direction = direction
                return

        # 3. Nothing perpendicular is safe either. Keep going only if the
        #    current heading survives; otherwise take any legal escape.
        if self.is_safe(self.direction):
            return
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if not self.is_reversal(direction) and self.is_safe(direction):
                self.direction = direction
                return

    def step(self):
        if self.dead:
            return
        self.last_dir = self.direction        # this is the move being executed
        hx, hy = self.body[-1]
        nx, ny = hx + self.direction[0], hy + self.direction[1]
        if not (0 <= nx < COLS and 0 <= ny < ROWS) or (nx, ny) in self.body:
            self.dead = True
            return
        self.prev_head = (hx, hy)             # for smooth rendering
        self.body.append((nx, ny))
        if (nx, ny) == self.food:
            self.score += 1
            self.removed_tail = None          # grew: nothing retracts
            self.place_food()
        else:
            self.removed_tail = self.body.popleft()

    @staticmethod
    def _segment(canvas, fx, fy, colour, pad):
        """Draw one body square at a FLOAT cell position (allows sliding)."""
        cv2.rectangle(canvas,
                      (int(fx * CELL + pad), int(fy * CELL + pad)),
                      (int((fx + 1) * CELL - pad), int((fy + 1) * CELL - pad)),
                      colour, -1)

    def draw(self, canvas, progress=1.0):
        """progress = how far through the current tick (0..1).

        Only the two ends move between ticks: the head slides out of the neck
        and the vacated tail cell retracts into the body. The middle is
        already where it belongs, so nothing else needs to animate.
        """
        for x in range(0, CANVAS_W, CELL):
            cv2.line(canvas, (x, 0), (x, PLAY_H), GRID, 1)
        for y in range(0, PLAY_H + 1, CELL):
            cv2.line(canvas, (0, y), (CANVAS_W, y), GRID, 1)

        if self.food:
            fx = self.food[0] * CELL + CELL // 2
            fy = self.food[1] * CELL + CELL // 2
            if SHOW_FOOD_ZONE:
                cv2.circle(canvas, (fx, fy), FOOD_ASSIST_RADIUS, (40, 48, 44), 1)
            # gentle breathing so the board never looks frozen
            r = CELL // 3 + int(2 * math.sin(time.time() * 3.0))
            cv2.circle(canvas, (fx, fy), r, FOOD, -1)

        t = 0.0 if self.dead else min(max(progress, 0.0), 1.0)

        # Tail retracting into the next segment
        if self.removed_tail is not None and t < 1.0 and len(self.body) > 1:
            tx, ty = self.removed_tail
            nx, ny = self.body[0]
            self._segment(canvas, tx + (nx - tx) * t, ty + (ny - ty) * t,
                          SNAKE_BODY, 5)

        # Settled middle segments
        for x, y in list(self.body)[:-1]:
            self._segment(canvas, x, y, SNAKE_BODY, 5)

        # Head sliding out of the neck toward its new cell
        hx, hy = self.body[-1]
        px, py = self.prev_head
        self._segment(canvas, px + (hx - px) * t, py + (hy - py) * t,
                      SNAKE_HEAD, 3)


class Button:
    """Dwell to lock, then blink within LOCK_WINDOW to press.

    The lock matters because closing your eyes makes the gaze model output
    nonsense for a frame or two — without it, the blink that should press the
    button would first fling the cursor off it. Once locked, gaze is ignored
    entirely until the window expires.
    """

    def __init__(self, label, x, y, w, h, colour):
        self.label = label
        self.rect = (x, y, x + w, y + h)
        self.colour = colour
        self.hover_since = None
        self.locked_until = None
        self.refractory_until = 0.0

    def contains(self, p):
        x0, y0, x1, y1 = self.rect
        return p is not None and x0 <= p[0] <= x1 and y0 <= p[1] <= y1

    def center(self):
        x0, y0, x1, y1 = self.rect
        return (x0 + x1) // 2, (y0 + y1) // 2

    @property
    def locked(self):
        return self.locked_until is not None

    def release(self, now, refractory=True):
        self.locked_until = None
        self.hover_since = None
        if refractory:
            self.refractory_until = now + LOCK_REFRACTORY

    def update(self, gaze_px, now):
        """Advance the state machine. Returns True while locked."""
        if self.locked:
            if now >= self.locked_until:      # no blink in time -> let go
                self.release(now)
            return self.locked

        if now < self.refractory_until:
            self.hover_since = None
            return False

        if self.contains(gaze_px):
            if self.hover_since is None:
                self.hover_since = now
            elif now - self.hover_since >= BUTTON_DWELL:
                self.locked_until = now + LOCK_WINDOW
                self.hover_since = None
        else:
            self.hover_since = None
        return self.locked

    def draw(self, canvas, now):
        x0, y0, x1, y1 = self.rect
        if self.locked:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), self.colour, -1)
            cv2.rectangle(canvas, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (255, 255, 255), 2)
            text_colour = (20, 20, 20)
            hint = "BLINK NOW"
        else:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (46, 40, 36), -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), self.colour, 2)
            text_colour = self.colour
            hint = ""

        (tw, th), _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, 0.95, 2)
        cv2.putText(canvas, self.label, ((x0 + x1 - tw) // 2, (y0 + y1 + th) // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, text_colour, 2)
        if hint:
            (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            cv2.putText(canvas, hint, ((x0 + x1 - hw) // 2, y1 - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_colour, 2)

        if self.locked:
            # countdown bar draining left-to-right over the lock window
            frac = max(0.0, (self.locked_until - now) / LOCK_WINDOW)
            cv2.rectangle(canvas, (x0, y1 - 6), (int(x0 + (x1 - x0) * frac), y1 - 1),
                          (20, 20, 20), -1)
        elif self.hover_since is not None:
            frac = min((now - self.hover_since) / BUTTON_DWELL, 1.0)
            cv2.line(canvas, (x0, y1 - 3), (int(x0 + (x1 - x0) * frac), y1 - 3),
                     self.colour, 3)


def magnet(gaze_px, centres, radius, snap):
    """Pull the point toward the nearest centre: full lock inside `snap`,
    linear fade out to `radius`, nothing beyond."""
    if gaze_px is None or not centres:
        return gaze_px
    nearest, best_d = None, radius
    for cx, cy in centres:
        d = ((gaze_px[0] - cx) ** 2 + (gaze_px[1] - cy) ** 2) ** 0.5
        if d < best_d:
            nearest, best_d = (cx, cy), d
    if nearest is None:
        return gaze_px
    if best_d <= snap:
        return nearest
    pull = (radius - best_d) / float(radius - snap)
    return (int(gaze_px[0] + (nearest[0] - gaze_px[0]) * pull),
            int(gaze_px[1] + (nearest[1] - gaze_px[1]) * pull))


def button_assist(gaze_px, buttons):
    """Magnetism toward the buttons, active only near the button strip so it
    can never grab the cursor mid-game."""
    if gaze_px is None or gaze_px[1] < ASSIST_MIN_Y:
        return gaze_px
    return magnet(gaze_px, [b.center() for b in buttons],
                  ASSIST_RADIUS, ASSIST_SNAP)


def food_assist(gaze_px, snake):
    """Magnetism toward the food, so homing in does not need pixel-perfect
    gaze. Only inside the play area, and weaker than the buttons so you can
    still deliberately steer away."""
    if gaze_px is None or snake.food is None or gaze_px[1] >= PLAY_H:
        return gaze_px
    fx = snake.food[0] * CELL + CELL // 2
    fy = snake.food[1] * CELL + CELL // 2
    return magnet(gaze_px, [(fx, fy)], FOOD_ASSIST_RADIUS, FOOD_ASSIST_SNAP)


def draw_hud(canvas, snake, ear, blink_flash, status, now, eyes_live=True):
    cv2.putText(canvas, f"SCORE {snake.score}", (20, PLAY_H + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, TEXT, 2)

    # Eye-openness bar — shows the blink detector working
    bx, by = 20, PLAY_H + 66
    cv2.putText(canvas, "eyes", (bx, by + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, DIM, 1)
    if ear is not None:
        w = int(min(ear / 0.35, 1.0) * 120)
        shut = ear < BLINK_EAR
        cv2.rectangle(canvas, (bx + 46, by), (bx + 46 + w, by + 14),
                      (0, 0, 255) if shut else (0, 220, 0), -1)
        cv2.rectangle(canvas, (bx + 46, by), (bx + 166, by + 14), (70, 65, 62), 1)
        thr = bx + 46 + int(BLINK_EAR / 0.35 * 120)
        cv2.line(canvas, (thr, by - 2), (thr, by + 16), (200, 200, 200), 1)

    if now - blink_flash < 0.25:
        cv2.putText(canvas, "BLINK", (bx + 180, by + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if status is not None and not status["in_zone"]:
        msg = "HOLD POSITION - " + " / ".join(status["messages"])
        cv2.putText(canvas, msg, (CANVAS_W // 2 - 260, PLAY_H + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2)
    elif status is None:
        cv2.putText(canvas, "NO FACE", (CANVAS_W // 2 - 60, PLAY_H + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2)
    elif not eyes_live and not snake.dead:
        cv2.putText(canvas, "PAUSED - OPEN BOTH EYES",
                    (CANVAS_W // 2 - 200, PLAY_H + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 220), 2)

    if not snake.dead and now < snake.start_at:
        left = snake.start_at - now
        msg = f"GET READY  {left:.0f}"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        cv2.putText(canvas, msg, ((CANVAS_W - tw) // 2, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 220), 3)
        hint = "look where you want the snake to go"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(canvas, hint, ((CANVAS_W - hw) // 2, 122),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, DIM, 1)

    if snake.dead:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (CANVAS_W, PLAY_H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
        cv2.putText(canvas, "GAME OVER", (CANVAS_W // 2 - 210, PLAY_H // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
        cv2.putText(canvas, f"score {snake.score}  -  look at RESTART and blink",
                    (CANVAS_W // 2 - 250, PLAY_H // 2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, TEXT, 2)


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

    window_name = "Gaze Snake"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    reset_bbox_smoothing()
    reader = FrameReader(cap, PositioningGate())

    try:
        if not phase_a_positioning(reader, window_name):
            return
        drift = measure_drift(reader, window_name, model)
        if drift is None:
            return

        cursor = GazeCursor(model, drift, smoothing=CURSOR_SMOOTHING)
        snake = Snake()
        blinks = BlinkDetector()
        btn_y, btn_h, btn_w = PLAY_H + 22, 76, 260
        restart = Button("RESTART", CANVAS_W // 2 - btn_w - 20, btn_y, btn_w, btn_h,
                         (120, 255, 180))
        quit_btn = Button("QUIT", CANVAS_W // 2 + 20, btn_y, btn_w, btn_h,
                          (110, 130, 255))
        buttons = [restart, quit_btn]

        last_tick = time.time()
        blink_flash = 0.0
        gaze_px = None                   # persists so a blink can't move it
        eyes_closed_since = None

        while True:
            now = time.time()
            reading = reader.read(with_gaze=True, with_features=True)
            status = reading["status"]
            feats = reading["features"]

            ear_l = feats["ear_left"] if feats else None
            ear_r = feats["ear_right"] if feats else None
            ear = min(ear_l, ear_r) if ear_l is not None else None   # worse eye
            eyes_ok = eyes_are_open(ear_l, ear_r)

            # A blink is not a command. Track how long the eyes have been shut:
            # brief closures are ignored by the game entirely (the cursor still
            # freezes, so nothing jumps), and only a sustained closure pauses.
            if eyes_ok:
                eyes_closed_since = None
            elif eyes_closed_since is None:
                eyes_closed_since = now
            eyes_lost = (eyes_closed_since is not None
                         and now - eyes_closed_since >= EYES_CLOSED_GRACE)

            # Freeze the cursor unless BOTH eyes are open: a shut or half-shut
            # eye makes the gaze model output nonsense, which would otherwise
            # fling the cursor off the button at the moment you blink to press.
            if reading["pitch"] is not None and eyes_ok:
                gx, gy = cursor.update(reading["pitch"], reading["yaw"])
                gaze_px = (int(gx * CANVAS_W), int(gy * CANVAS_H))

            blinked = blinks.update(ear_l, ear_r, now)
            if blinked:
                blink_flash = now

            in_zone = status is not None and status["in_zone"]
            aim_px = button_assist(gaze_px, buttons) if in_zone else None

            # Buttons first — a blink there must not also steer the snake
            locked_button = None
            on_button = False
            for b in buttons:
                if b.update(aim_px, now):
                    locked_button = b
                on_button = on_button or b.contains(aim_px)

            if blinked and in_zone and locked_button is not None:
                locked_button.release(now, refractory=False)
                if locked_button.label == "RESTART":
                    snake.reset(now)
                    last_tick = now
                elif locked_button.label == "QUIT":
                    break

            # Steering + ticking only while positioned, not on/locked to a button
            busy = on_button or locked_button is not None
            ready = now >= snake.start_at            # grace period over?

            # The snake keeps going through a blink; it only stops if the eyes
            # stay shut (or half-shut) past the grace period.
            live = in_zone and not eyes_lost

            # Aiming works during the grace period — that is the point of it.
            steer_px = food_assist(gaze_px, snake) if live else gaze_px
            if live and steer_px is not None and not busy and not snake.dead:
                if steer_px[1] < PLAY_H:
                    snake.steer(steer_px)
            if live and ready and not snake.dead and now - last_tick >= TICK:
                snake.step()
                last_tick = now
            if not live or not ready:
                last_tick = now          # pause rather than fast-forward

            # How far through the current tick we are. While paused the snake
            # sits fully settled on its cells rather than frozen mid-slide.
            animating = live and ready and not snake.dead
            progress = min((now - last_tick) / TICK, 1.0) if animating else 1.0

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            canvas[:] = BG
            snake.draw(canvas, progress)
            cv2.line(canvas, (0, PLAY_H), (CANVAS_W, PLAY_H), (70, 65, 62), 2)
            for b in buttons:
                b.draw(canvas, now)
            draw_hud(canvas, snake, ear, blink_flash, status, now, not eyes_lost)

            # Intent line: head -> where you are looking, plus the direction the
            # snake has actually taken. Makes any steering mismatch obvious.
            if (SHOW_INTENT and steer_px is not None and not snake.dead
                    and not busy and steer_px[1] < PLAY_H):
                hx, hy = snake.head_px()
                cv2.line(canvas, (hx, hy), steer_px, (70, 90, 80), 1)
                dx, dy = snake.direction
                cv2.arrowedLine(canvas, (hx, hy),
                                (hx + dx * CELL, hy + dy * CELL),
                                SNAKE_HEAD, 2, tipLength=0.4)

            # While locked the cursor is pinned to the button, so it visibly
            # stays put through the blink.
            draw_px = locked_button.center() if locked_button else aim_px or gaze_px
            if draw_px is not None:
                ring = (255, 255, 255) if locked_button else CURSOR
                cv2.circle(canvas, draw_px, 16 if locked_button else 14, ring, 2)
                cv2.circle(canvas, draw_px, 3, ring, -1)

            cv2.imshow(window_name, canvas)
            if aborted():
                break

        print(f"Final score: {snake.score}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
