"""
gaze_server.py — Phase 1 of the gaze-controlled game integration.

Streams the LIVE calibrated gaze point to the browser over a local WebSocket,
so a web page can use "where you are looking" as an input device.

Pipeline (all reused from the existing milestones, no OpenCV window):
    webcam frame
      -> cv2.flip(frame, 1)            # mirror, to match how calibration was recorded
      -> get_gaze_reading(...)         # MediaPipe face + L2CS ONNX -> (pitch, yaw)
      -> apply_calibration(model, ...) # -> normalized screen (x, y) in [0, 1]
      -> EMA smoothing                 # steady the jittery raw signal
      -> broadcast {"x", "y"} as JSON  # ~30 msgs/sec to every connected browser

Architecture:
    * A background THREAD runs the (blocking) webcam + inference loop and writes
      the newest reading into the module-level LATEST dict.
    * The asyncio WebSocket server broadcasts LATEST to all clients on a fixed
      timer. This decouples the ~10-15 FPS inference from a smooth send rate and
      lets many browser tabs connect at once.

Message format (one per broadcast tick):
    {"type": "gaze", "ok": true,  "x": 0.42, "y": 0.71, "t": 1690000000.123}
    {"type": "gaze", "ok": false, "x": null, "y": null, "t": 1690000000.123}   # no face / out of zone

Run (from anywhere — the script cd's to its own folder so models/ resolves):
    gaze_env\\Scripts\\python.exe gaze_server.py
Then open gaze_test.html in a browser. Press Ctrl+C here to stop.

Requires: models/calibration_model.pkl (run milestone4_calibration.py first).
"""

import os
import pathlib

# Resolve models/, data/ and face_landmarker.task relative to THIS file, so the
# server runs correctly no matter what directory it is launched from. Must run
# before importing the gaze modules — gaze_pipeline computes the ONNX model path
# from the current working directory at import time.
os.chdir(pathlib.Path(__file__).resolve().parent)

import asyncio
import json
import math
import threading
import time

import cv2
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from websockets.asyncio.server import serve, broadcast

from gaze_pipeline import get_gaze_reading, reset_bbox_smoothing
from calibration_utils import load_calibration, apply_calibration

# --- Config ---------------------------------------------------------------
HOST = "localhost"
PORT = 8765
FACE_MODEL_PATH = "face_landmarker.task"
CAM_INDEX = 0
CAM_W, CAM_H = 1280, 720      # 720p @ ~31fps via the MSMF backend (DSHOW capped 720p at 10fps)
SEND_HZ = 60                  # how often to push the latest reading to browsers

# 1€ filter (Casiez et al.) — adaptive smoothing that fixes the shaky-vs-laggy
# tradeoff: it smooths hard when the gaze is still (kills jitter) and eases off
# when it moves fast (stays responsive). Tune:
#   MIN_CUTOFF ↓  -> steadier when still (but a touch more lag on slow moves)
#   BETA       ↑  -> snappier on fast moves (but more jitter passes through)
ONE_EURO_MIN_CUTOFF = 0.7
ONE_EURO_BETA = 0.6
ONE_EURO_DCUTOFF = 1.0

# --- Shared state ---------------------------------------------------------
# The gaze thread REPLACES this dict wholesale (an atomic reference swap under
# the GIL), so the broadcaster always reads a self-consistent snapshot without
# needing a lock.
LATEST = {"type": "gaze", "ok": False, "x": None, "y": None, "t": 0.0}
_running = True

# Newest camera frame — produced by the capture thread, consumed by the gaze
# loop. Decoupling capture from inference means the slow webcam read never
# stalls the GPU inference, so cursor updates arrive as fast as frames do.
_frame_lock = threading.Lock()
_latest_frame = None
_frame_seq = 0


def _capture_thread(cap):
    global _latest_frame, _frame_seq
    while _running:
        ret, frame = cap.read()
        if not ret:
            continue
        with _frame_lock:
            _latest_frame = frame
            _frame_seq += 1


class OneEuro:
    """1€ filter for a single scalar stream. Call .filter(value, timestamp)."""

    def __init__(self, min_cutoff, beta, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None
        self.freq = 30.0

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x
        if t > self.t_prev:
            self.freq = 1.0 / (t - self.t_prev)
        self.t_prev = t
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        self.dx_prev = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(self.dx_prev)
        a = self._alpha(cutoff)
        self.x_prev = a * x + (1 - a) * self.x_prev
        return self.x_prev


def build_landmarker():
    """MediaPipe FaceLandmarker configured exactly as the calibration used
    (VIDEO mode, one face, head-pose matrices on) so live readings match."""
    base = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def gaze_loop():
    """Blocking webcam + inference loop. Runs on a background thread and keeps
    LATEST up to date. Never touches the event loop or the client set."""
    global LATEST, _running

    try:
        model = load_calibration()
    except FileNotFoundError:
        print("ERROR: models/calibration_model.pkl not found — "
              "run milestone4_calibration.py first.")
        _running = False
        return

    landmarker = build_landmarker()
    reset_bbox_smoothing()

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_MSMF)   # Media Foundation: 720p @ 31fps (DSHOW was 10fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print(f"ERROR: could not open webcam at index {CAM_INDEX}.")
        _running = False
        return

    # Capture on its own thread; inference always grabs the freshest frame.
    threading.Thread(target=_capture_thread, args=(cap,), daemon=True).start()
    print("Gaze loop running (GPU / threaded capture / 1€ filter) — look around.  (Ctrl+C to stop)")
    fx = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_DCUTOFF)
    fy = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_DCUTOFF)
    frame_idx = 0
    last_seq = -1
    try:
        while _running:
            with _frame_lock:
                frame = _latest_frame
                seq = _frame_seq
            if frame is None or seq == last_seq:
                time.sleep(0.002)      # no new frame yet — don't busy-spin
                continue
            last_seq = seq
            # Mirror the frame: calibration was recorded through FrameReader,
            # which flips horizontally. Skipping this inverts left/right.
            frame = cv2.flip(frame, 1)

            pitch, yaw, _bbox = get_gaze_reading(frame, landmarker, frame_idx)
            frame_idx += 1

            now = time.time()
            if pitch is None:
                LATEST = {"type": "gaze", "ok": False,
                          "x": None, "y": None, "t": now}
                fx.reset(); fy.reset()    # forget history so we don't glide from a stale point
                continue

            x, y = apply_calibration(model, pitch, yaw)   # normalized [0, 1]
            sx = fx.filter(x, now)
            sy = fy.filter(y, now)
            LATEST = {"type": "gaze", "ok": True,
                      "x": round(sx, 4), "y": round(sy, 4), "t": now}
    finally:
        cap.release()
        print("Camera released.")


# --- WebSocket server -----------------------------------------------------
CLIENTS = set()


async def handler(websocket):
    """One coroutine per connected browser. We only push data, so just hold the
    connection open and drop it from the broadcast set when it closes."""
    CLIENTS.add(websocket)
    print(f"Browser connected  (clients: {len(CLIENTS)})")
    try:
        await websocket.wait_closed()
    finally:
        CLIENTS.discard(websocket)
        print(f"Browser disconnected  (clients: {len(CLIENTS)})")


async def broadcaster():
    """Fan the newest reading out to every client on a fixed timer."""
    period = 1.0 / SEND_HZ
    while _running:
        if CLIENTS:
            broadcast(CLIENTS, json.dumps(LATEST))   # non-blocking, best-effort to all
        await asyncio.sleep(period)


async def main():
    thread = threading.Thread(target=gaze_loop, daemon=True)
    thread.start()

    # Give the loop a beat to fail fast on a missing model / camera.
    await asyncio.sleep(0.5)
    if not _running:
        return

    async with serve(handler, HOST, PORT):
        print(f"WebSocket server ready at ws://{HOST}:{PORT}")
        print("Open gaze_test.html in your browser to see your gaze as a dot.")
        await broadcaster()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _running = False
        print("\nStopping…")
