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

# Gesture control is optional: if the module or its model is missing, the server
# still runs as a pure gaze server and simply reports gesture as unavailable.
try:
    from gesture_bridge import GesturePointer
    GESTURE_AVAILABLE = True
    GESTURE_IMPORT_ERROR = ""
except Exception as _exc:                      # ImportError, missing model, ...
    GesturePointer = None
    GESTURE_AVAILABLE = False
    GESTURE_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"

# --- Config ---------------------------------------------------------------
HOST = "localhost"
PORT = 8765

# Input modes. "gaze" is the default so existing behaviour is unchanged unless
# somebody deliberately switches.
#
# There is deliberately no hybrid mode. It would need the gaze and gesture
# pipelines running at the same time, which means sharing one camera between
# them — and measurement showed that costs ~20% of the hand-detection rate,
# because MediaPipe holds the GIL during inference and starves the shared
# capture thread. Exclusive ownership per mode restores standalone accuracy,
# and that was judged worth more than aiming with the eyes while pinching.
MODE_GAZE, MODE_GESTURE = "gaze", "gesture"
VALID_MODES = (MODE_GAZE, MODE_GESTURE)

# Detect on every frame, exactly as the standalone module does. This was 2 for a
# while as a CPU saving, but halving the detection rate measurably degraded
# tracking: the module's 1€ filter and its frame-voting stabiliser are both tuned
# for a full-rate stream. Accuracy wins over the saving. Raise it only if a weak
# machine genuinely cannot keep up.
GESTURE_DETECT_EVERY = 1
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

# Camera ownership. Only one pipeline may hold the webcam at a time — Windows
# enforces that, and it is also what restores standalone gesture accuracy: with
# the camera to itself the gesture module runs its own sequential read+detect
# loop instead of competing with the gaze capture thread for the GIL.
_gaze_camera_on = threading.Event()      # set while the gaze pipeline owns it
_gaze_camera_free = threading.Event()    # set once the gaze capture thread let go

# --- Input mode -----------------------------------------------------------
# The gesture engine is created on the first switch into a mode that needs it
# and torn down on the way out, so a player who never leaves gaze mode pays
# nothing for hand tracking. _mode_lock guards both together.
_mode = MODE_GAZE
_gesture: "GesturePointer | None" = None
_mode_lock = threading.Lock()


def open_gaze_camera() -> bool:
    """Take the webcam for the gaze pipeline and start its capture thread."""
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_MSMF)   # MSMF: 720p @ 30fps
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        cap.release()
        return False
    _gaze_camera_free.clear()
    _gaze_camera_on.set()
    threading.Thread(target=_capture_thread, args=(cap,), daemon=True).start()
    return True


def release_gaze_camera(timeout: float = 4.0) -> bool:
    """Ask the gaze capture thread to let go, and wait until it actually has.

    Waiting matters: opening the camera again before the previous owner has
    released it fails on Windows, which would leave gesture mode with no camera.
    """
    if not _gaze_camera_on.is_set():
        return True
    _gaze_camera_free.clear()
    _gaze_camera_on.clear()
    ok = _gaze_camera_free.wait(timeout)
    time.sleep(0.3)          # the driver needs a beat before it can be reopened
    return ok


def read_shared_frame():
    """The newest camera frame, **as captured**, and its sequence number.

    This is the seam the gesture engine feeds from: the capture thread already
    owns the camera, so nothing else ever opens one.

    Note the frame is NOT mirrored — ``gaze_loop`` flips its own local copy for
    calibration, which never reaches this slot. Consumers that need a mirrored
    image (as the hand tracker does) must flip it themselves.
    """
    with _frame_lock:
        return _latest_frame, _frame_seq


def set_mode(mode: str) -> dict:
    """Switch input mode, starting or stopping hand tracking as needed.

    Returns the resulting status. Safe to call with the mode already active —
    switching to the same mode is a no-op rather than a restart, so repeated
    UI clicks cannot stutter the engine.
    """
    global _mode, _gesture

    if mode not in VALID_MODES:
        return mode_status(error=f"unknown mode {mode!r}")
    if mode != MODE_GAZE and not GESTURE_AVAILABLE:
        return mode_status(error="gesture module unavailable: " + GESTURE_IMPORT_ERROR)

    with _mode_lock:
        if mode == _mode:
            return mode_status()

        if mode == MODE_GESTURE:
            # Hand the camera over completely. The gesture module then opens it
            # itself through MediaPipeHandTracker and runs the same sequential
            # read->detect loop it uses standalone — which is the whole point:
            # sharing frames with the gaze capture thread cost ~20% of the
            # detection rate, because MediaPipe holds the GIL during inference.
            if not release_gaze_camera():
                return mode_status(error="gaze pipeline would not release the camera")
            try:
                _gesture = GesturePointer(own_camera=True,
                                          camera_index=CAM_INDEX,
                                          frame_size=(CAM_W, CAM_H)).start()
                print("Gesture engine started with EXCLUSIVE camera (standalone path)")
            except Exception as exc:
                _gesture = None
                open_gaze_camera()          # roll back so gaze still works
                return mode_status(error=f"could not start gesture engine: {exc}")
        else:                                # MODE_GAZE
            if _gesture is not None:
                _gesture.stop()              # closes its tracker, frees the camera
                _gesture = None
                time.sleep(0.3)              # let the driver settle before reopening
                print("Gesture engine stopped, camera returned to gaze")
            if not open_gaze_camera():
                return mode_status(error="could not reopen the camera for gaze")

        _mode = mode
        print(f"Input mode -> {mode}")
        return mode_status()


def mode_status(error: str = "") -> dict:
    """Status payload the browser uses to render the mode indicator."""
    if not error and not GESTURE_AVAILABLE:
        error = GESTURE_IMPORT_ERROR
    return {
        "type": "mode",
        "mode": _mode,
        "gesture_available": GESTURE_AVAILABLE,
        "error": error,
        "t": time.time(),
    }


def _capture_thread(cap):
    """Owns the camera for the GAZE pipeline only.

    Exits and releases the camera when _gaze_camera_on clears, so gesture mode
    can take exclusive ownership. Windows allows exactly one owner of a webcam.
    """
    global _latest_frame, _frame_seq
    while _running and _gaze_camera_on.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        with _frame_lock:
            _latest_frame = frame
            _frame_seq += 1
    cap.release()
    with _frame_lock:
        _latest_frame = None
    _gaze_camera_free.set()          # tell whoever is waiting the camera is theirs


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

    # Capture runs on its own thread; inference always grabs the freshest frame.
    # open_gaze_camera() starts that thread, and gesture mode can later take the
    # camera away and hand it back.
    if not open_gaze_camera():
        print(f"ERROR: could not open webcam at index {CAM_INDEX}.")
        _running = False
        return
    print("Gaze loop running (GPU / threaded capture / 1€ filter) — look around.  (Ctrl+C to stop)")
    fx = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_DCUTOFF)
    fy = OneEuro(ONE_EURO_MIN_CUTOFF, ONE_EURO_BETA, ONE_EURO_DCUTOFF)
    frame_idx = 0
    last_seq = -1
    try:
        while _running:
            # Gesture mode owns the camera outright. Idle completely: no frame
            # polling, no inference, nothing competing for the GIL — the gesture
            # module gets the process to itself, as it does standalone.
            if not _gaze_camera_on.is_set():
                if LATEST["ok"]:
                    LATEST = {"type": "gaze", "ok": False,
                              "x": None, "y": None, "t": time.time()}
                    fx.reset(); fy.reset()   # don't resume from a stale point
                last_seq = -1                # frames restart when we get it back
                time.sleep(0.05)
                continue

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


def build_input_message() -> dict:
    """Resolve the active input source into one message the game can consume.

    The browser never inspects which device is driving — it reads x/y and reacts
    to click_seq. That is what lets the same game code run on either input.

      gaze    - eyes aim, dwell clicks (unchanged behaviour)
      gesture - fingertip aims, pinch clicks

    Exactly one of the two pipelines is alive at a time, because they cannot
    share the camera without degrading hand tracking.
    """
    gaze, mode = LATEST, _mode
    gesture = _gesture.snapshot() if _gesture is not None else None

    if mode == MODE_GESTURE and gesture is not None:
        ok, x, y = gesture["ok"], gesture["x"], gesture["y"]
    else:
        ok, x, y = gaze["ok"], gaze["x"], gaze["y"]

    msg = {
        "type": "input",
        "mode": mode,
        "source": MODE_GESTURE if mode == MODE_GESTURE else MODE_GAZE,
        "ok": bool(ok),
        "x": x,
        "y": y,
        # Present in every mode so the HUD can show hand state even while the
        # eyes are steering.
        "hand_ok": bool(gesture["ok"]) if gesture else False,
        "gesture": gesture["gesture"] if gesture else "",
        # Counters, not booleans: the client fires once per increment, so a
        # dropped frame can never swallow a click or replay an old one.
        "click_seq": gesture["click_seq"] if gesture else 0,
        "clear_seq": gesture["clear_seq"] if gesture else 0,
        "swipe_seq": gesture["swipe_seq"] if gesture else 0,
        "swipe": gesture["swipe"] if gesture else "",
        "t": time.time(),
    }
    return msg


async def handler(websocket):
    """One coroutine per connected browser.

    Mostly we push data, but the client may also send commands — currently only
    a mode switch, which is what makes runtime switching possible without
    restarting the server or the game.
    """
    CLIENTS.add(websocket)
    print(f"Browser connected  (clients: {len(CLIENTS)})")
    try:
        await websocket.send(json.dumps(mode_status()))   # so the HUD is right immediately
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue
            if msg.get("cmd") == "mode":
                # Off the event loop: a switch now hands the camera between
                # pipelines and builds a MediaPipe landmarker, which takes
                # seconds. Running that inline would freeze the broadcaster and
                # every other client for the duration.
                status = await asyncio.to_thread(set_mode, str(msg.get("mode", "")))
                broadcast(CLIENTS, json.dumps(status))    # tell every tab, not just this one
    except Exception as exc:
        # Never swallow silently: a failure in here used to close the socket with
        # no trace, which looked exactly like a hang from the browser's side.
        import traceback
        print(f"WS handler error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        CLIENTS.discard(websocket)
        print(f"Browser disconnected  (clients: {len(CLIENTS)})")


async def broadcaster():
    """Fan the newest reading out to every client on a fixed timer.

    Every JSON encode here competes for the GIL with MediaPipe inference, so we
    only send what actually carries news: the raw gaze feed is re-sent only when
    the gaze loop produces a new reading (~15-19/s), not blindly every tick.
    The resolved input message still goes out every tick, because gesture state
    can change between gaze readings.
    """
    period = 1.0 / SEND_HZ
    last_gaze_t = None
    while _running:
        if CLIENTS:
            # The raw gaze feed keeps its original shape and name so existing
            # consumers (gaze_test.html, and the game before it was migrated)
            # keep working untouched.
            if LATEST["t"] != last_gaze_t:
                last_gaze_t = LATEST["t"]
                broadcast(CLIENTS, json.dumps(LATEST))
            broadcast(CLIENTS, json.dumps(build_input_message()))
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
        if GESTURE_AVAILABLE:
            print("Gesture control available — press [m] in the game to switch modes.")
        else:
            print(f"Gesture control UNAVAILABLE ({GESTURE_IMPORT_ERROR})")
            print("  Gaze still works normally; see docs/INTEGRATION.md to fix.")
        print("Open gaze_test.html in your browser to see your gaze as a dot.")
        await broadcaster()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        _running = False
        print("\nStopping…")
    finally:
        # Release the hand tracker too, or its thread keeps the process alive.
        if _gesture is not None:
            _gesture.stop()
