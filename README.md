# Gaze- and Gesture-Controlled AI Quest — Setup & Launch Guide

A browser game you play **with your eyes or with your hands**. A Python program watches your
webcam and streams a cursor position to the browser over a local WebSocket. You "click" either
by looking at something and holding still for about a second (dwell), or by **pinching** your
thumb and index finger.

Press **`m`** in the game to switch between the two at any time — no restart, no lost progress.

The project is three parts:

| Part | What it is | Runs on |
|---|---|---|
| **Gaze module** (`gaze-detection/`) | Python: webcam → MediaPipe face/iris → L2CS-Net gaze model (ONNX) → per-person calibration → 1€ filter | `ws://localhost:8765` |
| **Gesture module** (`gesture-control-module/`) | Python: webcam → MediaPipe Hands → gesture classifier → pointer, pinch, swipe | same socket |
| **Game** (`game/workingGameTemplate/`) | Plain HTML/JS/Three.js — a 2D quest (`index.html`) and a 3D forest quest (`forest.html`) | `http://localhost:8000` |

Both input modules are served by the **same** `gaze_server.py`, so there is still only one
server to start. How they were wired together is documented in
[`docs/INTEGRATION.md`](docs/INTEGRATION.md).

> **Only one input mode runs at a time.** Windows lets just one program own a webcam, and
> sharing frames between the two pipelines measurably degraded hand tracking. Whichever mode is
> active gets the camera to itself, so switching takes about 1.5 seconds.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Windows 10 / 11** | The gaze server uses the Media Foundation camera backend and DirectML for GPU inference. |
| **Python 3.12 – 3.14** | The reference environment is Python 3.14. `python --version` to check. |
| **Webcam** | A plain 720p laptop camera is enough. No extra sensors. **Check it isn't muted** — many laptops have an `Fn` key or a physical shutter, and a muted camera streams a grey padlock image that neither face nor hand detection can read. |
| **A Chromium browser** | Chrome or Edge. Fullscreen is required for accuracy (see §5). |
| **Git** | Two of the Python dependencies install straight from GitHub. |
| **GPU (optional)** | Any DirectX 12 GPU gives ~15 ms gaze inference via DirectML instead of ~104 ms on CPU. It falls back to CPU automatically. |

> **No Node.js / npm required.** Three.js and its loaders are vendored in
> `game/workingGameTemplate/vendor/`, so the game has zero JavaScript package installs.

---

## 2. Required libraries

All Python dependencies are pinned in [`gaze-detection/requirements.txt`](gaze-detection/requirements.txt).
The ones that matter, and why:

### Core gaze pipeline
| Library | Version | Purpose |
|---|---|---|
| `mediapipe` | 0.10.35 | Face + iris landmarks and the head-pose transformation matrix |
| `l2cs` | git @ `4a0f978` | L2CS-Net pretrained gaze-estimation model |
| `face-detection` | git @ `786fbab` | Face detector required by the L2CS package |
| `onnxruntime-directml` | 1.24.4 | Runs the gaze model on the GPU through DirectX 12 |
| `onnx` / `onnxscript` / `onnx-ir` | 1.22.0 / 0.7.1 / 0.2.1 | One-time PyTorch → ONNX conversion of the gaze model |
| `torch` + `torchvision` | 2.13.0+cpu / 0.28.0+cpu | Only needed to export the model to ONNX (CPU build is fine) |
| `opencv-python` / `opencv-contrib-python` | 5.0.0.93 | Webcam capture, image ops, on-screen overlays |
| `numpy` | 2.5.1 | Everything numeric |
| `scikit-learn` | 1.9.0 | Fits the degree-3 calibration polynomial |
| `scipy` | 1.18.0 | Robust outlier rejection during calibration |
| `websockets` | 16.1.1 | The `ws://localhost:8765` bridge to the browser |

### Gesture pipeline
The gesture module needs **no extra packages** — it reuses `mediapipe`, `opencv-python` and
`numpy`, which the gaze environment already installs. That is why both run from the one venv.

| Library | Purpose |
|---|---|
| `mediapipe` | Hand landmarks (21 points per hand) via the Tasks API |
| `opencv-python` | Webcam capture and colour conversion |
| `numpy` | Landmark normalisation, the geometric classifier and the k-NN |

> The module supports both the current MediaPipe **Tasks** API and the older `mp.solutions`
> one, picking automatically. On this project's MediaPipe 0.10.35 it uses Tasks.

### Supporting
| Library | Purpose |
|---|---|
| `pandas` | Reads/writes the gaze sample dataset (`data/gaze_samples.csv`) |
| `matplotlib` | Accuracy plots in the analysis scripts |
| `PyAutoGUI`, `PyGetWindow`, `PyScreeze`, `MouseInfo` | Screen size + optional real-mouse control (`milestone8_live_cursor.py`) |
| `sounddevice` | Audio cues in the calibration/game milestones |
| `joblib` | Saving / loading `models/calibration_model.pkl` |
| `pillow` | Image handling |

### Model files — **not in Git** (too large)
These three are excluded by `.gitignore` and must be fetched during setup:

```
gaze-detection/face_landmarker.task              # ~3.8 MB — MediaPipe face landmarker
gaze-detection/models/L2CSNet_gaze360.pkl        # ~91 MB — L2CS PyTorch weights
gaze-detection/models/l2cs_gaze360.onnx          # ~91 MB — exported ONNX model (runs at runtime)
gesture-control-module/models/hand_landmarker.task  # ~7 MB — MediaPipe hand landmarker
```

The hand model downloads itself on first use if it is missing, so it usually needs no action.

---

## 3. Setup, step by step

### Step 1 — Clone and enter the project

```powershell
git clone https://github.com/arsid69/game-gaze.git
cd game-gaze\gaze-detection
```

### Step 2 — Create and activate the virtual environment

```powershell
python -m venv gaze_env
.\gaze_env\Scripts\Activate.ps1
```

If PowerShell blocks the activation script:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Step 3 — Install the Python libraries

`torch` is pinned to a `+cpu` build, so the PyTorch index has to be reachable:

```powershell
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

If `torch` still fails, install it separately first and then re-run the line above:
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**No DirectX 12 GPU?** Swap the ONNX runtime for the CPU build — everything still works, just slower
(~104 ms per frame instead of ~15 ms):
```powershell
pip uninstall -y onnxruntime-directml
pip install onnxruntime
```

The two GitHub packages (`l2cs`, `face-detection`) are already listed in
`requirements.txt`. If they were skipped, install them explicitly:
```powershell
pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
```

### Step 4 — Download the MediaPipe face landmarker

Run from `gaze-detection/`:

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"
```

### Step 5 — Get the gaze model weights

**Option A (fastest)** — if a teammate sends you `l2cs_gaze360.onnx`, drop it into
`gaze-detection/models/` and skip to Step 6.

**Option B (from scratch)** — download the PyTorch weights and convert them:

```powershell
mkdir models -Force
Invoke-WebRequest -Uri "https://huggingface.co/dorni/SpeakerVid-5M-data-curation-models/resolve/5c6e04d7fa3321e6228e79162f8ec98466bf308a/L2CSNet_gaze360.pkl" -OutFile "models\L2CSNet_gaze360.pkl"
python export_to_onnx.py
```

### Step 6 — Verify the pipeline (optional but recommended)

Each script opens a webcam window; press `q` to close it.

```powershell
python milestone1_face_mesh.py      # face + iris landmarks appear
python milestone2_eye_headpose.py   # eye crops + pitch/yaw/roll
python milestone3_gaze_model.py     # a gaze direction arrow
```

### Step 7 — Set your camera focal length

```powershell
python milestone5_positioning.py
```
Sit **50 cm** from the screen, face centred, and press `c` once. This writes
`models/camera_focal.json` and makes the distance gate accurate. Press `q` to exit.

### Step 8 — Calibrate (required, per person)

```powershell
python milestone4_calibration.py
```

Hold the green positioning box, then **follow the moving dot with your eyes** all the way
through, including the 5 static validation dots at the end. This writes
`models/calibration_model.pkl` — the file the game actually depends on.

> Calibration is **per person, per laptop, per sitting position**. The committed
> `calibration_model.pkl` belongs to whoever ran it last — a new player must re-run this step.

Check your accuracy at any time:
```powershell
python milestone6_test_accuracy.py
```

### Step 9 — Confirm the folder looks like this

```
gaze-detection/
├── face_landmarker.task
├── models/
│   ├── l2cs_gaze360.onnx
│   ├── calibration_model.pkl
│   └── camera_focal.json
├── gaze_server.py
├── gaze_pipeline.py
├── calibration_utils.py
└── requirements.txt
```

---

## 4. Launching the game — two servers

Both must be running at the same time, in **two separate PowerShell windows**.

### Terminal 1 — the gaze server (port 8765)

```powershell
cd "C:\game integration\gaze-detection"
.\gaze_env\Scripts\python.exe gaze_server.py
```

Wait for:
```
Gaze loop running (GPU / threaded capture / 1€ filter) — look around.
WebSocket server ready at ws://localhost:8765
```
**Leave this window open.** It owns the webcam.

### Terminal 2 — the game's static file server (port 8000)

```powershell
cd "C:\game integration\game\workingGameTemplate"
python -m http.server 8000
```

> The 3D game uses ES modules and an import map — it **will not run** if you double-click
> the HTML file. It must be served over `http://`.

### Open in the browser

| Game | URL |
|---|---|
| **3D Forest AI Quest** (the current game) | http://localhost:8000/forest.html |
| 2D AI Quest (earlier version) | http://localhost:8000/index.html |
| Gaze pipe test — just a dot that follows your eyes | open `gaze-detection/gaze_test.html` |

---

## 5. Playing

The HUD in the corner always shows which input mode is live: `input GAZE` or `input GESTURE`.
The game starts in gaze mode.

### 5a. Gaze mode — the two-key ritual (every single time)

1. Press **`F`** → fullscreen. **Do not skip this.** Calibration maps your gaze to the whole
   physical screen; a windowed viewport is smaller and pushed down by the browser chrome, so
   the cursor lands off by the height of the toolbar. It also unlocks sound.
2. Press **`C`** → recenter. **The two games differ here:**
   - **3D forest** (`forest.html`) — a yellow **LOOK HERE** target appears. Keep your eyes on it
     until it disappears (~1.3 s); the offset is captured at the end of that hold.
   - **2D quest** (`index.html`) — look at the middle of the screen *first*, then press `C`.
     The offset is captured the instant you press the key.

Then look at an orb or button and **hold still for ~1 second** — a ring charges around the
cursor and it activates.

### 5b. Gesture mode — press `m`

The HUD shows `SWITCHING…` for about 1.5 seconds while the camera changes hands, then reads
`input GESTURE`. Watch for `hand: yes` — that means you are being tracked.

1. Hold your hand up, **palm towards the camera**, roughly **40–70 cm** away.
2. **Point** with your index finger — the cursor follows your fingertip.
3. **Pinch** thumb and index together to select whatever the cursor is on.
   There is **no 1-second wait** in this mode: a pinch fires immediately.
4. **Open your palm** to clear the selection.
5. **Swipe** left or right to turn the forest and find more orbs.

Press `m` again to go back to gaze. Your progress is kept across switches.

> `C` (recenter) does nothing in gesture mode, and that is deliberate — it corrects *gaze*
> drift, and your fingertip is already an absolute position.

### Controls

| Key / action | Effect |
|---|---|
| **`m`** | **Switch input mode: gaze ↔ gesture** (~1.5 s camera handover) |
| Look + hold ~1 s | *(gaze)* Click / collect the highlighted thing |
| **Pinch** thumb + index | *(gesture)* Click / collect immediately |
| **Open palm** | *(gesture)* Clear the selection |
| **Swipe left / right** | *(gesture)* Turn the forest |
| `F` | Fullscreen on / off (also enables sound) |
| `C` | Recenter — gaze mode only |
| `G` | Input control off / on (mouse still works) |
| Mouse drag / click | Orbit the camera (3D forest); works as a fallback everywhere |
| Scroll | Zoom (3D forest) |
| Look at left/right edge | *(gaze)* **3D:** turn the forest · **2D:** pan the Collect field |
| Look at top/bottom edge | *(gaze)* Scroll a long page |

### Gestures the game understands

| Gesture | Meaning |
|---|---|
| Point (index finger out) | Move the cursor |
| Pinch (thumb + index touching) | Select / click |
| Open palm | Clear selection |
| Swipe left / right | Turn the view |

Anything else you record yourself (§5c) is reported to the game by name, ready to be bound to
new actions.

### 5c. Recording your own gestures

The gesture module ships a recorder. New gestures need no code changes — the engine picks them
up the next time it starts.

**Stop the gaze server first (`Ctrl+C`)**, because the recorder needs the camera:

```powershell
cd "C:\game integration\gesture-control-module"
..\gaze-detection\gaze_env\Scripts\python.exe -m gesture_control.recorder record --name thumbs_up --samples 60
```

Hold the pose, press **Space** to start capturing, and **move your hand around slightly** while
it records — varied samples generalise far better than 60 identical frames. 30–60 is plenty.

```powershell
# list everything recorded
..\gaze-detection\gaze_env\Scripts\python.exe -m gesture_control.recorder list

# live view of what it recognises right now
..\gaze-detection\gaze_env\Scripts\python.exe -m gesture_control.recorder test

# delete one
..\gaze-detection\gaze_env\Scripts\python.exe -m gesture_control.recorder delete --name thumbs_up
```

Recordings are stored in `gestures_dataset.json` **next to the folder you run the command
from** — so always run it from `gesture-control-module/`, or the server will not find them.
Restart the gaze server afterwards to load them.

### The quest — 4 steps

1. **Collect Data** — turn through the night forest and collect the glowing data orbs.
2. **Clean Data** — discard the noisy or irrelevant sources (look at a card and hold to toggle
   keep ↔ discard).
3. **Train AI** — answer Professor Skye's train/test question, then test the model.
   **Your final accuracy depends on how well you cleaned the data** — up to 90%.
4. **Apply** — deploy the forecast and protect Market Marshes.

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| Blank page / nothing loads | The game server (Terminal 2) isn't running, or you opened the file directly instead of `http://localhost:8000/forest.html` |
| Stuck forever on **"LOADING FOREST…"** | You double-clicked the file and are on `file:///C:/...`. Browsers refuse ES modules over `file://`. Use `http://localhost:8000/forest.html` |
| Your edits to the game don't show up | Hard-reload: **Ctrl + Shift + R**. A normal reload serves the cached copy |
| `ERROR: models/calibration_model.pkl not found` | Run `python milestone4_calibration.py` (Step 8) |
| `ERROR: could not open webcam at index 0` | Another app (Teams, Zoom, a previous `gaze_server.py`) holds the camera. Close it, or change `CAM_INDEX` in `gaze_server.py` |
| Cursor is in the wrong place | You're not fullscreen — press `F`, then look at centre and press `C` |
| Cursor drifted over time | Look at the middle, press `C` again |
| Chip bottom-left shows `face: —` | Check the gaze server window is still running and your face is lit and in frame |
| **Camera shows a grey padlock; nothing is ever detected** | The webcam is muted at hardware level — look for an `Fn` key with a crossed-out camera icon, a physical shutter, or your laptop vendor's privacy utility. Windows privacy settings can read "Allow" and it can still be muted this way. |
| **`m` does nothing** | The server reports gesture as unavailable — check its startup output for the reason |
| **HUD stuck on `SWITCHING…`** | The camera handover failed. Check the server window; switching back and forth again usually recovers it |
| **`hand: —` in gesture mode** | Hand out of frame, too close/far (aim for 40–70 cm), or poorly lit. Palm towards the camera |
| **Pinch selects nothing** | Aim first — the cursor must be on the target before you pinch. Pinch fully, thumb touching index |
| **Pinch fires by accident** | Lower `gestures.pinch_close` in `gesture-control-module/gesture_control/config.py` |
| **Cursor jitters in gesture mode** | Raise `pointer.min_cutoff` in the same file |
| **Custom gesture never recognised** | Make sure you ran the recorder from `gesture-control-module/`, then restarted the server |
| `socket: disconnected` | Terminal 1 died or was Ctrl+C'd — restart it; the browser reconnects automatically |
| Really shaky or laggy | Close every other game tab; only one should be open |
| Gaze feels sluggish generally | Check the gaze server's first line: `[gaze_pipeline] ONNX running on: DmlExecutionProvider`. If it says `CPUExecutionProvider`, install `onnxruntime-directml` |
| No sound | Browsers require a key-press first — hit `F` once |
| Port 8000 or 8765 already in use (`Errno 10048`) | Stop the old server, or change `PORT` in `gaze_server.py` (and `WS_URL` in `gaze-client.js` / `GAZE_WS` in `forest.js`) |

### When you're finished

Press **Ctrl + C** in **both** PowerShell windows. This frees the webcam and the ports.

If a window is unresponsive, force-stop everything:
```powershell
taskkill /F /IM python.exe
```

Then confirm both ports are actually free — **no output means all clear**:
```powershell
netstat -ano | findstr LISTENING | findstr "8000 8765"
```

---

## 7. Tuning reference

**`gaze-detection/gaze_server.py`**

| Constant | Default | Meaning |
|---|---|---|
| `PORT` | 8765 | WebSocket port |
| `CAM_INDEX` | 0 | Which webcam |
| `CAM_W, CAM_H` | 1280, 720 | Capture resolution (do not drop to 480p — it crops the sensor and skews distance) |
| `SEND_HZ` | 60 | Broadcast rate to the browser |
| `ONE_EURO_MIN_CUTOFF` | 0.7 | ↓ = steadier when your gaze is still |
| `ONE_EURO_BETA` | 0.6 | ↑ = snappier on fast eye movements |

**`gesture-control-module/gesture_control/config.py`** — every gesture tunable

| Constant | Default | Meaning |
|---|---|---|
| `gestures.pinch_close` | 0.34 | How tight a pinch must be to register |
| `gestures.pinch_open` | 0.48 | When it lets go again (the gap stops flickering) |
| `gestures.stable_frames` | 3 | Frames a gesture must hold before it counts. ↑ steadier, slower |
| `pointer.min_cutoff` | 1.2 | ↑ = steadier cursor when your hand is still |
| `pointer.beta` | 0.02 | ↑ = snappier on fast hand movement |
| `primary_hand` | `"any"` | `"Right"` / `"Left"` to ignore the other hand |
| `tracker.max_hands` | 2 | Hands tracked at once |

**`game/workingGameTemplate/gaze-client.js`**

| Constant | Default | Meaning |
|---|---|---|
| `DWELL_MS` | 1050 | Hold time to trigger a click |
| `HOVER_PAD` / `MAGNET` | 85 / 0.15 | Data-button reach (px) / pull toward centre |
| `PROCEED_PAD` / `PROCEED_MAGNET` | 200 / 0.65 | Navigation-button reach / pull |
| `CARD_PAD` / `CARD_MAGNET` | 30 / 0.40 | Clean-stage card reach / pull |
| `HYSTERESIS` | 60 | Target stickiness, stops flicker between neighbours |
| `EDGE_ZONE` | 0.09 | Fraction of the screen edge that pans/scrolls |

---

## 8. Project layout

```
game integration/
├── game/workingGameTemplate/
│   ├── forest.html          # 3D forest quest — the current game
│   ├── forest.js            # Three.js scene + built-in gaze cursor
│   ├── quest.js             # Quest wrapper: Collect → Clean → Train → Apply
│   ├── index.html           # 2D quest (earlier version)
│   ├── main.js              # 2D game logic
│   ├── input-manager.js     # Input layer: the only WebSocket client; gaze/gesture + [m]
│   ├── gaze-client.js       # Input overlay for the 2D game
│   ├── styles.css
│   ├── models/              # .glb assets for the 3D scene
│   └── vendor/three/        # Vendored Three.js — no npm install needed
├── gesture-control-module/          # The gesture module (used as-is, unmodified)
│   ├── gesture_control/
│   │   ├── engine.py        # Interaction state machine, emits gesture events
│   │   ├── tracker.py       # MediaPipe Hands backends
│   │   ├── gestures.py      # Rule + k-NN classifiers
│   │   ├── recorder.py      # CLI for recording custom gestures
│   │   └── config.py        # Every gesture tunable
│   └── models/hand_landmarker.task
├── gaze-detection/
│   ├── gaze_server.py       # WebSocket bridge for BOTH inputs — run this
│   ├── gesture_bridge.py    # Adapter joining the gesture module to the server
│   ├── gaze_pipeline.py     # Face detect + crop + ONNX gaze inference
│   ├── calibration_utils.py # Fit / save / load / apply the calibration polynomial
│   ├── positioning_gate.py  # Distance + centering constraints
│   ├── export_to_onnx.py    # One-time PyTorch → ONNX conversion
│   ├── milestone1..9_*.py   # Step-by-step build & test scripts
│   ├── gaze_test.html       # Standalone "dot follows your eyes" test
│   ├── requirements.txt
│   └── models/
├── docs/
│   ├── forest_cheatsheet.html   # Player-facing quick reference
│   └── forest_technical.html    # 3D game technical write-up
└── GAZE_SYSTEM_DOCS.md          # Full system documentation + cheat sheet
```

## Further reading

- [`docs/INTEGRATION.md`](docs/INTEGRATION.md) — how the gesture module was joined to the gaze
  game: architecture, input flow, every file changed and why, and the measurements behind the
  exclusive-camera decision
- [`gesture-control-module/README.md`](gesture-control-module/README.md) — the gesture module's
  own documentation: event API, gesture recording, tuning
- [`GAZE_SYSTEM_DOCS.md`](GAZE_SYSTEM_DOCS.md) — architecture, design decisions, performance figures
- [`gaze-detection/documents/TECHNICAL_DOCUMENTATION.md`](gaze-detection/documents/TECHNICAL_DOCUMENTATION.md) — algorithms and math
- [`gaze-detection/documents/ROADMAP.md`](gaze-detection/documents/ROADMAP.md) — path to higher accuracy
- [`docs/forest_cheatsheet.html`](docs/forest_cheatsheet.html) — one-page player cheat sheet
