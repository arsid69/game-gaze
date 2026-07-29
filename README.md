# Gaze-Controlled AI Quest — Setup & Launch Guide

A browser game you play **with your eyes**. A Python program watches your webcam, works out
where you are looking on screen, and streams that point to the browser over a local WebSocket.
You "click" by looking at something and holding still for about a second (dwell-to-click).

The project is two halves that run at the same time:

| Half | What it is | Runs on |
|---|---|---|
| **Gaze module** (`gaze-detection/`) | Python: webcam → MediaPipe face/iris → L2CS-Net gaze model (ONNX) → per-person calibration → 1€ filter → WebSocket broadcast | `ws://localhost:8765` |
| **Game** (`game/workingGameTemplate/`) | Plain HTML/JS/Three.js — a 2D quest (`index.html`) and a 3D forest quest (`forest.html`) | `http://localhost:8000` |

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Windows 10 / 11** | The gaze server uses the Media Foundation camera backend and DirectML for GPU inference. |
| **Python 3.12 – 3.14** | The reference environment is Python 3.14. `python --version` to check. |
| **Webcam** | A plain 720p laptop camera is enough. No extra sensors. |
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
gaze-detection/face_landmarker.task          # ~3.8 MB — MediaPipe face landmarker
gaze-detection/models/L2CSNet_gaze360.pkl    # ~91 MB — L2CS PyTorch weights
gaze-detection/models/l2cs_gaze360.onnx      # ~91 MB — exported ONNX model (what runs at runtime)
```

---

## 3. Setup, step by step

### Step 1 — Clone and enter the project

```powershell
git clone <your-repo-url> "game integration"
cd "game integration\gaze-detection"
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

## 5. Playing — the two-key ritual (every single time)

1. Press **`F`** → fullscreen. **Do not skip this.** Calibration maps your gaze to the whole
   physical screen; a windowed viewport is smaller and pushed down by the browser chrome, so
   the cursor lands off by the height of the toolbar. It also unlocks sound.
2. Look at the **middle of the screen**, then press **`C`** → recenters the cursor on your eyes.

Then look at an orb or button and **hold still for ~1 second** — a ring charges around the
cursor and it clicks.

### Controls

| Key / action | Effect |
|---|---|
| Look + hold ~1 s | Click / collect the highlighted thing |
| `F` | Fullscreen on / off (also enables sound) |
| `C` | Recenter — press whenever the cursor feels shifted |
| `G` | Gaze control off / on (mouse still works) |
| Mouse drag | Orbit the camera (3D forest) |
| Scroll | Zoom (3D forest) |
| Look at left/right edge | Pan the field (2D Collect stage) |
| Look at top/bottom edge | Scroll a long page |

---

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| Blank page / nothing loads | The game server (Terminal 2) isn't running, or you opened the file directly instead of `http://localhost:8000/forest.html` |
| `ERROR: models/calibration_model.pkl not found` | Run `python milestone4_calibration.py` (Step 8) |
| `ERROR: could not open webcam at index 0` | Another app (Teams, Zoom, a previous `gaze_server.py`) holds the camera. Close it, or change `CAM_INDEX` in `gaze_server.py` |
| Cursor is in the wrong place | You're not fullscreen — press `F`, then look at centre and press `C` |
| Cursor drifted over time | Look at the middle, press `C` again |
| Chip bottom-left shows `face: —` | Check the gaze server window is still running and your face is lit and in frame |
| `socket: disconnected` | Terminal 1 died or was Ctrl+C'd — restart it; the browser reconnects automatically |
| Really shaky or laggy | Close every other game tab; only one should be open |
| No sound | Browsers require a key-press first — hit `F` once |
| Port 8000 or 8765 already in use | Kill the old process, or change `PORT` in `gaze_server.py` (and `WS_URL` in `gaze-client.js` / `GAZE_WS` in `forest.js`) |

### When you're finished

Press **Ctrl + C** in **both** PowerShell windows. This frees the webcam and the ports.

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
│   ├── gaze-client.js       # Gaze overlay for the 2D game
│   ├── styles.css
│   ├── models/              # .glb assets for the 3D scene
│   └── vendor/three/        # Vendored Three.js — no npm install needed
├── gaze-detection/
│   ├── gaze_server.py       # WebSocket bridge — run this
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

- [`GAZE_SYSTEM_DOCS.md`](GAZE_SYSTEM_DOCS.md) — architecture, design decisions, performance figures
- [`gaze-detection/documents/TECHNICAL_DOCUMENTATION.md`](gaze-detection/documents/TECHNICAL_DOCUMENTATION.md) — algorithms and math
- [`gaze-detection/documents/ROADMAP.md`](gaze-detection/documents/ROADMAP.md) — path to higher accuracy
- [`docs/forest_cheatsheet.html`](docs/forest_cheatsheet.html) — one-page player cheat sheet
