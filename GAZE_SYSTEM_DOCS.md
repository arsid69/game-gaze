# Gaze-Controlled Flood Response — Technical Documentation & Cheat Sheet

**Project:** Controlling the *AI Quest — Flood Response* browser game with webcam gaze instead of a mouse.
**Status:** Working end-to-end (Start → Collect → Clean → Train → Implement, all by gaze).
**Last updated:** 2026-07-26

---

# PART 1 — TECHNICAL DOCUMENTATION (formal)

## 1. Overview

The system lets a user play a 2D browser game using **eye gaze** as the pointing device and a **dwell** (look-and-hold) gesture as the click. It connects two previously separate systems:

- A **Python gaze-tracking pipeline** (MediaPipe + an L2CS ONNX gaze model + a per-user calibration model) that outputs where the user is looking as normalized screen coordinates.
- A **browser game** (plain HTML/DOM; the only WebGL is a decorative background) whose every action is an HTML `<button>` and whose field panning is bound to the arrow keys.

The integration is **additive and non-invasive**: the game's application logic (`main.js`) is unmodified. Gaze control is delivered as an overlay script plus one `<script>` tag, expressed entirely through interactions the game already understands (native button `.click()` and arrow-key events). Mouse and keyboard remain fully functional.

## 2. System Architecture

```
┌────────────┐  frames   ┌───────────────────────────────────────────┐
│  Webcam    │ ────────▶ │  gaze_server.py  (Python)                 │
└────────────┘           │                                            │
                         │  capture thread (MSMF, 1280×720, ~31fps)   │
                         │        │ freshest frame                    │
                         │        ▼                                    │
                         │  inference loop:                           │
                         │    MediaPipe FaceLandmarker → face crop     │
                         │    L2CS ONNX gaze model (DirectML / GPU)    │
                         │    → (pitch, yaw)                           │
                         │    apply_calibration() → (x, y) ∈ [0,1]     │
                         │    1€ filter (adaptive smoothing)           │
                         │        │                                    │
                         │  WebSocket server (asyncio) broadcasts      │
                         └────────┼───────────────────────────────────┘
                                  │ ws://localhost:8765
                                  │ {"type":"gaze","ok":true,"x":..,"y":..}
                                  ▼
                         ┌───────────────────────────────────────────┐
                         │  gaze-client.js  (browser overlay)         │
                         │    • render gaze cursor + dwell ring        │
                         │    • recenter offset ([c])                  │
                         │    • aim-assist (3 tiers) + occlusion test   │
                         │    • dwell → element.click()                │
                         │    • edge gaze → ArrowLeft/Right / scroll    │
                         └────────┼───────────────────────────────────┘
                                  │ native DOM events
                                  ▼
                         ┌───────────────────────────────────────────┐
                         │  main.js  (game logic — UNMODIFIED)        │
                         └───────────────────────────────────────────┘
```

## 3. Components

### 3.1 `gaze_server.py` (Python bridge)
Runs the gaze pipeline and streams results to the browser.

- **Capture thread.** A dedicated thread continuously reads frames via the **Media Foundation** backend (`cv2.CAP_MSMF`) at **1280×720**, writing the newest frame to a shared slot. Decoupling capture from inference prevents the webcam read from stalling the GPU.
- **Inference loop.** Consumes the freshest frame, horizontally mirrors it (to match calibration, which was recorded mirrored), and runs:
  1. `get_gaze_reading()` — MediaPipe FaceLandmarker (VIDEO mode) locates the face; a smoothed square crop is fed to the **L2CS gaze model** (`l2cs_gaze360.onnx`) to produce `(pitch, yaw)`.
  2. `apply_calibration()` — a per-user polynomial model maps `(pitch, yaw)` → normalized screen `(x, y)` in `[0, 1]`.
  3. **1€ filter** — adaptive smoothing on `x` and `y` (see §5.4).
- **ONNX execution provider.** The session prefers `DmlExecutionProvider` (**DirectML → GPU**) and falls back to CPU. On the target machine (RTX 4060) this reduces model latency from ~104 ms to ~15 ms.
- **WebSocket server.** An `asyncio` server (`ws://localhost:8765`) broadcasts the latest reading to all connected clients at `SEND_HZ` (60). When no face is detected it emits `ok:false`.

### 3.2 `gaze-client.js` (browser overlay)
Injected after `main.js` via one `<script>` tag. Self-contained IIFE.

- **Coordinate mapping.** Incoming `(x, y)` are normalized to the **whole physical screen** (that is how calibration is defined). They are mapped to `window.innerWidth/innerHeight`. This is correct **only in fullscreen**; in a windowed browser the viewport is smaller and offset by the tab/URL bar, producing a systematic error (see §5.3).
- **Recenter offset.** Pressing `c` stores `drift = rawGaze − screenCentre`; subsequent readings subtract it, correcting per-session head-position drift.
- **Aim-assist (three tiers).** Each frame the nearest eligible button to the gaze point is selected; a magnet then pulls the rendered cursor toward it. Tiers by CSS selector:

  | Tier | Selector | Reach (`pad`) | Magnet | Rationale |
  |---|---|---|---|---|
  | Navigation | `.proceed-btn, #start-btn` | 200 px | 0.65 | Large, isolated, in the weak lower screen — must be easy to hit |
  | Clean cards | `.clean-card` | 30 px | 0.40 | Large but tightly packed — small reach prevents grabbing neighbours |
  | Data buttons | (default) | 85 px | 0.15 | Standard forgiveness without over-grabbing |

- **Occlusion test.** A button is a candidate only if it is the top-most element at its own centre (`document.elementFromPoint`). This prevents "clicking through" an overlay (e.g. the intro/Start card) to buttons behind it.
- **Stickiness (hysteresis).** The current target keeps a 60 px distance bonus so the selection does not flicker between adjacent buttons.
- **Dwell-to-click.** Holding gaze on a target for `DWELL_MS` (1050 ms) fires the element's native `.click()`, then a 500 ms cooldown and a "leave before re-arm" rule prevent accidental repeats.
- **Edge navigation.** Gaze in the horizontal edge zone dispatches `ArrowLeft/ArrowRight` (the game's own pan handler) during the Collect stage; gaze in the vertical edge zone scrolls the page when it overflows.
- **Disabled buttons** (e.g. collected data sources) are excluded from all targeting and dimmed via injected CSS.

### 3.3 Calibration (`milestone4_calibration.py`)
Produces `models/calibration_model.pkl`. A smooth-pursuit routine records ~1,500+ `(pitch, yaw) → (target_x, target_y)` samples as the user's eyes follow a moving dot, fits a degree-3 polynomial (with robust outlier rejection), then measures held-out error on 5 static points. **Its camera settings (MSMF, 1280×720) must match `gaze_server.py`** so the model trains on the same face-crop scale it is later used at.

## 4. Message Format

One JSON object per broadcast tick:
```json
{ "type": "gaze", "ok": true,  "x": 0.42, "y": 0.71, "t": 1785063539.1 }
{ "type": "gaze", "ok": false, "x": null, "y": null, "t": 1785063539.2 }
```
`ok:false` denotes no face detected. Coordinates are normalized `[0, 1]`, origin top-left.

## 5. Key Technical Decisions & Rationale

**5.1 GPU via DirectML.** The L2CS model on CPU cost ~104 ms/inference (~10 fps ceiling). Switching `onnxruntime` → `onnxruntime-directml` and preferring `DmlExecutionProvider` runs it on the GPU at ~15 ms. DirectML was chosen over CUDA because it requires no CUDA/cuDNN installation on Windows and works through DirectX 12.

**5.2 MSMF camera backend + 720p.** With DirectShow (`CAP_DSHOW`) the webcam delivered 720p at only ~10 fps (the dominant source of lag). The Media Foundation backend (`CAP_MSMF`) delivers the same 1280×720 at ~31 fps. 720p is retained (over 480p) because the sharper face crop yields better gaze accuracy.

**5.3 Fullscreen requirement.** Calibration maps gaze to the entire screen. A windowed browser viewport is smaller than, and vertically offset from, the screen, so the mapping is wrong by the height of the browser chrome. Running fullscreen makes viewport ≡ screen and removes the offset. This was the single largest accuracy issue observed.

**5.4 1€ filter (adaptive smoothing).** Fixed-strength smoothing forces a trade-off: heavy smoothing is steady but laggy; light smoothing is responsive but shaky. The 1€ filter varies its cutoff with gaze speed — smoothing hard when the gaze is still (removing jitter) and easing off during fast movement (preserving responsiveness). Parameters: `MIN_CUTOFF = 0.7`, `BETA = 0.6`.

**5.5 Dwell-to-click.** Gaze cannot reliably produce a discrete "click." A dwell (sustained fixation) is the standard gaze-selection gesture and tolerates the pipeline's ~30 fps update rate.

## 6. Configuration Reference

**`gaze_server.py`**
| Constant | Value | Meaning |
|---|---|---|
| `PORT` | 8765 | WebSocket port |
| `CAM_W, CAM_H` | 1280, 720 | Capture resolution (MSMF) |
| `SEND_HZ` | 60 | Broadcast rate |
| `ONE_EURO_MIN_CUTOFF` | 0.7 | ↓ = steadier at rest |
| `ONE_EURO_BETA` | 0.6 | ↑ = snappier on fast moves |

**`gaze-client.js`**
| Constant | Value | Meaning |
|---|---|---|
| `DWELL_MS` | 1050 | Hold time to click |
| `HOVER_PAD` / `MAGNET` | 85 / 0.15 | Data-button reach / pull |
| `PROCEED_PAD` / `PROCEED_MAGNET` | 200 / 0.65 | Navigation-button reach / pull |
| `CARD_PAD` / `CARD_MAGNET` | 30 / 0.40 | Clean-card reach / pull |
| `HYSTERESIS` | 60 | Target stickiness (px) |
| `EDGE_ZONE` | 0.09 | Edge fraction that pans/scrolls |

## 7. Performance Characteristics (target machine, RTX 4060)

| Metric | Before | After |
|---|---|---|
| Gaze model latency | ~104 ms (CPU) | ~15 ms (GPU/DirectML) |
| Camera throughput @720p | ~10 fps (DSHOW) | ~31 fps (MSMF) |
| End-to-end gaze update rate | ~5 fps | ~30 fps |
| Calibration held-out error | — | ~8.9 % of screen (top/centre 3–8 %, bottom 13–14 %) |

## 8. Files & Locations

```
C:\game integration\
├── game\workingGameTemplate\
│   ├── index.html        (game; one <script src="gaze-client.js"> added)
│   ├── main.js           (game logic — UNMODIFIED)
│   └── gaze-client.js    (gaze overlay — NEW)
└── gaze-detection\
    ├── gaze_server.py            (WebSocket bridge — NEW)
    ├── gaze_test.html            (standalone pipe test — NEW)
    ├── gaze_pipeline.py          (edited: prefers DirectML)
    ├── milestone4_calibration.py (edited: MSMF/720p to match runtime)
    ├── models\calibration_model.pkl
    └── gaze_env\                 (Python 3.14 venv; pyvenv.cfg → C:\Python314)
```

---

# PART 2 — CHEAT SHEET (plain English)

## What this is
A way to **play the game with your eyes**. A Python program watches your webcam, figures out where you're looking, and sends that to the game. You "click" a button by **looking at it and holding still for about a second**.

## Start it up (2 steps)

**1. Start the eye-tracking program.** Open PowerShell (press Windows key, type `powershell`, Enter) and paste:
```
cd "C:\game integration\gaze-detection"; .\gaze_env\Scripts\python.exe gaze_server.py
```
Wait until it says **"WebSocket server ready."** Leave this window open.

**2. Open the game**, then do the two-key ritual below.

## The two-key ritual (do this EVERY time)
1. Press **`F`** → makes the game fill the whole screen. *(Skip this and the cursor will be off — this is the #1 thing people forget.)*
2. Look at the **middle of the screen**, then press **`c`** → lines the cursor up with your eyes.

That's it. Now look at buttons and hold to click.

## Controls
| Key / action | What it does |
|---|---|
| **Look + hold ~1 sec** | Clicks the button you're looking at (a ring fills up, then it clicks) |
| **`F`** | Fullscreen on/off |
| **`c`** | Recenter — do it whenever the cursor feels shifted |
| **`g`** | Turn gaze control off/on (mouse still works) |
| Look at **left/right edge** | Scrolls the field sideways (Collect stage) |
| Look at **top/bottom edge** | Scrolls up/down if the page is long |

## How "clicking" works
The button you're looking at **glows cyan** — that's the one about to be clicked. A ring fills around the cursor; when it's full, it clicks. If the **wrong** button is glowing, just look more directly at the one you want and it'll switch.

## If something feels wrong
| It feels like… | Do this |
|---|---|
| Cursor is in the wrong place | You're probably not fullscreen — press **`F`**, then look at centre and press **`c`** |
| Cursor drifted off over time | Look at the middle, press **`c`** again |
| It clicked the wrong button | Look more squarely at the button you want; hold steadier |
| Nothing happens / no cursor | Check the eye-tracker window is still running and says a face is detected; check the chip (bottom-left) shows `face: yes` |
| Really shaky or laggy | Make sure only ONE game tab is open; close the others |

## When to redo calibration (the "follow the dot" step)
Only if accuracy is genuinely bad **even after** fullscreen + recenter — e.g. a different person is using it, or the lighting/seating changed a lot. Then:
```
cd "C:\game integration\gaze-detection"; .\gaze_env\Scripts\python.exe milestone4_calibration.py
```
Sit still, hold the green box, **follow the dot with your eyes** all the way through the 5 dots at the end, wait for the score, done. Then restart the eye-tracker program.

## When you're finished
Click the eye-tracker PowerShell window and press **Ctrl + C** to stop it (frees the webcam).
