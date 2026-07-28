# Project Context — Webcam Gaze Detection

**Purpose of this file:** a complete, self-contained briefing for an AI agent
that has never seen this project, so it can understand the system and propose a
better roadmap. Everything here is measured from the code and dataset as of
2026-07-21, not aspirational. Where a claim is an opinion or unverified, it is
marked *(assumption)*.

The current roadmap lives in `ROADMAP.md`. **You are being asked to critique and
improve it.** Read this whole file first; the open questions are in §10.

---

## 1. What the project is

A gaze tracker that estimates where a person is looking on their laptop screen
using **only a standard 720p webcam** — no infrared, no depth sensor, no
head-mounted hardware. Built incrementally as numbered "milestone" scripts.

**Hardware it was built on:** Lenovo Legion R7000 ARP8, Ryzen 7 7735H, RTX 4060
(unused — inference is CPU), Realtek 1280×720 webcam, 15.6" 2560×1440 display
(34.4 × 19.4 cm). Windows 11, Python venv at `gaze_env/`.

**Context:** a student/KPITB project. Not production. The user (Arsi) is
learning as it is built. Priorities in practice: accuracy, and a demonstrable,
explainable pipeline.

---

## 2. The core pipeline (how one gaze estimate is made)

```
webcam frame 1280x720, mirrored
   │
   ▼
MediaPipe FaceLandmarker (VIDEO mode)  →  478 landmarks + 4x4 head transform matrix
   │
   ├──► positioning gate: is the user 45-65 cm away and centred?  (gate, not estimator)
   │
   ├──► square face crop (padded 30%, EMA-smoothed) → L2CS-Net ONNX → gaze_pitch, gaze_yaw
   │
   └──► gaze_features.extract_features() → 32 numbers describing head + eyes
   │
   ▼
per-person calibration: polynomial maps (gaze_pitch, gaze_yaw) → screen (x, y) in [0,1]²
```

**Two models are involved and must not be confused:**

1. **L2CS-Net** — a pretrained ResNet-50 CNN (trained on Gaze360), run via ONNX
   Runtime on CPU. Takes the face image, outputs raw gaze **angles**. Frozen; we
   do not train it. Outputs two 90-bin softmax distributions → expectation over
   bins → continuous angle. **Gotcha:** the ONNX file mislabels its outputs;
   tensor[0] is yaw despite being named `pitch_bins`. Handled in
   `gaze_pipeline._predict_gaze_radians()`.

2. **The calibration model** — a tiny polynomial (`sklearn`) fitted **per person**
   that maps gaze angles to a specific screen. This is the only thing "trained"
   in normal use, and it is the subject of the roadmap.

---

## 3. File inventory (16 Python files, ~3,800 lines)

### Libraries (shared logic)
| File | Lines | Role |
|---|---|---|
| `gaze_pipeline.py` | 151 | Face detect, square smoothed crop, ONNX gaze inference. `detect_face()`, `predict_gaze_from_crop()`, `get_gaze_reading()`. |
| `gaze_features.py` | 179 | Extracts all 32 measured features from one frame. `extract_features()`, `iris_ratio()`, `eye_aspect_ratio()`, `head_pose_from_matrix()`. |
| `positioning_gate.py` | 241 | Distance (pinhole/IPD) + centring constraint, parking-sensor overlay. `PositioningGate.evaluate()`. |
| `calibration_utils.py` | 340 | Polynomial fit, robust MAD refit, model + dataset I/O, schema migration, data-dictionary generation. |

### Data tooling
| File | Role |
|---|---|
| `inspect_dataset.py` | Plain-language dataset summary (rows, sources, sessions, coverage grid). |
| `analyze_dataset.py` | **The analysis layer.** Feature-signal correlations, redundancy, session-drift measurement, and a session-held-out model comparison that scores 5 candidate feature sets. Prints a recommendation. |
| `export_to_onnx.py` | One-time PyTorch→ONNX conversion of L2CS weights. |

### Milestones (runnable apps/demos)
| File | Writes data? | Role |
|---|---|---|
| `milestone1_face_mesh.py` | no | Landmark visualisation. |
| `milestone2_eye_headpose.py` | no | Live feature inspector — draws all 32 features + the landmarks they come from. |
| `milestone3_gaze_model.py` | no | Raw uncalibrated gaze arrow. |
| `milestone4_calibration.py` | **yes** (`milestone4_pursuit`) | Calibration: positioning gate → 86 s smooth-pursuit sweep → 5-dot held-out validation. Also the de-facto shared runtime module (exports `FrameReader`, `phase_a_positioning`, `measure_drift`, `CANVAS_W/H`). |
| `milestone5_positioning.py` | no | Positioning-gate demo; `c` key measures focal length at 50 cm. |
| `milestone6_test_accuracy.py` | no | Accuracy test on 9 fresh dots, reports cm error + interpretation. The final evaluation tool. |
| `milestone7_gaze_game.py` | **yes** (`milestone7_game`) | Look-to-pop target game. Collects fixation data, aim assist, auto-retrains after each game. Defines `GazeCursor`. |
| `milestone8_live_cursor.py` | no | Live gaze cursor; `m` drives the real OS mouse. |
| `milestone9_snake_game.py` | no (by choice) | Gaze-steered, blink-to-click Snake. Demo only. |

### Docs
`README.md`, `ROADMAP.md` (the thing to improve), `TECHNICAL_DOCUMENTATION.md`
(exhaustive), `SETUP_INSTRUCTIONS.md`, plus HTML/PDF explainers
(`cheatsheet.html`, `dataset_documentation.html`, `milestone9_documentation.html`,
`technical_documentation.html`).

---

## 4. The dataset

`data/gaze_samples.csv`, schema v3, **39 columns**, currently **6,373 rows**.

**One row = one video frame where the true look-point was known** (the program
drew the dot and told the user to look at it). Supervised pair: 32 measured
features (X) → `target_x, target_y` (Y, the on-screen dot in [0,1]).

**No pixels are stored** — only the numbers computed per frame. This blocks any
CNN fine-tuning without a pipeline change (see §7).

### The 32 features (the model's possible inputs)
| Group | Columns | How derived |
|---|---|---|
| gaze | `gaze_pitch`, `gaze_yaw` | L2CS-Net (the only NN-derived features) |
| head rotation | `head_pitch/yaw/roll` | Euler decomposition of MediaPipe's 4×4 matrix |
| head translation | `head_tx/ty/tz` | translation column of that matrix (units ≠ cm; tz always negative, OpenGL −Z convention) |
| iris raw | `iris_{left,right}_{x,y}` | landmarks 468/473, normalized |
| iris ratio | `iris_*_ratio_{x,y}` | iris projected onto the eye-corner axis, normalized by eye width — **head-position invariant** |
| eye corners | `eye_{l,r}_{inner,outer}_{x,y}` | landmarks 33/133/362/263 |
| openness | `ear_left`, `ear_right` | eye aspect ratio, 6-point formula |
| face size | `face_width`, `face_height`, `ipd_px` | landmark spans; ipd is iris-to-iris pixels |
| position | `pos_distance_cm`, `pos_face_dx/dy` | pinhole distance from ipd_px; nose offset from frame centre |

Label = `target_x, target_y`. Metadata = `source, user, session_id, timestamp,
schema_version`.

### Dataset composition (measured today)
```
rows       6,373
sources    4,984 pursuit + 1,389 game
people     3  (arsi 2,631 · default 2,382 · asif 1,360)   ← 37% untagged "default"
sessions   13, all inside a single 14-hour window (20-21 Jul)   ← low time diversity
coverage   even across a 3x3 screen grid (thinnest cell 445 rows)
```

---

## 5. What the analysis layer already found (`analyze_dataset.py`)

These are **measured**, session-held-out (train on 10 sessions, test on the
newest 3, no drift correction):

**Feature signal (|correlation| with label):**
- `iris_left_ratio_x` → target_x = **0.942** — *higher than* `gaze_yaw` (0.866).
  The geometric feature beats the neural net on the horizontal axis.
- `gaze_pitch` → target_y = 0.860; iris ratios ~0.81 on vertical. NN wins vertical.
- Conclusion: gaze angles and iris ratios are **complementary**.

**Redundancy:** 54 feature pairs correlate > 0.95. `head_tx ≈ pos_face_dx ≈
eye_*_x`, `head_ty ≈ iris_*_y ≈ eye_*_y`. Head-translation / raw-iris / eye-corner
/ gate-position all encode "where the face sits."

**Session drift:** per-session gaze bias spread 0.067 rad (pitch), 0.069 rad
(yaw); full range ~13–14°. Posture varies a lot between sittings.

**Model comparison (session-held-out cm error):**
| Feature set | Params | Error |
|---|---|---|
| 2 (current, deg-3) | 10 | 4.43 cm |
| **6 (+iris ratios, deg-2 ridge)** | 28 | **3.00 cm** ← best |
| 9 (+head pose) | 55 | 3.28 cm |
| 12 (+position) | 91 | 9.85 cm ← worse |
| 32 (everything, deg-2) | 561 | 10.95 cm ← worse |

**Key finding that overruled intuition:** adding position/head-translation
features *hurt* generalisation, because those values are near-constant within a
session and differ between sessions — the model uses them as session
fingerprints. The roadmap's Stage-1 pick (6 features, ridge, deg-2) came from
this measurement, not theory.

**Two different accuracy numbers, both true:**
- Same-session, drift-corrected (milestone 6, live UX): **2.0–2.4 cm**.
- New-session, no drift correction (held-out, honest generalisation): **3.00 cm** best.

---

## 6. Key design decisions and why (so you don't re-litigate them)

- **Positioning gate is a hard constraint, not a feature.** Out-of-zone frames
  are discarded, never saved. This keeps the calibration valid but limits data
  diversity to one distance band (45–65 cm).
- **Smooth-pursuit sweep + perimeter pass** for calibration coverage. A plain
  snake left corners untrained (extrapolation error up to 6.2 cm); a perimeter
  loop fixed the worst corner to 0.7 cm.
- **Degree-3 polynomial** was chosen because looking down (eyelid occlusion) is a
  nonlinearity deg-2 couldn't fit. But the analysis now favours deg-2 + more
  features + ridge over deg-3 + fewer features.
- **Drift correction**: a one-dot recentre at session start subtracts constant
  posture offset. Works, but is a patch for the model not seeing posture.
- **Data integrity in the game:** labels come from *raw* gaze during a verified
  fixation (stability-based, not closeness-based), never from the aim-assisted
  cursor — otherwise the assist would manufacture its own ground truth.
- **Session-held-out evaluation** is mandated everywhere; random splits leak
  because consecutive frames are near-identical.

---

## 7. Honest limitations / blockers

1. **Per-person only.** A model trained on one person doesn't transfer. Only 3
   people in the data, one-third untagged.
2. **Low session diversity.** 13 sittings, all in 14 hours. Lighting/posture
   variety is thin — held-out numbers may be optimistic *(assumption: more
   spread-out data would raise the honest error before improvements lower it)*.
3. **Single distance band.** The gate only ever admits 45–65 cm, so the model
   never sees near/far and can't generalise outside it.
4. **No pixels stored.** Fine-tuning L2CS (the only path to a genuinely "large"
   model) needs the 448×448 crops saved (~30–60 KB/frame, ~1.5–3 GB/50k). Not
   currently collected.
5. **CPU-bound.** 8–15 FPS; the CNN dominates. Limits sample rate and adds
   cursor lag.
6. **The model ignores 30 of 32 collected features.** Everything past
   gaze_pitch/yaw is written but unused at fit time. Stage 1 begins fixing this.
7. **`head_t*` units are not cm** and are collinear with several other columns;
   naive inclusion destabilises the fit (measured).

---

## 8. The roadmap as it currently stands (summary of `ROADMAP.md`)

A 4-layer loop: **analyse → decide → train → verify**, repeated as data grows.
Scaling ladder by parameter count (rule: 10–20 rows/param):

| Stage | Model | Params | Rows needed | Status |
|---|---|---|---|---|
| 0 | poly-3, 2 feat | 10 | ~200 | current, superseded |
| 1 | ridge poly-2, 6 feat | 28 | ~600 | **ready now**, predicted 3.00 cm |
| 1b | + per-session posture normalisation | ~60 | ~1,200 | ready, experiment |
| 2 | boosting / small MLP | ~4,000 | 40,000+ | need ~6× more rows |
| 3 | person-independent + embedding | ~10,000 | 100,000+, 20 people | need people |
| 4 | fine-tuned L2CS CNN | ~11 M | 100,000 images | blocked (no pixels) |

Immediate recommended action in the roadmap: **build Stage 1** (code task, data
already exists), compare honestly, keep collecting spread out, always set
`GAZE_USER`.

---

## 9. Things a fresh reviewer often suggests — and the project's current stance

- *"Just use a bigger neural net."* → Blocked without stored face crops (§7.4);
  and per-person data volume (6k rows) is far below what an MLP needs (§8).
- *"Add all the features."* → Measured to hurt (§5). Redundancy + session
  fingerprinting.
- *"Use head pose to fix drift."* → Plausible but raw inclusion made it worse;
  the untested idea is per-session normalisation (Stage 1b).
- *"Random train/test split."* → Rejected; leaks via near-identical adjacent
  frames.
- *"Collect more data."* → Only helps up to the current model's ceiling; the
  binding constraint right now is model/features, not row count.

---

## 10. Open questions for you (the reviewing AI)

The roadmap is defensible but built by the same people who built the system, so
it may share their blind spots. Specifically, please pressure-test:

1. **Is the staged polynomial→MLP→CNN ladder the right spine at all?** Would a
   different model family (e.g. per-user Gaussian Process, k-NN in feature space,
   a small shared MLP with per-user calibration head) beat it at *this* data
   scale, given the measured feature correlations in §5?
2. **Is per-session normalisation (Stage 1b) the correct fix for drift**, or is
   there a cleaner formulation (e.g. treating session as a random effect /
   mixed-effects model, or an explicit recentre baked into features)?
3. **Should the single-distance-band constraint be relaxed** to collect
   multi-distance data, accepting harder learning for real generalisation?
4. **Is chasing sub-cm accuracy (Stage 4) ever worth the pixel-storage cost**
   for the likely applications (dwell-click UI, attention/region tracking)?
5. **What is the highest-value next data to collect** — more people, more
   sessions per person, more distances, or stored crops — given the goal is a
   model that works for a new user on a new day?
6. **Is there a better evaluation protocol** than session-held-out cm error for
   this problem?

Assume you can propose changes to feature engineering, model choice, data
collection strategy, and evaluation — but respect the hard constraints:
webcam-only, CPU inference, and (for now) no stored images unless you argue
convincingly that Stage 4 justifies starting to store them.

### Fast orientation commands (if the reviewer can run code)
```powershell
python analyze_dataset.py     # the measured basis for all §5 numbers
python inspect_dataset.py     # dataset composition
# key source: gaze_features.py, calibration_utils.py, analyze_dataset.py, milestone4_calibration.py
```
