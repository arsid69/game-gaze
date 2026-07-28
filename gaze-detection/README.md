# Real-Time Eye Gaze Detection

Webcam-based gaze tracking with a plain 720p laptop camera — no extra sensors.
Face/iris landmarks → head pose → pretrained gaze model → positioning
constraints (distance + centering, car-reverse-sensor style) → per-person
calibration → measured accuracy, a game, and a live gaze cursor.

## Documentation

| Document | Contents |
|---|---|
| **`TECHNICAL_DOCUMENTATION.md`** | Full technical reference — architecture, algorithms and math (distance estimation, calibration, drift correction, aim assist), data formats, API reference, configuration, limitations, troubleshooting |
| **`ROADMAP.md`** | Why the dataset is valid training data, how much data is needed, and the phased plan to higher accuracy |
| **`SETUP_INSTRUCTIONS.md`** | Installation and dependencies |

## Layout

```text
milestone1_face_mesh.py      # MediaPipe face + iris landmark detection
milestone2_eye_headpose.py   # Eye crops + head pose from the facial transformation matrix
milestone3_gaze_model.py     # Pretrained L2CS-Net gaze model (ONNX) -> raw gaze arrow
milestone4_calibration.py    # Gated smooth-pursuit calibration + held-out validation
milestone5_positioning.py    # Positioning gate demo: distance levels + centering ('c' = focal calibration)
milestone6_test_accuracy.py  # Accuracy test -> per-point cm errors + interpreted report
milestone7_gaze_game.py      # Gamified test: pop 10 targets with your gaze
milestone8_live_cursor.py    # Free gaze cursor; 'm' drives the real Windows mouse

positioning_gate.py          # Distance (IPD pinhole) + centering constraints and overlay
gaze_pipeline.py             # Shared face detect + square smoothed crop + ONNX gaze inference
calibration_utils.py         # Fit / save / load / apply the calibration polynomial
export_to_onnx.py            # One-time PyTorch -> ONNX conversion for the gaze model
models/                      # ONNX model, calibration_model.pkl, camera_focal.json
face_landmarker.task         # MediaPipe face landmark model file
```

## Setup

See `SETUP_INSTRUCTIONS.md`.

## Workflow

```bash
# From project root, with gaze_env activated
python milestone1_face_mesh.py      # sanity check: face + iris landmarks
python milestone2_eye_headpose.py   # sanity check: eye crops + pitch/yaw/roll
python milestone3_gaze_model.py     # sanity check: raw gaze arrow
python milestone5_positioning.py    # sit right; press 'c' at 50 cm once for exact focal
python milestone4_calibration.py    # calibrate -> models/calibration_model.pkl
python milestone6_test_accuracy.py  # measure real accuracy in cm
python milestone7_gaze_game.py      # play the target game
python milestone8_live_cursor.py    # live cursor; 'm' = control the real mouse
```

Press `q` in any window to quit.

## Important

- The gaze model's raw output order is `(yaw_bins, pitch_bins)` — handled in
  `gaze_pipeline.py`; don't "fix" it back.
- Calibration is **per person, per laptop, per sitting position**. The
  positioning gate (45–65 cm, face centered) enforces the same zone at
  calibration time and at use time.
- Run all scripts at the camera's native 1280x720 (already set in code);
  640x480 crops the sensor and skews the distance estimate.
