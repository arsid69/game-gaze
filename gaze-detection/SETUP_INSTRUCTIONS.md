# Setup Instructions — Gaze Detection

Follow these steps to get milestones 1–4 running on your machine.

## 1. Environment

```bash
python -m venv gaze_env
gaze_env\Scripts\activate          # Windows
# source gaze_env/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

If `torch` install is slow or fails, use the CPU-only build directly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 2. Get the L2CS-Net package

```bash
pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main
```

## 3. Download the face landmark model

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile "face_landmarker.task"
```

## 4. Get the gaze model weights (two options)

**Option A (recommended, faster):** ask your teammate for `models/l2cs_gaze360.onnx` directly (already converted) — just drop it in a `models/` folder and skip to step 5.

**Option B (from scratch):**
```powershell
mkdir models
Invoke-WebRequest -Uri "https://huggingface.co/dorni/SpeakerVid-5M-data-curation-models/resolve/5c6e04d7fa3321e6228e79162f8ec98466bf308a/L2CSNet_gaze360.pkl" -OutFile "models\L2CSNet_gaze360.pkl"
python export_to_onnx.py
```

## 5. Files you need in your project folder

```
gaze_detection/
├── face_landmarker.task
├── models/
│   └── l2cs_gaze360.onnx
├── gaze_pipeline.py
├── calibration_utils.py
├── milestone1_face_mesh.py
├── milestone2_eye_headpose.py
├── milestone3_gaze_model.py
├── milestone4_calibration.py
└── requirements.txt
```

## 6. Run the milestones, in order

```bash
python milestone1_face_mesh.py      # face + iris landmarks
python milestone2_eye_headpose.py   # eye crops + head pose
python milestone3_gaze_model.py     # gaze direction arrow
python milestone4_calibration.py    # 9-point calibration
```

Each opens a webcam window; press `q` to quit. Milestone 4 walks you through
looking at 9 dots on screen and saves a per-person calibration model to
`models/calibration_model.pkl`.
