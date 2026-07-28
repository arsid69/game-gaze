"""
calibration_utils.py — shared functions for fitting and applying the
gaze (pitch, yaw) -> normalized screen (x, y) correction model.

Used by milestone4_calibration.py to create and save the model, and by
apply_calibration() to turn a live gaze reading into a screen position.
"""

import csv
import os
import pickle
from datetime import datetime

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from gaze_features import FEATURE_COLUMNS

CALIBRATION_PATH = "models/calibration_model.pkl"
DATASET_PATH = "data/gaze_samples.csv"


def fit_calibration(samples, degree=2):
    """
    samples: list of (pitch, yaw, x_norm, y_norm) tuples collected during
             the calibration routine.
    degree: polynomial degree for the feature expansion (2 = quadratic).
    Returns a dict containing the fitted poly feature transformer and the
    two regressors (one for x, one for y).
    """
    data = np.array(samples)
    gaze = data[:, 0:2]        # pitch, yaw
    targets_x = data[:, 2]     # normalized screen x
    targets_y = data[:, 3]     # normalized screen y

    poly = PolynomialFeatures(degree=degree)
    gaze_poly = poly.fit_transform(gaze)

    reg_x = LinearRegression().fit(gaze_poly, targets_x)
    reg_y = LinearRegression().fit(gaze_poly, targets_y)

    model = {"poly": poly, "reg_x": reg_x, "reg_y": reg_y, "degree": degree}

    # Training-fit error (rough sanity check, not a true held-out validation —
    # good enough for now, revisit properly in Milestone 6)
    pred_x = reg_x.predict(gaze_poly)
    pred_y = reg_y.predict(gaze_poly)
    err_x = np.abs(pred_x - targets_x)
    err_y = np.abs(pred_y - targets_y)
    mean_err = np.mean(np.sqrt(err_x**2 + err_y**2))
    model["fit_error_normalized"] = float(mean_err)

    return model


def apply_calibration(model, pitch, yaw):
    """Returns (x_norm, y_norm) predicted screen position for a given gaze reading."""
    gaze_poly = model["poly"].transform([[pitch, yaw]])
    x_norm = float(model["reg_x"].predict(gaze_poly)[0])
    y_norm = float(model["reg_y"].predict(gaze_poly)[0])
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))
    return x_norm, y_norm


# --- Dataset schema (v3) ---------------------------------------------------
# Column groups, in write order:
#   MEASURED  ("user data")  — everything observed from the webcam
#   LABEL     ("actual data")— the on-screen target the user was looking at
#   META      — provenance: who, which session, from what activity, when
#
# The label columns are named target_* so ground truth is never confused with
# a prediction. Nothing in this file ever stores a model prediction.
# Column order is chosen for a human opening the CSV in Excel:
#   who/what this row is  ->  THE ANSWER  ->  what the camera measured  ->  when
ID_COLUMNS = ["source", "user", "session_id"]
LABEL_COLUMNS = ["target_x", "target_y"]          # the fixed dot position
TRAIL_COLUMNS = ["timestamp", "schema_version"]
META_COLUMNS = ID_COLUMNS + TRAIL_COLUMNS
DATASET_HEADER = ID_COLUMNS + LABEL_COLUMNS + FEATURE_COLUMNS + TRAIL_COLUMNS

SCHEMA_VERSION = 3

# Older layouts, kept so existing data can be migrated rather than discarded.
_HEADER_V1 = ["pitch", "yaw", "x", "y", "source", "timestamp"]
_HEADER_V2 = ["pitch", "yaw", "x", "y",
              "distance_cm", "face_dx", "face_dy", "source", "timestamp"]

# Where a legacy column lands in the v3 schema.
_LEGACY_MAP = {"pitch": "gaze_pitch", "yaw": "gaze_yaw",
               "x": "target_x", "y": "target_y",
               "distance_cm": "pos_distance_cm",
               "face_dx": "pos_face_dx", "face_dy": "pos_face_dy",
               "source": "source", "timestamp": "timestamp"}

CURRENT_USER = os.environ.get("GAZE_USER", "default")
SESSION_ID = datetime.now().strftime("%Y%m%d-%H%M%S")

# Source labels say which script and which activity produced the row, so the
# CSV is readable without having to ask anyone.
SOURCE_PURSUIT = "milestone4_pursuit"   # calibration sweep, following the moving dot
SOURCE_GAME = "milestone7_game"         # game, fixating on a target
_SOURCE_RENAMES = {"pursuit": SOURCE_PURSUIT, "game": SOURCE_GAME}

DICT_PATH = "data/README.md"


def _migrate_dataset(path):
    """Upgrade a v1/v2 CSV to the v3 schema in place.

    Legacy rows keep their gaze angles and labels (still valid training
    data); the columns that did not exist then are left blank, and
    schema_version records which layout each row came from.
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return

    old_header = rows[0]

    if set(old_header) == set(DATASET_HEADER):
        # Same columns — may just need reordering and/or source relabelling.
        needs_reorder = old_header != DATASET_HEADER
        needs_relabel = any(_SOURCE_RENAMES.get(dict(zip(old_header, r)).get("source"))
                            for r in rows[1:] if len(r) == len(old_header))
        if needs_reorder or needs_relabel:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(DATASET_HEADER)
                for r in rows[1:]:
                    if len(r) != len(old_header):
                        continue
                    d = dict(zip(old_header, r))
                    d["source"] = _SOURCE_RENAMES.get(d.get("source"), d.get("source"))
                    writer.writerow([d.get(c, "") for c in DATASET_HEADER])
        return
    if old_header == _HEADER_V1:
        version = 1
    elif old_header == _HEADER_V2:
        version = 2
    else:
        return  # unknown layout: leave untouched rather than corrupt it

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(DATASET_HEADER)
        for r in rows[1:]:
            if len(r) != len(old_header):
                continue
            old = dict(zip(old_header, r))
            new = {col: "" for col in DATASET_HEADER}
            for old_col, value in old.items():
                mapped = _LEGACY_MAP.get(old_col)
                if mapped:
                    new[mapped] = value
            new["user"] = "legacy"
            new["session_id"] = old.get("timestamp", "")[:13]  # hour granularity
            new["source"] = _SOURCE_RENAMES.get(new["source"], new["source"])
            new["schema_version"] = version
            writer.writerow([new[col] for col in DATASET_HEADER])


_DATA_DICTIONARY = """# What is this data?

Auto-generated by `calibration_utils.py`. Do not edit by hand.

`gaze_samples.csv` is the **training dataset** for the webcam gaze tracker.
One row = one video frame where we knew exactly where the person was looking.

## How a row is created

A dot is drawn on screen and the person looks at it. We record what the
webcam saw (the features) and where the dot was (the answer). That pairing
is what a model learns from.

| `source` value | Which script | What the person was doing |
|---|---|---|
| `milestone4_pursuit` | `milestone4_calibration.py` | Following a dot moving across the screen |
| `milestone7_game` | `milestone7_gaze_game.py` | Staring at a game target to pop it |

Only these two write data. Other scripts (the accuracy test, the live cursor,
the snake demo) read the calibration model but never add rows.

## Column groups

### 1. MEASURED — what the webcam saw (32 columns, the model's INPUT)

| Columns | Meaning |
|---|---|
| `gaze_pitch`, `gaze_yaw` | Eye direction from the L2CS neural network, radians. Strongest signal. |
| `head_pitch/yaw/roll` | Head rotation, degrees. |
| `head_tx/ty/tz` | Head position in space; `tz` is roughly distance from camera. |
| `iris_left_x/y`, `iris_right_x/y` | Where each iris sits in the camera image (0-1). |
| `iris_*_ratio_x/y` | Where each iris sits **inside its own eye** (0 = outer corner, 1 = inner). Best eye-direction feature. |
| `eye_*_inner/outer_x/y` | Eye corner positions (0-1). |
| `ear_left`, `ear_right` | Eye openness. ~0.3 = open, ~0.1 = blinking. |
| `face_width`, `face_height` | Face size in the image (0-1). |
| `ipd_px` | Pixels between the two pupils. Bigger = closer to the camera. |
| `pos_distance_cm` | Estimated distance from the screen, centimetres. |
| `pos_face_dx`, `pos_face_dy` | How far off-centre the face is (0 = centred). |

### 2. LABEL — the correct answer (2 columns, the model's OUTPUT)

| Column | Meaning |
|---|---|
| `target_x`, `target_y` | Where the dot was on screen. `0,0` = top-left, `1,1` = bottom-right. |

These are **ground truth**, not predictions. The program drew the dot, so it
knows the position exactly. Nothing in this file is ever a model guess.

### 3. META — where the row came from (5 columns)

| Column | Meaning |
|---|---|
| `source` | Which script/activity produced it (table above). |
| `user` | Who was sitting there. Set with `GAZE_USER` before running. |
| `session_id` | One ID per program run, `YYYYMMDD-HHMMSS`. Rows from one sitting share it. |
| `timestamp` | When the batch was written. |
| `schema_version` | Column layout version. `3` = current, `1`/`2` = older rows with blanks. |

## Why `session_id` matters

People sit slightly differently each time. The same eyes can point at the
same angle but be looking at a different spot if the head moved. Grouping
rows by session lets a model account for that instead of averaging it away.

## Quick look

```bash
python inspect_dataset.py
```
"""


def write_data_dictionary(path=DICT_PATH):
    """Write the plain-English explanation of the CSV next to it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(_DATA_DICTIONARY)


def append_dataset(samples, source, path=DATASET_PATH, user=None,
                   session_id=None):
    """Append samples to the growing dataset.

    Each sample is a dict: the feature dict from
    gaze_features.extract_features() plus "target_x" and "target_y".

    MEASURED features come from the webcam; target_x/target_y are the
    on-screen stimulus the user was looking at — genuine ground truth, since
    this program drew it. No model prediction is ever stored here.

    source: "pursuit" = calibration sweep, "game" = fixation on a game target.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        _migrate_dataset(path)
    new_file = not os.path.exists(path)
    if path == DATASET_PATH:
        write_data_dictionary()   # keep the explanation beside the data
    ts = datetime.now().isoformat(timespec="seconds")
    meta = {"source": source,
            "user": user or CURRENT_USER,
            "session_id": session_id or SESSION_ID,
            "timestamp": ts,
            "schema_version": SCHEMA_VERSION}

    def fmt(value):
        if value is None:
            return ""
        if isinstance(value, float):
            return f"{value:.6f}"
        return value

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(DATASET_HEADER)
        for s in samples:
            row = dict(s)
            row.update(meta)
            writer.writerow([fmt(row.get(col)) for col in DATASET_HEADER])


def load_dataset(path=DATASET_PATH):
    """Returns (samples, source_counts) for the current 2-feature fit:
    (gaze_pitch, gaze_yaw, target_x, target_y) tuples. Reads both the v3
    schema and any not-yet-migrated legacy column names."""
    samples, counts = [], {}
    if not os.path.exists(path):
        return samples, counts
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                samples.append((
                    float(row.get("gaze_pitch") or row["pitch"]),
                    float(row.get("gaze_yaw") or row["yaw"]),
                    float(row.get("target_x") or row["x"]),
                    float(row.get("target_y") or row["y"])))
            except (KeyError, TypeError, ValueError):
                continue
            counts[row.get("source", "?")] = counts.get(row.get("source", "?"), 0) + 1
    return samples, counts


def load_dataset_full(path=DATASET_PATH):
    """Returns every row as a dict (all features + labels + metadata),
    for training richer models or analysing the dataset."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def robust_fit_samples(samples, degree=3, mad_factor=2.5, min_keep=60):
    """Fit, drop residual outliers (blinks/glances) via MAD, refit.
    Returns (model, n_dropped)."""
    model = fit_calibration(samples, degree=degree)
    data = np.array(samples)
    preds = np.array([apply_calibration(model, p, yw) for p, yw in data[:, 0:2]])
    residuals = np.linalg.norm(preds - data[:, 2:4], axis=1)
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med)) + 1e-9
    keep = residuals <= med + mad_factor * 1.4826 * mad
    n_dropped = int((~keep).sum())
    if n_dropped and keep.sum() >= min_keep:
        model = fit_calibration([tuple(s) for s in data[keep]], degree=degree)
    return model, n_dropped


def save_calibration(model, path=CALIBRATION_PATH):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_calibration(path=CALIBRATION_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)
