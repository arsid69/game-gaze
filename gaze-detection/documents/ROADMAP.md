# Roadmap — data analysis first, then the model

**Updated:** 2026-07-21 · every number below is measured from
`data/gaze_samples.csv` by `analyze_dataset.py`, not estimated.

---

## 0. The principle

> **Analyse the data, then pick the model. Never the other way round.**

The previous version of this roadmap recommended a 12-feature model on
theoretical grounds. When that was actually measured against held-out sessions
it scored **9.85 cm** — more than three times worse than a 6-feature model.
The data overruled the theory. That is why the analysis layer exists.

```
LAYER 1  analyse   ->  what signal is in the data, what is redundant,
                       how much does it drift, how much is there
       │
LAYER 2  decide    ->  candidate models scored on held-out SESSIONS
       │
LAYER 3  train     ->  build only the winner
       │
LAYER 4  verify    ->  milestone6 on freshly collected dots
       │
       └─ repeat as the dataset grows
```

Run the analysis any time with:

```powershell
python analyze_dataset.py
```

---

## 1. Layer 1 — what the analysis says today

```
rows 6,373  ·  sessions 13  ·  people 3 (arsi 2,631 · default 2,382 · asif 1,360)
sources: 4,984 pursuit + 1,389 game  ·  all rows have all 32 features
```

### 1a. Which features actually carry signal

| Feature | vs `target_x` | vs `target_y` |
|---|---|---|
| `iris_left_ratio_x` | **0.942** | 0.033 |
| `iris_right_ratio_x` | **0.938** | 0.118 |
| `gaze_yaw` | 0.866 | 0.033 |
| `gaze_pitch` | 0.060 | **0.860** |
| `iris_right_ratio_y` | 0.219 | 0.814 |
| `iris_left_ratio_y` | 0.100 | 0.809 |
| `iris_right_x` (raw) | 0.453 | 0.121 |
| `head_yaw` | 0.413 | 0.151 |
| `ear_left` | 0.048 | 0.392 |

**The surprise:** `iris_left_ratio_x` correlates **0.942** with horizontal gaze
position — *better than the neural network's own* `gaze_yaw` at 0.866. The
simple geometric feature beats the deep model on the horizontal axis. Vertically
the network wins (`gaze_pitch` 0.860 vs iris ratio 0.814).

**Conclusion:** the two families are complementary, and both belong in the model.
Neither alone is best.

### 1b. Redundancy — 54 duplicated pairs

```
head_tx  ~ eye_left_inner_x    r = +0.981
head_tx  ~ pos_face_dx         r = +0.980
head_ty  ~ eye_left_outer_y    r = −0.990
head_ty  ~ iris_left_y         r = −0.986
… 50 more pairs above |r| > 0.95
```

Head translation, raw iris pixels, eye-corner coordinates and the gate's
position values are all **measuring the same thing** — where the face sits in
frame. Including all of them spends parameters without adding information, and
the collinearity actively destabilises the fit.

### 1c. Session drift is large and real

```
per-session bias spread   pitch 0.067 rad   yaw 0.069 rad
full range across 13      pitch 0.228 rad   yaw 0.248 rad   (≈13–14°)
```

Posture varies substantially between sittings. This is the single biggest
obstacle to a model that works on a new day.

### 1d. Screen coverage — healthy

```
   852   587   781
   445   583   514
  1041   787   783        thinnest cell 445 rows, no empty cells
```

---

## 2. Layer 2 — model options, scored

All candidates trained on 10 sessions and tested on the **3 newest sessions**,
which the model has never seen. No drift correction applied — this is the
honest "sit down on a new day" number.

| Option | Features | Degree | Params | **Held-out error** | Verdict |
|---|---|---|---|---|---|
| **A** current | 2 | 3 | 10 | 4.43 cm | baseline |
| A′ | 2 | 2 | 6 | 5.01 cm | worse |
| **B** + iris ratios | 6 | 2 + ridge | 28 | **3.00 cm** | ✅ **best** |
| B′ | 6 | 3 | 84 | 3.03 cm | tied, more params |
| **C** + head pose | 9 | 2 + ridge | 55 | 3.28 cm | worse than B |
| C′ | 9 | 3 | 220 | 4.30 cm | worse |
| **D** + position | 12 | 2 + ridge | 91 | 9.85 cm | ❌ much worse |
| D′ | 12 | 3 | 455 | 19.48 cm | ❌ collapses |
| **E** everything | 32 | 2 + ridge | 561 | 10.95 cm | ❌ too few rows |

### The winner

```python
features = [gaze_pitch, gaze_yaw,
            iris_left_ratio_x,  iris_left_ratio_y,
            iris_right_ratio_x, iris_right_ratio_y]
model    = StandardScaler → PolynomialFeatures(degree=2) → Ridge(alpha=1.0)
```

**3.00 cm vs 4.43 cm — a 32% improvement**, using 28 parameters against 6,373
rows. Comfortably supported by the data.

### Why more features made it worse

Options D and E look like they should win and don't. Two measured reasons:

1. **Collinearity** (§1b) — `head_tx`, `pos_face_dx` and the eye-corner columns
   are near-copies. Ridge dampens this but cannot fix a rank-deficient design.
2. **Session-specific posture** — distance and face offset are almost constant
   *within* a session and different *between* sessions. The model latches onto
   them as session fingerprints, which is exactly what a held-out session
   punishes. They memorise the sitting rather than the gaze.

> Position features are not useless — they are being used wrongly. The right way
> is per-session normalisation (subtract each session's mean posture) rather than
> feeding raw values. That is an experiment for Stage 2, not a reason to bolt
> them on now.

### Reading the two different error numbers

| Question | Method | Result |
|---|---|---|
| "How good is it right now, this sitting?" | milestone 6, drift-corrected | **2.0–2.4 cm** |
| "How good on a new day, no drift check?" | session-held-out, above | **3.00 cm** (best) |

Both are true. The first is the user experience today; the second is the honest
generalisation number and the one to optimise.

---

## 3. Layer 3 — the scaling ladder

Rule of thumb: **10–20 rows per parameter**.

| Stage | Model | Params | Rows needed | Have | Status |
|---|---|---|---|---|---|
| 0 | Poly-3, 2 features | 10 | ~200 | 6,373 | current, superseded |
| **1** | **Ridge poly-2, 6 features** | **28** | **~600** | **6,373** | ✅ **do this now** |
| 1b | + per-session posture normalisation | ~60 | ~1,200 | 6,373 | ✅ ready, experiment |
| 2 | Gradient boosting / small MLP | ~4,000 | 40,000+ | 6,373 | need ~6× more rows |
| 3 | Person-independent + embedding | ~10,000 | 100,000+, 20 people | 3 people | need people |
| 4 | Fine-tuned L2CS CNN | ~11 M | 100,000 **images** | 0 images | blocked — §5 |

---

## 4. Layer 4 — verification rules

1. **Hold out whole sessions, never random rows.** Consecutive frames are
   near-identical; a random split leaks and every model looks excellent.
2. **Keep the same held-out sessions** between experiments, or improvements
   cannot be compared.
3. **Report per screen region** — a good average hides a bad corner.
4. `milestone6_test_accuracy.py` is the final word: dots collected fresh, never
   seen in any form.

```python
from calibration_utils import load_dataset_full
rows = load_dataset_full()
sessions = sorted({r["session_id"] for r in rows})
test = set(sessions[-3:])                       # newest 3 sittings
train_rows = [r for r in rows if r["session_id"] not in test]
test_rows  = [r for r in rows if r["session_id"] in test]
```

---

## 5. The Stage 4 blocker

Fine-tuning L2CS-Net — a genuinely *large* model — is **impossible with this
dataset**, and no number of extra rows changes that.

> **The CSV has no pixels.** It stores 32 numbers per frame: the vision model's
> *output*, not its *input*. A CNN needs the face image.

Unlocking Stage 4 requires additionally saving the 448×448 face crop per row:

```
data/gaze_samples.csv          39 columns + crop_path
data/crops/<session_id>/000123.jpg
```

Cost ≈ 30–60 KB per frame → **1.5–3 GB per 50,000 frames**.

**Recommendation:** do not start saving crops until Stage 2 has plateaued.
Premature image collection burns disk and time for a stage the project may never
need — Stages 1–3 deliver most of the practical accuracy.

---

## 6. What to do next, in order

1. **Build the Stage 1 model** — 6 features, scaler → poly-2 → Ridge. The data
   already supports it; this is a code task, not a collection task.
   Expected: **4.43 → 3.00 cm** on a new session.
2. **Try 1b** — per-session posture normalisation, to rescue the position
   features that currently hurt.
3. **Re-run `analyze_dataset.py`** after every few hundred new rows. The
   recommendation is recomputed from the data, so it will tell you when the
   answer changes.
4. **Keep collecting, spread out.** All 13 sessions so far fall inside a single
   14-hour window — posture and lighting diversity is thin. One sitting a day
   beats ten in an afternoon.
5. **Fix the `default` rows.** 2,382 rows (37%) are untagged and effectively
   anonymous. Always `$env:GAZE_USER = "yourname"` before running.
6. **Recruit people** only if person-independence is a project goal. Three is
   not enough; twenty is the target.

---

## 7. Commands

```powershell
$env:GAZE_USER = "arsi"

python analyze_dataset.py            # LAYER 1+2: what the data supports
python milestone4_calibration.py     # ~500 rows, full screen coverage
python milestone7_gaze_game.py       # ~150 rows, retrains automatically
python milestone6_test_accuracy.py   # LAYER 4: honest measurement
python inspect_dataset.py            # plain-language inventory
```
