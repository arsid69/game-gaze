"""
analyze_dataset.py — the data analysis layer.

Answers "what does our data actually support?" before anyone picks a model.
Everything here is measured from data/gaze_samples.csv; nothing is assumed.

Run: python analyze_dataset.py
"""

import collections
import math
import sys

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from calibration_utils import load_dataset_full
from gaze_features import FEATURE_COLUMNS

# Candidate feature sets, smallest first.
FEATURE_SETS = {
    "A  current (2)": ["gaze_pitch", "gaze_yaw"],
    "B  + iris ratios (6)": [
        "gaze_pitch", "gaze_yaw",
        "iris_left_ratio_x", "iris_left_ratio_y",
        "iris_right_ratio_x", "iris_right_ratio_y"],
    "C  + head pose (9)": [
        "gaze_pitch", "gaze_yaw",
        "iris_left_ratio_x", "iris_left_ratio_y",
        "iris_right_ratio_x", "iris_right_ratio_y",
        "head_pitch", "head_yaw", "head_roll"],
    "D  + position (12)": [
        "gaze_pitch", "gaze_yaw",
        "iris_left_ratio_x", "iris_left_ratio_y",
        "iris_right_ratio_x", "iris_right_ratio_y",
        "head_pitch", "head_yaw", "head_roll",
        "pos_distance_cm", "pos_face_dx", "pos_face_dy"],
    "E  everything (32)": list(FEATURE_COLUMNS),
}

SCREEN_W_CM, SCREEN_H_CM = 34.4, 19.4


def rows_to_arrays(rows, cols):
    """Returns (X, Y, sessions) for rows where every requested column parses."""
    X, Y, S = [], [], []
    for r in rows:
        try:
            X.append([float(r[c]) for c in cols])
            Y.append([float(r["target_x"]), float(r["target_y"])])
            S.append(r["session_id"])
        except (KeyError, TypeError, ValueError):
            continue
    return np.array(X), np.array(Y), np.array(S)


def cm_error(pred, true):
    d = (pred - true) * np.array([SCREEN_W_CM, SCREEN_H_CM])
    return np.hypot(d[:, 0], d[:, 1])


def session_holdout_error(X, Y, sessions, degree, alpha, n_test=3):
    """Train on all but the newest n_test sessions, test on those.

    Whole sessions are held out, never random rows: consecutive frames are
    nearly identical, so a random split leaks and flatters every model.
    """
    uniq = sorted(set(sessions))
    if len(uniq) <= n_test:
        return None, None, 0
    test_ids = set(uniq[-n_test:])
    tr = np.array([s not in test_ids for s in sessions])
    te = ~tr
    if tr.sum() < 50 or te.sum() < 20:
        return None, None, 0

    poly = PolynomialFeatures(degree=degree, include_bias=True)
    scaler = StandardScaler()
    Xtr = poly.fit_transform(scaler.fit_transform(X[tr]))
    Xte = poly.transform(scaler.transform(X[te]))

    model = Ridge(alpha=alpha) if alpha else LinearRegression()
    preds = np.column_stack([
        model.fit(Xtr, Y[tr][:, i]).predict(Xte) for i in (0, 1)])
    preds = np.clip(preds, 0.0, 1.0)
    err = cm_error(preds, Y[te])
    return float(np.mean(err)), Xtr.shape[1], int(tr.sum())


def main():
    rows = load_dataset_full()
    if not rows:
        print("No data. Run milestone4_calibration.py first.")
        return 1

    print("=" * 68)
    print("  DATA ANALYSIS LAYER")
    print("=" * 68)

    # ---------- 1. inventory ----------
    users = collections.Counter(r["user"] for r in rows)
    sess = collections.Counter(r["session_id"] for r in rows)
    src = collections.Counter(r["source"] for r in rows)
    print(f"\n[1] INVENTORY")
    print(f"    rows      {len(rows)}")
    print(f"    sessions  {len(sess)}")
    print(f"    people    {len(users)}  ->  {dict(users)}")
    print(f"    sources   {dict(src)}")

    # ---------- 2. per-feature signal ----------
    print(f"\n[2] FEATURE SIGNAL  (|correlation| with the label)")
    X_all, Y_all, S_all = rows_to_arrays(rows, FEATURE_COLUMNS)
    if len(X_all) < 50:
        print("    not enough complete rows to analyse")
        return 1
    scored = []
    for i, name in enumerate(FEATURE_COLUMNS):
        col = X_all[:, i]
        if np.std(col) < 1e-9:
            scored.append((name, 0.0, 0.0))
            continue
        cx = abs(np.corrcoef(col, Y_all[:, 0])[0, 1])
        cy = abs(np.corrcoef(col, Y_all[:, 1])[0, 1])
        scored.append((name, cx, cy))
    scored.sort(key=lambda s: -max(s[1], s[2]))
    print(f"    {'feature':24s} {'vs target_x':>11s} {'vs target_y':>11s}")
    for name, cx, cy in scored[:12]:
        print(f"    {name:24s} {cx:11.3f} {cy:11.3f}")
    weak = [n for n, cx, cy in scored if max(cx, cy) < 0.05]
    print(f"    ... {len(scored) - 12} more")
    print(f"    near-zero signal on both axes: {len(weak)} features")

    # ---------- 3. redundancy ----------
    print(f"\n[3] REDUNDANCY  (feature pairs correlated > 0.95)")
    C = np.corrcoef(X_all.T)
    dupes = []
    for i in range(len(FEATURE_COLUMNS)):
        for j in range(i + 1, len(FEATURE_COLUMNS)):
            if not np.isnan(C[i, j]) and abs(C[i, j]) > 0.95:
                dupes.append((FEATURE_COLUMNS[i], FEATURE_COLUMNS[j], C[i, j]))
    for a, b, c in dupes[:10]:
        print(f"    {a:22s} ~ {b:22s} r={c:+.3f}")
    if not dupes:
        print("    none — no feature is a straight copy of another")
    elif len(dupes) > 10:
        print(f"    ... and {len(dupes) - 10} more pairs")
    print(f"    => {len(dupes)} redundant pairs: adding both wastes parameters")

    # ---------- 4. session drift ----------
    print(f"\n[4] SESSION DRIFT  (does posture shift between sittings?)")
    means = {}
    for r in rows:
        try:
            means.setdefault(r["session_id"], []).append(
                [float(r["gaze_pitch"]), float(r["gaze_yaw"]),
                 float(r["target_x"]), float(r["target_y"])])
        except (KeyError, TypeError, ValueError):
            continue
    offs = []
    for sid, vals in means.items():
        v = np.array(vals)
        if len(v) < 30:
            continue
        # average gaze angle minus average target: a per-session bias proxy
        offs.append((sid, v[:, 0].mean() - v[:, 2].mean(),
                     v[:, 1].mean() - v[:, 3].mean(), len(v)))
    if len(offs) >= 2:
        px = np.array([o[1] for o in offs])
        py = np.array([o[2] for o in offs])
        print(f"    sessions compared: {len(offs)}")
        print(f"    per-session bias spread:  pitch {px.std():.3f} rad"
              f"   yaw {py.std():.3f} rad")
        print(f"    range:                    pitch {px.max()-px.min():.3f}"
              f"   yaw {py.max()-py.min():.3f}")
        drifty = (px.std() > 0.05 or py.std() > 0.05)
        print(f"    => posture varies {'A LOT' if drifty else 'mildly'} between"
              f" sittings {'-> position features will help' if drifty else ''}")

    # ---------- 5. coverage ----------
    print(f"\n[5] SCREEN COVERAGE")
    grid = collections.Counter()
    for y in Y_all:
        grid[(min(int(y[0] * 3), 2), min(int(y[1] * 3), 2))] += 1
    for gy in range(3):
        print("    " + "".join(f"{grid.get((gx, gy), 0):>8}" for gx in range(3)))
    empty = [c for c in [(x, y) for y in range(3) for x in range(3)]
             if grid.get(c, 0) == 0]
    thin = min(grid.values()) if grid else 0
    print(f"    thinnest cell: {thin} rows"
          + ("  WARNING: empty cells " + str(empty) if empty else ""))

    # ---------- 6. model comparison ----------
    print(f"\n[6] MODEL COMPARISON  (session-held-out, newest 3 sittings as test)")
    print(f"    {'feature set':24s} {'deg':>4s} {'params':>7s} {'train':>7s} {'error cm':>9s}")
    results = []
    for label, cols in FEATURE_SETS.items():
        X, Y, S = rows_to_arrays(rows, cols)
        if len(X) < 100:
            print(f"    {label:24s}  (not enough complete rows)")
            continue
        for degree, alpha in ((3, 0.0), (2, 1.0)):
            if len(cols) > 12 and degree == 3:
                continue                    # would explode past the row count
            err, nparams, ntrain = session_holdout_error(X, Y, S, degree, alpha)
            if err is None:
                continue
            tag = f"deg{degree}" + ("+ridge" if alpha else "")
            enough = "" if ntrain >= 10 * nparams else "  <- too few rows"
            print(f"    {label:24s} {degree:>4d} {nparams:>7d} {ntrain:>7d}"
                  f" {err:>9.2f}{enough}")
            results.append((err, label, degree, alpha, nparams, ntrain))

    # ---------- 7. recommendation ----------
    print(f"\n[7] RECOMMENDATION")
    if results:
        results.sort()
        best_err, best_label, best_deg, best_alpha, best_p, best_n = results[0]
        base = [r for r in results if r[1].startswith("A")]
        base_err = min(b[0] for b in base) if base else None
        print(f"    best measured: {best_label}  degree {best_deg}"
              f"{' + ridge' if best_alpha else ''}"
              f"  ->  {best_err:.2f} cm")
        if base_err:
            delta = base_err - best_err
            pct = 100 * delta / base_err
            print(f"    current model: {base_err:.2f} cm")
            if delta > 0.1:
                print(f"    => SWITCH: {delta:.2f} cm better ({pct:.0f}% improvement)")
            else:
                print(f"    => KEEP current model; no candidate beat it meaningfully")
    n_rows, n_sess, n_users = len(rows), len(sess), len(users)
    print(f"\n    data supports:")
    print(f"      Stage 1 (Ridge, ~180 params)      "
          f"{'YES' if n_rows >= 2000 else 'no  (need 2,000 rows)'}")
    print(f"      Stage 2 (MLP/boosting, ~4k params) "
          f"{'YES' if n_rows >= 40000 else f'no  (need 40,000; have {n_rows})'}")
    print(f"      Stage 3 (person-independent)       "
          f"{'YES' if n_users >= 20 else f'no  (need 20 people; have {n_users})'}")
    print(f"      Stage 4 (fine-tune CNN)            no  (needs saved face crops, not just numbers)")
    if n_sess < 10:
        print(f"\n    NOTE: only {n_sess} sessions - collect across more days"
              f" before trusting held-out numbers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
