"""
inspect_dataset.py — plain-English summary of data/gaze_samples.csv.

Answers "what is this data?" without anyone having to read the code.

Run: python inspect_dataset.py
"""

import collections
import sys

from calibration_utils import load_dataset_full, DATASET_PATH, DICT_PATH
from gaze_features import FEATURE_COLUMNS

SOURCE_EXPLAIN = {
    "milestone4_pursuit": "milestone4_calibration.py - following the moving dot",
    "milestone7_game":    "milestone7_gaze_game.py   - staring at game targets",
    "pursuit":            "(old label) calibration sweep",
    "game":               "(old label) game fixations",
}


def bar(count, total, width=28):
    filled = int(width * count / total) if total else 0
    return "#" * filled + "." * (width - filled)


def main():
    rows = load_dataset_full()
    if not rows:
        print(f"No data yet - {DATASET_PATH} is empty or missing.")
        print("Run: python milestone4_calibration.py")
        return

    print("=" * 62)
    print(f"  {DATASET_PATH}  -  {len(rows)} rows")
    print("=" * 62)
    print("\nOne row = one video frame where we knew exactly where the")
    print("person was looking. Features = what the webcam saw.")
    print("Label (target_x, target_y) = where the dot actually was.\n")

    print("-- WHERE IT CAME FROM " + "-" * 40)
    for src, n in collections.Counter(r["source"] for r in rows).most_common():
        print(f"  {n:6d}  {bar(n, len(rows))}  {src}")
        print(f"          {SOURCE_EXPLAIN.get(src, 'unknown source')}")

    print("\n-- WHO " + "-" * 55)
    for user, n in collections.Counter(r["user"] for r in rows).most_common():
        note = "  <- set GAZE_USER to tag this properly" if user == "default" else ""
        print(f"  {n:6d}  {user}{note}")

    sessions = collections.Counter(r["session_id"] for r in rows)
    print(f"\n-- SITTINGS ({len(sessions)} separate runs) " + "-" * 30)
    for sid, n in sorted(sessions.items()):
        print(f"  {n:6d} rows   session {sid}")
    if len(sessions) < 3:
        print("  NOTE: fewer than 3 sittings - collect on different days so the")
        print("        model learns to handle posture changes.")

    print("\n-- FEATURE COVERAGE " + "-" * 42)
    complete = sum(1 for r in rows if r.get("ear_left"))
    print(f"  {complete:6d} rows have all 32 features (new format)")
    print(f"  {len(rows) - complete:6d} rows have only the old columns")

    print("\n-- SCREEN COVERAGE " + "-" * 43)
    grid = collections.Counter()
    for r in rows:
        try:
            gx = min(int(float(r["target_x"]) * 3), 2)
            gy = min(int(float(r["target_y"]) * 3), 2)
            grid[(gx, gy)] += 1
        except (ValueError, TypeError, KeyError):
            continue
    print("  how many samples per screen area:")
    for gy in range(3):
        cells = "".join(f"{grid.get((gx, gy), 0):>8}" for gx in range(3))
        print(f"    {cells}")
    empty = [f"({gx},{gy})" for gy in range(3) for gx in range(3)
             if grid.get((gx, gy), 0) == 0]
    if empty:
        print(f"  WARNING: no data for screen areas {', '.join(empty)}")

    times = sorted(r["timestamp"] for r in rows if r.get("timestamp"))
    if times:
        print(f"\n-- WHEN " + "-" * 54)
        print(f"  first: {times[0]}")
        print(f"  last:  {times[-1]}")

    print(f"\nFull column explanations: {DICT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
