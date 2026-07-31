"""Custom gesture recorder.

Record your own static gestures from the webcam and they become available to
the engine immediately -- no code changes, no retraining step.

Usage::

    python -m gesture_control.recorder record --name thumbs_down --samples 60
    python -m gesture_control.recorder list
    python -m gesture_control.recorder test
    python -m gesture_control.recorder delete --name thumbs_down

During ``record``: hold the pose, press SPACE to start capturing, ESC/q to
quit. Move your hand slightly while capturing -- varied samples generalise far
better than 60 copies of the same frame.
"""

from __future__ import annotations

import argparse
import sys
import time

from .config import GestureConfig, TrackerConfig
from .features import describe
from .gestures import (GestureDataset, KNNGestureClassifier, RuleClassifier,
                       CompositeClassifier)
from .tracker import MediaPipeHandTracker


def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError:  # pragma: no cover
        sys.exit("opencv-python is required: pip install -r requirements.txt")


def _draw_skeleton(cv2, image, landmarks, color=(0, 255, 120)):
    from .features import HAND_CONNECTIONS
    h, w = image.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(image, p, 3, (255, 255, 255), -1)


def _banner(cv2, image, lines, color=(255, 255, 255)):
    for i, line in enumerate(lines):
        y = 26 + i * 24
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color, 1, cv2.LINE_AA)


def cmd_record(args) -> int:
    cv2 = _require_cv2()
    config = GestureConfig(dataset_path=args.dataset)
    dataset = GestureDataset.load(config.dataset_path)

    captured = 0
    capturing = False
    last_capture = 0.0
    interval = 1.0 / max(args.rate, 1)

    with MediaPipeHandTracker(TrackerConfig(camera_index=args.camera)) as tracker:
        while captured < args.samples:
            frame = tracker.read()
            if frame is None:
                break
            image = frame.image
            hand = frame.hands[0] if frame.hands else None

            if hand is not None:
                _draw_skeleton(cv2, image, hand.landmarks,
                               (0, 200, 255) if capturing else (0, 255, 120))

            state = "RECORDING" if capturing else "paused"
            _banner(cv2, image, [
                f"gesture: {args.name}",
                f"{state}  {captured}/{args.samples}",
                "SPACE = start/stop, ESC = quit",
            ], (0, 200, 255) if capturing else (255, 255, 255))

            now = time.monotonic()
            if capturing and hand is not None and now - last_capture >= interval:
                dataset.add(args.name, hand.landmarks)
                captured += 1
                last_capture = now

            cv2.imshow("gesture recorder", image)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord(" "):
                capturing = not capturing

    cv2.destroyAllWindows()

    if captured == 0:
        print("No samples captured, nothing saved.")
        return 1

    dataset.save(config.dataset_path)
    print(f"Saved {captured} samples for '{args.name}' -> {config.dataset_path}")
    print("Dataset now holds:", dataset.labels())
    return 0


def cmd_list(args) -> int:
    dataset = GestureDataset.load(args.dataset)
    if not len(dataset):
        print(f"No custom gestures recorded yet ({args.dataset} is empty/missing).")
        return 0
    print(f"{args.dataset}: {len(dataset)} samples")
    for label, count in sorted(dataset.labels().items()):
        marker = "ok " if count >= 20 else "low"
        print(f"  [{marker}] {label:<20} {count} samples")
    if any(c < 20 for c in dataset.labels().values()):
        print("\nLabels marked 'low' have few samples; 30-60 works much better.")
    return 0


def cmd_delete(args) -> int:
    dataset = GestureDataset.load(args.dataset)
    removed = dataset.remove_label(args.name)
    dataset.save(args.dataset)
    print(f"Removed {removed} samples for '{args.name}'.")
    return 0


def cmd_test(args) -> int:
    """Live view of what the classifier currently thinks the hand is doing."""
    cv2 = _require_cv2()
    config = GestureConfig(dataset_path=args.dataset)
    dataset = GestureDataset.load(config.dataset_path)
    custom = KNNGestureClassifier(dataset, config) if len(dataset) else None
    classifier = CompositeClassifier(RuleClassifier(config), custom, config)

    with MediaPipeHandTracker(TrackerConfig(camera_index=args.camera)) as tracker:
        while True:
            frame = tracker.read()
            if frame is None:
                break
            image = frame.image
            lines = ["no hand detected"]
            if frame.hands:
                hand = frame.hands[0]
                _draw_skeleton(cv2, image, hand.landmarks)
                result = classifier.classify(hand.landmarks, hand.handedness)
                info = describe(hand.landmarks)
                fingers = "".join("1" if v else "0"
                                  for v in info["extended"].values())
                lines = [
                    f"{hand.handedness} hand",
                    f"{result.label}  ({result.confidence:.2f}, {result.source})",
                    f"fingers TIMRP={fingers}  pinch={info['pinch_ratio']}",
                ]
            _banner(cv2, image, lines)
            cv2.imshow("classifier test", image)
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                break

    cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gesture_control.recorder",
        description="Record and inspect custom hand gestures.")
    # Shared options, attached to every subcommand so they can be typed in
    # either order (`recorder --camera 1 list` and `recorder list --camera 1`).
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default=GestureConfig().dataset_path,
                        help="path to the gesture dataset JSON")
    common.add_argument("--camera", type=int, default=0, help="camera index")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", parents=[common],
                         help="record samples for a new gesture")
    rec.add_argument("--name", required=True, help="gesture label")
    rec.add_argument("--samples", type=int, default=50)
    rec.add_argument("--rate", type=float, default=10.0,
                     help="samples captured per second")
    rec.set_defaults(func=cmd_record)

    lst = sub.add_parser("list", parents=[common], help="show recorded gestures")
    lst.set_defaults(func=cmd_list)

    dele = sub.add_parser("delete", parents=[common], help="remove a gesture")
    dele.add_argument("--name", required=True)
    dele.set_defaults(func=cmd_delete)

    tst = sub.add_parser("test", parents=[common],
                         help="live classifier preview")
    tst.set_defaults(func=cmd_test)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
