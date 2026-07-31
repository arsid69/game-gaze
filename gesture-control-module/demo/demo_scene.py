"""Runnable demo: select and move 3D objects with your hand.

Renders a small 3D scene with a tiny software wireframe renderer drawn over
the webcam feed, so the only dependencies are the ones the module already
needs (OpenCV, NumPy, MediaPipe) -- no game engine to install.

Run it::

    python demo/demo_scene.py

Controls
--------
Point (index finger out)   move the cursor; objects light up when aimed at
Pinch (thumb + index)      grab the object under the cursor and drag it
Release the pinch          drop it
Open palm                  clear the selection
Swipe left / right         cycle the selection
r / ESC                    reset the scene / quit
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_control import (Camera, EngineConfig, EventType, GestureEngine,
                             Scene, SceneObject, build_tracker, ndc_to_pixel)
from gesture_control.features import HAND_CONNECTIONS, pinch_ratio

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("This demo needs opencv-python:  pip install -r requirements.txt")


WIDTH, HEIGHT = 1280, 720
WINDOW_NAME = "gesture control demo"

HOVER_COLOR = (255, 220, 60)
SELECT_COLOR = (60, 240, 255)
GRID_COLOR = (60, 60, 60)


# --------------------------------------------------------------------------
# Scene setup
# --------------------------------------------------------------------------
def build_scene() -> Scene:
    palette = [
        ("crate_a", (-3.0, 0.8, 9.0), (120, 200, 255)),
        ("crate_b", (0.0, 1.2, 11.0), (140, 255, 170)),
        ("crate_c", (3.2, 0.6, 9.5), (255, 170, 140)),
        ("crate_d", (-1.8, -1.2, 7.0), (220, 160, 255)),
        ("crate_e", (2.0, -1.4, 12.5), (255, 240, 150)),
    ]
    scene = Scene()
    for obj_id, pos, color in palette:
        scene.add(SceneObject(
            id=obj_id,
            position=np.array(pos, dtype=float),
            half_extents=np.array([0.7, 0.7, 0.7]),
            color=color,
        ))
    return scene


# --------------------------------------------------------------------------
# Minimal wireframe renderer
# --------------------------------------------------------------------------
def project(camera: Camera, point, width: int, height: int):
    """World point -> pixel coordinates, or None when behind the camera."""
    ndc = camera.world_to_ndc(point)
    if ndc is None:
        return None
    x, y, depth = ndc
    px, py = ndc_to_pixel(x, y, width, height)
    return px, py, depth


CUBE_EDGES = ((0, 1), (1, 3), (3, 2), (2, 0),
              (4, 5), (5, 7), (7, 6), (6, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def cube_corners(center: np.ndarray, half: np.ndarray) -> list[np.ndarray]:
    signs = [(-1, -1, -1), (1, -1, -1), (-1, 1, -1), (1, 1, -1),
             (-1, -1, 1), (1, -1, 1), (-1, 1, 1), (1, 1, 1)]
    return [center + half * np.array(s, dtype=float) for s in signs]


def draw_grid(image, camera: Camera) -> None:
    """A floor grid: without it there is no sense of depth at all."""
    h, w = image.shape[:2]
    y = -2.5
    for i in range(-6, 7):
        for a, b in (((i, y, 2.0), (i, y, 18.0)),
                     ((-6.0, y, i + 8.0), (6.0, y, i + 8.0))):
            pa, pb = project(camera, a, w, h), project(camera, b, w, h)
            if pa and pb:
                cv2.line(image, pa[:2], pb[:2], GRID_COLOR, 1, cv2.LINE_AA)


def draw_object(image, camera: Camera, obj: SceneObject,
                hovered: bool, selected: bool, grabbed: bool) -> None:
    h, w = image.shape[:2]
    half = obj.half_extents if obj.half_extents is not None \
        else np.array([obj.radius] * 3)
    projected = [project(camera, c, w, h) for c in cube_corners(obj.position, half)]
    if any(p is None for p in projected):
        return

    color = obj.color
    thickness = 2
    if selected:
        color, thickness = SELECT_COLOR, 3
    if hovered:
        color, thickness = HOVER_COLOR, 3
    if grabbed:
        thickness = 4

    for a, b in CUBE_EDGES:
        cv2.line(image, projected[a][:2], projected[b][:2], color,
                 thickness, cv2.LINE_AA)

    center = project(camera, obj.position, w, h)
    if center:
        label = f"{obj.id}  {center[2]:.1f}m"
        cv2.putText(image, label, (center[0] - 40, center[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_scene(image, camera: Camera, scene: Scene, hover_id, grab_id) -> None:
    draw_grid(image, camera)
    # Painter's algorithm: furthest first, so near objects draw on top.
    def depth(o: SceneObject) -> float:
        return float(np.dot(o.position - camera.position, camera.forward))

    for obj in sorted(scene.objects, key=depth, reverse=True):
        draw_object(image, camera, obj,
                    hovered=obj.id == hover_id,
                    selected=scene.is_selected(obj.id),
                    grabbed=obj.id == grab_id)


def draw_hand(image, landmarks) -> None:
    h, w = image.shape[:2]
    pts = [(int(p[0] * w), int(p[1] * h)) for p in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], (0, 255, 120), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(image, p, 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_cursor(image, ndc, gesture: str) -> None:
    h, w = image.shape[:2]
    x, y = ndc_to_pixel(ndc[0], ndc[1], w, h)
    color = SELECT_COLOR if gesture == "pinch" else (255, 255, 255)
    cv2.circle(image, (x, y), 14, color, 2, cv2.LINE_AA)
    cv2.circle(image, (x, y), 2, color, -1, cv2.LINE_AA)
    cv2.line(image, (x - 22, y), (x - 16, y), color, 1)
    cv2.line(image, (x + 16, y), (x + 22, y), color, 1)


def draw_hud(image, lines) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 26 + 22 * len(lines)),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (14, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (235, 235, 235), 1, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Window
# --------------------------------------------------------------------------
def desktop_work_area(fallback=(1920, 1040)):
    """Screen size minus the taskbar, so the window can open maximised."""
    try:
        import ctypes
        from ctypes import wintypes
        rect = wintypes.RECT()
        # SPI_GETWORKAREA = 0x0030
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0,
                                                      ctypes.byref(rect), 0):
            return rect.right - rect.left, rect.bottom - rect.top
    except Exception:      # not Windows, or the call is unavailable
        pass
    return fallback


def open_window() -> None:
    """A resizable window filling the desktop. OpenCV scales the frame to fit."""
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *desktop_work_area())
    cv2.moveWindow(WINDOW_NAME, 0, 0)


def toggle_fullscreen() -> None:
    prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_NORMAL if prop == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main() -> int:
    config = EngineConfig()
    config.tracker.frame_width = WIDTH
    config.tracker.frame_height = HEIGHT

    camera = Camera(position=np.array([0.0, 0.0, 0.0]),
                    forward=np.array([0.0, 0.0, 1.0]),
                    fov_y_deg=60.0,
                    aspect=WIDTH / HEIGHT)

    scene = build_scene()
    engine = GestureEngine(scene=scene, camera=camera, config=config)

    log: list[str] = []

    @engine.on()
    def record(event):
        """Keep the last few interesting events for the on-screen log."""
        if event.type in (EventType.POINTER_MOVE, EventType.GRAB_MOVE):
            return
        target = f" -> {event.object_id}" if event.object_id else ""
        extra = f" {event.data.get('direction', '')}".rstrip()
        log.append(f"{event.type.value}{target}{extra}")
        del log[:-6]

    try:
        tracker = build_tracker(config.tracker)
    except Exception as exc:
        print(f"Could not start the camera: {exc}")
        return 1

    print(__doc__)
    open_window()
    fps, last = 0.0, time.monotonic()

    try:
        while True:
            frame = tracker.read()
            if frame is None:
                break
            engine.update(frame)

            image = frame.image
            image = cv2.addWeighted(image, 0.35,
                                    np.zeros_like(image), 0.65, 0)  # dim the feed

            state = engine.primary
            draw_scene(image, camera, scene,
                       hover_id=state.hover_id if state else None,
                       grab_id=state.grab_id if state else None)

            if frame.hands:
                draw_hand(image, frame.hands[0].landmarks)
            if state is not None:
                draw_cursor(image, state.ndc, state.gesture)

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            # Show the raw pinch distance and the two thresholds: if grabbing
            # misbehaves, watching this number cross them explains why.
            pinch = "-"
            if frame.hands:
                ratio = pinch_ratio(frame.hands[0].landmarks)
                close = config.gestures.pinch_close
                op = config.gestures.pinch_open
                pinch = f"{ratio:.2f} (grab<{close:.2f} release>{op:.2f})"

            hud = [
                f"fps {fps:4.1f}   gesture: {state.gesture if state else '-'}"
                f"   selected: {', '.join(scene.selected) or '-'}",
                f"pinch {pinch}"
                + (f"   holding: {state.grab_id}" if state and state.grab_id else ""),
                "point = aim | pinch = grab & drag | open palm = clear | "
                "swipe = cycle | r = reset | f = fullscreen | ESC = quit",
            ] + log[-3:]
            draw_hud(image, hud)

            cv2.imshow(WINDOW_NAME, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("f"):
                toggle_fullscreen()
            if key == ord("r"):
                scene = build_scene()
                engine.scene = scene
                log.clear()
    finally:
        tracker.close()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
