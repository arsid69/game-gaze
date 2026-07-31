"""How to wire the gesture engine into your own game loop.

Two integration styles are shown:

1. ``polling_example``  - drain events once per tick (fits most game engines)
2. ``callback_example`` - react immediately via subscribed handlers

Neither needs the demo renderer; both work with any engine that has an update
loop and objects with positions.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_control import (Camera, EventType, GestureEngine, Scene,
                             SceneObject)


def make_world() -> tuple[Scene, Camera]:
    scene = Scene([
        SceneObject("sword", position=np.array([-2.0, 0.0, 8.0]), radius=0.8),
        SceneObject("shield", position=np.array([0.0, 0.0, 9.0]), radius=0.8),
        SceneObject("potion", position=np.array([2.0, 0.0, 8.0]), radius=0.6),
        # A wall you can aim at but never pick up.
        SceneObject("wall", position=np.array([0.0, 0.0, 16.0]),
                    half_extents=np.array([8.0, 4.0, 0.2]), grabbable=False),
    ])
    camera = Camera(position=np.array([0.0, 0.0, 0.0]),
                    forward=np.array([0.0, 0.0, 1.0]),
                    fov_y_deg=60.0, aspect=16 / 9)
    return scene, camera


def polling_example(seconds: float = 30.0) -> None:
    """Camera runs on a background thread; the game reads events each tick."""
    scene, camera = make_world()
    engine = GestureEngine(scene=scene, camera=camera)
    engine.start()                       # spawns the tracking thread

    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            # ---- your game tick starts here ----
            for event in engine.poll():
                if event.type is EventType.HOVER_ENTER:
                    print(f"aiming at {event.object_id}")
                elif event.type is EventType.SELECT:
                    print(f"selected {event.object_id}")
                elif event.type is EventType.GRAB_BEGIN:
                    print(f"picked up {event.object_id}")
                elif event.type is EventType.GRAB_END:
                    obj = scene.get(event.object_id)
                    where = np.round(obj.position, 2) if obj else "?"
                    print(f"dropped {event.object_id} at {where}")
                elif event.type is EventType.SWIPE:
                    print(f"swipe {event.data['direction']}")
                elif event.type is EventType.CLEAR_SELECTION:
                    print("selection cleared")
            # ---- render your frame here ----
            time.sleep(1 / 60)
    finally:
        engine.stop()


def callback_example(seconds: float = 30.0) -> None:
    """Handlers fire the moment an event happens."""
    scene, camera = make_world()
    engine = GestureEngine(scene=scene, camera=camera)

    @engine.on(EventType.SELECT)
    def on_select(event):
        print(f"[select] {event.object_id}")

    @engine.on(EventType.GRAB_MOVE)
    def on_drag(event):
        # Called every frame while dragging - keep this cheap.
        obj = scene.get(event.object_id)
        if obj is not None:
            obj.position[1] = max(obj.position[1], 0.0)   # clamp above the floor

    @engine.on(EventType.GESTURE_BEGIN)
    def on_gesture(event):
        # Custom gestures recorded with the recorder arrive here by name.
        if event.gesture not in ("point", "pinch", "open_palm"):
            print(f"[gesture] {event.gesture} ({event.data.get('source')})")

    engine.start()
    try:
        time.sleep(seconds)
    finally:
        engine.stop()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "polling"
    if mode == "callback":
        callback_example()
    else:
        polling_example()
