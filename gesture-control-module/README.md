# Gesture Detection Module for 3D Games

A Python module that turns webcam hand gestures into game-level interaction
events: point at an object to highlight it, pinch to select and drag it,
open your palm to clear the selection, swipe to cycle through objects.

Built on **MediaPipe Hands** (21 3D landmarks per hand) + **OpenCV** + **NumPy**.
It ships as a library with a clean event API, plus a runnable 3D demo so you can
see it working before wiring it into an engine.

---

## Install and run

```bash
pip install -r requirements.txt

python demo/demo_scene.py            # the interactive 3D demo
python tests/test_gesture_control.py # 25 tests, no camera needed
```

Python 3.9-3.12, 64-bit. MediaPipe publishes no wheels for 3.13+.

The first run downloads Google's hand landmark model (~7 MB) into `models/`. That
needs an internet connection once; afterwards it works offline. If the download is
blocked, fetch [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)
in a browser and save it to `models/hand_landmarker.task` yourself.

### Which MediaPipe API?

Google removed the old `mp.solutions` API in MediaPipe releases after 0.10.9.
`tracker.py` supports both: it uses the current **Tasks API** where available and
falls back to the legacy one on older installs. Force a choice with
`config.tracker.backend = "tasks"` or `"legacy"` if you ever need to.

---

## The demo

`demo/demo_scene.py` renders five wireframe crates in 3D over a dimmed webcam
feed, using a tiny software renderer so no game engine is required.

| Gesture | Action |
|---|---|
| Point (index finger out) | Move the cursor; crates highlight when aimed at |
| Pinch (thumb + index) | Select the highlighted crate and pick it up |
| Pinch and move | Drag it around at a constant distance from the camera |
| Release the pinch | Drop it |
| Open palm | Clear the selection |
| Swipe left / right | Cycle the selection through the scene |
| `r` / `Esc` | Reset the scene / quit |

---

## Using it in your own game

Define your scene, then read events once per tick:

```python
from gesture_control import GestureEngine, Scene, SceneObject, Camera, EventType

scene = Scene([
    SceneObject("sword",  position=(-2, 0, 8), radius=0.8),
    SceneObject("shield", position=( 0, 0, 9), radius=0.8),
    SceneObject("wall",   position=( 0, 0, 16),
                half_extents=(8, 4, 0.2), grabbable=False),
])
camera = Camera(position=(0, 0, 0), forward=(0, 0, 1), fov_y_deg=60, aspect=16/9)

engine = GestureEngine(scene=scene, camera=camera)
engine.start()                       # camera + tracking on a background thread

while running:                       # your game loop
    for event in engine.poll():
        if event.type is EventType.SELECT:
            print("selected", event.object_id)
        elif event.type is EventType.GRAB_MOVE:
            sync_to_engine(event.object_id, event.position)

engine.stop()
```

Or subscribe to callbacks instead of polling:

```python
@engine.on(EventType.HOVER_ENTER)
def highlight(event):
    ...
```

Both styles are shown end to end in `examples/basic_usage.py`.

### Events

| Event | Fired when |
|---|---|
| `HAND_DETECTED` / `HAND_LOST` | A hand enters or leaves the frame |
| `GESTURE_BEGIN` / `GESTURE_END` | A stable gesture starts or stops — including your custom ones |
| `POINTER_MOVE` | Every frame, with the smoothed cursor position |
| `HOVER_ENTER` / `HOVER_EXIT` | The ray starts or stops touching an object |
| `SELECT` / `DESELECT` / `CLEAR_SELECTION` | Selection changes |
| `GRAB_BEGIN` / `GRAB_MOVE` / `GRAB_END` | Drag lifecycle |
| `SWIPE` | A fast directional flick (`event.data["direction"]`) |

Every event carries `timestamp`, `hand` (`"Right"`/`"Left"`), `gesture`,
`object_id`, `position` (world space) and `ndc` (cursor in `[-1, 1]`).

Objects the ray can hit are `SceneObject`s with either a `radius` (sphere) or
`half_extents` (box). Set `selectable=False` to make something invisible to the
ray, or `grabbable=False` to allow selecting but not dragging. Hang your own
engine object off `payload`.

---

## Recording your own gestures

The built-in gestures are hand-written rules. Anything else you want, you record:

```bash
python -m gesture_control.recorder record --name thumbs_down --samples 60
python -m gesture_control.recorder list
python -m gesture_control.recorder test      # live view of what it recognises
python -m gesture_control.recorder delete --name thumbs_down
```

Hold the pose, press `Space` to start capturing, and **move your hand around
slightly while recording** — varied samples generalise far better than 60 copies
of one frame. 30-60 samples per gesture is plenty.

Samples land in `gestures_dataset.json` and the engine picks them up on the next
run; they arrive as `GESTURE_BEGIN` events under the name you recorded. No
retraining step, no code changes.

---

## How it works

```
webcam ─► MediaPipe Hands ─► normalise ─► classify ─► stabilise ─┐
                                                                 │
                        One Euro smoothing ◄── fingertip ────────┤
                                 │                               │
                                 ▼                               ▼
                         camera ray ─► raycast ─► hover/select/drag ─► events
```

**Normalisation** (`features.py`) is what makes recognition work at all. Raw
landmarks depend on where your hand is, how far away it is and how it is turned.
Each hand is re-expressed in a coordinate frame built from the palm itself —
origin at the wrist, one axis up towards the middle knuckle, one across the
knuckles — and divided by hand size. What comes out describes only the *pose*.
The test suite verifies this is invariant to translation, scale and rotation.

**Classification** (`gestures.py`) runs two classifiers. Geometric rules handle
the built-ins from finger-bend angles and the thumb/index distance; a
distance-weighted k-NN handles anything you recorded, and reports `unknown`
rather than guessing when nothing is close enough. Rules win for `pinch`, since
that drives grabbing.

**Stability** matters more than raw accuracy here. Three mechanisms:

- *Hysteresis on pinch* — closes at 0.34, only reopens at 0.48, so a hand
  hovering near the threshold does not drop what it is holding.
- *Frame voting* — a label is committed only after N identical frames, so one
  bad frame can never fire a spurious select.
- *One Euro filter* — filters hard when the hand is still (killing tremor) and
  barely at all when it moves fast (keeping the cursor responsive). A plain
  low-pass filter forces you to choose between jitter and lag; this one adapts.

**Interaction** (`engine.py`) is a per-hand state machine. The fingertip becomes
a normalised device coordinate, the camera turns that into a ray, the ray picks
the nearest object. On pinch, the object's distance along the ray and its offset
from it are stored — so it neither snaps to the cursor nor drifts nearer or
further while you drag.

---

## Layout

```
gesture_control/
├── gesture_control/
│   ├── config.py      every tunable number, in one place
│   ├── tracker.py     HandTracker interface + MediaPipe backend
│   ├── features.py    landmark normalisation, finger/pinch features
│   ├── gestures.py    rule + k-NN classifiers, dataset, stabiliser
│   ├── motion.py      swipe detection
│   ├── pointer.py     One Euro filter, camera, rays, scene, raycasting
│   ├── engine.py      the interaction state machine and event source
│   ├── recorder.py    CLI for recording custom gestures
│   └── synthetic.py   generates fake hands for tests
├── demo/demo_scene.py
├── examples/basic_usage.py
├── tests/test_gesture_control.py
└── requirements.txt
```

---

## Tuning

Everything adjustable lives in `config.py`:

```python
from gesture_control import EngineConfig, GestureEngine

config = EngineConfig()
config.tracker.model_complexity = 0     # faster tracking on a weak laptop
config.gestures.pinch_close = 0.30      # require a tighter pinch
config.gestures.stable_frames = 5       # steadier, slightly slower to respond
config.pointer.beta = 0.05              # more responsive cursor, more jitter
config.primary_hand = "Right"           # ignore the other hand entirely

engine = GestureEngine(config=config)
```

Common problems:

- *Cursor drifts or feels laggy* — raise `pointer.beta`, or `pointer.min_cutoff`.
- *Pinch triggers by accident* — lower `gestures.pinch_close`.
- *Objects get dropped mid-drag* — raise `gestures.pinch_open` and
  `gestures.stable_frames`.
- *Grabbing misses what you aimed at* — raise `pointer.grab_assist`.
- *A held object feels floaty* — lower `pointer.drag_smoothing` (0 = no easing).
- *Custom gestures fire on the wrong pose* — lower `gestures.knn_max_distance`,
  or record more varied samples.
- *Low frame rate* — drop the capture resolution in `tracker.frame_width` /
  `frame_height`.
- *`module 'mediapipe' has no attribute 'solutions'`* — an old copy of this
  module. `tracker.py` handles both MediaPipe APIs; make sure you are running
  the current version of it.

---

## Other engines

`GestureEngine` never draws anything and never imports a game framework, so it
drops into Panda3D, Ursina, Pygame + PyOpenGL or anything else: call
`engine.update(frame)` from your loop (or `engine.start()` and poll), and apply
`event.position` to your own objects.

For Unity or Unreal, run it as a separate process and forward `engine.poll()`
over UDP as JSON — the events are already flat and serialisable.
