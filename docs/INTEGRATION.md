# Gesture + Gaze Integration

How the existing gesture-control module was connected to the existing
gaze-controlled game, what changed, and how to use it.

Written to be readable without knowing either codebase.

---

## 1. What this integration does

Before, the game could only be controlled by **eye gaze**. A separate, finished
Python module could recognise **hand gestures** but had nothing to do with the game.

Now both drive the same game, and you can switch between them **while playing**,
without restarting anything.

| Mode | You aim with | You select by |
|---|---|---|
| **Gaze** | your eyes | holding still ~1 s (dwell) |
| **Gesture** | your index fingertip | pinching thumb + index |

Press **`m`** to switch. The switch takes about 1.5 seconds, because the two
pipelines hand the camera to each other.

### Why there is no hybrid mode

An earlier version had one — eyes aiming, hand clicking — which needed both
pipelines running at once, sharing one camera through a common capture thread.

Measurement showed that cost about **20% of the hand-detection rate**
(19.1 → 15.3 detections/sec). The cause is that MediaPipe holds the Python GIL
while it runs inference, so the shared capture thread could only run *between*
detections and was starved down to the detection rate — delivering fewer and
staler frames. Because a gesture must hold for 3 consecutive frames, a slower
frame rate also made gestures slower and less reliable to commit.

Giving whichever pipeline is active **exclusive** ownership of the camera
restores the standalone module's accuracy exactly, at the cost of hybrid mode
and a ~1.5 s pause when switching. That trade was made deliberately.

---

## 2. The one idea that made this simple

The gaze system already published a very small message:

```json
{"type": "gaze", "ok": true, "x": 0.42, "y": 0.71}
```

`x` and `y` are just "where on the screen, from 0 to 1". The game never cared
*how* that point was produced — only where it was.

The gesture module happens to produce the same kind of thing: a pointer position
in its own units (`ndc`, from -1 to 1). Converting one to the other is one line
of arithmetic.

So the integration is mostly **plumbing, not new logic**: teach the server to
produce that same point from a second device, and let the browser pick which
device is feeding it.

Nothing in the gaze algorithm changed. Nothing in the gesture module changed.

---

## 3. Architecture

Exactly one pipeline owns the camera at a time. Switching mode hands it over.

```
   GAZE MODE                            GESTURE MODE
   ─────────                            ────────────
   webcam                               webcam
     │                                    │
     ▼                                    ▼
   capture thread                       MediaPipeHandTracker
     │  _latest_frame                     │  (the standalone class,
     ▼                                    ▼   read → detect, one thread)
   gaze pipeline                        GestureEngine  (no scene)
   face → L2CS → calibration → 1€         │  pointer + pinch + swipe
     │                                    │
     └──────────────┬─────────────────────┘
                    ▼
       gaze_server.py  publishes whichever is active
                    │
    {"type":"input", "x", "y", "click_seq", ...}
                    │  ws://localhost:8765
                    ▼
         input-manager.js   (the only socket)
                    │
     ┌──────────────┴──────────────┐
     ▼                             ▼
forest.js (3D)            gaze-client.js (2D)
```

### Why exclusive ownership

Windows will not let two programs open the same webcam, so the first design had
the gaze server's capture thread own it and share frames with the gesture engine.
That worked, but measurably degraded hand tracking (see "Why there is no hybrid
mode" above).

Now, in gesture mode the gesture module opens the camera **itself**, through the
very same `MediaPipeHandTracker` it uses standalone. There is no adapter in the
path at all — which is the strongest possible guarantee that behaviour matches
the reference implementation.

`SharedFrameTracker` still exists in `gesture_bridge.py` for the shared-camera
case, but the server no longer uses it.

### Why the gesture module's 3D scene is not used

The module can do more than report gestures: it has its own 3D scene, ray
casting, hover and selection. The game already does all of that in three.js.

Using both would put selection logic in two places that could disagree, and
would mean sending the whole scene to Python every frame. So the engine runs
with an **empty scene** and is used purely as an input device. The game stays in
charge of what is hovered and selected.

---

## 4. Files

### New

| File | What it is |
|---|---|
| `gaze-detection/gesture_bridge.py` | The adapter. Lets the gesture engine read the gaze server's camera frames, and converts its pointer into the game's coordinate system. |
| `game/workingGameTemplate/input-manager.js` | The browser's input layer. Owns the only WebSocket, exposes one cursor, and handles mode switching. |

### Changed

| File | Why | What changed |
|---|---|---|
| `gaze-detection/gaze_server.py` | It owns the camera, so it is where a second input source has to join | Added mode state, start/stop of the gesture engine, the `input` message, and handling of `mode` commands from the browser. **The gaze pipeline itself is untouched**, and the original `gaze` message is still sent unchanged. |
| `game/workingGameTemplate/forest.js` | The 3D game read the socket directly | Its ~14-line WebSocket block was replaced by a subscription to the input manager. Dwell now only applies in gaze mode; pinch fires the same action. Swipe turns the forest. HUD shows the mode. |
| `game/workingGameTemplate/gaze-client.js` | Same, for the 2D game | Same change, same reasoning. |
| `forest.html`, `index.html` | Load the input layer before the code that uses it | One `<script>` tag each. |

### Untouched on purpose

- **The whole gesture module** (`gesture-control-module/`) — not one line.
- **The gaze algorithm** — `gaze_pipeline.py`, `calibration_utils.py`, the
  calibration milestones.
- **`gaze_test.html`** — still works, because the original `gaze` message is
  still broadcast exactly as before.

---

## 5. How input flows, step by step

1. The capture thread grabs a webcam frame and mirrors it.
2. The gaze pipeline turns it into a screen point, if it can see your face.
3. If a gesture mode is active, `SharedFrameTracker` hands the same frame to
   MediaPipe Hands, and `GestureEngine` turns it into a pointer plus a gesture
   label.
4. `gaze_server.py` picks which point is the live cursor, based on the mode.
5. It sends one `input` message about 40 times a second.
6. `input-manager.js` receives it, updates `GameInput.x` / `.y` / `.ok`, and
   fires a `click` event if the pinch counter went up.
7. The game reads that cursor and does exactly what it always did.

### Why clicks are counters, not flags

The server sends `click_seq: 7` rather than `clicked: true`. The browser fires
when the number *increases*. If a message is dropped or two arrive at once, you
still get exactly one click — a boolean could be missed entirely or seen twice.

---

## 6. User guide

### Switching modes

Press **`m`** to cycle: **Gaze → Gesture → Hybrid → Gaze**.

The HUD always shows the current mode. Your progress is never lost — switching
does not reload the page or restart the game.

If gesture control is unavailable, `m` does nothing and the server prints why.

### Playing with gestures

1. Hold your hand up, palm towards the camera, about 40–70 cm away.
2. Point with your index finger — the cursor follows your fingertip.
3. **Pinch** thumb and index together to select what the cursor is on.
4. **Open palm** clears the selection.
5. **Swipe** left or right to turn the forest and find more orbs.

The HUD shows `hand: yes` when your hand is being tracked.

### Recording your own gestures

The gesture module has a recorder built in, and it needs no game changes — the
engine picks up new gestures the next time it starts.

**Stop the gaze server first** (`Ctrl+C`), because the recorder needs the camera:

```powershell
cd "C:\game integration\gesture-control-module"
..\gaze-detection\gaze_env\Scripts\python.exe -m gesture_control.recorder record --name thumbs_up --samples 60
```

Hold the pose, press **Space** to start capturing, and **move your hand around a
little** while it records — varied samples work much better than 60 identical
frames. 30–60 samples is plenty.

Managing what you have recorded:

```powershell
# see everything recorded
..\gaze-detection\gaze_env\Scripts\python.exe -m gesture_control.recorder list

# live view of what it recognises right now
..\gaze-detection\gaze_env\Scripts\python.exe -m gesture_control.recorder test

# remove one
..\gaze-detection\gaze_env\Scripts\python.exe -m gesture_control.recorder delete --name thumbs_up
```

Gestures are stored in `gestures_dataset.json`. Restart the gaze server to pick
up changes.

Custom gestures arrive in the browser as `GameInput.gesture`, so game code can
react to them. `pinch` and `open_palm` are the two the game currently acts on.

### Troubleshooting

| Problem | Fix |
|---|---|
| `m` does nothing | The server reports gesture as unavailable. Check its startup output for the reason. |
| Cursor does not move in gesture mode | Check the HUD says `hand: yes`. Hands need reasonable light and to be fully in frame. |
| Pinch does not select | Aim first — the cursor must be on the target before you pinch. Pinch fully; the module needs thumb and index close. |
| Pinch selects the wrong thing | Lower `pinch_close` in `gesture-control-module/gesture_control/config.py`. |
| Cursor jitters | Raise `pointer.min_cutoff` in the same file. |
| Gaze got slower after switching | Switch back to gaze mode; hand tracking stops and its cost goes away. |
| Everything is slower | Raise `GESTURE_DETECT_EVERY` in `gaze_server.py` to 3 or 4. |

---

## 7. Adding another input later

The point of the input layer is that a third device does not touch the games.

1. Produce a screen point (`x`, `y` in 0–1) in Python.
2. Add a mode name to `VALID_MODES` and a branch in `build_input_message()`.

`input-manager.js`, `forest.js` and `gaze-client.js` need no changes at all.

---

## 8. Tuning

| Where | Setting | Effect |
|---|---|---|
| `gaze_server.py` | `GESTURE_DETECT_EVERY` | Frames per hand detection. Higher = cheaper, slightly less responsive. |
| `gesture_control/config.py` | `gestures.pinch_close` / `pinch_open` | How hard you must pinch, and when it lets go. |
| `gesture_control/config.py` | `gestures.stable_frames` | Frames a gesture must hold before it counts. Higher = steadier, slower. |
| `gesture_control/config.py` | `pointer.min_cutoff` / `beta` | Cursor smoothing vs responsiveness. |
| `gesture_control/config.py` | `primary_hand` | `"Right"`, `"Left"` or `"any"`. |
| `forest.js` / `gaze-client.js` | `DWELL_MS` | Gaze dwell time. Ignored in gesture and hybrid modes. |
