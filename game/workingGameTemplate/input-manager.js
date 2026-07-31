/* ============================================================
   INPUT MANAGER — one input layer for gaze and gesture
   ============================================================

   The problem this solves
   -----------------------
   Both games used to open their own WebSocket and read gaze straight off the
   wire, so "where is the cursor" was hard-wired to one device in two places.
   Adding hand tracking that way would have meant a second copy of the same
   plumbing in each game.

   Instead the server resolves the active device and publishes ONE cursor. This
   file is the only WebSocket client in the browser; the games read a cursor and
   subscribe to activation events, and never learn which device produced them.

       server ──► input-manager.js ──► forest.js  (3D)
                        │          └─► gaze-client.js (2D)
                        └─ mode UI + [m] key

   Adding a third input later (head tracking, a gamepad) means teaching the
   server one more source. Nothing in this file or the games has to change.

   Public API (window.GameInput)
   -----------------------------
     .ok / .x / .y        cursor, normalised [0,1], y down — same contract gaze
                          always used, so existing game maths is untouched
     .mode                "gaze" | "gesture" | "hybrid"
     .source              which device is actually aiming right now
     .handOk / .gesture   hand state, reported in every mode
     .connected           socket state, for the HUD
     .gestureAvailable    false when the server has no gesture module
     .onClick(fn)         activation (pinch). Dwell stays in the game.
     .onSwipe(fn)         fn({direction})
     .onClear(fn)         open palm
     .onStatus(fn)        connection / mode changed — repaint the HUD
     .setMode(m) / .cycleMode()
   ============================================================ */
(() => {
  "use strict";

  const WS_URL = "ws://localhost:8765";
  const RECONNECT_MS = 1200;

  const listeners = { click: [], swipe: [], clear: [], status: [] };

  // Sequence counters, not booleans. The server increments a counter per event
  // and we fire on the delta, so a dropped or coalesced frame can never swallow
  // a click or replay an old one. null means "not yet synced" — see below.
  let lastClick = null, lastSwipe = null, lastClear = null;

  const api = {
    ok: false, x: 0.5, y: 0.5,
    mode: "gaze", source: "gaze",
    handOk: false, gesture: "",
    connected: false, gestureAvailable: false,

    onClick:  (fn) => listeners.click.push(fn),
    onSwipe:  (fn) => listeners.swipe.push(fn),
    onClear:  (fn) => listeners.clear.push(fn),
    onStatus: (fn) => listeners.status.push(fn),

    setMode(mode) {
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ cmd: "mode", mode }));
      }
    },

    /** Toggle gaze <-> gesture. Bound to [m].
     *  There is no hybrid mode: the two pipelines cannot share the camera
     *  without measurably degrading hand tracking, so exactly one runs at a
     *  time. Switching hands the camera over, which takes a moment. */
    cycleMode() {
      if (!api.gestureAvailable) {
        emit("status");                    // let the HUD explain why nothing happened
        return;
      }
      if (api.switching) return;           // ignore mashing during the handover
      api.switching = true;
      emit("status");
      api.setMode(api.mode === "gaze" ? "gesture" : "gaze");
    },

    /** Human-readable label for HUDs. */
    modeLabel() {
      if (api.switching) return "SWITCHING…";
      return { gaze: "GAZE", gesture: "GESTURE" }[api.mode] || api.mode;
    },

    /** True while the camera is being handed between pipelines. */
    switching: false,
  };

  function emit(kind, payload) {
    for (const fn of listeners[kind]) {
      try { fn(payload); } catch (err) { console.error("[input]", kind, err); }
    }
  }

  // --- Socket ----------------------------------------------------------
  let ws = null;

  function connect() {
    try { ws = new WebSocket(WS_URL); } catch { return setTimeout(connect, RECONNECT_MS); }

    ws.onopen = () => { api.connected = true; emit("status"); };

    ws.onmessage = (ev) => {
      let m; try { m = JSON.parse(ev.data); } catch { return; }

      if (m.type === "mode") {
        api.mode = m.mode;
        api.gestureAvailable = !!m.gesture_available;
        api.switching = false;             // handover finished (or failed)
        if (m.error) console.warn("[input] server:", m.error);
        emit("status");
        return;
      }

      // "gaze" messages are the raw device feed, kept for gaze_test.html. The
      // game consumes the resolved "input" message instead.
      if (m.type !== "input") return;

      api.ok = !!m.ok;
      if (m.ok) { api.x = m.x; api.y = m.y; }
      api.handOk = !!m.hand_ok;
      api.gesture = m.gesture || "";
      if (m.mode && m.mode !== api.mode) { api.mode = m.mode; emit("status"); }
      api.source = m.source || api.mode;

      // First message after connecting: adopt the server's counters without
      // firing. Otherwise reconnecting mid-game would replay every click the
      // server has counted since it started.
      if (lastClick === null) {
        lastClick = m.click_seq; lastSwipe = m.swipe_seq; lastClear = m.clear_seq;
        return;
      }
      if (m.click_seq > lastClick) { lastClick = m.click_seq; emit("click"); }
      if (m.clear_seq > lastClear) { lastClear = m.clear_seq; emit("clear"); }
      if (m.swipe_seq > lastSwipe) {
        lastSwipe = m.swipe_seq;
        emit("swipe", { direction: m.swipe || "" });
      }
    };

    ws.onclose = () => {
      api.connected = false; api.ok = false; api.handOk = false;
      lastClick = lastSwipe = lastClear = null;   // resync on the next connection
      emit("status");
      setTimeout(connect, RECONNECT_MS);
    };
    ws.onerror = () => ws.close();
  }

  // [m] switches input mode from anywhere, in either game.
  window.addEventListener("keydown", (e) => {
    if (e.key === "m" || e.key === "M") api.cycleMode();
  });

  connect();
  window.GameInput = api;
})();
