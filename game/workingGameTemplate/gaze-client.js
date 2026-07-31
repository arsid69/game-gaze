/* ============================================================
   GAZE CLIENT — Phase 2 (v2: cursor-follows-eyes)
   Non-invasive gaze control overlay. Nothing in main.js changes.

   Design (fixed after v1 felt "stuck"):
     • The cursor follows your RAW gaze — it goes where you look,
       and is NOT snapped onto buttons. Only a gentle magnet (30%)
       settles it once it is already near a button, so dwell is
       stable without trapping the cursor.
     • Any button within HOVER_PAD px of your gaze becomes the
       target: it lights up, a ring charges, and after DWELL_MS it
       fires its own .click(). Generous tolerance so ~70%-accurate
       gaze still lands reliably.
     • Look at a screen EDGE to pan (Collect stage) or scroll
       (any stage) so off-screen buttons like "Proceed" are reachable.

   Requires gaze_server.py (ws://localhost:8765).
   Keys:  g = gaze on/off   ·   f = fullscreen (recommended!)
   ============================================================ */
(() => {
  "use strict";

  // --- Tunables --------------------------------------------------------
  // Socket URL now lives in input-manager.js, the only WebSocket client.
  const HOVER_PAD    = 85;    // gaze within this many px of a button (edge) targets it
  const MAGNET       = 0.15;  // gentle settle toward a targeted button (0 = none, 1 = snap)
  const HYSTERESIS   = 60;    // stickiness: current target keeps a px bonus so it doesn't flicker to a neighbor
  // Proceed/Start buttons live in the weak lower screen (13-14% error there), so
  // give ONLY those a big reach + strong magnet — easy to hit despite the drift.
  const PRIORITY_SEL   = ".proceed-btn, #start-btn";
  const PROCEED_PAD    = 200;  // reach for priority buttons (vs 85 for data buttons)
  const PROCEED_MAGNET = 0.65; // strong snap for priority buttons (vs 0.15)
  // Clean-stage cards: big but packed tight, so a wide reach would grab the
  // neighbor. Give them a SMALL reach (you must look on the card) + a firm
  // magnet so the cursor settles onto the one you're actually looking at.
  const CARD_SEL       = ".clean-card";
  const CARD_PAD       = 30;
  const CARD_MAGNET    = 0.40;
  const DWELL_MS     = 1050;  // hold gaze on a button this long to click it (deliberate)
  const COOLDOWN_MS  = 500;   // min gap between two clicks
  const CURSOR_LERP  = 0.45;  // render smoothing (higher = snappier, lower = smoother)
  const EDGE_ZONE    = 0.09;  // fraction of screen at each side/edge that triggers pan/scroll
  const EDGE_PAN_MS  = 120;   // horizontal pan nudge interval (Collect stage)
  const SCROLL_STEP  = 26;    // px per vertical scroll tick

  // --- State -----------------------------------------------------------
  let enabled = true, faceOk = false;
  let rawX = 0.5, rawY = 0.5;
  let driftX = 0, driftY = 0;   // manual recenter offset — set by the [c] key
  let curX = innerWidth / 2, curY = innerHeight / 2;
  let target = null, prevTarget = null;
  let dwellStart = 0, armed = true, cooldownUntil = 0, lastPan = 0;

  // --- Styles + overlay ------------------------------------------------
  const style = document.createElement("style");
  style.textContent = `
    #gaze-cursor { position: fixed; top:0; left:0; width:34px; height:34px;
      margin:-17px 0 0 -17px; z-index:99999; pointer-events:none; transition:opacity .15s; }
    #gaze-cursor svg { display:block; overflow:visible; }
    #gaze-cursor .ring   { fill:none; stroke:rgba(120,240,245,.4); stroke-width:2.5; }
    #gaze-cursor .charge { fill:none; stroke:#6ff0f5; stroke-width:4; stroke-linecap:round;
      transform:rotate(-90deg); transform-origin:17px 17px; filter:drop-shadow(0 0 5px rgba(110,240,245,.9)); }
    #gaze-cursor .core   { fill:#eafeff; }
    #gaze-cursor.lost    { opacity:0; }

    .gaze-hover { outline:3px solid rgba(110,240,245,.95) !important; outline-offset:3px;
      box-shadow:0 0 16px rgba(110,240,245,.55) !important; transition:box-shadow .1s; }

    /* Collected data buttons are already gaze-ignored (disabled); dim them so
       it's obvious they're done and your eye goes to the ones left to collect. */
    .data-btn:disabled { opacity:0.28 !important; filter:grayscale(0.5); }

    /* Implement stage: push "Back" and "Restart" to opposite ends so their
       strong magnets can't mix up — they were only 10px apart. */
    #stage-implement .stage-nav { justify-content: space-between !important; }

    #gaze-chip { position:fixed; bottom:14px; left:14px; z-index:99999;
      font:12px/1.5 "JetBrains Mono", ui-monospace, Consolas, monospace;
      background:rgba(6,14,19,.82); color:#9fe9ef; border:1px solid rgba(90,160,170,.3);
      border-radius:9px; padding:8px 11px; pointer-events:none; backdrop-filter:blur(5px); white-space:pre; }
    #gaze-chip b { color:#eafeff; } #gaze-chip .on{color:#56e39f} #gaze-chip .off{color:#ff9a9a}
    #gaze-chip .warn{color:#ffcf5c}
  `;
  document.head.appendChild(style);

  const RR = 15, CIRC = 2 * Math.PI * RR;
  const cursor = document.createElement("div");
  cursor.id = "gaze-cursor"; cursor.className = "lost";
  cursor.innerHTML = `<svg width="34" height="34" viewBox="0 0 34 34">
      <circle class="ring"   cx="17" cy="17" r="${RR}"></circle>
      <circle class="charge" cx="17" cy="17" r="${RR}" stroke-dasharray="${CIRC}" stroke-dashoffset="${CIRC}"></circle>
      <circle class="core"   cx="17" cy="17" r="2.5"></circle></svg>`;
  document.body.appendChild(cursor);
  const chargeEl = cursor.querySelector(".charge");

  const chip = document.createElement("div");
  chip.id = "gaze-chip"; document.body.appendChild(chip);
  function setChip(sock) {
    const gi = window.GameInput;
    const on = enabled ? `<span class="on">ON</span>` : `<span class="off">OFF</span>`;
    const fs = document.fullscreenElement ? "" : `   <span class="warn">press [f] to fullscreen</span>`;
    // Gesture modes care about the hand, gaze mode about the face.
    const tracked = gi && gi.mode !== "gaze"
      ? `hand:<b>${gi.handOk ? "yes" : "—"}</b>`
      : `face:<b>${faceOk ? "yes" : "—"}</b>`;
    const activate = gi && gi.mode !== "gaze" ? "pinch to click" : "hold gaze to click";
    chip.innerHTML = `INPUT <b>${gi ? gi.modeLabel() : "GAZE"}</b>   control ${on}   `
      + `socket:<b>${sock}</b>   ${tracked}${fs}\n`
      + `[m] input mode   [f] fullscreen   [c] recenter   [g] toggle   ·   ${activate}`;
  }

  // --- Targets = enabled, on-screen buttons ----------------------------
  function candidates() {
    const out = [];
    for (const el of document.querySelectorAll("button")) {
      if (el.disabled) continue;
      const r = el.getBoundingClientRect();
      if (r.width <= 0 || r.height <= 0) continue;
      if (r.right < 0 || r.left > innerWidth || r.bottom < 0 || r.top > innerHeight) continue;
      // Occlusion test: is this button actually the visible, topmost element at
      // its own center? If something covers it (e.g. the intro/Start overlay),
      // elementFromPoint returns that overlay instead — so we skip the button
      // and never "click through" what the player can't see. (The cursor + chip
      // are pointer-events:none, so they're transparent to this test.)
      const cx = Math.min(Math.max(r.left + r.width / 2, 1), innerWidth - 1);
      const cy = Math.min(Math.max(r.top + r.height / 2, 1), innerHeight - 1);
      const top = document.elementFromPoint(cx, cy);
      if (!top || (top !== el && !el.contains(top))) continue;
      out.push({ el, r });
    }
    return out;
  }

  // Nearest button to the gaze by distance-to-RECT (0 if inside). Returns the
  // element if it is within HOVER_PAD, else null — so open space targets nothing
  // and the cursor stays free.
  const padFor = (el) => el.matches(PRIORITY_SEL) ? PROCEED_PAD
                       : el.matches(CARD_SEL) ? CARD_PAD
                       : HOVER_PAD;

  function pickTarget(px, py, current) {
    let best = null, bestScore = Infinity;
    for (const { el, r } of candidates()) {
      const dx = Math.max(r.left - px, 0, px - r.right);
      const dy = Math.max(r.top - py, 0, py - r.bottom);
      const d = Math.hypot(dx, dy);
      if (d > padFor(el)) continue;                          // outside this button's reach
      const score = (el === current) ? d - HYSTERESIS : d;   // sticky: keep current target
      if (score < bestScore) { bestScore = score; best = { el, r, d }; }
    }
    return best;
  }

  // --- Edge navigation so far-away / off-screen buttons are reachable ---
  function edgeNav(gx, gy, hasTarget) {
    const now = performance.now();
    // Horizontal: pan the Collect field via the game's own arrow-key handler.
    const collect = document.getElementById("stage-collect");
    if (collect && collect.classList.contains("active") && !hasTarget && now - lastPan >= EDGE_PAN_MS) {
      if (gx < EDGE_ZONE)      { window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowLeft"  })); lastPan = now; }
      else if (gx > 1 - EDGE_ZONE) { window.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowRight" })); lastPan = now; }
    }
    // Vertical: scroll the page if it overflows (reaches a Proceed button below the fold).
    const scrollable = document.scrollingElement &&
      document.scrollingElement.scrollHeight > innerHeight + 4;
    if (scrollable) {
      if (gy > 1 - EDGE_ZONE) window.scrollBy(0, SCROLL_STEP);
      else if (gy < EDGE_ZONE) window.scrollBy(0, -SCROLL_STEP);
    }
  }

  // --- Loop ------------------------------------------------------------
  function frame() {
    requestAnimationFrame(frame);
    syncInput();                 // pull the active device's cursor for this frame
    if (!enabled || !faceOk) {
      cursor.classList.add("lost");
      chargeEl.setAttribute("stroke-dashoffset", CIRC);
      clearTarget();
      return;
    }

    // Apply the manual recenter offset, then map to viewport pixels. The offset
    // corrects GAZE drift only — a fingertip is already an absolute position, so
    // applying it there would push the cursor off the buttons entirely.
    const useDrift = !Input || Input.source !== "gesture";
    const gx = Math.min(Math.max(rawX - (useDrift ? driftX : 0), 0), 1);
    const gy = Math.min(Math.max(rawY - (useDrift ? driftY : 0), 0), 1);
    const px = gx * innerWidth, py = gy * innerHeight;
    const hit = pickTarget(px, py, target);

    // Cursor position: raw gaze, gently magnetized toward a target if there is one.
    let tx = px, ty = py;
    if (hit) {
      const cx = hit.r.left + hit.r.width / 2, cy = hit.r.top + hit.r.height / 2;
      const m = hit.el.matches(PRIORITY_SEL) ? PROCEED_MAGNET
              : hit.el.matches(CARD_SEL) ? CARD_MAGNET
              : MAGNET;
      tx = px + (cx - px) * m;
      ty = py + (cy - py) * m;
    }
    curX += (tx - curX) * CURSOR_LERP;
    curY += (ty - curY) * CURSOR_LERP;
    cursor.style.transform = `translate(${curX}px, ${curY}px)`;
    cursor.classList.remove("lost");

    edgeNav(gx, gy, !!hit);

    // Target + dwell
    const el = hit ? hit.el : null;
    if (el !== target) {
      clearTarget();
      target = el;
      if (target) target.classList.add("gaze-hover");
      dwellStart = performance.now();
      armed = true;
    }

    let progress = 0;
    const now = performance.now();
    if (target && dwellActive()) {
      progress = Math.min((now - dwellStart) / DWELL_MS, 1);
      if (progress >= 1 && armed && now >= cooldownUntil) {
        const toClick = target;
        clearTarget();                 // remove highlight before the click navigates
        toClick.click();               // fire the game's own handler
        armed = false; cooldownUntil = now + COOLDOWN_MS;
        cursor.animate([{ filter: "brightness(2.4)" }, { filter: "brightness(1)" }],
                       { duration: 250, easing: "ease-out" });
        progress = 0;
      }
    }
    chargeEl.setAttribute("stroke-dashoffset", CIRC * (1 - progress));
  }

  function clearTarget() {
    if (target) target.classList.remove("gaze-hover");
    target = null;
  }

  // --- Input source ----------------------------------------------------
  // input-manager.js owns the socket and resolves gaze vs gesture, so this file
  // only asks "where is the cursor" and "was that a click". Everything below
  // (targets, magnet, dwell, edge nav) is unchanged by the integration.
  const Input = window.GameInput;

  function syncInput() {
    if (!Input) return;
    faceOk = Input.ok;
    if (Input.ok) { rawX = Input.x; rawY = Input.y; }
  }

  // Dwell exists because eyes cannot click. A hand can, so pinch replaces it
  // rather than running alongside it and double-firing.
  function dwellActive() { return !Input || Input.mode === "gaze"; }

  function fireTarget() {
    if (!target || performance.now() < cooldownUntil) return;
    const toClick = target;
    clearTarget();                 // drop the highlight before the click navigates
    toClick.click();
    armed = false; cooldownUntil = performance.now() + COOLDOWN_MS;
  }

  if (Input) {
    Input.onClick(fireTarget);
    Input.onStatus(() => setChip(Input.connected ? "connected" : "disconnected"));
  }

  window.addEventListener("keydown", (e) => {
    if (e.key === "g" || e.key === "G") { enabled = !enabled; if (!enabled) clearTarget(); }
    if (e.key === "f" || e.key === "F") {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
      else document.exitFullscreen?.();
    }
    // Recenter: look at the middle of the screen and press C. We zero the offset
    // so "looking at center" maps to center, correcting per-session head drift.
    // Meaningless for a hand pointer, so it is ignored in gesture mode.
    if (e.key === "c" || e.key === "C") {
      if (faceOk && (!Input || Input.source !== "gesture")) {
        driftX = rawX - 0.5; driftY = rawY - 0.5;
      }
    }
    setChip(faceOk || enabled ? "connected" : "connecting…");
  });
  document.addEventListener("fullscreenchange", () => setChip("connected"));

  setInterval(() => setChip(Input && Input.connected ? "connected" : "disconnected"), 1000);
  requestAnimationFrame(frame);
})();
