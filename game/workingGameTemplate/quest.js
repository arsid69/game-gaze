/* ============================================================
   QUEST WRAPPER — Phase 4
   The AI-Quest story around the 3D forest (forest.js):
     • an opening "Your Quest" page showing all 4 steps
     • Professor Skye mentor between stages
     • STEP 1 Collect (the forest) → STEP 2 Clean the data
     • STEP 3 Train / STEP 4 Apply come later (stubbed)
   Every choice is gaze-selectable (via forest.js domTargets) OR mouse.
   ============================================================ */
(() => {
  "use strict";
  const F = () => window.__forest;
  const setCollect = (on) => F() && F().setCollectEnabled && F().setCollectEnabled(on);
  const setButtons = (els) => { const d = F() && F().domTargets; if (!d) return; d.length = 0; els.forEach((e) => d.push(e)); };
  const clearButtons = () => { const d = F() && F().domTargets; if (d) d.length = 0; };
  const drop = (id) => { const e = document.getElementById(id); if (e) e.remove(); };
  const dropAll = () => ["q-steps", "q-skye", "q-clean", "q-train", "q-accuracy", "q-apply", "q-complete"].forEach(drop);

  const STEPS = [
    { name: "Collect Data", desc: "Gather data from the forest" },
    { name: "Clean Data",   desc: "Remove noisy & irrelevant data" },
    { name: "Train AI",     desc: "Train your flood-prediction model" },
    { name: "Apply",        desc: "Deploy the flood forecast" },
  ];

  const style = document.createElement("style");
  style.textContent = `
    .q-layer{position:fixed;inset:0;z-index:80;display:grid;place-items:center;text-align:center;padding:20px;overflow:auto;
      font-family:ui-monospace,"Cascadia Code",Consolas,monospace;color:#eafeff}
    .q-layer .eyebrow{font-size:12px;letter-spacing:.22em;color:#8fd3dc;margin-bottom:10px}
    .q-btn{font:600 16px ui-monospace,Consolas,monospace;color:#2a1a12;background:#f4a58e;border:none;border-radius:999px;
      padding:14px 34px;margin-top:26px;cursor:pointer;pointer-events:auto;transition:.12s}
    .q-btn:hover{filter:brightness(1.06)}
    .q-btn.gz-hover{outline:3px solid #6ff0f5;outline-offset:4px;box-shadow:0 0 22px rgba(110,240,245,.7)}
    /* steps page */
    #q-steps{background:radial-gradient(circle at 50% 34%,#12303c,#05111a 78%)}
    .q-steps-box{width:min(1000px,95vw)}
    .q-steps-title{font-family:system-ui,"Segoe UI",sans-serif;font-size:clamp(28px,5vw,50px);font-weight:800;margin:2px 0 26px;text-wrap:balance}
    .q-steps-row{display:flex;gap:14px;justify-content:center;flex-wrap:wrap}
    .q-step{flex:1;min-width:168px;max-width:220px;background:rgba(14,33,41,.8);border:1px solid rgba(120,240,245,.16);
      border-radius:16px;padding:18px 16px;transition:.2s}
    .q-step.current{border-color:#6ff0f5;box-shadow:0 0 26px rgba(110,240,245,.28);background:rgba(16,44,52,.92)}
    .q-step.upcoming{opacity:.5}
    .q-step-n{width:38px;height:38px;border-radius:50%;display:grid;place-items:center;font-weight:700;font-size:16px;margin:0 auto 12px;
      border:2px solid rgba(120,240,245,.4);color:#cfeef2}
    .q-step.done .q-step-n{background:#2fae8a;border-color:#2fae8a;color:#06131a}
    .q-step.current .q-step-n{background:#6ff0f5;border-color:#6ff0f5;color:#06131a}
    .q-step-name{font-family:system-ui,"Segoe UI",sans-serif;font-weight:700;font-size:16px;margin-bottom:6px}
    .q-step-desc{font-size:12px;color:#8fb0b6;line-height:1.45}
    .q-step-badge{margin-top:12px;font-size:10px;letter-spacing:.14em;text-transform:uppercase;color:#6f97a0}
    .q-step.done .q-step-badge{color:#3fcf90}
    .q-step.current .q-step-badge{color:#6ff0f5}
    /* skye device */
    #q-skye{background:rgba(6,16,22,.55);backdrop-filter:blur(3px)}
    .q-device{width:min(680px,92vw);background:linear-gradient(180deg,#12242c,#0b1a20);border:1px solid rgba(120,240,245,.25);
      border-radius:22px;padding:20px;box-shadow:0 24px 64px rgba(0,0,0,.55)}
    .q-dhead{display:flex;align-items:center;gap:12px;margin-bottom:14px}
    .q-ava{width:54px;height:54px;border-radius:50%;flex:none;display:grid;place-items:center;font-size:27px;
      background:radial-gradient(circle at 40% 35%,#a7ead0,#3f9e86);border:2px solid rgba(120,240,245,.45)}
    .q-dhead .who{text-align:left}.q-dhead .who .t{font-size:10px;letter-spacing:.18em;color:#7fa6ad}.q-dhead .who .n{font-weight:700;font-size:16px}
    .q-bubbles{display:flex;flex-direction:column;gap:10px;text-align:left}
    .q-bub{background:#0e2129;border:1px solid rgba(120,240,245,.18);border-radius:14px;padding:12px 15px;font-size:15px;line-height:1.5}
    .q-bub .lbl{font-size:10px;letter-spacing:.16em;color:#6ff0f5;display:block;margin-bottom:5px}
    /* cleaning */
    #q-clean{background:rgba(5,14,20,.74);backdrop-filter:blur(3px)}
    .q-clean-box{width:min(940px,95vw)}
    .q-clean-sub{color:#9fc4cc;font-size:14px;max-width:62ch;margin:6px auto 20px;line-height:1.5}
    .q-clean-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;max-width:840px;margin:0 auto}
    .q-card{font-family:ui-monospace,Consolas,monospace;text-align:left;background:rgba(14,33,41,.9);border:1px solid rgba(120,240,245,.28);
      border-radius:12px;padding:14px 16px;cursor:pointer;pointer-events:auto;display:flex;flex-direction:column;gap:8px;transition:.12s;color:#eafeff}
    .q-card .q-card-name{font-weight:600;font-size:14px}
    .q-card .q-card-state{font-size:12px;color:#3fcf90;letter-spacing:.02em}
    .q-card.discard{opacity:.5;border-color:rgba(240,115,109,.55)}
    .q-card.discard .q-card-state{color:#f0736d}
    .q-card.gz-hover{outline:3px solid #6ff0f5;outline-offset:3px;box-shadow:0 0 16px rgba(110,240,245,.5)}`;
  document.head.appendChild(style);

  const style2 = document.createElement("style");
  style2.textContent = `
    #q-train, #q-accuracy{background:radial-gradient(circle at 50% 40%,#10222c,#05111a 80%)}
    .q-panel{width:min(640px,94vw);background:linear-gradient(180deg,#12242c,#0a1820);border:1px solid rgba(120,240,245,.25);
      border-radius:22px;padding:26px 28px;box-shadow:0 24px 64px rgba(0,0,0,.55)}
    .q-panel-title{font-family:system-ui,"Segoe UI",sans-serif;font-size:clamp(20px,3.4vw,30px);font-weight:800;margin:6px 0 8px;line-height:1.15;text-wrap:balance}
    .q-panel-sub{color:#9fc4cc;font-size:14px;line-height:1.5;max-width:52ch;margin:0 auto}
    .q-btnrow{display:flex;gap:12px;justify-content:center;flex-wrap:wrap;margin-top:10px}
    .q-testviz{font-size:44px;margin:18px 0 6px}
    .q-spin{width:44px;height:44px;margin:22px auto 6px;border:4px solid rgba(120,240,245,.25);border-top-color:#6ff0f5;border-radius:50%;animation:qspin .8s linear infinite}
    @keyframes qspin{to{transform:rotate(360deg)}}
    .q-acc-big{font-family:system-ui,"Segoe UI",sans-serif;font-weight:800;font-size:64px;line-height:1;color:#6ff0f5;margin:8px 0}
    .q-acc-big span{font-size:28px;color:#9fc4cc}
    .q-acc-bar{position:relative;height:12px;background:rgba(255,255,255,.1);border-radius:7px;max-width:320px;margin:0 auto 16px}
    .q-acc-fill{height:100%;border-radius:7px;background:linear-gradient(90deg,#4fd8e0,#7ff0a0)}
    .q-acc-suff{position:absolute;top:-4px;left:70%;width:2px;height:20px;background:#eafeff;opacity:.7}
    .q-legend{display:flex;gap:18px;justify-content:center;font-size:12px;color:#9fc4cc;margin-top:8px}
    .q-legend i{display:inline-block;width:14px;height:3px;margin-right:6px;vertical-align:middle;border-radius:2px}
    @media (prefers-reduced-motion:reduce){.q-spin{animation:none}}
    /* Apply stage — see-through overlays so the 3D flood shows behind */
    #q-apply{background:rgba(4,10,16,.42)!important;align-items:flex-end}
    .q-hud{margin-bottom:12vh;background:rgba(6,16,22,.8);border:1px solid rgba(120,240,245,.3);border-radius:14px;padding:16px 26px;backdrop-filter:blur(4px)}
    .q-hud-line{font-size:clamp(16px,2.4vw,22px);font-weight:600;color:#eafeff}
    #q-complete{background:rgba(4,10,16,.5)!important}
    .q-complete-box{width:min(560px,94vw);background:linear-gradient(180deg,rgba(18,36,44,.94),rgba(10,24,30,.94));
      border:1px solid rgba(120,240,245,.3);border-radius:22px;padding:26px 28px;box-shadow:0 24px 64px rgba(0,0,0,.6)}
    .q-badge{display:inline-block;font-family:var(--mono);font-size:11px;letter-spacing:.16em;padding:5px 12px;border-radius:999px;margin-bottom:8px}
    .q-badge.good{background:rgba(63,207,144,.18);color:#7ff0a0;border:1px solid rgba(63,207,144,.5)}
    .q-badge.mid{background:rgba(224,178,92,.18);color:#e8cf72;border:1px solid rgba(224,178,92,.5)}
    .q-badge.bad{background:rgba(240,115,109,.18);color:#f0736d;border:1px solid rgba(240,115,109,.5)}
    .q-stats{display:flex;gap:10px;justify-content:center;margin:16px 0 6px;flex-wrap:wrap}
    .q-stat{background:rgba(10,24,30,.8);border:1px solid rgba(120,240,245,.18);border-radius:12px;padding:12px 16px;min-width:96px}
    .q-stat b{display:block;font-family:system-ui,sans-serif;font-size:26px;color:#6ff0f5;line-height:1}
    .q-stat span{font-size:11px;color:#9fc4cc;letter-spacing:.04em}
    .q-conf{position:fixed;inset:0;pointer-events:none;z-index:79;overflow:hidden}
    .q-conf i{position:absolute;top:-14px;width:9px;height:14px;border-radius:2px;animation:qfall linear forwards}
    @keyframes qfall{to{transform:translateY(108vh) rotate(720deg);opacity:.9}}`;
  document.head.appendChild(style2);

  function mkBtn(text, onClick) {
    const b = document.createElement("button"); b.className = "q-btn"; b.textContent = text;
    b.onclick = () => { clearButtons(); onClick(); };
    return b;
  }

  // --- Opening / progress: the 4-step roadmap -------------------------
  function showSteps(current, btnText, onGo) {
    dropAll(); setCollect(false);
    const l = document.createElement("div"); l.className = "q-layer"; l.id = "q-steps";
    const box = document.createElement("div"); box.className = "q-steps-box";
    const cards = STEPS.map((s, i) => {
      const st = i + 1 < current ? "done" : (i + 1 === current ? "current" : "upcoming");
      const badge = st === "done" ? "Complete" : st === "current" ? "Now" : "Locked";
      return `<div class="q-step ${st}">
        <div class="q-step-n">${st === "done" ? "✓" : i + 1}</div>
        <div class="q-step-name">${s.name}</div>
        <div class="q-step-desc">${s.desc}</div>
        <div class="q-step-badge">${badge}</div>
      </div>`;
    }).join("");
    box.innerHTML = `<div class="eyebrow">FLOOD FORECASTING · AI QUEST</div>
      <h2 class="q-steps-title">Your Quest — Data Forest</h2>
      <div class="q-steps-row">${cards}</div>`;
    const btn = mkBtn(btnText, onGo);
    box.appendChild(btn); l.appendChild(box); document.body.appendChild(l);
    setButtons([btn]);
  }

  // --- Professor Skye dialog ------------------------------------------
  function skye(lines, btnText, onDone) {
    dropAll();
    const l = document.createElement("div"); l.className = "q-layer"; l.id = "q-skye";
    const dev = document.createElement("div"); dev.className = "q-device";
    dev.innerHTML = `<div class="q-dhead">
        <div class="q-ava">🧑‍🔬</div>
        <div class="who"><div class="t">VIDEO TRANSMISSION</div><div class="n">Professor Skye</div></div>
      </div>
      <div class="q-bubbles">${lines.map((t) => `<div class="q-bub"><span class="lbl">PROFESSOR SKYE</span>${t}</div>`).join("")}</div>`;
    const btn = mkBtn(btnText, onDone);
    dev.appendChild(btn); l.appendChild(dev); document.body.appendChild(l);
    setButtons([btn]);
  }

  // --- FLOW -----------------------------------------------------------
  function start() {
    showSteps(1, "Begin Quest", () => skye([
      "Welcome! I'm Professor Skye, your AI mentor — I'll guide you as you make crucial decisions about your AI.",
      "First, to build your flood-prediction tool you need data. Head into the forest — <b>look left/right to scroll</b>, and hold your gaze on a data orb to collect it.",
    ], "Let's go!", () => { dropAll(); setCollect(true); }));
  }

  // Forest calls this when every data orb is collected.
  window.__questOnComplete = () => {
    showSteps(2, "Continue", () => skye([
      "Great work — you've collected the data! But not all of it is useful.",
      "Some sources are <b>noisy or irrelevant</b>. Let's clean the dataset: keep what helps predict floods, discard the junk.",
    ], "Clean the data", showCleaning));
  };

  // --- STEP 2: Clean the data -----------------------------------------
  function showCleaning() {
    dropAll();
    const data = (F() && F().data) || [];
    const keep = {}; data.forEach((d) => (keep[d.id] = true));
    const l = document.createElement("div"); l.className = "q-layer"; l.id = "q-clean";
    const box = document.createElement("div"); box.className = "q-clean-box";
    box.innerHTML = `<div class="eyebrow">STEP 2 · CLEAN THE DATA</div>
      <div class="q-clean-sub">Some of the data you collected is noisy or irrelevant. <b>Discard the junk</b> — keep only the sources that genuinely help predict floods.</div>`;
    const grid = document.createElement("div"); grid.className = "q-clean-grid";
    const cards = data.map((d) => {
      const c = document.createElement("button"); c.className = "q-card keep"; c.dataset.id = d.id;
      c.innerHTML = `<span class="q-card-name">${d.label}</span><span class="q-card-state">✓ keep</span>`;
      c.onclick = () => {
        keep[d.id] = !keep[d.id];
        c.classList.toggle("keep", keep[d.id]);
        c.classList.toggle("discard", !keep[d.id]);
        c.querySelector(".q-card-state").textContent = keep[d.id] ? "✓ keep" : "✕ discard";
      };
      grid.appendChild(c); return c;
    });
    box.appendChild(grid);
    const confirm = mkBtn("Confirm dataset", () => cleaningDone(keep, data));
    box.appendChild(confirm);
    l.appendChild(box); document.body.appendChild(l);
    setButtons([...cards, confirm]);
  }

  function cleaningDone(keep, data) {
    const goodKept = data.filter((d) => d.clean && keep[d.id]).length;
    const badKept = data.filter((d) => !d.clean && keep[d.id]).length;
    const totalGood = data.filter((d) => d.clean).length;
    window.__questClean = { goodKept, badKept, totalGood };   // used later by Train stage
    const perfect = badKept === 0 && goodKept === totalGood;
    const msg = perfect
      ? "Perfect — you kept every useful source and removed all the noise. That's exactly how you get a trustworthy model."
      : `You kept ${goodKept}/${totalGood} useful sources and left ${badKept} noisy one(s) in. Cleaner data means a more accurate model — you can always refine it.`;
    skye([msg], "Continue", () => showSteps(3, "Start Training", startTraining));
  }

  // ===== STEP 3: TRAIN & TEST =====
  function bigPanel(id, innerHTML, btns) {
    dropAll();
    const l = document.createElement("div"); l.className = "q-layer"; l.id = id;
    const box = document.createElement("div"); box.className = "q-panel";
    box.innerHTML = innerHTML;
    const row = document.createElement("div"); row.className = "q-btnrow";
    const els = btns.map(([t, cb]) => { const b = mkBtn(t, cb); row.appendChild(b); return b; });
    box.appendChild(row); l.appendChild(box); document.body.appendChild(l);
    setButtons(els);
  }

  // Accuracy reflects how well the player cleaned the data in Step 2 (capped at 90%).
  function computeAccuracy() {
    const c = window.__questClean || { goodKept: 5, badKept: 0, totalGood: 5 };
    const coverage = c.totalGood ? c.goodKept / c.totalGood : 0;
    return Math.round(Math.max(0.35, Math.min(1, coverage - c.badKept * 0.15)) * 90);
  }

  function chartSVG(acc) {
    const W = 320, H = 130, pad = 12, m = 12, actual = [], ai = [];
    for (let i = 0; i < m; i++) {
      const t = i / (m - 1);
      const base = 0.28 + 0.5 * Math.sin(t * Math.PI) + 0.08 * Math.sin(t * 7);
      actual.push(base);
      ai.push(Math.max(0.02, Math.min(0.98, base + (1 - acc / 90) * Math.sin(i * 1.7) * 0.28)));
    }
    const pts = (a) => a.map((v, i) => `${(pad + i / (m - 1) * (W - 2 * pad)).toFixed(1)},${(H - pad - v * (H - 2 * pad)).toFixed(1)}`).join(" ");
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:360px;display:block;margin:14px auto 0">
      <polyline points="${pts(actual)}" fill="none" stroke="#e8cf72" stroke-width="2.5"></polyline>
      <polyline points="${pts(ai)}" fill="none" stroke="#6ff0f5" stroke-width="2.5" stroke-dasharray="5 3"></polyline>
    </svg>
    <div class="q-legend"><span><i style="background:#e8cf72"></i>Actual event</span><span><i style="background:#6ff0f5"></i>Your AI prediction</span></div>`;
  }

  function startTraining() {
    skye([
      "Now let's <b>train and test</b> your model. Here's the key idea:",
      "We must test it on data it has <b>NOT seen before</b> — otherwise it's like handing it the answers to a test. 🤔",
    ], "Got it", askTestQuestion);
  }

  function askTestQuestion() {
    bigPanel("q-train", `<div class="eyebrow">STEP 3 · TRAIN THE AI</div>
      <div class="q-panel-title">Should we test your AI on data it has <u>already seen</u>?</div>
      <div class="q-panel-sub">Think about what really shows whether it learned…</div>`,
      [
        ["Yes", () => skye(["Not quite — if the AI has already seen the data, it's like giving it the answers. We need to see if it can handle <b>new</b> problems before the real world."], "I see", showTesting)],
        ["No", () => skye(["Exactly! Testing on <b>new</b> data it never saw is the only way to know if your model truly learned to predict floods."], "Let's test", showTesting)],
      ]);
  }

  function showTesting() {
    bigPanel("q-train", `<div class="eyebrow">STEP 3 · AI MODEL TESTING</div>
      <div class="q-panel-title">Test your flood model</div>
      <div class="q-panel-sub">Run your model against months it never saw, and compare its forecast to what actually happened.</div>
      <div class="q-testviz">🌊📈</div>`,
      [["Test Model", runTest]]);
  }

  function runTest() {
    bigPanel("q-train", `<div class="eyebrow">STEP 3 · TESTING…</div>
      <div class="q-panel-title">Running your model…</div><div class="q-spin"></div>`, []);
    setTimeout(() => showAccuracy(computeAccuracy()), 1500);
  }

  function showAccuracy(acc) {
    window.__questAccuracy = acc;
    const msg = acc >= 80
      ? "Your accuracy is in the 70–90% range — that's great! Your model is reliable enough for real-world use. (The max is 90% — no forecast is ever 100% certain.)"
      : acc >= 55
      ? `${acc}% — decent, but noisy data is dragging it down. Cleaner data would push this higher.`
      : `${acc}% — the model can't find a reliable signal. Too much noise, or too little good data, went in.`;
    bigPanel("q-accuracy", `<div class="eyebrow">STEP 3 · FORECAST ACCURACY</div>
      <div class="q-acc-big">${acc}<span>%</span></div>
      <div class="q-acc-bar"><div class="q-acc-fill" style="width:${acc}%"></div><div class="q-acc-suff"></div></div>
      ${chartSVG(acc)}
      <div class="q-panel-sub" style="margin-top:16px">${msg}</div>`,
      [
        ["Improve (re-clean)", showCleaning],
        ["Generate Forecast →", () => showSteps(4, "Deploy the forecast", startApply)],
      ]);
  }

  // ===== STEP 4: APPLY / DEPLOY =====
  function startApply() {
    skye([
      "Your model is trained and ready — now let's <b>deploy it</b> to protect Market Marshes. 🌊",
      "First, generate the flood forecast so we know when the water will rise.",
    ], "Generate forecast", forecastScreen);
  }

  function riskChart(acc) {
    const W = 320, H = 120, pad = 12, m = 12, risk = [];
    for (let i = 0; i < m; i++) risk.push(0.15 + 0.8 * Math.pow(Math.sin((i / (m - 1)) * Math.PI), 1.4));
    const bw = (W - 2 * pad) / m - 4;
    const bars = risk.map((v, i) => {
      const x = pad + i * ((W - 2 * pad) / m), h = v * (H - 2 * pad), y = H - pad - h;
      const col = v > 0.6 ? "#f0736d" : v > 0.35 ? "#e8cf72" : "#4fd8e0";
      return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${bw.toFixed(1)}" height="${h.toFixed(1)}" rx="2" fill="${col}"></rect>`;
    }).join("");
    const warnY = (H - pad - 0.6 * (H - 2 * pad)).toFixed(1);
    return `<svg viewBox="0 0 ${W} ${H}" width="100%" style="max-width:360px;display:block;margin:14px auto 2px">
      ${bars}<line x1="${pad}" y1="${warnY}" x2="${W - pad}" y2="${warnY}" stroke="#eafeff" stroke-dasharray="4 3" stroke-width="1"></line>
    </svg>
    <div class="q-legend"><span style="color:#eafeff">- - warning threshold</span><span>bars = predicted flood risk / month</span></div>`;
  }

  function forecastScreen() {
    const acc = window.__questAccuracy != null ? window.__questAccuracy : computeAccuracy();
    bigPanel("q-train", `<div class="eyebrow">STEP 4 · GENERATE FORECAST</div>
      <div class="q-panel-title">Flood-risk forecast</div>
      <div class="q-panel-sub">Your model projects flood risk across the monsoon season. Risk climbs past the warning line — issue the early warning.</div>
      ${riskChart(acc)}`,
      [["Issue Early Warning", () => deploySequence(acc)]]);
  }

  function deploySequence(acc) {
    if (F() && F().startFlood) F().startFlood(acc >= 80);   // 3D rain + rising floodwater
    hudSequence([
      "🌧️ Monsoon rains arriving over Market Marshes…",
      "📊 Forecasting river levels for the days ahead…",
      "📡 Early-warning system broadcasting to the marshes…",
    ], 0, () => questComplete(acc));
  }

  function hudSequence(lines, i, done) {
    dropAll();
    const l = document.createElement("div"); l.className = "q-layer"; l.id = "q-apply";
    l.innerHTML = `<div class="q-hud"><div class="q-hud-line">${lines[i]}</div></div>`;
    document.body.appendChild(l);
    setTimeout(() => (i + 1 < lines.length ? hudSequence(lines, i + 1, done) : done()), 1700);
  }

  function confetti() {
    const c = document.createElement("div"); c.className = "q-conf"; c.id = "q-conf";
    const cols = ["#6ff0f5", "#7ff0a0", "#e8cf72", "#f4a58e", "#b79cf0"];
    for (let i = 0; i < 90; i++) {
      const p = document.createElement("i");
      p.style.left = Math.random() * 100 + "vw";
      p.style.background = cols[i % cols.length];
      p.style.animationDuration = (1.8 + Math.random() * 1.8) + "s";
      p.style.animationDelay = (Math.random() * 0.6) + "s";
      c.appendChild(p);
    }
    document.body.appendChild(c);
    setTimeout(() => c.remove(), 4400);
  }

  function questComplete(acc) {
    const c = window.__questClean || { goodKept: 5, badKept: 0, totalGood: 5 };
    const removed = 3 - c.badKept;   // 3 noisy sources total
    let cls, tag, head, body;
    if (acc >= 80) {
      cls = "good"; tag = "MARKET MARSHES · DEPLOYED"; head = "The marshes are safe! 🎉";
      body = "Your early-warning system flagged the flood well before peak rainfall — communities had real time to evacuate before the water rose, and the waters receded.";
      if (F() && F().emote) F().emote("emote-yes");
      confetti();
    } else if (acc >= 55) {
      cls = "mid"; tag = "MARKET MARSHES · PARTIAL"; head = "Warnings issued — but late.";
      body = "Noisy data cost precious lead time, so the flood caught part of the marshes. Cleaner data would protect more next time.";
    } else {
      cls = "bad"; tag = "MARKET MARSHES · AT RISK"; head = "The forecast wasn't reliable enough.";
      body = "The warnings came too late to trust. Head back, clean the dataset, and retrain to protect the marshes.";
    }
    dropAll();
    const l = document.createElement("div"); l.className = "q-layer"; l.id = "q-complete";
    const box = document.createElement("div"); box.className = "q-complete-box";
    box.innerHTML = `<div class="q-badge ${cls}">${tag}</div>
      <div class="q-panel-title">${head}</div>
      <div class="q-panel-sub" style="margin:6px auto 4px">${body}</div>
      <div class="q-stats">
        <div class="q-stat"><b>8</b><span>data collected</span></div>
        <div class="q-stat"><b>${removed}/3</b><span>noise removed</span></div>
        <div class="q-stat"><b>${acc}%</b><span>accuracy</span></div>
      </div>`;
    const row = document.createElement("div"); row.className = "q-btnrow";
    const btns = (acc >= 80 ? [["Play Again", replayQuest]] : [["Improve data", showCleaning], ["Play Again", replayQuest]])
      .map(([t, cb]) => { const b = mkBtn(t, cb); row.appendChild(b); return b; });
    box.appendChild(row); l.appendChild(box); document.body.appendChild(l);
    setButtons(btns);
  }

  function replayQuest() {
    const conf = document.getElementById("q-conf"); if (conf) conf.remove();
    if (F() && F().stopFlood) F().stopFlood();
    if (F() && F().reset) F().reset();
    window.__questClean = null; window.__questAccuracy = null;
    start();
  }

  (function boot() { if (F() && F().domTargets) start(); else setTimeout(boot, 120); })();
})();
