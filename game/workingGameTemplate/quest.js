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
  const dropAll = () => ["q-steps", "q-skye", "q-clean"].forEach(drop);

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
    skye([msg], "Continue", () => showSteps(3, "Continue", trainStub));
  }

  // --- STEP 3/4 stub (built next) -------------------------------------
  function trainStub() {
    skye([
      "Next: <b>Train the AI</b> on your cleaned data, then <b>Apply</b> the forecast to protect the marshes.",
      "Those stages are coming next — great progress so far! 🌊",
    ], "Back to quest", () => showSteps(3, "Continue", trainStub));
  }

  (function boot() { if (F() && F().domTargets) start(); else setTimeout(boot, 120); })();
})();
