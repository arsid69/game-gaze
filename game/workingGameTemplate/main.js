/* ============================================================
   1. DATA SOURCES
   `clean: false` marks redundant/irrelevant entries the player
   is expected to filter out in Process 2.
   ============================================================ */
const DATA_SOURCES = [
  { id: "rain_gauge",     tag: "SRC://rain_gauge_07",     label: "Rainfall Gauge Records",        clean: true  },
  { id: "river_level",    tag: "SRC://river_gauge_12",    label: "River Water Level (Gauge)",     clean: true  },
  { id: "sat_imagery",    tag: "SRC://sat_flood_ir",      label: "Satellite Flood Imagery",       clean: true  },
  { id: "dem",            tag: "SRC://terrain_dem_30m",   label: "Digital Elevation Model",       clean: true  },
  { id: "soil_moisture",  tag: "SRC://soil_probe_04",     label: "Soil Moisture Readings",        clean: true  },
  { id: "weather_fc",     tag: "SRC://forecast_feed",     label: "Weather Forecast Feed",         clean: true  },
  { id: "land_use",       tag: "SRC://landcover_map",     label: "Land Use / Land Cover Map",     clean: true  },
  { id: "drainage",       tag: "SRC://infra_drainage",    label: "Drainage Network Map",          clean: true  },
  { id: "flood_history",  tag: "SRC://archive_1998_2025", label: "Historical Flood Records",      clean: true  },
  { id: "reservoir",      tag: "SRC://dam_discharge",     label: "Reservoir Discharge Data",       clean: true  },
  { id: "rain_dupe",      tag: "SRC://rain_gauge_07_copy",label: "Duplicate Rain Gauge Log",        clean: false },
  { id: "traffic_cam",    tag: "SRC://cam_feed_ring_rd",  label: "Unrelated Traffic Camera Feed",  clean: false },
  { id: "celeb_tweet",    tag: "SRC://social_misc",       label: "Celebrity Weather Tweet",        clean: false },
  { id: "corrupt_backup", tag: "SRC://backup_2019_dmg",   label: "Old Sensor Backup (corrupted)",  clean: false },
];

/* Player progress state, carried across stages */
const collectedData = [];      // sources pulled in during Process 1
const selection = {};          // id -> included-for-training bool, set in Process 2
let lastAccuracy = null;       // result of the most recent Process 3 training run

/* ============================================================
   2. STAGE MACHINE
   Four sections (#stage-collect/clean/train/implement) are
   swapped via a simple "active" class. The stepper at the top
   reflects the current stage and marks earlier ones as done.
   ============================================================ */
const STAGES = ["collect", "clean", "train", "implement"];

function goToStage(name) {
  document.querySelectorAll(".stage").forEach(s => s.classList.remove("active"));
  document.getElementById(`stage-${name}`).classList.add("active");

  const idx = STAGES.indexOf(name);
  document.querySelectorAll(".step").forEach(el => {
    const stepIdx = STAGES.indexOf(el.dataset.stage);
    el.classList.toggle("active", stepIdx === idx);
    el.classList.toggle("done", stepIdx < idx);
  });

  if (name === "clean") renderCleanStage();
  if (name === "train") renderTrainStage();
  if (name === "implement") renderImplementStage();
}

document.getElementById("start-btn").addEventListener("click", () => {
  document.body.classList.add("started");
  goToStage("collect");
});

/* ============================================================
   3. PROCESS 1 — COLLECT DATA
   ============================================================ */
const viewport    = document.getElementById("grid-panel");
const world       = document.getElementById("world");
const minimap     = document.getElementById("minimap");
const minimapThumb= document.getElementById("minimap-thumb");
const panHint     = document.getElementById("pan-hint");
const listEl      = document.getElementById("collected-list");
const emptyHint   = document.getElementById("empty-hint");
const countEl     = document.getElementById("progress-count");
const fillEl      = document.getElementById("progress-fill");
const statusLine  = document.getElementById("status-line");
const toCleanBtn  = document.getElementById("to-clean-btn");

countEl.textContent = `0 / ${DATA_SOURCES.length}`;

/* --- Scatter layout: buttons placed across a wide, jittered
   3-row grid so nothing overlaps but nothing lines up neatly. --- */
const ROWS = 3;
const CELL_W = 260;
const BTN_W = 190;
const EDGE_PAD = 60;

const placementOrder = [...DATA_SOURCES.keys()];
for (let i = placementOrder.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [placementOrder[i], placementOrder[j]] = [placementOrder[j], placementOrder[i]];
}

const cols = Math.ceil(DATA_SOURCES.length / ROWS);
const worldWidth = cols * CELL_W + EDGE_PAD * 2;
world.style.width = worldWidth + "px";

function layoutButtons() {
  const panelHeight = viewport.clientHeight;
  const rowHeight = panelHeight / ROWS;

  placementOrder.forEach((sourceIndex, i) => {
    const source = DATA_SOURCES[sourceIndex];
    const row = i % ROWS;
    const col = Math.floor(i / ROWS);

    const jitterX = (Math.random() - 0.5) * (CELL_W - BTN_W - 20);
    const jitterY = (Math.random() - 0.5) * (rowHeight * 0.35);

    const left = EDGE_PAD + col * CELL_W + jitterX;
    const top = row * rowHeight + rowHeight / 2 - 40 + jitterY;

    const btn = document.getElementById(`btn-${source.id}`);
    btn.style.left = `${left}px`;
    btn.style.top = `${Math.max(10, top)}px`;
  });
}

DATA_SOURCES.forEach(source => {
  const btn = document.createElement("button");
  btn.className = "data-btn";
  btn.id = `btn-${source.id}`;
  btn.innerHTML = `<span class="tag">${source.tag}</span><span class="label">${source.label}</span>`;
  btn.addEventListener("click", () => {
    if (dragMoved) return; // a drag just ended on top of this button — not a click
    collectData(source, btn);
  });
  world.appendChild(btn);
});

layoutButtons();

/* --- Drag-to-pan navigation --- */
let currentX = 0;
let isPanning = false;
let dragMoved = false;
let startPointerX = 0;
let startX = 0;
let minTranslate = 0;

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function applyTransform() {
  world.style.transform = `translateX(${currentX}px)`;

  const viewportW = viewport.clientWidth;
  const mapW = minimap.clientWidth;
  const thumbW = Math.max(24, (viewportW / worldWidth) * mapW);
  const scrollRatio = minTranslate === 0 ? 0 : currentX / minTranslate;
  const thumbLeft = scrollRatio * (mapW - thumbW);
  minimapThumb.style.width = `${thumbW}px`;
  minimapThumb.style.left = `${thumbLeft}px`;
}

function onPointerDown(e) {
  isPanning = true;
  dragMoved = false;
  startPointerX = e.clientX;
  startX = currentX;
  viewport.classList.add("panning");
}

function onPointerMove(e) {
  if (!isPanning) return;
  const dx = e.clientX - startPointerX;
  if (Math.abs(dx) > 5) {
    dragMoved = true;
    panHint.style.opacity = "0";
  }
  currentX = clamp(startX + dx, minTranslate, 0);
  applyTransform();
}

function onPointerUp() {
  isPanning = false;
  viewport.classList.remove("panning");
}

viewport.addEventListener("pointerdown", onPointerDown);
window.addEventListener("pointermove", onPointerMove);
window.addEventListener("pointerup", onPointerUp);
window.addEventListener("pointercancel", onPointerUp);

window.addEventListener("keydown", (e) => {
  const step = 120;
  if (e.key === "ArrowRight") { currentX = clamp(currentX - step, minTranslate, 0); applyTransform(); panHint.style.opacity = "0"; }
  if (e.key === "ArrowLeft")  { currentX = clamp(currentX + step, minTranslate, 0); applyTransform(); panHint.style.opacity = "0"; }
});

function handleResize() {
  minTranslate = Math.min(0, viewport.clientWidth - worldWidth);
  currentX = clamp(currentX, minTranslate, 0);
  layoutButtons();
  applyTransform();
}
window.addEventListener("resize", handleResize);
handleResize();

placementOrder.forEach((sourceIndex) => {
  const source = DATA_SOURCES[sourceIndex];
  const btn = document.getElementById(`btn-${source.id}`);
  const dot = document.createElement("div");
  dot.className = "minimap-dot";
  dot.id = `dot-${source.id}`;
  const ratio = (parseFloat(btn.style.left) + BTN_W / 2) / worldWidth;
  dot.style.left = `${ratio * 100}%`;
  minimap.appendChild(dot);
});

function collectData(source, btn) {
  btn.disabled = true;
  collectedData.push(source);
  selection[source.id] = true; // included for training by default

  const dot = document.getElementById(`dot-${source.id}`);
  if (dot) dot.classList.add("done");

  if (emptyHint.parentNode) emptyHint.remove();

  const li = document.createElement("li");
  li.textContent = source.label;
  listEl.appendChild(li);
  listEl.scrollTop = listEl.scrollHeight;

  countEl.textContent = `${collectedData.length} / ${DATA_SOURCES.length}`;
  fillEl.style.width = `${(collectedData.length / DATA_SOURCES.length) * 100}%`;

  if (collectedData.length === DATA_SOURCES.length) {
    statusLine.textContent = "All sources collected — proceed to cleaning.";
    toCleanBtn.disabled = false;
  } else {
    statusLine.textContent = `${DATA_SOURCES.length - collectedData.length} source(s) still out there.`;
  }
}

toCleanBtn.addEventListener("click", () => goToStage("clean"));

/* ============================================================
   4. PROCESS 2 — CLEAN THE DATA
   Every collected source starts included; the player toggles
   off anything that looks redundant or irrelevant.
   ============================================================ */
const cleanGrid    = document.getElementById("clean-grid");
const cleanSummary = document.getElementById("clean-summary");
const toTrainBtn   = document.getElementById("to-train-btn");

function renderCleanStage() {
  cleanGrid.innerHTML = "";

  collectedData.forEach(source => {
    const card = document.createElement("button");
    card.className = "clean-card" + (selection[source.id] ? "" : " excluded");
    card.innerHTML = `
      <span class="tag">${source.tag}</span>
      <span class="label">${source.label}</span>
      <span class="state">${selection[source.id] ? "✓ included" : "✕ excluded"}</span>
    `;
    card.addEventListener("click", () => {
      selection[source.id] = !selection[source.id];
      card.classList.toggle("excluded", !selection[source.id]);
      card.querySelector(".state").textContent = selection[source.id] ? "✓ included" : "✕ excluded";
      updateCleanSummary();
    });
    cleanGrid.appendChild(card);
  });

  updateCleanSummary();
}

function updateCleanSummary() {
  const total = collectedData.length;
  const included = collectedData.filter(s => selection[s.id]).length;
  cleanSummary.textContent = `${included} of ${total} collected sources currently included for training.`;
  toTrainBtn.disabled = included === 0;
}

toTrainBtn.addEventListener("click", () => goToStage("train"));

/* ============================================================
   5. PROCESS 3 — TRAIN THE AI
   ============================================================ */
const trainSummary   = document.getElementById("train-summary");
const trainBtn       = document.getElementById("train-btn");
const trainResult    = document.getElementById("train-result");
const toImplementBtn = document.getElementById("to-implement-btn");

function renderTrainStage() {
  const included = collectedData.filter(s => selection[s.id]).length;
  trainSummary.textContent = `Training will use ${included} selected source${included === 1 ? "" : "s"} from your cleaned dataset.`;
}

trainBtn.addEventListener("click", () => {
  const selectedSources = collectedData.filter(s => selection[s.id]);
  const cleanSelected = selectedSources.filter(s => s.clean).length;
  const noisySelected = selectedSources.length - cleanSelected;
  const totalClean = DATA_SOURCES.filter(s => s.clean).length;

  const coverage = cleanSelected / totalClean;
  const penalty = noisySelected * 0.15;
  const accuracy = Math.round(Math.max(0, Math.min(1, coverage - penalty)) * 100);
  lastAccuracy = accuracy;

  let message;
  if (noisySelected === 0 && coverage === 1) {
    message = "Clean, complete dataset — the model trained on exactly the right sources.";
  } else if (accuracy >= 80) {
    message = "Solid result. A little more cleanup could push this higher.";
  } else if (accuracy >= 50) {
    message = "The model is noisy — some redundant or irrelevant sources are dragging it down.";
  } else {
    message = "The model can't find a reliable signal — too much bad data, or too little good data, went in.";
  }

  trainResult.innerHTML = `
    <div class="accuracy-line"><span>MODEL ACCURACY</span><span>${accuracy}%</span></div>
    <div class="accuracy-track"><div class="accuracy-fill" style="width:${accuracy}%"></div></div>
    <div class="accuracy-msg">${message}</div>
  `;

  toImplementBtn.disabled = false;
});

toImplementBtn.addEventListener("click", () => goToStage("implement"));

/* ============================================================
   6. PROCESS 4 — IMPLEMENT
   The narrative payoff: what the trained AI is actually able
   to do about the monsoon flooding, scaled to how good the
   training run was.
   ============================================================ */
const implementContent = document.getElementById("implement-content");
const backToCleanBtn   = document.getElementById("back-to-clean-btn");
const restartBtn       = document.getElementById("restart-btn");

function renderImplementStage() {
  const accuracy = lastAccuracy ?? 0;
  let tagClass, tagText, heading, body;

  if (accuracy >= 80) {
    tagClass = "good";
    tagText = "Deployed";
    heading = "Early-warning system live";
    body = "With clean, well-covered training data, your model reliably flags high-risk zones along flood-prone rivers and issues warnings well ahead of peak monsoon rainfall — giving communities real time to evacuate before the water rises.";
  } else if (accuracy >= 50) {
    tagClass = "mid";
    tagText = "Partially deployed";
    heading = "Warnings issued, but not fully trusted";
    body = "The model can flag likely flood zones, but noisy or incomplete training data limits its lead time and confidence. Some warnings arrive late, and false alarms start to erode trust. Going back to clean the dataset further would help.";
  } else {
    tagClass = "bad";
    tagText = "Not deployable";
    heading = "Predictions aren't reliable enough";
    body = "Too much irrelevant or redundant data made it into training, or too little of the genuinely useful data did. The model can't be trusted to warn communities in time. Head back to Process 2 and clean the dataset before retraining.";
  }

  implementContent.innerHTML = `
    <span class="result-tag ${tagClass}">${tagText}</span>
    <h3>${heading}</h3>
    <p>${body}</p>
    <div class="recap">FINAL MODEL ACCURACY: ${accuracy}%</div>
  `;
}

backToCleanBtn.addEventListener("click", () => goToStage("clean"));
restartBtn.addEventListener("click", () => window.location.reload());

/* ============================================================
   7. WEBGL BACKGROUND — animated water surface
   Runs continuously behind every stage. A subtle parallax
   offset (tied to how far the player has panned in Process 1)
   makes it feel layered rather than static.
   ============================================================ */
const canvas = document.getElementById("bg-canvas");
const gl = canvas.getContext("webgl");

if (!gl) {
  console.warn("WebGL not supported; using flat background.");
} else {
  const vertexSrc = `
    attribute vec2 aPosition;
    void main() {
      gl_Position = vec4(aPosition, 0.0, 1.0);
    }
  `;

  const fragmentSrc = `
    precision mediump float;
    uniform vec2 uResolution;
    uniform float uTime;
    uniform float uOffset;

    float waterHeight(vec2 uv, float t) {
      float v = 0.0;
      v += sin(uv.x * 9.0  + t * 0.6) * 0.5 + 0.5;
      v += sin(uv.x * 21.0 - t * 0.9 + uv.y * 5.0) * 0.5 + 0.5;
      v += sin((uv.x + uv.y) * 15.0 + t * 1.2) * 0.5 + 0.5;
      return v / 3.0;
    }

    void main() {
      vec2 uv = gl_FragCoord.xy / uResolution.xy;
      uv.x += uOffset;

      float waves = waterHeight(uv, uTime);

      vec3 deep  = vec3(0.027, 0.086, 0.125);
      vec3 mid   = vec3(0.055, 0.16, 0.21);
      vec3 glow  = vec3(0.30, 0.85, 0.88);

      vec3 color = mix(deep, mid, uv.y);
      color = mix(color, glow, waves * 0.10);

      float vignette = smoothstep(1.1, 0.25, distance(uv, vec2(0.5)));
      color *= mix(0.65, 1.0, vignette);

      gl_FragColor = vec4(color, 1.0);
    }
  `;

  function compileShader(type, src) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error(gl.getShaderInfoLog(shader));
    }
    return shader;
  }

  const program = gl.createProgram();
  gl.attachShader(program, compileShader(gl.VERTEX_SHADER, vertexSrc));
  gl.attachShader(program, compileShader(gl.FRAGMENT_SHADER, fragmentSrc));
  gl.linkProgram(program);
  gl.useProgram(program);

  const positions = new Float32Array([
    -1, -1,
     3, -1,
    -1,  3,
  ]);
  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

  const aPosition = gl.getAttribLocation(program, "aPosition");
  gl.enableVertexAttribArray(aPosition);
  gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);

  const uResolution = gl.getUniformLocation(program, "uResolution");
  const uTime = gl.getUniformLocation(program, "uTime");
  const uOffset = gl.getUniformLocation(program, "uOffset");

  function resizeGL() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
  }
  window.addEventListener("resize", resizeGL);
  resizeGL();

  function render(t) {
    gl.uniform2f(uResolution, canvas.width, canvas.height);
    gl.uniform1f(uTime, t * 0.001);
    gl.uniform1f(uOffset, (-currentX / worldWidth) * 0.15);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}
