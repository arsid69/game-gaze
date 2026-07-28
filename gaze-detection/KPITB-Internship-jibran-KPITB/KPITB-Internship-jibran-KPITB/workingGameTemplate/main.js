/* ============================================================
   1. DATA SOURCES
   `clean: false` marks redundant/irrelevant entries — not used
   in Stage 1, but kept here so Stage 2 (cleaning) can filter
   on this same array without redefining the data.
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

/* Holds whatever the player has collected so far. */
const collectedData = [];

/* Tracks which collected sources are currently included for
   training, keyed by source id. Defaults to true on collection —
   the player then excludes anything that looks redundant or
   irrelevant before hitting "Train Model". */
const selection = {};

/* ============================================================
   2. UI ELEMENT REFS
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
const trainBtn    = document.getElementById("train-btn");
const trainResult = document.getElementById("train-result");

countEl.textContent = `0 / ${DATA_SOURCES.length}`;

/* ============================================================
   3. SCATTER LAYOUT
   Buttons are placed across a world much wider than the
   viewport, in a jittered 3-row grid so nothing overlaps but
   nothing lines up too neatly either — the player has to look.
   ============================================================ */
const ROWS = 3;
const CELL_W = 260;
const BTN_W = 190;
const EDGE_PAD = 60;

// Shuffle placement order (not the underlying data) so buttons
// aren't just scattered in the same order they're defined.
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

/* ============================================================
   4. DRAG-TO-PAN NAVIGATION
   ============================================================ */
let currentX = 0;      // current translateX of #world
let isPanning = false;
let dragMoved = false;
let startPointerX = 0;
let startX = 0;
let minTranslate = 0;  // computed on resize

function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

function applyTransform() {
  world.style.transform = `translateX(${currentX}px)`;

  // Update minimap thumb position/size
  const viewportW = viewport.clientWidth;
  const mapW = minimap.clientWidth;
  const thumbW = Math.max(24, (viewportW / worldWidth) * mapW);
  const scrollRatio = minTranslate === 0 ? 0 : currentX / minTranslate; // 0..1
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

// Keyboard support: left/right arrows nudge the view
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

/* Minimap dots — one per data source, positioned by its
   button's location in the world, filled in green once
   collected. */
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

/* ============================================================
   5. DATA COLLECTION
   ============================================================ */
function collectData(source, btn) {
  btn.disabled = true;
  collectedData.push(source);
  selection[source.id] = true; // included for training by default

  const dot = document.getElementById(`dot-${source.id}`);
  if (dot) dot.classList.add("done");

  if (emptyHint.parentNode) emptyHint.remove();

  const li = document.createElement("li");
  li.textContent = source.label;
  li.dataset.id = source.id;
  li.title = "Click to include/exclude from training";
  li.addEventListener("click", () => toggleSelection(source.id, li));
  listEl.appendChild(li);
  listEl.scrollTop = listEl.scrollHeight;

  countEl.textContent = `${collectedData.length} / ${DATA_SOURCES.length}`;
  fillEl.style.width = `${(collectedData.length / DATA_SOURCES.length) * 100}%`;

  updateSelectionSummary();
}

/* ============================================================
   6. TRAINING SELECTION
   The player picks which collected sources to actually train
   with. Excluding noisy/duplicate/irrelevant sources (clean:
   false) and including the real ones (clean: true) yields a
   higher accuracy score.
   ============================================================ */
function toggleSelection(id, li) {
  selection[id] = !selection[id];
  li.classList.toggle("excluded", !selection[id]);
  updateSelectionSummary();
}

function updateSelectionSummary() {
  const selectedCount = Object.values(selection).filter(Boolean).length;
  trainBtn.disabled = selectedCount === 0;

  if (selectedCount === 0) {
    statusLine.textContent = collectedData.length
      ? "Select at least one source to train the model."
      : "";
  } else {
    statusLine.textContent = `${selectedCount} of ${collectedData.length} collected source${collectedData.length === 1 ? "" : "s"} selected for training.`;
  }
}

trainBtn.addEventListener("click", () => {
  const selectedSources = collectedData.filter(s => selection[s.id]);
  const cleanSelected = selectedSources.filter(s => s.clean).length;
  const noisySelected = selectedSources.length - cleanSelected;
  const totalClean = DATA_SOURCES.filter(s => s.clean).length;

  // Coverage of the genuinely useful data, penalized for every
  // noisy/irrelevant source that made it into the training set.
  const coverage = cleanSelected / totalClean;
  const penalty = noisySelected * 0.15;
  const accuracy = Math.round(Math.max(0, Math.min(1, coverage - penalty)) * 100);

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
});

/* ============================================================
   7. WEBGL BACKGROUND — animated water surface
   A subtle parallax offset (tied to how far the player has
   panned) makes the backdrop feel like it sits behind the
   explorable field rather than static wallpaper.
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
    // Small fraction of the pan offset, so the backdrop drifts
    // slower than the foreground field (parallax depth cue).
    gl.uniform1f(uOffset, (-currentX / worldWidth) * 0.15);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);
}
