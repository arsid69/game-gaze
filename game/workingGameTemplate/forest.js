/* ============================================================
   DATA FOREST — Phase 3 (Step 3: the game)
   A low-poly 3D forest built from the kennye'sMiniforest GLB
   pack. The forest is scenery; the GAME is collecting glowing
   data-source orbs by GAZE (look + hold). The archer reacts,
   and a win screen shows when the full dataset is gathered.
   Gaze reuses the Phase 2 server (ws://localhost:8765).
   Keys: f fullscreen · c recenter · g gaze on/off · drag = orbit
   ============================================================ */
import * as THREE from "three";
import { GLTFLoader } from "./vendor/three/GLTFLoader.js";
import { OrbitControls } from "./vendor/three/OrbitControls.js";

const canvas = document.getElementById("app");
const loadFill = document.getElementById("load-fill");
const loadingEl = document.getElementById("loading");

// --- Renderer --------------------------------------------------------
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;

// --- Scene / atmosphere ----------------------------------------------
const SKY = 0x070b18;   // night
const scene = new THREE.Scene();
scene.background = new THREE.Color(SKY);
scene.fog = new THREE.Fog(SKY, 34, 80);

// --- Camera: LATERAL PAN (side-scroller) — slides left/right, no rotation --
// The view translates along X across a wide forest strip; the look direction
// stays parallel, so it reads as "scrolling the window", not orbiting.
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 300);
const STRIP = 30;                                   // half-width of the forest strip (bigger map)
const CAM_HEIGHT = 6, CAM_Z = 13, LOOK_Y = 2.4;
const CAM_EDGE = 0.16, CAM_PAN_SPEED = 12;          // gaze edge zone + pan speed (units/s)
let camX = 0;
function applyCamera() {
  camX = Math.min(STRIP, Math.max(-STRIP, camX));   // inline clamp (clamp const is defined later)
  camera.position.set(camX, CAM_HEIGHT, CAM_Z);
  camera.lookAt(camX, LOOK_Y, 0);                   // target rides with camX → pure sideways pan
}
applyCamera();

// Mouse drag also pans horizontally (fallback).
let camDragging = false, camLastX = 0;
addEventListener("pointerdown", (e) => { camDragging = true; camLastX = e.clientX; });
addEventListener("pointerup", () => { camDragging = false; });
addEventListener("pointermove", (e) => { if (camDragging) { camX -= (e.clientX - camLastX) * 0.03; camLastX = e.clientX; } });

// Gaze at the left/right edge slides the view to find orbs (unless one is targeted).
function updateCamera(dt) {
  if (gazeOn && faceOk && !hovered) {
    const gx = clamp(rawX - drift()[0], 0, 1);   // gesture pointer takes no gaze offset
    if (gx < CAM_EDGE) camX -= CAM_PAN_SPEED * dt * ((CAM_EDGE - gx) / CAM_EDGE);
    else if (gx > 1 - CAM_EDGE) camX += CAM_PAN_SPEED * dt * ((gx - (1 - CAM_EDGE)) / CAM_EDGE);
  }
  applyCamera();
}

// --- Lighting --------------------------------------------------------
scene.add(new THREE.HemisphereLight(0x3a4a70, 0x0c1018, 0.85));   // cool night ambient
const sun = new THREE.DirectionalLight(0xcdd8ff, 1.7);            // cool moonlight
sun.position.set(-16, 30, 10);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 1; sun.shadow.camera.far = 60;
sun.shadow.camera.left = -52; sun.shadow.camera.right = 52;
sun.shadow.camera.top = 30; sun.shadow.camera.bottom = -30;
sun.shadow.bias = -0.0004;
scene.add(sun);

// --- Ground ----------------------------------------------------------
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(150, 70),
  new THREE.MeshStandardMaterial({ color: 0x4f7a44, roughness: 1, metalness: 0 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// --- Night sky: starfield + moon -------------------------------------
{
  const N = 1400, pos = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const r = 90 + Math.random() * 110, th = Math.random() * Math.PI * 2, ph = Math.random() * Math.PI * 0.6;
    pos[i * 3]     = r * Math.sin(ph) * Math.cos(th);
    pos[i * 3 + 1] = r * Math.cos(ph) + 8;               // mostly overhead
    pos[i * 3 + 2] = r * Math.sin(ph) * Math.sin(th) - 30; // biased behind the scene
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({
    color: 0xcfe0ff, size: 1.7, sizeAttenuation: false, fog: false, transparent: true, opacity: 0.9,
  })));

  const moon = new THREE.Mesh(new THREE.SphereGeometry(5, 32, 32),
    new THREE.MeshBasicMaterial({ color: 0xeaf0ff, fog: false }));
  moon.position.set(-46, 58, -95);
  scene.add(moon);
  const moonGlow = new THREE.Mesh(new THREE.SphereGeometry(8.5, 32, 32),
    new THREE.MeshBasicMaterial({ color: 0x9fb6ff, transparent: true, opacity: 0.15, fog: false }));
  moonGlow.position.copy(moon.position);
  scene.add(moonGlow);
}

// --- Helpers ---------------------------------------------------------
const rand = (a, b) => a + Math.random() * (b - a);
const TAU = Math.PI * 2;
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const lerp = (a, b, t) => a + (b - a) * t;

// Scenery props are decorative (NOT gaze-selectable). Only data orbs are.
function place(gltf, x, z, { ry = 0, s = 1, y = 0 } = {}) {
  const o = gltf.scene.clone(true);
  o.position.set(x, y, z);
  o.rotation.y = ry;
  o.scale.setScalar(s);
  o.traverse((n) => { if (n.isMesh) { n.castShadow = true; n.receiveShadow = true; } });
  scene.add(o);
  return o;
}

// --- Model loading ---------------------------------------------------
const manager = new THREE.LoadingManager();
manager.onProgress = (url, loaded, total) => { loadFill.style.width = (loaded / total * 100) + "%"; };
manager.onLoad = () => { loadingEl.classList.add("done"); };
const loader = new GLTFLoader(manager);
const NAMES = [
  "patch-grass", "patch-dirt", "tree", "tree-high", "plant",
  "rocks-high", "rocks-low", "rocks-ramp", "stones", "tent",
  "bridge", "platform", "building-structure", "building-roof",
  "building-platform", "flag", "fence", "ladder", "target", "character-archer",
];
const load = (name) => new Promise((res, rej) =>
  loader.load(`./models/${name}.glb`, (g) => res(g), undefined, rej));

// --- Archer state (rigged, 32 clips) ---------------------------------
let mixer = null, archerClips = {}, archerIdleAction = null, archerObj = null;

(async function build() {
  const loaded = await Promise.all(NAMES.map(load));
  const M = {};
  NAMES.forEach((n, i) => (M[n] = loaded[i]));

  // Dense forest — many clones of the same assets across the wider strip.
  const trees = [M["tree"], M["tree-high"]];
  for (let i = 0; i < 62; i++) {
    const z = i < 42 ? rand(-22, -9) : rand(-9, -3);   // back rows + midground
    place(trees[i % 2], rand(-STRIP - 10, STRIP + 10), z, { ry: rand(0, TAU), s: rand(0.9, 1.8) });
  }
  const rocks = [M["rocks-high"], M["rocks-low"], M["stones"], M["rocks-ramp"]];
  for (let i = 0; i < 28; i++) place(rocks[i % rocks.length], rand(-STRIP - 6, STRIP + 6), rand(-8, 6), { ry: rand(0, TAU), s: rand(0.7, 1.3) });
  for (let i = 0; i < 36; i++) place(Math.random() < 0.5 ? M["plant"] : M["patch-grass"], rand(-STRIP - 6, STRIP + 6), rand(-7, 6), { ry: rand(0, TAU), s: rand(0.7, 1.3) });
  // Landmarks (a few clones each) scattered along the strip
  place(M["tent"], -25, -9, { ry: 0.5 });
  place(M["tent"], 21, -11, { ry: -0.7, s: 0.9 });
  place(M["platform"], 12, -10, { ry: -0.3 });
  place(M["building-structure"], 12, -10.1, { ry: -0.3 });
  place(M["building-roof"], 12, -10.1, { ry: -0.3 });
  place(M["building-structure"], -13, -12, { ry: 0.4, s: 0.9 });
  place(M["building-roof"], -13, -12, { ry: 0.4, s: 0.9 });
  place(M["flag"], 4, -7, { ry: 1.2, s: 1.1 });
  place(M["flag"], -19, -8, { ry: -0.6 });
  place(M["bridge"], -6, -12, { ry: 0 });
  place(M["bridge"], 27, -13, { ry: 0.2 });

  // Archer — centre stage, playing idle
  const archer = M["character-archer"];
  archer.scene.position.set(0, 0, 0);
  archer.scene.rotation.y = Math.PI;
  archer.scene.scale.setScalar(1.15);
  archer.scene.traverse((n) => { if (n.isMesh) { n.castShadow = true; n.receiveShadow = true; } });
  scene.add(archer.scene);
  archerObj = archer.scene;
  mixer = new THREE.AnimationMixer(archer.scene);
  (archer.animations || []).forEach((c) => { archerClips[c.name] = c; });
  archerIdleAction = mixer.clipAction(archerClips["idle"] || archer.animations[0]);
  archerIdleAction.play();

  createDataNodes();   // the collectibles
})().catch((e) => {
  console.error("Scene build failed:", e);
  document.querySelector("#loading .lbl").textContent = "Load error — see console";
});

/* ============================================================
   DATA SOURCES — the game objects (glowing orbs to collect)
   ============================================================ */
const DATA = [
  { id: "rain",   label: "Rainfall Gauge",         color: 0x5fe0e8, clean: true },
  { id: "river",  label: "River Level",            color: 0x5ab6ff, clean: true },
  { id: "sat",    label: "Satellite Imagery",      color: 0x7fe6a4, clean: true },
  { id: "dem",    label: "Elevation Model",        color: 0xe8cf72, clean: true },
  { id: "soil",   label: "Soil Moisture",          color: 0xd39457, clean: true },
  { id: "tweet",  label: "Celebrity Weather Tweet", color: 0xc06bd0, clean: false },
  { id: "backup", label: "Corrupted Old Backup",    color: 0x9a6b52, clean: false },
  { id: "dupe",   label: "Duplicate Rain Log",      color: 0x6a8fd0, clean: false },
];
const dataNodes = [];
let collected = 0, gameWon = false;

const labelWrap = document.createElement("div");
labelWrap.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:40";
document.body.appendChild(labelWrap);

// --- Juice: particle burst on collect --------------------------------
const particles = [];
const PARTICLE_GEO = new THREE.OctahedronGeometry(0.07, 0);
function spawnBurst(pos, colorHex) {
  const color = new THREE.Color(colorHex);
  for (let i = 0; i < 18; i++) {
    const m = new THREE.Mesh(PARTICLE_GEO, new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 1.6 }));
    m.position.copy(pos);
    m.userData.vel = new THREE.Vector3(rand(-1, 1), rand(0.3, 1.5), rand(-1, 1)).normalize().multiplyScalar(rand(2, 4.6));
    m.userData.life = 1.0;
    scene.add(m); particles.push(m);
  }
}
function updateParticles(dt) {
  for (let i = particles.length - 1; i >= 0; i--) {
    const p = particles[i];
    p.userData.life -= dt / 0.8;
    if (p.userData.life <= 0) { scene.remove(p); p.material.dispose(); particles.splice(i, 1); continue; }
    p.userData.vel.y -= dt * 6;
    p.position.addScaledVector(p.userData.vel, dt);
    p.scale.setScalar(Math.max(0.001, p.userData.life));
    p.rotation.y += dt * 8;
  }
}

// --- Juice: synthesized sound (Web Audio, CSP-safe, no files) ---------
let audioCtx = null;
function ensureAudio() {
  if (!audioCtx) { try { audioCtx = new (window.AudioContext || window.webkitAudioContext)(); } catch {} }
  if (audioCtx && audioCtx.state === "suspended") audioCtx.resume();
}
function beep(freq, t0, dur, type, vol, slideTo) {
  if (!audioCtx) return;
  const o = audioCtx.createOscillator(), g = audioCtx.createGain();
  o.type = type; o.frequency.setValueAtTime(freq, t0);
  if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, t0 + dur);
  g.gain.setValueAtTime(0, t0);
  g.gain.linearRampToValueAtTime(vol, t0 + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(audioCtx.destination); o.start(t0); o.stop(t0 + dur + 0.02);
}
function playPop() { if (audioCtx) beep(680, audioCtx.currentTime, 0.14, "triangle", 0.2, 240); }
function playWin() { if (audioCtx) [523, 659, 784, 1047].forEach((f, i) => beep(f, audioCtx.currentTime + i * 0.12, 0.24, "triangle", 0.16)); }

// --- Win-screen gaze-selectable buttons (2D) -------------------------
const domTargets = [];
let domHovered = null, domDwellStart = 0, domArmed = true, domFireAt = 0;
const btnStyle = document.createElement("style");
btnStyle.textContent = `
  .gz-btn{font:600 15px ui-monospace,Consolas,monospace;color:#eafeff;background:rgba(79,216,224,.14);
    border:1px solid rgba(120,240,245,.5);border-radius:10px;padding:13px 24px;margin:8px;cursor:pointer;pointer-events:auto;transition:.12s}
  .gz-btn:hover{background:rgba(79,216,224,.24)}
  .gz-btn.gz-hover{outline:3px solid rgba(120,240,245,.95);outline-offset:3px;box-shadow:0 0 18px rgba(110,240,245,.6)}`;
document.head.appendChild(btnStyle);
function gazeButton(text, onClick) { const b = document.createElement("button"); b.textContent = text; b.className = "gz-btn"; b.onclick = onClick; return b; }

function createDataNodes() {
  // Randomly scattered across the strip (with a little spacing so none overlap),
  // so the player scrolls left/right to hunt them down — different every game.
  const used = [];
  DATA.forEach((d) => {
    let x, tries = 0;
    do { x = rand(-STRIP + 3, STRIP - 3); tries++; }
    while (tries < 40 && used.some((u) => Math.abs(u - x) < 4.5));
    used.push(x);
    createDataNode(d, x, rand(1.5, 3), rand(-4, 4));
  });
}

function createDataNode(d, x, y, z) {
  const mat = new THREE.MeshStandardMaterial({
    color: d.color, emissive: d.color, emissiveIntensity: 0.6,
    roughness: 0.25, metalness: 0.1, flatShading: true,
  });
  const orb = new THREE.Mesh(new THREE.IcosahedronGeometry(0.42, 0), mat);
  orb.position.set(x, y, z);
  orb.castShadow = true;
  orb.userData = {
    selectable: true, isData: true, data: d, collected: false, dying: 0,
    baseY: y, baseScale: 1, k: 0, phase: Math.random() * TAU,
  };
  orb.add(new THREE.PointLight(d.color, 5, 4.5));
  scene.add(orb);

  const el = document.createElement("div");
  el.textContent = d.label;
  el.style.cssText = "position:absolute;transform:translate(-50%,-150%);font:600 12px ui-monospace,Consolas,monospace;"
    + "color:#eafeff;background:rgba(6,20,24,.72);border:1px solid rgba(120,240,245,.35);border-radius:7px;"
    + "padding:3px 8px;white-space:nowrap;transition:opacity .15s,border-color .15s,transform .15s";
  labelWrap.appendChild(el);
  orb.userData.label = el;

  dataNodes.push(orb);
  selectables.push(orb);
}

function collectNode(orb) {
  if (orb.userData.collected) return;
  orb.userData.collected = true;
  orb.userData.dying = 1.0;
  collected++;
  orb.userData.label.style.opacity = "0";
  const i = selectables.indexOf(orb);
  if (i >= 0) selectables.splice(i, 1);   // no longer targetable
  if (hovered === orb) hovered = null;
  faceArcherTo(orb.position);
  archerPlay("pick-up");
  spawnBurst(orb.position, orb.userData.data.color);
  playPop();
  chip("connected");
  if (collected >= DATA.length && !gameWon) {
    if (window.__questOnComplete) { gameWon = true; window.__questOnComplete(); }
    else winGame();
  }
}

function faceArcherTo(pos) {
  if (!archerObj) return;
  const a = Math.atan2(pos.x - archerObj.position.x, pos.z - archerObj.position.z);
  archerObj.rotation.y = a;   // Kenney character faces +Z
}

let winOverlay = null, winInner = null;
function winGame() {
  gameWon = true;
  if (archerObj) archerObj.rotation.y = Math.PI;
  archerPlay("emote-yes");
  playWin();
  winOverlay = document.createElement("div");
  winOverlay.style.cssText = "position:fixed;inset:0;display:grid;place-items:center;z-index:70;"
    + "background:rgba(6,16,20,.58);backdrop-filter:blur(3px);text-align:center;"
    + "font-family:ui-monospace,Consolas,monospace;color:#eafeff";
  winInner = document.createElement("div");
  winOverlay.appendChild(winInner);
  document.body.appendChild(winOverlay);
  showWinDataset();
}
function showWinDataset() {
  winInner.innerHTML = `<div style="font-size:13px;letter-spacing:.24em;color:#6ff0f5">DATASET COMPLETE</div>
    <div style="font-size:clamp(24px,4vw,38px);font-weight:700;margin:10px 0">All ${DATA.length} data sources collected 🎉</div>
    <div style="color:#9fc4cc;max-width:46ch;margin:0 auto 8px">Clean, complete dataset — train the model to deploy the flood early-warning system.</div>`;
  const b = gazeButton("Train the Model", trainModel);
  winInner.appendChild(b);
  domTargets.length = 0; domTargets.push(b); domHovered = null;
}
function trainModel() {
  const acc = 100;
  winInner.innerHTML = `<div style="font-size:13px;letter-spacing:.24em;color:#6ff0f5">MODEL TRAINED</div>
    <div style="font-size:clamp(22px,4vw,34px);font-weight:700;margin:10px 0">Accuracy ${acc}%</div>
    <div style="width:280px;height:10px;border-radius:6px;background:rgba(255,255,255,.12);overflow:hidden;margin:0 auto 12px"><div style="height:100%;width:${acc}%;background:linear-gradient(90deg,#4fd8e0,#7ff0a0)"></div></div>
    <div style="color:#9fc4cc;max-width:48ch;margin:0 auto">Early-warning system deployed — communities get real time to evacuate before the water rises.</div>`;
  const b = gazeButton("Play Again", playAgain);
  winInner.appendChild(b);
  domTargets.length = 0; domTargets.push(b); domHovered = null;
  playWin();
}
function playAgain() {
  if (winOverlay) { winOverlay.remove(); winOverlay = null; }
  domTargets.length = 0; domHovered = null;
  gameWon = false; collected = 0; hovered = null;
  for (const orb of dataNodes) { scene.remove(orb); orb.userData.label.remove(); }
  dataNodes.length = 0; selectables.length = 0;
  if (archerObj) archerObj.rotation.y = Math.PI;
  createDataNodes();
  chip("connected");
}

// Crossfade the archer into a one-shot clip, then back to idle.
function archerPlay(name) {
  if (!mixer || !archerClips[name] || !archerIdleAction) return;
  const action = mixer.clipAction(archerClips[name]);
  action.reset(); action.setLoop(THREE.LoopOnce, 1); action.clampWhenFinished = true;
  archerIdleAction.fadeOut(0.15);
  action.fadeIn(0.15).play();
  const onFin = (e) => {
    if (e.action !== action) return;
    mixer.removeEventListener("finished", onFin);
    action.fadeOut(0.25);
    archerIdleAction.reset().fadeIn(0.25).play();
  };
  mixer.addEventListener("finished", onFin);
}

function updateDataNodes(dt) {
  for (const orb of dataNodes) {
    // collect animation: spin, float up, shrink out
    if (orb.userData.dying > 0) {
      orb.userData.dying = Math.max(0, orb.userData.dying - dt / 0.5);
      const t = orb.userData.dying;
      orb.scale.setScalar(orb.userData.baseScale * t * t);
      orb.rotation.y += dt * 16;
      orb.position.y = orb.userData.baseY + (1 - t) * 1.8;
      orb.material.emissiveIntensity = 0.6 + (1 - t) * 2.5;
      if (t <= 0) orb.visible = false;
      continue;
    }
    if (orb.userData.collected) continue;

    orb.userData.k = lerp(orb.userData.k, orb === hovered ? 1 : 0, clamp(dt * 12, 0, 1));
    orb.userData.phase += dt;
    orb.rotation.y += dt * 0.7;
    const k = orb.userData.k;
    orb.position.y = orb.userData.baseY + Math.sin(orb.userData.phase * 1.6) * 0.12 + k * 0.18;
    orb.scale.setScalar(orb.userData.baseScale * (1 + k * 0.14));
    orb.material.emissiveIntensity = 0.55 + k * 0.9 + Math.sin(orb.userData.phase * 3) * 0.05;

    const p = projScreen(orb.position);
    const el = orb.userData.label;
    if (p.behind) { el.style.opacity = "0"; }
    else {
      el.style.left = p.x + "px";
      el.style.top = p.y + "px";
      el.style.opacity = "1";
      el.style.transform = `translate(-50%,-150%) scale(${1 + k * 0.12})`;
      el.style.borderColor = orb === hovered ? "rgba(120,240,245,.95)" : "rgba(120,240,245,.35)";
    }
  }
}

/* ============================================================
   GAZE CONTROL IN 3D — raycast + aim-assist + dwell
   ============================================================ */
const DWELL_MS = 1000;   // socket URL now lives in input-manager.js, the only WS client
const ASSIST_RADIUS = 120;   // px — forgiving snap to the nearest orb on screen
const HYSTERESIS = 55;       // px — sticky current target
const MAGNET = 0.5;          // cursor pull onto the locked orb

const raycaster = new THREE.Raycaster();
const selectables = [];
let rawX = 0.5, rawY = 0.5, driftX = 0, driftY = 0, faceOk = false, gazeOn = true;
let hovered = null, dwellStart = 0, armed = true;
let recentering = false, recenterAt = 0;
let collectEnabled = true;   // quest.js gates collection during story/dialog phases

// Cursor + charge ring
const RR = 16, C2 = 2 * Math.PI * RR;
const gzCur = document.createElement("div");
gzCur.style.cssText = "position:fixed;top:0;left:0;width:36px;height:36px;margin:-18px 0 0 -18px;z-index:50;pointer-events:none;opacity:0;transition:opacity .15s";
gzCur.innerHTML = `<svg width="36" height="36" viewBox="0 0 36 36">
  <circle cx="18" cy="18" r="${RR}" fill="none" stroke="rgba(120,240,245,.4)" stroke-width="2.5"></circle>
  <circle id="gz-charge" cx="18" cy="18" r="${RR}" fill="none" stroke="#6ff0f5" stroke-width="4" stroke-linecap="round"
    stroke-dasharray="${C2}" stroke-dashoffset="${C2}" transform="rotate(-90 18 18)" style="filter:drop-shadow(0 0 5px rgba(110,240,245,.9))"></circle>
  <circle cx="18" cy="18" r="2.5" fill="#eafeff"></circle></svg>`;
document.body.appendChild(gzCur);
const gzCharge = gzCur.querySelector("#gz-charge");

// Guided recenter target
const centerMark = document.createElement("div");
centerMark.style.cssText = "position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);z-index:60;pointer-events:none;opacity:0;transition:opacity .18s;text-align:center";
centerMark.innerHTML = `<svg width="64" height="64" viewBox="0 0 64 64">
  <circle cx="32" cy="32" r="24" fill="none" stroke="#ffd24a" stroke-width="2"></circle>
  <circle cx="32" cy="32" r="4" fill="#ffd24a"></circle>
  <path d="M32 3v11M32 50v11M3 32h11M50 32h11" stroke="#ffd24a" stroke-width="2" stroke-linecap="round"></path>
</svg><div style="color:#ffd24a;font:12px ui-monospace,monospace;margin-top:6px;letter-spacing:.12em">LOOK HERE</div>`;
document.body.appendChild(centerMark);

const hud = document.getElementById("hud");
function chip(status) {
  const gi = window.GameInput;
  const mode = gi ? gi.modeLabel() : "GAZE";
  // In gesture modes the useful signal is "can it see my hand", not "my face".
  const tracked = gi && gi.mode !== "gaze"
    ? `hand:<b>${gi.handOk ? "yes" : "—"}</b>`
    : `face:<b>${faceOk ? "yes" : "—"}</b>`;
  const activate = gi && gi.mode !== "gaze" ? "pinch to select" : "hold gaze to select";
  hud.innerHTML = `DATA FOREST · input <b>${mode}</b>  control <b>${gazeOn ? "ON" : "OFF"}</b>  `
    + `socket:<b>${status}</b>  ${tracked}  collected:<b>${collected}/${DATA.length}</b>\n`
    + `[m] input mode   [f] fullscreen   [c] recenter   [g] toggle   ·   ${activate}`;
}

/* Input comes from input-manager.js, which owns the only socket and hides
   whether the cursor is being driven by eyes or by a hand. Pulling the values
   into the existing rawX / rawY / faceOk variables means every calculation
   below this point is untouched by the gesture integration. */
const Input = window.GameInput;

function syncInput() {
  if (!Input) return;
  faceOk = Input.ok;                       // "is the cursor live", whatever the device
  if (Input.ok) { rawX = Input.x; rawY = Input.y; }
}

/* Dwell is a gaze affordance: with no way to click, you hold still instead.
   A hand can click, so pinch replaces dwell in the modes that have one —
   otherwise a pinch and the dwell timer would both fire on the same orb. */
function dwellActive() { return !Input || Input.mode === "gaze"; }

/* Fire whatever is under the cursor right now. Shared by dwell and pinch so
   activation lives in one place and both routes behave identically. */
function fireTarget() {
  if (domTargets.length) {                 // win-screen / card buttons take priority
    if (domHovered) {
      domHovered.click();
      domArmed = false; domFireAt = performance.now();
    }
    return;
  }
  if (hovered && collectEnabled) { selectObject(hovered); armed = false; }
}

if (Input) {
  Input.onClick(fireTarget);               // pinch
  Input.onStatus(() => chip(Input.connected ? "connected" : "disconnected"));
  // A swipe turns the forest, so orbs behind you are reachable without
  // having to hold your gaze at the screen edge.
  Input.onSwipe(({ direction }) => {
    if (direction === "left") camX -= 6;
    else if (direction === "right") camX += 6;
  });
}

addEventListener("keydown", (e) => {
  ensureAudio();
  if (e.key === "g" || e.key === "G") { gazeOn = !gazeOn; if (!gazeOn && hovered) hovered = null; chip("connected"); }
  if (e.key === "f" || e.key === "F") { document.fullscreenElement ? document.exitFullscreen?.() : document.documentElement.requestFullscreen?.(); }
  // Recenter only means something for gaze. In gesture mode it would capture
  // wherever your hand happened to be as a permanent offset.
  if ((e.key === "c" || e.key === "C") && !(Input && Input.source === "gesture")) {
    recentering = true; recenterAt = performance.now() + 1300; centerMark.style.opacity = "1";
  }
});

function findSelectable(o) { while (o) { if (o.userData && o.userData.selectable) return o; o = o.parent; } return null; }
function selectObject(root) { if (root.userData.isData) collectNode(root); }

const _pv = new THREE.Vector3();
function projScreen(w) {
  _pv.copy(w).project(camera);
  return { x: (_pv.x * 0.5 + 0.5) * innerWidth, y: (-_pv.y * 0.5 + 0.5) * innerHeight, behind: _pv.z > 1 };
}

// 2D gaze selection for the win-screen buttons (dwell-to-click, like Phase 2).
function update2DGaze(gpx, gpy) {
  const PAD = 60;
  let best = null, bestD = PAD, bcx = 0, bcy = 0;
  for (const el of domTargets) {
    const r = el.getBoundingClientRect();
    const dx = Math.max(r.left - gpx, 0, gpx - r.right);
    const dy = Math.max(r.top - gpy, 0, gpy - r.bottom);
    let d = Math.hypot(dx, dy);
    if (el === domHovered) d -= 40;
    if (d < bestD) { bestD = d; best = el; bcx = r.left + r.width / 2; bcy = r.top + r.height / 2; }
  }
  let cxp = gpx, cyp = gpy;
  if (best) { cxp = lerp(gpx, bcx, 0.5); cyp = lerp(gpy, bcy, 0.5); }
  gzCur.style.transform = `translate(${cxp}px, ${cyp}px)`;
  gzCur.style.opacity = "1";
  domTargets.forEach((el) => el.classList.toggle("gz-hover", el === best));
  const now = performance.now();
  const REFIRE_PAUSE = 550;   // after a toggle, pause before the same card can toggle again
  if (best !== domHovered) { domHovered = best; domDwellStart = now; domArmed = true; }
  // Re-arm on the SAME target after a short pause, so a card can be toggled
  // keep ↔ discard just by continuing to look at it (charge ring recharges).
  else if (domHovered && !domArmed && now - domFireAt >= REFIRE_PAUSE) { domArmed = true; domDwellStart = now; }
  let progress = 0;
  if (domHovered && domArmed && dwellActive()) {
    progress = Math.min((now - domDwellStart) / DWELL_MS, 1);
    if (progress >= 1) { domHovered.click(); domArmed = false; domFireAt = now; }
  }
  gzCharge.setAttribute("stroke-dashoffset", C2 * (1 - progress));
}

/* The recenter offset corrects GAZE drift — where your eyes land versus where
   you think you are looking. A hand pointer has no such error: the fingertip is
   already an absolute position. Subtracting the gaze offset from it just shoves
   the cursor sideways, which put menu buttons out of reach entirely (their hit
   radius is 60px, and a 0.1 offset is 128px at 1280 wide).
   Hybrid aims with the eyes, so it keeps the correction. */
function drift() {
  return (Input && Input.source === "gesture") ? [0, 0] : [driftX, driftY];
}

function updateGaze() {
  syncInput();                 // pull the active device's cursor for this frame
  const [dX, dY] = drift();

  // Guided recenter: hold the target ~1.3s, then capture the offset.
  if (recentering) {
    if (faceOk && performance.now() >= recenterAt) {
      driftX = rawX - 0.5; driftY = rawY - 0.5;
      recentering = false; centerMark.style.opacity = "0";
    } else {
      if (faceOk) {
        const gx0 = clamp(rawX - dX, 0, 1), gy0 = clamp(rawY - dY, 0, 1);
        gzCur.style.transform = `translate(${gx0 * innerWidth}px, ${gy0 * innerHeight}px)`;
        gzCur.style.opacity = "1";
      }
      hovered = null;
      gzCharge.setAttribute("stroke-dashoffset", C2);
      return;
    }
  }

  if (!gazeOn || !faceOk) {
    gzCur.style.opacity = "0";
    hovered = null;
    gzCharge.setAttribute("stroke-dashoffset", C2);
    return;
  }

  const gx = clamp(rawX - dX, 0, 1), gy = clamp(rawY - dY, 0, 1);
  const gpx = gx * innerWidth, gpy = gy * innerHeight;

  // Win-screen buttons up? Do 2D gaze selection on them instead of the 3D scene.
  if (domTargets.length) { update2DGaze(gpx, gpy); return; }

  // Direct raycast, then aim-assist fallback to the nearest orb on screen.
  raycaster.setFromCamera({ x: gx * 2 - 1, y: -(gy * 2 - 1) }, camera);
  const hits = raycaster.intersectObjects(selectables, true);
  let root = hits.length ? findSelectable(hits[0].object) : null;
  let lockPt = null;
  if (!root) {
    let bestD = ASSIST_RADIUS, best = null, bestPt = null;
    for (const s of selectables) {
      const p = projScreen(s.position);
      if (p.behind) continue;
      let d = Math.hypot(p.x - gpx, p.y - gpy);
      if (s === hovered) d -= HYSTERESIS;
      if (d < bestD) { bestD = d; best = s; bestPt = p; }
    }
    root = best; lockPt = bestPt;
  } else {
    lockPt = projScreen(root.position);
  }

  let cxp = gpx, cyp = gpy;
  if (lockPt) { cxp = lerp(gpx, lockPt.x, MAGNET); cyp = lerp(gpy, lockPt.y, MAGNET); }
  gzCur.style.transform = `translate(${cxp}px, ${cyp}px)`;
  gzCur.style.opacity = "1";

  if (root !== hovered) { hovered = root; dwellStart = performance.now(); armed = true; }
  let progress = 0;
  if (hovered && dwellActive()) {
    progress = Math.min((performance.now() - dwellStart) / DWELL_MS, 1);
    if (progress >= 1 && armed && collectEnabled) { selectObject(hovered); armed = false; }
  }
  // With dwell off the ring stays empty rather than frozen full, so it never
  // looks like a charge that failed to fire.
  gzCharge.setAttribute("stroke-dashoffset", C2 * (1 - progress));
}

// ===== FLOOD + RAIN — the Apply-stage payoff =====
let floodWater = null, rain = null, flood = null;
function startFlood(protectedOutcome) {
  if (!floodWater) {
    floodWater = new THREE.Mesh(new THREE.PlaneGeometry(180, 100),
      new THREE.MeshStandardMaterial({ color: 0x2f7fb0, transparent: true, opacity: 0.6, roughness: 0.15, metalness: 0.35 }));
    floodWater.rotation.x = -Math.PI / 2;
    scene.add(floodWater);
  }
  if (!rain) {
    const N = 2600, p = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) { p[i * 3] = rand(-45, 45); p[i * 3 + 1] = rand(0, 32); p[i * 3 + 2] = rand(-22, 14); }
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(p, 3));
    rain = new THREE.Points(g, new THREE.PointsMaterial({ color: 0xbcd8ff, size: 0.1, transparent: true, opacity: 0.65, sizeAttenuation: true }));
    scene.add(rain);
  }
  rain.visible = true; floodWater.visible = true; floodWater.position.y = -8;
  flood = { t: 0, phase: "rise", prot: !!protectedOutcome, peak: protectedOutcome ? 0.7 : 1.8 };
}
function stopFlood() { if (rain) rain.visible = false; if (floodWater) floodWater.visible = false; flood = null; }
function updateFlood(dt) {
  if (!flood) return;
  const p = rain.geometry.attributes.position;
  for (let i = 0; i < p.count; i++) { let y = p.getY(i) - dt * 24; if (y < -1) y = rand(20, 32); p.setY(i, y); }
  p.needsUpdate = true;
  if (flood.phase === "rise") {
    flood.t = Math.min(1, flood.t + dt * 0.4);
    floodWater.position.y = lerp(-8, flood.peak, flood.t);
    if (flood.t >= 1) { flood.phase = flood.prot ? "recede" : "hold"; flood.t = 0; }
  } else if (flood.phase === "recede") {
    flood.t = Math.min(1, flood.t + dt * 0.4);
    floodWater.position.y = lerp(flood.peak, -4, flood.t);
    if (flood.t >= 1) { flood.phase = "done"; rain.visible = false; }
  }
  floodWater.material.opacity = 0.5 + Math.sin(performance.now() * 0.003) * 0.06;
}

// --- Render loop -----------------------------------------------------
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (mixer) mixer.update(dt);
  updateCamera(dt);
  updateGaze();
  updateDataNodes(dt);
  updateParticles(dt);
  updateFlood(dt);
  renderer.render(scene, camera);
}
animate();

// Debug hook
window.__forest = {
  THREE, scene, renderer, camera, selectables, dataNodes, domTargets, data: DATA,
  get collected() { return collected; },
  get hovered() { return hovered; },
  setGaze(x, y) { rawX = x; rawY = y; faceOk = true; },
  setCollectEnabled(v) { collectEnabled = v; },
  emote(name) { if (archerObj) archerPlay(name); },   // trigger an archer animation (e.g. emote-yes)
  startFlood(p) { startFlood(p); },                   // Apply payoff: rain + rising floodwater
  stopFlood() { stopFlood(); },
  reset() {                                            // full replay: respawn orbs, clear progress
    stopFlood();
    gameWon = false; collected = 0; hovered = null;
    for (const orb of dataNodes) { scene.remove(orb); if (orb.userData && orb.userData.label) orb.userData.label.remove(); }
    dataNodes.length = 0; selectables.length = 0;
    createDataNodes();
    collectEnabled = false;
  },
};

addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
