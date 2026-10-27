// Interactive 3-D heart for a result: each finding lights up the part of the heart it usually comes
// from (stylised anatomy, not a scan of the patient's heart). three.js comes from the CDN import map
// in index.html; without WebGL or the CDN a plain list is shown instead.
import { CLASS_NAMES, esc } from "./ui.js";

export const REGIONS = {
  MI: { where: "Front wall of the left ventricle, fed by the coronary arteries",
    plain: "Signs of a heart attack point to heart muscle that may not be getting enough blood, now or in the past.", color: "#c8102e", view: [1.6, 0.2, 3.6] },
  STTC: { where: "Both lower chambers (ventricles)",
    plain: "ST/T changes are about how the ventricles reset between beats. They can go with reduced blood flow or strain.", color: "#e0761a", view: [0.2, -0.4, 4.2] },
  CD: { where: "SA node, AV node and the bundle branches down the septum",
    plain: "Conduction problems are delays or blocks somewhere along the path the heartbeat signal travels.", color: "#d6a400", view: [-2.4, 0.6, 3.2] },
  HYP: { where: "Wall of the left ventricle",
    plain: "Hypertrophy means a heart wall may be thicker than usual, often from long-term high blood pressure.", color: "#6a4fc4", view: [4.2, 0.2, 2.4] },
  NORM: { where: "The whole heart",
    plain: "A normal pattern means the heart's electrical activity looks typical.", color: "#1fa463", view: [0, 0.3, 4.4] },
};

const STATE_WORD = { found: "Found", unclear: "Unclear", not_found: "Not found" };

/** el: container. perClass: {CODE: {result: found|not_found|unclear, probability}}. bpm: heart rate. */
export async function mountHeart(el, { perClass, bpm = 72 } = {}) {
  const codes = ["MI", "STTC", "CD", "HYP", "NORM"];
  const state = (c) => (perClass[c] ? perClass[c].result : "not_found");
  const lit = codes.filter((c) => c !== "NORM" && state(c) !== "not_found");
  el.innerHTML = `<div class="heart3d">
    <div class="heart3d-stage" id="h3-stage"><div class="heart3d-hint">Drag to turn · scroll to zoom</div></div>
    <div><p class="small ink2" style="margin-bottom:10px">${lit.length ? "Tap a finding to see where on the heart it usually comes from." : "Nothing to point at: none of the four warning signs was found."}</p>
      <div class="region-list">${codes.map((c) => { const st = state(c); const on = c === "NORM" ? st === "found" : st !== "not_found";
        return `<button class="region" data-c="${c}" ${on ? "" : 'style="opacity:.55"'}><i style="background:${REGIONS[c].color}"></i>
          <span><b>${esc(CLASS_NAMES[c])} · ${STATE_WORD[st] || ""}</b><span class="small">${esc(REGIONS[c].where)}</span></span>
          <span class="tiny muted">${perClass[c] ? Math.round(perClass[c].probability * 100) + "%" : ""}</span></button>`; }).join("")}</div>
      <p class="small" id="h3-plain" style="margin-top:12px;color:var(--ink-2)"></p>
      <p class="small muted" style="margin-top:6px">The 3-D view shows where each finding usually comes from. It is not an image of your heart.</p></div></div>`;
  const stage = el.querySelector("#h3-stage");
  const plain = el.querySelector("#h3-plain");
  let THREE, OrbitControls;
  try {
    THREE = await import("three");
    ({ OrbitControls } = await import("three/addons/controls/OrbitControls.js"));
    const probe = document.createElement("canvas");
    if (!(probe.getContext("webgl2") || probe.getContext("webgl"))) throw new Error("no WebGL");
  } catch (e) {
    stage.innerHTML = `<div class="heart3d-fallback small">The 3-D heart needs WebGL and an internet connection. The list shows the same information.</div>`;
    bindList(null);
    return () => {};
  }

  // ------------------------------------------------------------------ scene
  const w = () => Math.max(1, stage.clientWidth), h = () => Math.max(1, stage.clientHeight);   // the stage has a fixed CSS height
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(w(), h(), false);
  stage.prepend(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(36, w() / h(), 0.1, 100);
  camera.position.set(0, 0.4, 7.2);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.enablePan = false; controls.minDistance = 4; controls.maxDistance = 11;
  controls.target.set(0, 0.15, 0);
  controls.autoRotate = true; controls.autoRotateSpeed = 0.9;
  scene.add(new THREE.HemisphereLight(0xfff4ec, 0x2a1a22, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(3, 4, 5); scene.add(key);
  const rim = new THREE.DirectionalLight(0xff8a8a, 0.8); rim.position.set(-4, 2, -3); scene.add(rim);

  const heart = new THREE.Group();
  heart.rotation.z = -0.32; heart.rotation.y = -0.25;
  scene.add(heart);
  const muscle = (color) => new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05, emissive: 0x000000 });
  const blob = (r, sx, sy, sz, x, y, z, mat) => {
    const m = new THREE.Mesh(new THREE.SphereGeometry(r, 48, 32), mat);
    m.scale.set(sx, sy, sz); m.position.set(x, y, z); heart.add(m); return m;
  };
  const tube = (pts, r, mat) => {
    const curve = new THREE.CatmullRomCurve3(pts.map((p) => new THREE.Vector3(...p)));
    const m = new THREE.Mesh(new THREE.TubeGeometry(curve, 64, r, 16, false), mat); heart.add(m); return m;
  };
  const lvMat = muscle(0xc9505c), rvMat = muscle(0xc65a63), atMat = muscle(0xb84a58);
  const lv = blob(1, 0.95, 1.3, 0.92, 0.32, -0.3, 0, lvMat);
  const rv = blob(1, 0.85, 1.05, 0.78, -0.38, -0.12, 0.28, rvMat);
  const la = blob(0.5, 1, 0.85, 1, 0.38, 0.78, -0.25, atMat);
  const ra = blob(0.55, 1, 0.9, 1, -0.6, 0.62, 0.02, atMat);
  tube([[0.05, 0.55, 0], [0.05, 1.35, 0.05], [0.25, 1.75, -0.05], [0.65, 1.62, -0.3], [0.75, 0.95, -0.55]], 0.2,
    new THREE.MeshStandardMaterial({ color: 0xa3192f, roughness: 0.4, metalness: 0.1 }));                   // aorta
  tube([[-0.2, 0.45, 0.45], [-0.1, 1.15, 0.4], [0.2, 1.45, 0.2], [0.55, 1.35, 0.0]], 0.17,
    new THREE.MeshStandardMaterial({ color: 0x5a6fa0, roughness: 0.45 }));                                   // pulmonary trunk
  tube([[-0.75, 1.0, -0.1], [-0.7, 1.6, -0.15]], 0.15, new THREE.MeshStandardMaterial({ color: 0x41557a, roughness: 0.5 })); // vena cava
  // coronary arteries (part of the MI region)
  const corMat = new THREE.MeshStandardMaterial({ color: 0xe0485c, roughness: 0.35, emissive: 0x000000 });
  const lad = tube([[0.0, 0.55, 0.78], [0.05, 0.0, 0.9], [0.12, -0.7, 0.82], [0.25, -1.35, 0.45]], 0.045, corMat);
  const lcx = tube([[0.05, 0.55, 0.7], [0.55, 0.45, 0.62], [0.95, 0.1, 0.2], [0.98, -0.4, -0.25]], 0.04, corMat);
  const rca = tube([[-0.15, 0.5, 0.75], [-0.75, 0.25, 0.7], [-1.05, -0.3, 0.3], [-0.8, -0.9, -0.1]], 0.04, corMat);
  // the zone most often affected: anterior wall of the LV
  const zoneMat = new THREE.MeshStandardMaterial({ color: 0xc8102e, transparent: true, opacity: 0, roughness: 0.4, emissive: 0xc8102e, emissiveIntensity: 0 });
  const zone = blob(0.55, 1, 1.2, 0.45, 0.45, -0.45, 0.68, zoneMat);
  // conduction system
  const wireMat = new THREE.MeshStandardMaterial({ color: 0xd6a400, emissive: 0xd6a400, emissiveIntensity: 0.25, transparent: true, opacity: 0.55 });
  const sa = blob(0.09, 1, 1, 1, -0.72, 0.98, 0.32, wireMat);
  const av = blob(0.08, 1, 1, 1, -0.12, 0.28, 0.62, wireMat);
  const wires = [
    tube([[-0.72, 0.98, 0.32], [-0.45, 0.6, 0.55], [-0.12, 0.28, 0.62]], 0.025, wireMat),
    tube([[-0.12, 0.28, 0.62], [-0.02, -0.2, 0.75], [0.2, -0.9, 0.7], [0.32, -1.4, 0.4]], 0.03, wireMat),
    tube([[-0.12, 0.28, 0.62], [-0.22, -0.25, 0.8], [-0.35, -0.9, 0.62], [-0.3, -1.15, 0.35]], 0.03, wireMat),
  ];
  const parts = { MI: [zone, lad, lcx, rca], STTC: [lv, rv], CD: [sa, av, ...wires], HYP: [lv], NORM: [lv, rv, la, ra] };

  // ------------------------------------------------------------------ highlight + beat
  const col = (hex) => new THREE.Color(hex);
  const levels = {};
  codes.forEach((c) => { const st = state(c); levels[c] = c === "NORM" ? (st === "found" ? 0.5 : 0) : st === "found" ? 1 : st === "unclear" ? 0.45 : 0; });
  let focus = null;
  const base = { lv: lvMat.color.clone(), rv: rvMat.color.clone() };
  const hyp = levels.HYP > 0 ? 1 + 0.12 * levels.HYP : 1;
  const lvScale = lv.scale.clone();

  function paint(t) {
    const pulse = (k) => 0.55 + 0.45 * Math.sin(t * k);
    const amt = (c) => levels[c] * (focus && focus !== c ? 0.35 : 1);
    zoneMat.opacity = 0.85 * amt("MI");
    zoneMat.emissiveIntensity = 0.9 * amt("MI") * pulse(4);
    corMat.emissive = col(0xc8102e); corMat.emissiveIntensity = 0.6 * amt("MI") * pulse(4);
    const s = amt("STTC") * pulse(2.2);
    lvMat.color.copy(base.lv).lerp(col(REGIONS.STTC.color), 0.45 * s);
    rvMat.color.copy(base.rv).lerp(col(REGIONS.STTC.color), 0.45 * s);
    lvMat.emissive = col(REGIONS.HYP.color); lvMat.emissiveIntensity = 0.35 * amt("HYP") * pulse(1.6);
    wireMat.emissiveIntensity = 0.25 + 1.6 * amt("CD") * pulse(6);
    wireMat.opacity = 0.55 + 0.45 * Math.min(1, amt("CD") * 1.5);
    const n = amt("NORM") * pulse(1.2);
    rvMat.emissive = col(REGIONS.NORM.color); rvMat.emissiveIntensity = 0.18 * n;
    atMat.emissive = col(REGIONS.NORM.color); atMat.emissiveIntensity = 0.18 * n;
  }

  const clock = new THREE.Clock();
  const period = 60 / Math.max(35, Math.min(180, bpm || 72));
  let raf = 0, visible = true;
  function frame() {
    raf = requestAnimationFrame(frame);
    if (!visible) return;
    const t = clock.getElapsedTime();
    const ph = (t % period) / period;
    const beat = Math.exp(-((ph - 0.12) ** 2) / 0.004) + 0.4 * Math.exp(-((ph - 0.32) ** 2) / 0.006); // lub-dub
    const sq = 1 - 0.045 * beat;
    lv.scale.set(lvScale.x * hyp * sq, lvScale.y * sq, lvScale.z * hyp * sq);
    rv.scale.set(0.85 * sq, 1.05 * sq, 0.78 * sq);
    paint(t);
    controls.update();
    renderer.render(scene, camera);
  }
  frame();

  // ------------------------------------------------------------------ interaction
  const ray = new THREE.Raycaster(), ptr = new THREE.Vector2();
  const owner = (obj) => codes.find((c) => c !== "NORM" && levels[c] > 0 && parts[c].includes(obj)) || null;
  renderer.domElement.addEventListener("click", (ev) => {
    const r = renderer.domElement.getBoundingClientRect();
    ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ptr, camera);
    const hit = ray.intersectObjects(heart.children, false).map((x) => owner(x.object)).find(Boolean);
    if (hit) select(hit);
  });
  function flyTo(v) {
    const from = camera.position.clone(), to = new THREE.Vector3(...v).setLength(7);
    const t0 = performance.now();
    controls.autoRotate = false;
    const step = (now) => { const k = Math.min(1, (now - t0) / 700); const e = 1 - (1 - k) ** 3;
      camera.position.lerpVectors(from, to, e); if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  }
  function select(c) {
    focus = focus === c ? null : c;
    el.querySelectorAll(".region").forEach((b) => b.classList.toggle("on", b.dataset.c === focus));
    plain.textContent = focus ? REGIONS[focus].plain : "";
    if (focus) flyTo(REGIONS[focus].view); else controls.autoRotate = true;
  }
  bindList(select);
  const first = codes.find((c) => c !== "NORM" && state(c) === "found") || codes.find((c) => c !== "NORM" && state(c) === "unclear");
  if (first) setTimeout(() => select(first), 900);

  const ro = new ResizeObserver(() => { renderer.setSize(w(), h(), false); camera.aspect = w() / h(); camera.updateProjectionMatrix(); });
  ro.observe(stage);
  const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting; });
  io.observe(stage);
  return () => { cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); renderer.dispose(); };

  function bindList(onSelect) {
    el.querySelectorAll(".region").forEach((b) => b.addEventListener("click", () => {
      if (onSelect) onSelect(b.dataset.c);
      else { plain.textContent = REGIONS[b.dataset.c].plain; el.querySelectorAll(".region").forEach((x) => x.classList.toggle("on", x === b)); }
    }));
  }
}
