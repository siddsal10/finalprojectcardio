// 12-lead ECG renderer on standard ECG paper (25 mm/s, 10 mm/mV by default).
import { esc, tip } from "./ui.js";

const LAYOUT_3x4 = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]];
const SPEED = 25; // mm per second
const CAL_MM = 6; // left margin with the 1 mV calibration pulse

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/**
 * Mounts an interactive viewer into `root`.
 * signal: {fs, leads: [...12], data: [[...], ...]} in mV.
 */
export function mountEcg(root, signal, { layout = "standard", gain = 10, toolbar = true, rhythmLead = "II" } = {}) {
  const state = { layout, gain, lead: rhythmLead };
  root.innerHTML = `
    ${toolbar ? `<div class="ecg-toolbar no-print">
      <div class="seg" data-k="layout">
        <button data-v="standard">3×4 + rhythm</button><button data-v="stacked">12 × 10 s</button><button data-v="single">Single lead</button>
      </div>
      <div class="seg" data-k="gain"><button data-v="5">5 mm/mV</button><button data-v="10">10 mm/mV</button><button data-v="20">20 mm/mV</button></div>
      <select class="input" data-k="lead" style="width:auto;padding:6px 10px">${signal.leads.map((l) => `<option ${l === state.lead ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>
      <span class="muted small" style="margin-left:auto">25 mm/s · ${signal.fs} Hz · 0.5–40 Hz band-pass</span>
    </div>` : ""}
    <div class="ecg-wrap"><canvas></canvas></div>
    <div class="ecg-legend">Small squares 1 mm (0.04 s, ${(0.1 * 10 / state.gain).toFixed(2)} mV) · large squares 5 mm (0.2 s). Hover the trace to read time and amplitude.</div>`;
  const canvas = root.querySelector("canvas");
  const wrap = root.querySelector(".ecg-wrap");
  const legend = root.querySelector(".ecg-legend");
  const leadIdx = Object.fromEntries(signal.leads.map((l, i) => [l, i]));
  let geo = null;

  function sync() {
    root.querySelectorAll(".seg").forEach((seg) => {
      seg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", String(state[seg.dataset.k]) === b.dataset.v));
    });
    const sel = root.querySelector("select[data-k=lead]");
    if (sel) sel.style.display = state.layout === "single" ? "" : "none";
    legend.firstChild && (legend.innerHTML = `Small squares 1 mm (0.04 s, ${(1 / state.gain).toFixed(2)} mV) · large squares 5 mm (0.2 s, ${(5 / state.gain).toFixed(1)} mV). Hover the trace to read time and amplitude.`);
  }

  function draw() {
    const seconds = signal.data[0].length / signal.fs;
    const widthMm = CAL_MM + seconds * SPEED;
    const avail = (wrap.clientWidth || 900) - 2;
    const pxmm = Math.max(3.2, avail / widthMm);
    const rowMm = state.layout === "standard" ? Math.max(24, 2.6 * state.gain) : state.layout === "stacked" ? Math.max(14, 1.9 * state.gain) : Math.max(40, 4 * state.gain);
    const rows = state.layout === "standard" ? 4 : state.layout === "stacked" ? 12 : 1;
    const heightMm = rows * rowMm + 4;
    const dpr = window.devicePixelRatio || 1;
    const W = Math.round(widthMm * pxmm), H = Math.round(heightMm * pxmm);
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = cssVar("--paper"); ctx.fillRect(0, 0, W, H);
    // grid
    for (let mm = 0; mm <= widthMm; mm++) {
      ctx.strokeStyle = mm % 5 ? cssVar("--paper-minor") : cssVar("--paper-major");
      ctx.lineWidth = mm % 5 ? 0.5 : 1;
      ctx.beginPath(); ctx.moveTo(mm * pxmm + .5, 0); ctx.lineTo(mm * pxmm + .5, H); ctx.stroke();
    }
    for (let mm = 0; mm <= heightMm; mm++) {
      ctx.strokeStyle = mm % 5 ? cssVar("--paper-minor") : cssVar("--paper-major");
      ctx.lineWidth = mm % 5 ? 0.5 : 1;
      ctx.beginPath(); ctx.moveTo(0, mm * pxmm + .5); ctx.lineTo(W, mm * pxmm + .5); ctx.stroke();
    }
    const trace = cssVar("--trace");
    const segs = []; // for hover: {x0, x1, yBase, lead, t0}
    const ink = cssVar("--ink-2");
    function drawSeg(lead, t0, t1, x0Mm, baseMm) {
      const i = leadIdx[lead];
      if (i === undefined) return;
      const s = signal.data[i];
      const a = Math.round(t0 * signal.fs), b = Math.min(s.length, Math.round(t1 * signal.fs));
      ctx.strokeStyle = trace; ctx.lineWidth = 1.3; ctx.lineJoin = "round";
      ctx.beginPath();
      for (let k = a; k < b; k++) {
        const x = (x0Mm + ((k - a) / signal.fs) * SPEED) * pxmm;
        const y = (baseMm - s[k] * state.gain) * pxmm;
        k === a ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.fillStyle = ink; ctx.font = `600 ${Math.max(10, 2.6 * pxmm)}px Inter, system-ui, sans-serif`;
      ctx.fillText(lead, (x0Mm + 1) * pxmm, (baseMm - rowMm * 0.32) * pxmm);
      segs.push({ x0: x0Mm * pxmm, x1: (x0Mm + (t1 - t0) * SPEED) * pxmm, base: baseMm * pxmm, yTop: (baseMm - rowMm / 2) * pxmm, yBot: (baseMm + rowMm / 2) * pxmm, lead, t0, i });
    }
    function cal(baseMm) {
      ctx.strokeStyle = trace; ctx.lineWidth = 1.3;
      const x = 1 * pxmm, y = baseMm * pxmm, h = state.gain * pxmm;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + pxmm, y); ctx.lineTo(x + pxmm, y - h); ctx.lineTo(x + 4 * pxmm, y - h); ctx.lineTo(x + 4 * pxmm, y); ctx.lineTo(x + 5 * pxmm, y); ctx.stroke();
    }
    if (state.layout === "standard") {
      const colS = seconds / 4;
      LAYOUT_3x4.forEach((row, r) => {
        const base = 2 + rowMm * (r + 0.55);
        cal(base);
        row.forEach((lead, c) => drawSeg(lead, c * colS, (c + 1) * colS, CAL_MM + c * colS * SPEED, base));
        // column separators
        ctx.strokeStyle = ink; ctx.lineWidth = 1;
        for (let c = 1; c < 4; c++) { const x = (CAL_MM + c * colS * SPEED) * pxmm; ctx.beginPath(); ctx.moveTo(x, (base - 3) * pxmm); ctx.lineTo(x, (base + 3) * pxmm); ctx.stroke(); }
      });
      const base = 2 + rowMm * 3.55;
      cal(base);
      drawSeg(state.lead, 0, seconds, CAL_MM, base);
    } else if (state.layout === "stacked") {
      signal.leads.forEach((lead, r) => { const base = 2 + rowMm * (r + 0.55); cal(base); drawSeg(lead, 0, seconds, CAL_MM, base); });
    } else {
      const base = 2 + rowMm * 0.6; cal(base); drawSeg(state.lead, 0, seconds, CAL_MM, base);
    }
    geo = { segs, pxmm };
  }

  canvas.addEventListener("mousemove", (ev) => {
    if (!geo) return;
    const r = canvas.getBoundingClientRect();
    const x = ev.clientX - r.left, y = ev.clientY - r.top;
    const seg = geo.segs.find((s) => x >= s.x0 && x <= s.x1 && y >= s.yTop && y <= s.yBot);
    if (!seg) return tip(null);
    const t = seg.t0 + (x - seg.x0) / geo.pxmm / SPEED;
    const v = signal.data[seg.i][Math.min(signal.data[seg.i].length - 1, Math.round(t * signal.fs))];
    tip(`<b>Lead ${esc(seg.lead)}</b><br>t = ${t.toFixed(2)} s · ${v.toFixed(2)} mV`, ev);
  });
  canvas.addEventListener("mouseleave", () => tip(null));
  root.querySelectorAll(".seg").forEach((seg) => seg.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    state[seg.dataset.k] = seg.dataset.k === "gain" ? Number(b.dataset.v) : b.dataset.v;
    sync(); draw();
  }));
  const sel = root.querySelector("select[data-k=lead]");
  sel && sel.addEventListener("change", () => { state.lead = sel.value; draw(); });
  sync(); draw();
  const ro = new ResizeObserver(() => draw());
  ro.observe(wrap);
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  mq.addEventListener("change", draw);
  window.addEventListener("cs-theme", draw);
  return { redraw: draw };
}
