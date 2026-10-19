// Animated walkthrough of what happened to an uploaded ECG, built from the server's real pipeline record.
import { CLASSES, CLASS_NAMES, LEVELS, esc, fileSize, icon } from "./ui.js";

const READER_COLORS = { A: "var(--s1)", B: "var(--s2)", C: "var(--s3)", D: "var(--s4)" };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const ms = (v) => (v === null || v === undefined ? "" : v < 1 ? "<1 ms" : `${Math.round(v)} ms`);

function tracePath(values, w = 600, h = 90) {
  if (!values || !values.length) return "";
  const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
  return values.map((v, i) => `${i ? "L" : "M"}${((i / (values.length - 1)) * w).toFixed(1)},${(h - 6 - ((v - lo) / span) * (h - 12)).toFixed(1)}`).join("");
}

const STEPS = [
  ["file", "Reading your file"],
  ["activity", "Cleaning the signal"],
  ["cpu", "Four AI readers"],
  ["layers", "Combining their votes"],
  ["target", "Confidence check"],
  ["heart", "Your result"],
];

/** Skeleton shown while the upload is still running. */
export function pipelineShell(el, filename) {
  el.innerHTML = `<div class="card pipe">
    <div class="pipe-head"><div><div class="eyebrow">Analysing</div><h2>${esc(filename || "your ECG")}</h2></div>
      <div class="row"><span class="small muted" id="pipe-time"></span><button class="btn sm" id="pipe-skip" disabled>Skip animation</button></div></div>
    <div class="pipe-bar"><i id="pipe-progress" style="width:4%"></i></div>
    <ol class="pipe-steps">${STEPS.map(([ic, t], i) => `<li class="pipe-step ${i === 0 ? "active" : ""}" data-i="${i}">
      <div class="pipe-dot">${icon(ic)}</div><div class="pipe-main"><div class="pipe-title"><b>${t}</b><span class="pipe-ms"></span></div><div class="pipe-body"></div></div></li>`).join("")}</ol></div>`;
  el.querySelector('.pipe-step[data-i="0"] .pipe-body').innerHTML = `<div class="row small muted"><span class="spinner" style="width:16px;height:16px;border-width:2px"></span>Uploading…</div>`;
}

/**
 * Plays the walkthrough inside a container that already holds pipelineShell().
 * opts: {perClass: {CODE: {result, probability}}, level, onDone(), doneLabel}
 */
export async function playPipeline(el, pl, { perClass, level, onDone, doneLabel = "See my full result" } = {}) {
  let fast = false;
  const skip = el.querySelector("#pipe-skip");
  skip.disabled = false;
  skip.addEventListener("click", () => { fast = true; skip.disabled = true; });
  const wait = (t) => (fast ? Promise.resolve() : sleep(t));
  const steps = [...el.querySelectorAll(".pipe-step")];
  const prog = el.querySelector("#pipe-progress");
  el.querySelector("#pipe-time").textContent = pl.total_ms ? `Analysed in ${(pl.total_ms / 1000).toFixed(2)} s` : "";
  const open = (i, html, t) => {
    steps.forEach((s, k) => { s.classList.toggle("active", k === i); if (k < i) s.classList.add("done"); });
    steps[i].querySelector(".pipe-body").innerHTML = html;
    steps[i].querySelector(".pipe-ms").textContent = ms(t);
    prog.style.width = `${Math.round(((i + 1) / steps.length) * 100)}%`;
    if (!fast) steps[i].scrollIntoView({ block: "nearest", behavior: "smooth" });
    return steps[i].querySelector(".pipe-body");
  };
  if (fast) el.classList.add("fast");
  skip.addEventListener("click", () => el.classList.add("fast"));

  // 1 · reading
  const inp = pl.input || {};
  let b = open(0, `<div class="pipe-facts">
      ${inp.filename ? `<span class="chip">${icon("file")}${esc(inp.filename)}</span>` : ""}${inp.format && !String(inp.filename || "").startsWith(inp.format) ? `<span class="chip">${esc(inp.format)}</span>` : ""}
      ${inp.size_bytes ? `<span class="chip">${fileSize(inp.size_bytes)}</span>` : ""}<span class="chip">${inp.fs_in || 100} samples / second</span><span class="chip">${inp.duration_s || 10} seconds</span></div>
    <div class="leads">${(inp.leads || ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]).map((l, i) => `<span style="animation-delay:${i * 70}ms">${esc(l)}</span>`).join("")}</div>
    <p class="small muted">${inp.n_leads || 12} leads found${inp.bad_values_fixed ? ` · ${inp.bad_values_fixed} missing values repaired` : ""}.</p>`, pl.read_ms);
  await wait(1500);

  // 2 · cleaning
  const cl = pl.clean || {};
  b = open(1, `<div class="grid g2" style="gap:14px;align-items:start">
      <ul class="ticks">${(cl.steps || []).map((s, i) => `<li style="animation-delay:${i * 260}ms">${icon("check")}${esc(s)}</li>`).join("")}</ul>
      <div class="trace-box">${cl.raw_ii ? `<div class="tiny muted">Lead II as uploaded</div><svg viewBox="0 0 600 90" preserveAspectRatio="none"><path class="draw raw" d="${tracePath(cl.raw_ii)}"/></svg>` : ""}
        <div class="tiny muted">Lead II, cleaned</div><svg viewBox="0 0 600 90" preserveAspectRatio="none"><path class="draw clean" style="animation-delay:${cl.raw_ii ? 700 : 0}ms" d="${tracePath(cl.clean_ii)}"/></svg></div></div>`, (cl.ms || 0) + (pl.scale_ms || 0));
  await wait(2000);

  // 3 · readers
  b = open(2, `<p class="small muted" style="margin-bottom:10px">Four separately trained models read your ECG on their own. Each gives a likelihood for the five findings.</p>
    <div class="readers">${pl.readers.map((r, i) => `<div class="reader" style="animation-delay:${i * 180}ms">
      <div class="row between"><b style="color:${READER_COLORS[r.key]}">Reader ${r.key}</b><span class="tiny muted">${esc(r.arch)} · ${ms(r.ms)}</span></div>
      <div class="tiny muted" style="margin-bottom:8px">${esc(r.role)}</div>
      ${CLASSES.map((c) => `<div class="rbar"><span>${c}</span><div class="meter"><i style="--w:${Math.round(r.probs[c] * 100)}%;background:${READER_COLORS[r.key]};animation-delay:${i * 180 + 200}ms"></i></div><b class="num">${Math.round(r.probs[c] * 100)}</b></div>`).join("")}</div>`).join("")}</div>`,
    pl.readers.reduce((a, r) => a + r.ms, 0));
  await wait(2300);

  // 4 · combining
  b = open(3, `<p class="small muted" style="margin-bottom:10px">The four votes are averaged, each reader counting equally. Dots are the readers, the bar is the combined likelihood.</p>
    ${CLASSES.map((c, k) => { const avg = pl.readers.reduce((a, r) => a + r.probs[c], 0) / pl.readers.length; const p = perClass[c].probability;
      return `<div class="combo"><span class="combo-name">${esc(CLASS_NAMES[c])}</span><div class="combo-track">
        ${pl.readers.map((r, i) => `<i class="vote" style="left:${r.probs[c] * 100}%;background:${READER_COLORS[r.key]};animation-delay:${k * 120 + i * 60}ms" title="Reader ${r.key}: ${Math.round(r.probs[c] * 100)}%"></i>`).join("")}
        <div class="combo-fill" style="--w:${Math.round((p ?? avg) * 100)}%;animation-delay:${k * 120 + 350}ms"></div></div><b class="num">${Math.round((p ?? avg) * 100)}%</b></div>`; }).join("")}`, pl.combine && pl.combine.ms);
  await wait(2100);

  // 5 · confidence
  const word = { found: ["crit", "Found"], not_found: ["good", "Not found"], unclear: ["warn", "Unclear"] };
  b = open(4, `<p class="small muted" style="margin-bottom:10px">Each finding is only reported when the likelihood is clearly high or clearly low. Anything in between is marked unclear for a doctor to check.</p>
    <div class="conf">${CLASSES.map((c, i) => { const r = perClass[c].result; const [tone, lab] = c === "NORM" && r !== "unclear" ? [r === "found" ? "good" : "neutral", r === "found" ? "Yes" : "No"] : word[r];
      return `<div class="conf-item" style="animation-delay:${i * 220}ms"><span>${esc(CLASS_NAMES[c])}</span><span class="badge ${tone}">${lab}</span></div>`; }).join("")}</div>`);
  await wait(1900);

  // 6 · result
  const lv = LEVELS[level];
  b = open(5, `<div class="pipe-result ${lv.tone}"><div class="row" style="flex-wrap:nowrap"><span class="kpi-icon">${icon(lv.icon)}</span><div><b style="font-size:17px">${lv.title}</b>
      <div class="small">${level === "reassuring" ? "No warning signs among the five findings." : level === "check" ? "Some findings are unclear. Sharing with your doctor is recommended." : "Your doctor should look at this."}</div></div></div></div>
    ${onDone ? `<button class="btn primary lg" id="pipe-done" style="margin-top:14px">${doneLabel} ${icon("arrowRight")}</button>` : ""}`);
  steps[5].classList.add("done");
  skip.disabled = true;
  if (onDone) b.querySelector("#pipe-done").addEventListener("click", onDone);
}
