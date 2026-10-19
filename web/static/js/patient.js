// Patient app: heart checks, results, symptom journal, sharing with the doctor.
import { api, blobUrl, get, patch, post } from "./api.js";
import { mountEcg } from "./ecg.js";
import { mountMonitor } from "./monitor.js";
import { pipelineShell, playPipeline } from "./pipeline.js";
import {
  $, CLASSES, CLASS_NAMES, LEVELS, MOODS, PLAIN, date, dateTime, esc, fileSize, findingList, icon, initials, modal,
  moodFace, statusBadge, toast,
} from "./ui.js";

const shell = () => import("./app.js");
const refresh = () => shell().then((m) => m.refreshCounts());
const query = () => new URLSearchParams(location.hash.split("?")[1] || "");
const head = (title, sub = "", actions = "", crumbs = "") => `<div class="page-head"><div>${crumbs ? `<div class="crumbs">${crumbs}</div>` : ""}<h1>${title}</h1>${sub ? `<p>${sub}</p>` : ""}</div>${actions ? `<div class="row">${actions}</div>` : ""}</div>`;
const WHEN = ["Just now", "Earlier today", "Yesterday", "A few days ago", "Over a week ago"];

const levelOf = (e) => (e.ai ? e.ai.level : (e.findings.some((c) => c !== "NORM") ? "see_doctor" : e.findings.length ? "reassuring" : "check"));
const shareState = (e) => (e.submission ? statusBadge(e.submission.status) : e.self_check ? `<span class="badge neutral">${icon("lock")}Private</span>` : `<span class="badge info">${icon("stethoscope")}From your clinic</span>`);

// ======================================================================= home
async function home(el) {
  const [o, me] = await Promise.all([get("/api/me/overview"), shell().then((m) => m.app.user)]);
  const l = o.latest;
  const lv = l ? LEVELS[levelOf(l)] : null;
  const h = new Date().getHours();
  el.innerHTML = `
    ${head(`Good ${h < 12 ? "morning" : h < 18 ? "afternoon" : "evening"}, ${esc(me.name.split(" ")[0])}`, new Date().toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" }))}
    <div class="grid g-2-1">
      <div class="stack">
        ${l ? `<div class="hero-status ${lv.tone}"><div class="monitor-inline" id="home-mon"></div>
            <div class="eyebrow">Your latest heart check · ${date(l.recorded_at)}</div><h2>${lv.title}</h2>
            <p>${esc(l.ai ? l.ai.advice : l.summary)}</p>
            <div class="hero-meta"><div><b>${l.heart_rate ? Math.round(l.heart_rate) + " bpm" : "–"}</b>Heart rate</div>
              <div><b>${l.ai ? `${CLASSES.filter((c) => l.ai.per_class[c].result !== "unclear").length} / 5` : "Reviewed"}</b>${l.ai ? "Clear answers" : "By your doctor"}</div>
              <div><b>${l.submission ? { sent: "Sent", seen: "Seen", replied: "Replied" }[l.submission.status] : l.self_check ? "Private" : "Clinic"}</b>Doctor</div></div>
            <div class="row" style="margin-top:18px"><a class="btn" href="#/results/${l.id}">View my result ${icon("arrowRight")}</a><a class="btn ghost-light" href="#/check">New check</a></div></div>`
          : `<div class="hero-status none"><div class="monitor-inline" id="home-mon"></div><div class="eyebrow">Welcome to CardioSentinel</div><h2>Do your first heart check</h2>
            <p>Upload your ECG, or try an example recording, and see five early warning signs explained in plain language.</p>
            <div class="row" style="margin-top:18px"><a class="btn" href="#/check">${icon("activity")}Check my heart</a></div></div>`}
        ${l && l.ai && l.ai.urgent ? urgentBox() : ""}
        <div class="card"><div class="card-head"><div><h2>How are you feeling today?</h2><p>A quick note helps your doctor see the whole picture.</p></div><a class="btn sm ghost" href="#/journal">Open journal ${icon("arrowRight")}</a></div>
          <div class="card-body"><div class="moods" id="moods">${MOODS.map(([v, f, t]) => `<button class="mood" data-v="${v}"><span>${f}</span>${t}</button>`).join("")}</div></div></div>
        <div class="grid g4">
          ${[["#/check", "activity", "crit", "Check my heart", "Upload an ECG"], ["#/journal", "smile", "warn", "Log symptoms", "How do you feel?"],
            ["#/share", "share", "good", "Share with doctor", "Send checks & photos"], ["#/appointments", "calendar", "info", "Book a visit", "Request a time"]]
            .map(([hr, i, t, b, s]) => `<a class="card quick" href="${hr}"><span class="kpi-icon" style="background:var(--${t}-soft);color:var(--${t})">${icon(i)}</span><b>${b}</b><span>${s}</span></a>`).join("")}
        </div>
      </div>
      <div class="stack">
        <div class="card"><div class="card-head"><h2>My doctor</h2>${o.unread_messages ? `<span class="badge crit">${o.unread_messages} new</span>` : ""}</div>
          <div class="card-body"><div class="row" style="flex-wrap:nowrap"><span class="avatar">${initials(o.patient.clinician_name || "?")}</span><div><b>${esc(o.patient.clinician_name || "Not assigned")}</b><div class="small muted">Cardiology</div></div></div>
          <div class="row" style="margin-top:14px"><a class="btn sm primary" href="#/share">${icon("share")}Share</a><a class="btn sm" href="#/doctor?tab=messages">${icon("message")}Messages</a></div></div></div>
        <div class="card"><div class="card-head"><h2>Next appointment</h2></div><div class="card-body">${o.next_appointment ? apptMini(o.next_appointment) : `<p class="muted small">Nothing booked.</p><a class="btn sm" href="#/appointments">Request a visit</a>`}</div></div>
        <div class="card"><div class="card-head"><h2>Recent notes</h2><a class="btn sm ghost" href="#/journal">All</a></div>
          <div class="list" style="margin-top:8px">${o.recent_notes.length ? o.recent_notes.map((n) => `<div class="list-item"><span style="font-size:24px">${moodFace(n.feeling)}</span><div class="grow"><div class="title">${esc(n.symptoms.join(", ") || n.body || "Note")}</div><div class="small muted">${dateTime(n.created_at)}</div></div>${n.shared ? `<span class="badge good">${icon("check")}Shared</span>` : ""}</div>`).join("") : `<div class="empty small">No notes yet.</div>`}</div></div>
      </div>
    </div>`;
  el.querySelectorAll("#moods .mood").forEach((b) => b.addEventListener("click", () => noteModal(Number(b.dataset.v), () => home(el))));
  mountMonitor($("#home-mon"), { bpm: (l && l.heart_rate) || 72, color: "#ffffff", height: 120, glow: true });
}

const urgentBox = () => `<div class="callout crit">${icon("phone")}<div><b>Possible signs of a heart attack.</b> If you have chest pain or pressure, shortness of breath, sweating, nausea or feel faint, <b>call your local emergency number now</b>. Don't wait for a message reply.</div></div>`;
const apptMini = (a) => `<div class="row" style="flex-wrap:nowrap;align-items:flex-start"><div style="text-align:center;min-width:56px;padding:8px;border-radius:12px;background:var(--primary-soft);color:var(--primary)"><div class="tiny" style="text-transform:uppercase;font-weight:700">${new Date(a.starts_at).toLocaleDateString(undefined, { month: "short" })}</div><div style="font-size:22px;font-weight:800">${new Date(a.starts_at).getDate()}</div></div>
  <div><b>${esc(a.kind)}</b><div class="small muted">${dateTime(a.starts_at)}</div><div class="small">${esc(a.location || "")}</div><div style="margin-top:6px">${statusBadge(a.status)}</div></div></div>`;

// ======================================================================= check my heart
async function check(el) {
  const samples = await get("/api/me/samples").catch(() => []);
  let files = [], sym = new Set(), mood = null;
  el.innerHTML = `
    ${head("Check my heart", "Upload your ECG and get your heart check in seconds. It stays private until you choose to share it.")}
    <div class="steps"><span class="step on"><i>1</i>Add your ECG</span><span class="step on"><i>2</i>How you feel</span><span class="step"><i>3</i>Your result</span></div>
    <div class="grid g-2-1">
      <div class="stack" id="check-form">
        <div class="card"><div class="card-head"><div><h2>1 · Add your ECG</h2><p>A 12-lead ECG file from your clinic or device.</p></div></div><div class="card-body">
          <div class="dropzone" id="dz">${icon("upload")}<p style="margin:10px 0 2px"><b>Drop your ECG file here</b> or click to choose</p><p class="muted small">Digital ECG files: CSV, NPY or WFDB (.hea + .dat)</p><input type="file" id="file" multiple hidden accept=".npy,.csv,.txt,.hea,.dat"></div>
          <div id="file-list" class="small" style="margin-top:10px"></div>
          ${samples.length ? `<div class="divider"></div><div class="label" style="margin-bottom:8px">No ECG file? Try an example recording</div><div class="picks">${samples.map((s, i) => `<button class="pick" data-sample="${s.ecg_id}">${icon("activity")}Example ${i + 1}</button>`).join("")}</div>` : ""}
          <div class="callout info" style="margin-top:16px">${icon("camera")}<div class="small">Only have a photo or PDF of your ECG? You can <a href="#/share">share it with your doctor</a>. Automatic checks of ECG photos are coming soon.</div></div>
        </div></div>
        <div class="card"><div class="card-head"><div><h2>2 · How do you feel right now?</h2><p>Optional. Saved to your journal with this check.</p></div></div><div class="card-body">
          <div class="moods" id="c-moods" style="margin-bottom:14px">${MOODS.map(([v, f, t]) => `<button class="mood" data-v="${v}"><span>${f}</span>${t}</button>`).join("")}</div>
          <div class="picks" id="c-sym"></div>
          <textarea class="input" id="c-note" placeholder="Anything else? e.g. racing heart when climbing stairs" style="margin-top:12px;min-height:70px"></textarea>
        </div></div>
        <button class="btn brand lg" id="run" style="align-self:flex-start">${icon("heart")}Check my heart</button>
      </div>
      <div class="stack">
        <div class="card"><div class="card-body small ink2"><h3 style="margin-bottom:8px;color:var(--ink)">What you'll get</h3>
          <p>${icon("check")} A clear overall result</p><p>${icon("check")} Five warning signs explained in plain language, each marked found, not found or unclear</p><p>${icon("check")} Your heart rhythm and heart rate</p><p>${icon("check")} One tap to share it with your doctor</p></div></div>
        <div class="callout warn">${icon("alert")}<div class="small">This is an AI pre-read, not a diagnosis. It cannot rule out a heart attack. If you feel unwell now, call your local emergency number.</div></div>
      </div>
    </div>`;
  const notes = await get("/api/me/notes");
  $("#c-sym").innerHTML = notes.symptom_options.filter((s) => s !== "None").map((s) => `<button class="pick" data-s="${esc(s)}">${esc(s)}</button>`).join("");
  el.querySelectorAll("#c-sym .pick").forEach((b) => b.addEventListener("click", () => { const s = b.dataset.s; sym.has(s) ? sym.delete(s) : sym.add(s); b.classList.toggle("on"); }));
  el.querySelectorAll("#c-moods .mood").forEach((b) => b.addEventListener("click", () => { mood = Number(b.dataset.v); el.querySelectorAll("#c-moods .mood").forEach((x) => x.classList.toggle("on", x === b)); }));
  const dz = $("#dz"), input = $("#file");
  const show = () => { $("#file-list").innerHTML = files.map((f) => `${icon("file")} ${esc(f.name)} <span class="muted">(${fileSize(f.size)})</span>`).join("<br>"); };
  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { files = [...input.files]; show(); });
  dz.addEventListener("dragover", (ev) => { ev.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (ev) => { ev.preventDefault(); dz.classList.remove("drag"); files = [...ev.dataTransfer.files]; show(); });

  const run = async (fn, label) => {
    const noteBody = $("#c-note").value.trim();
    const hasNote = mood !== null || sym.size || noteBody;
    el.innerHTML = `${head("Analysing your ECG", "Watch each step as it happens. It only takes a moment.")}<div id="pipe"></div>`;
    pipelineShell($("#pipe"), label);
    try {
      const r = await fn();
      let noteId = null;
      if (hasNote) noteId = (await post("/api/me/notes", { feeling: mood, symptoms: [...sym], body: noteBody, started: "Just now" })).id;
      const go = () => { location.hash = `#/results/${r.id}${noteId ? `?note=${noteId}` : ""}`; };
      if (r.pipeline) await playPipeline($("#pipe"), r.pipeline, { perClass: r.ai.per_class, level: r.ai.level, onDone: go });
      else go();
    } catch (ex) {
      toast(ex.message, true);
      check(el);
    }
  };
  $("#run").addEventListener("click", () => {
    if (!files.length) return toast("Choose your ECG file first, or try an example recording.", true);
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    run(() => api("/api/me/analyze", { method: "POST", form }), files.map((f) => f.name).join(" + "));
  });
  el.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => run(() => post(`/api/me/analyze/sample/${b.dataset.sample}`), "Example recording")));
}

// ======================================================================= results
async function results(el) {
  const rows = await get("/api/me/ecgs");
  let tab = "all";
  const draw = () => {
    const list = rows.filter((e) => tab === "all" || (tab === "shared" ? e.released && (e.submission || !e.self_check) : e.released && e.self_check && !e.submission));
    $("#res-list").innerHTML = list.length ? list.map((e) => {
      if (!e.released) return `<div class="list-item"><span class="kpi-icon" style="background:var(--surface-3);color:var(--ink-3)">${icon("clock")}</span><div class="grow"><div class="title">Clinic ECG · result on the way</div><div class="small muted">${dateTime(e.recorded_at)} · Your doctor is reviewing it</div></div>${statusBadge("pending")}</div>`;
      const lv = LEVELS[levelOf(e)];
      return `<a class="list-item" href="#/results/${e.id}"><span class="kpi-icon" style="background:var(--${lv.tone}-soft);color:var(--${lv.tone})">${icon(lv.icon)}</span>
        <div class="grow"><div class="title">${lv.title}</div><div class="small muted">${dateTime(e.recorded_at)} · ${e.self_check ? "Heart check" : "Clinic ECG"}${e.heart_rate ? ` · ${Math.round(e.heart_rate)} bpm` : ""}</div></div>
        <div class="row">${shareState(e)}${icon("arrowRight")}</div></a>`;
    }).join("") : `<div class="empty">${icon("activity")}<p>No results here yet.</p><a class="btn primary" href="#/check">Check my heart</a></div>`;
  };
  el.innerHTML = `${head("My results", "Every heart check and clinic ECG in one place.", `<a class="btn brand" href="#/check">${icon("activity")}New heart check</a>`)}
    <div class="tabs" id="r-tabs"><button class="active" data-t="all">All</button><button data-t="private">Private</button><button data-t="shared">Shared with doctor</button></div>
    <div class="card"><div class="list" id="res-list" style="margin-top:-1px"></div></div>`;
  el.querySelectorAll("#r-tabs button").forEach((b) => b.addEventListener("click", () => { tab = b.dataset.t; el.querySelectorAll("#r-tabs button").forEach((x) => x.classList.toggle("active", x === b)); draw(); }));
  draw();
}

async function result(el, { id }) {
  const e = await get(`/api/ecgs/${id}`);
  const lv = LEVELS[levelOf(e)];
  const noteId = query().get("note");
  el.innerHTML = `
    ${head(e.self_check ? "Your heart check" : "Your clinic ECG", `${dateTime(e.recorded_at)}${e.heart_rate ? ` · heart rate about ${Math.round(e.heart_rate)} beats per minute` : ""}`, `${e.pipeline ? `<button class="btn no-print" id="replay">${icon("cpu")}How this was analysed</button>` : ""}<button class="btn no-print" onclick="window.print()">${icon("printer")}Print</button>`, `<a href="#/results">My results</a> / ${date(e.recorded_at)}`)}
    <div class="grid g-2-1">
      <div class="stack">
        <div class="hero-status ${lv.tone}"><div class="monitor-inline" id="res-mon"></div><div class="eyebrow">${e.self_check ? "AI heart check" : e.reviewed_by ? `Reviewed by ${esc(e.reviewed_by)}` : "Clinic ECG"}</div>
          <h2>${lv.title}</h2><p>${esc(e.ai ? e.ai.advice : e.summary)}</p>
          ${e.ai && e.ai.unclear.length ? `<p style="margin-top:6px"><b>Unclear:</b> ${e.ai.unclear.map((c) => esc(CLASS_NAMES[c])).join(", ")}</p>` : ""}</div>
        ${e.ai && e.ai.urgent ? urgentBox() : ""}
        ${e.reviewed_by && e.self_check ? `<div class="callout good">${icon("stethoscope")}<div><b>${esc(e.reviewed_by)} reviewed this check.</b> ${esc(e.summary)}</div></div>` : ""}
        ${e.ai ? `<div class="card"><div class="card-head"><div><h2>What we checked</h2><p>Five early warning signs. "Unclear" means the AI could not decide, so a doctor should look.</p></div></div><div class="card-body flush">${findingList(e.ai.per_class)}</div></div>`
          : `<div class="card"><div class="card-head"><div><h2>What your doctor found</h2></div></div><div class="card-body flush">${CLASSES.map((c) => { const y = e.findings.includes(c); return `<div class="finding"><div class="ic" style="background:var(--${y ? (c === "NORM" ? "good" : "crit") : "surface-3"}${y ? "-soft" : ""});color:var(--${y ? (c === "NORM" ? "good" : "crit") : "ink-3"})">${icon(y ? (c === "NORM" ? "check" : "alert") : "x")}</div><div><h3>${esc(CLASS_NAMES[c])}</h3><p>${esc(PLAIN[c])}</p></div><div class="res"><span class="badge ${y ? (c === "NORM" ? "good" : "crit") : "neutral"}">${y ? (c === "NORM" ? "Yes" : "Found") : c === "NORM" ? "No" : "Not found"}</span></div></div>`; }).join("")}</div></div>`}
        <div class="card"><div class="card-head" style="padding-bottom:12px"><div><h2>Your heart rhythm</h2><p>10 seconds of lead II. Each tall spike is one heartbeat.</p></div></div><div id="rhythm"></div></div>
      </div>
      <div class="stack">
        <div class="card"><div class="card-head"><h2>Share with your doctor</h2>${shareState(e)}</div><div class="card-body">
          ${e.submission ? `<div class="progress">${["sent", "seen", "replied"].map((s, i) => `${i ? '<span class="bar"></span>' : ""}<span class="p ${["sent", "seen", "replied"].indexOf(e.submission.status) >= i ? "done" : ""}"><i></i>${{ sent: "Sent", seen: "Seen", replied: "Replied" }[s]}</span>`).join("")}</div>
              <a class="btn sm" style="margin-top:12px" href="#/doctor/${e.submission.id}">View what you sent ${icon("arrowRight")}</a>`
            : e.self_check ? `<p class="small ink2">Your doctor can't see this check yet. Share it with notes on how you feel, photos or documents.</p><a class="btn primary" href="#/share?ecg=${e.id}${noteId ? `&note=${noteId}` : ""}" style="width:100%;justify-content:center">${icon("share")}Share with my doctor</a>`
            : `<p class="small ink2">This ECG was recorded at your clinic, so your doctor already has it.</p><a class="btn sm" href="#/doctor?tab=messages">${icon("message")}Ask a question</a>`}
        </div></div>
        <div class="card"><div class="card-body"><h3 style="margin-bottom:10px">Next steps</h3>
          <a class="list-item" style="padding:10px 0;border:0" href="#/journal">${icon("smile")}<span class="grow">Log how you feel</span>${icon("arrowRight")}</a>
          <a class="list-item" style="padding:10px 0" href="#/appointments">${icon("calendar")}<span class="grow">Book a visit</span>${icon("arrowRight")}</a>
          <a class="list-item" style="padding:10px 0" href="#/learn">${icon("book")}<span class="grow">What do these results mean?</span>${icon("arrowRight")}</a></div></div>
        <div class="callout">${icon("info")}<div class="small">An AI pre-read, not a diagnosis. It cannot rule out heart disease or predict cardiac arrest. Talk to your doctor about any symptoms.</div></div>
      </div>
    </div>`;
  const sig = await get(`/api/ecgs/${id}/signal`);
  mountEcg($("#rhythm"), sig, { layout: "single", toolbar: false, rhythmLead: "II" });
  mountMonitor($("#res-mon"), { bpm: e.heart_rate || 72, color: "#ffffff", height: 120 });
  const rp = $("#replay");
  const replay = () => modal({
    title: "How your ECG was analysed", body: `<div id="pipe-modal"></div>`,
    onMount: (m) => {
      m.querySelector(".modal").classList.add("wide");
      const box = m.querySelector("#pipe-modal");
      pipelineShell(box, e.pipeline.input && e.pipeline.input.filename);
      box.querySelector(".pipe").style.cssText = "box-shadow:none;border:0;padding:0";
      playPipeline(box, e.pipeline, { perClass: e.ai.per_class, level: e.ai.level });
    },
  });
  if (rp) {
    rp.addEventListener("click", replay);
    if (query().get("replay")) replay();
  }
}

// ======================================================================= journal
function noteForm(opts, pre = {}) {
  return `<div class="label" style="margin-bottom:8px">How do you feel overall?</div>
    <div class="moods" id="n-moods" style="margin-bottom:16px">${MOODS.map(([v, f, t]) => `<button type="button" class="mood ${pre.feeling === v ? "on" : ""}" data-v="${v}"><span>${f}</span>${t}</button>`).join("")}</div>
    <div class="label" style="margin-bottom:8px">Any symptoms?</div>
    <div class="picks" id="n-sym" style="margin-bottom:16px">${opts.filter((s) => s !== "None").map((s) => `<button type="button" class="pick" data-s="${esc(s)}">${esc(s)}</button>`).join("")}</div>
    <div class="grid g2" style="gap:0 14px"><div class="field"><label>How strong? <b id="n-sev-v">5</b>/10</label><input type="range" id="n-sev" min="0" max="10" value="5"></div>
    <div class="field"><label>When did it start?</label><select class="input" id="n-when">${WHEN.map((w) => `<option>${w}</option>`).join("")}</select></div></div>
    <div class="field"><label>In your own words</label><textarea class="input" id="n-body" placeholder="e.g. My heart races when I climb stairs, and I felt dizzy this morning."></textarea></div>`;
}
function bindNoteForm(root, state) {
  root.querySelectorAll("#n-moods .mood").forEach((b) => b.addEventListener("click", () => { state.feeling = Number(b.dataset.v); root.querySelectorAll("#n-moods .mood").forEach((x) => x.classList.toggle("on", x === b)); }));
  root.querySelectorAll("#n-sym .pick").forEach((b) => b.addEventListener("click", () => { const s = b.dataset.s; state.sym.has(s) ? state.sym.delete(s) : state.sym.add(s); b.classList.toggle("on"); }));
  const sev = root.querySelector("#n-sev");
  sev.addEventListener("input", () => { root.querySelector("#n-sev-v").textContent = sev.value; });
  return async () => post("/api/me/notes", { feeling: state.feeling, symptoms: [...state.sym], severity: state.sym.size ? Number(sev.value) : null,
    started: root.querySelector("#n-when").value, body: root.querySelector("#n-body").value });
}
async function noteModal(feeling, done) {
  const { symptom_options } = await get("/api/me/notes");
  const state = { feeling, sym: new Set() };
  modal({
    title: "How are you feeling?", body: noteForm(symptom_options, { feeling }),
    foot: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="n-save">Save note</button>`,
    onMount: (m, close) => { const save = bindNoteForm(m, state); m.querySelector("#n-save").addEventListener("click", async () => { try { await save(); close(); toast("Saved to your journal"); done && done(); } catch (ex) { toast(ex.message, true); } }); },
  });
}

async function journal(el) {
  const { notes, symptom_options } = await get("/api/me/notes");
  const state = { feeling: null, sym: new Set() };
  el.innerHTML = `${head("How I feel", "Keep a simple diary of symptoms and how you feel. Share entries with your doctor when you want.", `<a class="btn" href="#/share">${icon("share")}Share with doctor</a>`)}
    <div class="grid g-1-2">
      <div class="card" style="align-self:start"><div class="card-head"><h2>New entry</h2></div><div class="card-body">${noteForm(symptom_options)}<button class="btn primary" id="n-save">${icon("check")}Save entry</button></div></div>
      <div class="card"><div class="card-head"><div><h2>My journal</h2><p>${notes.length} entr${notes.length === 1 ? "y" : "ies"}</p></div></div>
        <div style="margin-top:12px">${notes.length ? notes.map((n) => `<div class="timeline-item"><div class="face">${moodFace(n.feeling)}</div><div>
          <div class="row between"><b>${dateTime(n.created_at)}</b><div class="row">${n.shared ? `<span class="badge good">${icon("check")}Shared</span>` : `<a class="btn sm" href="#/share?note=${n.id}">${icon("share")}Share</a><button class="btn sm ghost danger" data-del="${n.id}" aria-label="Delete">${icon("trash")}</button>`}</div></div>
          ${n.symptoms.length ? `<div class="picks" style="margin:8px 0">${n.symptoms.map((s) => `<span class="chip unclear">${esc(s)}</span>`).join("")}${n.severity !== null ? `<span class="chip">Strength ${n.severity}/10</span>` : ""}${n.started ? `<span class="chip">${esc(n.started)}</span>` : ""}</div>` : ""}
          ${n.body ? `<p class="ink2" style="margin:6px 0 0">${esc(n.body)}</p>` : ""}</div></div>`).join("") : `<div class="empty">${icon("smile")}<p>No entries yet. How are you feeling today?</p></div>`}</div></div>
    </div>`;
  const save = bindNoteForm(el, state);
  $("#n-save").addEventListener("click", async () => { try { await save(); toast("Saved to your journal"); journal(el); } catch (ex) { toast(ex.message, true); } });
  el.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("Delete this entry?")) return;
    await api(`/api/me/notes/${b.dataset.del}`, { method: "DELETE" }); journal(el);
  }));
}

// ======================================================================= share with doctor
export async function thumbsHtml(list, removable) {
  const out = [];
  for (const a of list) {
    let inner;
    if (a.mime.startsWith("image/")) {
      try { inner = `<img alt="${esc(a.filename)}" src="${await blobUrl(`/api/attachments/${a.id}`)}">`; } catch (e) { inner = `<div class="pdf">${icon("file")}${esc(a.filename)}</div>`; }
    } else inner = `<div class="pdf">${icon("file")}${esc(a.filename)}</div>`;
    out.push(`<div class="thumb" data-open="${a.id}" data-mime="${a.mime}" title="${esc(a.filename)} · ${fileSize(a.size)}">${inner}${removable ? `<button class="x" data-rm="${a.id}" aria-label="Remove">${icon("x")}</button>` : ""}</div>`);
  }
  return out.join("");
}
export function bindThumbOpen(root) {
  root.querySelectorAll("[data-open]").forEach((t) => t.addEventListener("click", async (ev) => {
    if (ev.target.closest("[data-rm]")) return;
    const url = await blobUrl(`/api/attachments/${t.dataset.open}`);
    if (t.dataset.mime.startsWith("image/")) modal({ title: t.title, body: `<img src="${url}" alt="" style="width:100%;border-radius:10px">` });
    else window.open(url, "_blank");
  }));
}

async function share(el) {
  const q = query();
  const [ecgs, { notes }, uploads, o] = await Promise.all([get("/api/me/ecgs"), get("/api/me/notes"), get("/api/me/uploads"), get("/api/me/overview")]);
  const checks = ecgs.filter((e) => e.released && e.self_check && !e.submission);
  const pre = q.get("ecg");
  const preNotes = new Set((q.get("note") || "").split(",").filter(Boolean));
  let atts = uploads;
  el.innerHTML = `${head("Share with my doctor", `Choose what ${esc(o.patient.clinician_name || "your doctor")} should see. Nothing is sent until you press Send.`, "", `<a href="#/doctor">My doctor</a> / Share`)}
    <div class="grid g-2-1">
      <div class="stack">
        <div class="card"><div class="card-head"><div><h2>${icon("activity")} Heart check</h2><p>Optional. Pick one of your private checks.</p></div></div><div class="card-body">
          ${checks.length ? `<div class="list" style="margin:0 -20px">${[{ id: "" }, ...checks].map((e) => `<label class="list-item" style="cursor:pointer"><input type="radio" name="ecg" value="${e.id}" ${String(e.id) === String(pre || "") ? "checked" : ""}>
            ${e.id ? `<div class="grow"><b>${LEVELS[levelOf(e)].title}</b><div class="small muted">${dateTime(e.recorded_at)}</div></div><span class="badge ${LEVELS[levelOf(e)].tone}">${LEVELS[levelOf(e)].short}</span>` : `<div class="grow muted">Don't include a heart check</div>`}</label>`).join("")}</div>`
            : `<p class="small muted">No private heart checks to share. <a href="#/check">Do a heart check</a></p>`}</div></div>
        <div class="card"><div class="card-head"><div><h2>${icon("smile")} How I've been feeling</h2><p>Pick journal entries to include.</p></div><a class="btn sm ghost" href="#/journal">${icon("plus")}New entry</a></div><div class="card-body">
          ${notes.filter((n) => !n.shared).length ? `<div class="list" style="margin:0 -20px">${notes.filter((n) => !n.shared).slice(0, 12).map((n) => `<label class="list-item" style="cursor:pointer"><input type="checkbox" name="note" value="${n.id}" ${preNotes.has(String(n.id)) ? "checked" : ""}><span style="font-size:22px">${moodFace(n.feeling)}</span><div class="grow"><div class="title">${esc(n.symptoms.join(", ") || n.body || "Note")}</div><div class="small muted">${dateTime(n.created_at)}${n.body && n.symptoms.length ? ` · ${esc(n.body.slice(0, 60))}` : ""}</div></div></label>`).join("")}</div>`
            : `<p class="small muted">No unshared journal entries. <a href="#/journal">Write one</a></p>`}</div></div>
        <div class="card"><div class="card-head"><div><h2>${icon("paperclip")} Photos & documents</h2><p>e.g. a photo of a paper ECG, lab results or a medication list. JPG, PNG, WEBP or PDF, up to 10 MB each.</p></div></div><div class="card-body">
          <div class="dropzone" id="adz" style="padding:22px">${icon("camera")}<p style="margin:8px 0 0"><b>Add photos or PDFs</b></p><input type="file" id="afile" multiple hidden accept="image/png,image/jpeg,image/webp,image/gif,application/pdf"></div>
          <div class="thumbs" id="athumbs" style="margin-top:12px"></div></div></div>
        <div class="card"><div class="card-head"><h2>${icon("message")} Message</h2></div><div class="card-body">
          <textarea class="input" id="s-msg" placeholder="e.g. I've had palpitations for a week. Could you look at my heart check?"></textarea></div></div>
        <button class="btn brand lg" id="s-send" style="align-self:flex-start">${icon("send")}Send to my doctor</button>
      </div>
      <div class="stack">
        <div class="card"><div class="card-body"><div class="row" style="flex-wrap:nowrap"><span class="avatar lg">${initials(o.patient.clinician_name || "?")}</span><div><b>${esc(o.patient.clinician_name || "Your doctor")}</b><div class="small muted">Cardiology · usually replies within 2 working days</div></div></div></div></div>
        <div class="callout">${icon("lock")}<div class="small">Only your doctor sees what you send. Shared heart checks also go to their review list.</div></div>
        <div class="callout crit">${icon("phone")}<div class="small"><b>Not for emergencies.</b> If you feel unwell now, call your local emergency number.</div></div>
      </div>
    </div>`;
  const drawAtts = async () => { $("#athumbs").innerHTML = await thumbsHtml(atts, true); bindRm(); bindThumbOpen(el); };
  const bindRm = () => el.querySelectorAll("[data-rm]").forEach((b) => b.addEventListener("click", async (ev) => {
    ev.stopPropagation(); await api(`/api/me/attachments/${b.dataset.rm}`, { method: "DELETE" }); atts = atts.filter((a) => String(a.id) !== b.dataset.rm); drawAtts();
  }));
  const upload = async (fileList) => {
    for (const f of fileList) {
      const form = new FormData(); form.append("file", f);
      try { atts = [await api("/api/me/attachments", { method: "POST", form }), ...atts]; } catch (ex) { toast(`${f.name}: ${ex.message}`, true); }
    }
    drawAtts();
  };
  const adz = $("#adz"), afile = $("#afile");
  adz.addEventListener("click", () => afile.click());
  afile.addEventListener("change", () => upload([...afile.files]));
  adz.addEventListener("dragover", (ev) => { ev.preventDefault(); adz.classList.add("drag"); });
  adz.addEventListener("dragleave", () => adz.classList.remove("drag"));
  adz.addEventListener("drop", (ev) => { ev.preventDefault(); adz.classList.remove("drag"); upload([...ev.dataTransfer.files]); });
  drawAtts();
  $("#s-send").addEventListener("click", async () => {
    const ecg = el.querySelector("input[name=ecg]:checked");
    const body = { ecg_id: ecg && ecg.value ? Number(ecg.value) : null, note_ids: [...el.querySelectorAll("input[name=note]:checked")].map((x) => Number(x.value)),
      attachment_ids: atts.map((a) => a.id), message: $("#s-msg").value };
    try {
      const s = await post("/api/me/submissions", body);
      toast("Sent to your doctor");
      location.hash = `#/doctor/${s.id}`;
    } catch (ex) { toast(ex.message, true); }
  });
}

// ======================================================================= my doctor
async function doctor(el) {
  const [subs, o] = await Promise.all([get("/api/me/submissions"), get("/api/me/overview")]);
  const tab = query().get("tab") || "shared";
  el.innerHTML = `${head("My doctor", "", `<a class="btn primary" href="#/share">${icon("share")}Share something new</a>`)}
    <div class="card" style="margin-bottom:18px"><div class="card-body row" style="flex-wrap:nowrap"><span class="avatar lg">${initials(o.patient.clinician_name || "?")}</span><div class="grow"><h2>${esc(o.patient.clinician_name || "Not assigned")}</h2><div class="small muted">Cardiology · replies usually within 2 working days</div></div><a class="btn" href="#/appointments">${icon("calendar")}Book a visit</a></div></div>
    <div class="tabs" id="d-tabs"><button data-t="shared" class="${tab === "shared" ? "active" : ""}">What I've shared (${subs.length})</button><button data-t="messages" class="${tab === "messages" ? "active" : ""}">Messages${o.unread_messages ? ` <span class="badge crit">${o.unread_messages}</span>` : ""}</button></div>
    <div id="d-body"></div>`;
  const show = async (t) => {
    el.querySelectorAll("#d-tabs button").forEach((b) => b.classList.toggle("active", b.dataset.t === t));
    if (t === "shared") {
      $("#d-body").innerHTML = `<div class="card"><div class="list" style="margin-top:-1px">${subs.length ? subs.map((s) => `<a class="list-item" href="#/doctor/${s.id}"><span class="kpi-icon" style="background:var(--primary-soft);color:var(--primary)">${icon("send")}</span>
        <div class="grow"><div class="title">${esc(s.message || "Shared with your doctor")}</div><div class="small muted">${dateTime(s.created_at)} · ${[s.ecg_id ? "heart check" : "", s.note_ids.length ? `${s.note_ids.length} note${s.note_ids.length > 1 ? "s" : ""}` : "", s.n_attachments ? `${s.n_attachments} file${s.n_attachments > 1 ? "s" : ""}` : ""].filter(Boolean).join(" · ") || "message"}</div>
        ${s.reply ? `<div class="small" style="margin-top:4px;color:var(--good)">${icon("message")} ${esc(s.reply.slice(0, 90))}${s.reply.length > 90 ? "…" : ""}</div>` : ""}</div>${statusBadge(s.status)}</a>`).join("")
        : `<div class="empty">${icon("share")}<p>You haven't shared anything yet.</p><a class="btn primary" href="#/share">Share with my doctor</a></div>`}</div></div>`;
    } else {
      const me = (await shell()).app.user;
      const draw = (msgs) => {
        $("#d-body").innerHTML = `<div class="card"><div class="chat-pane"><div class="chat-msgs" id="msgs" style="max-height:540px">${msgs.map((m) => `<div class="bubble ${m.sender_id === me.id ? "me" : "them"}">${esc(m.body)}<div class="meta">${esc(m.sender_name)} · ${dateTime(m.created_at)}</div></div>`).join("") || `<div class="empty">No messages yet.</div>`}</div>
          <form class="chat-input" id="send"><textarea class="input" placeholder="Write to your doctor…" required></textarea><button class="btn primary">${icon("send")}Send</button></form></div></div>
          <p class="tiny muted" style="margin-top:8px">${icon("phone")} Messages are not for emergencies.</p>`;
        const box = $("#msgs"); box.scrollTop = box.scrollHeight;
        $("#send").addEventListener("submit", async (ev) => { ev.preventDefault(); const ta = $("#send textarea"); if (!ta.value.trim()) return; draw(await post("/api/me/messages", { body: ta.value })); });
      };
      draw(await get("/api/me/messages"));
      refresh();
    }
  };
  el.querySelectorAll("#d-tabs button").forEach((b) => b.addEventListener("click", () => show(b.dataset.t)));
  show(tab);
}

async function submission(el, { id }) {
  const s = await get(`/api/me/submissions/${id}`);
  const steps = ["sent", "seen", "replied"], idx = steps.indexOf(s.status);
  el.innerHTML = `${head("What you shared", dateTime(s.created_at), "", `<a href="#/doctor">My doctor</a> / Shared`)}
    <div class="grid g-2-1"><div class="stack">
      <div class="card"><div class="card-body"><div class="progress">${steps.map((x, i) => `${i ? '<span class="bar"></span>' : ""}<span class="p ${idx >= i ? "done" : ""}"><i></i>${{ sent: "Sent", seen: "Seen by doctor", replied: "Doctor replied" }[x]}</span>`).join("")}</div></div></div>
      ${s.reply ? `<div class="card" style="border-color:color-mix(in srgb, var(--good) 40%, var(--border))"><div class="card-head"><div><h2>${icon("stethoscope")} ${esc(s.replied_by_name || "Your doctor")} replied</h2><p>${dateTime(s.replied_at)}</p></div></div><div class="card-body"><p style="white-space:pre-wrap">${esc(s.reply)}</p><a class="btn sm" href="#/doctor?tab=messages">${icon("message")}Reply</a></div></div>` : ""}
      ${s.message ? `<div class="card"><div class="card-head"><h2>Your message</h2></div><div class="card-body"><p style="white-space:pre-wrap">${esc(s.message)}</p></div></div>` : ""}
      ${s.ecg ? `<a class="card list-item" style="border-top:1px solid var(--border)" href="#/results/${s.ecg.id}"><span class="kpi-icon" style="background:var(--${LEVELS[levelOf(s.ecg)].tone}-soft);color:var(--${LEVELS[levelOf(s.ecg)].tone})">${icon("activity")}</span><div class="grow"><div class="title">Heart check · ${LEVELS[levelOf(s.ecg)].title}</div><div class="small muted">${dateTime(s.ecg.recorded_at)}</div></div>${icon("arrowRight")}</a>` : ""}
      ${s.notes.length ? `<div class="card"><div class="card-head"><h2>Journal entries</h2></div><div style="margin-top:12px">${s.notes.map((n) => `<div class="timeline-item"><div class="face">${moodFace(n.feeling)}</div><div><b>${dateTime(n.created_at)}</b><div class="picks" style="margin:6px 0">${n.symptoms.map((x) => `<span class="chip unclear">${esc(x)}</span>`).join("")}${n.severity !== null ? `<span class="chip">Strength ${n.severity}/10</span>` : ""}</div>${n.body ? `<p class="ink2">${esc(n.body)}</p>` : ""}</div></div>`).join("")}</div></div>` : ""}
      ${s.attachments.length ? `<div class="card"><div class="card-head"><h2>Photos & documents</h2></div><div class="card-body"><div class="thumbs" id="sthumbs"></div></div></div>` : ""}
    </div><div class="stack"><div class="callout">${icon("info")}<div class="small">Your doctor sees exactly what's on this page. To add something, <a href="#/share">share again</a> or send a message.</div></div></div></div>`;
  if (s.attachments.length) { $("#sthumbs").innerHTML = await thumbsHtml(s.attachments, false); bindThumbOpen(el); }
}

// ======================================================================= appointments, learn, profile
async function appointments(el) {
  const rows = await get("/api/me/appointments");
  const now = new Date().toISOString();
  const up = rows.filter((a) => a.starts_at >= now && a.status !== "cancelled").reverse();
  const past = rows.filter((a) => a.starts_at < now || a.status === "cancelled");
  const card = (a, canCancel) => `<div class="list-item">${apptMini(a)}<div class="grow"></div>${canCancel ? `<button class="btn sm danger" data-cancel="${a.id}">Cancel</button>` : ""}</div>`;
  el.innerHTML = `${head("Appointments", "Your visits with the cardiology team", `<button class="btn primary" id="req">${icon("plus")}Request a visit</button>`)}
    <div class="card" style="margin-bottom:18px"><div class="card-head"><h2>Upcoming</h2></div><div class="list" style="margin-top:8px">${up.length ? up.map((a) => card(a, true)).join("") : `<div class="empty small">No upcoming appointments.</div>`}</div></div>
    <div class="card"><div class="card-head"><h2>Past</h2></div><div class="list" style="margin-top:8px">${past.length ? past.map((a) => card(a, false)).join("") : `<div class="empty small">None yet.</div>`}</div></div>`;
  el.querySelectorAll("[data-cancel]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm("Cancel this appointment?")) return;
    await patch(`/api/appointments/${b.dataset.cancel}`, { status: "cancelled" }); toast("Appointment cancelled"); appointments(el);
  }));
  $("#req").addEventListener("click", () => {
    const d = new Date(Date.now() + 7 * 86400000); d.setHours(10, 0, 0, 0);
    modal({
      title: "Request a visit",
      body: `<form id="rq"><div class="field"><label>Preferred date & time</label><input class="input" type="datetime-local" name="starts_at" value="${d.toISOString().slice(0, 16)}" required></div>
        <div class="field"><label>Reason</label><select class="input" name="kind"><option>Discuss my heart check</option><option>New or worse symptoms</option><option>Cardiology consultation</option><option>Medication review</option></select></div>
        <div class="field"><label>Anything we should know? (optional)</label><textarea class="input" name="notes"></textarea></div>
        <p class="small muted">The clinic will confirm the exact time with you.</p></form>`,
      foot: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="rq-send">Send request</button>`,
      onMount: (m, close) => m.querySelector("#rq-send").addEventListener("click", async () => {
        const body = Object.fromEntries(new FormData(m.querySelector("#rq")).entries());
        body.starts_at = new Date(body.starts_at).toISOString();
        await post("/api/me/appointments", body); close(); toast("Request sent. The clinic will confirm."); appointments(el);
      }),
    });
  });
}

async function learn(el) {
  const more = {
    NORM: "A normal ECG is reassuring, but it is a 10-second snapshot. If you have symptoms, tell your doctor even when your check looks normal.",
    MI: "This can reflect an old heart attack, sometimes one that went unnoticed, or a recent one. Your doctor decides whether you need more tests, such as blood tests or a heart ultrasound.",
    STTC: "These changes have many causes, from reduced blood flow to the heart to medication or salt levels in the blood. They are judged together with your symptoms.",
    CD: "Bundle branch blocks and similar findings are common and often harmless, but some need monitoring.",
    HYP: "Often linked to long-term high blood pressure. It is usually confirmed with an ultrasound of the heart (echocardiogram).",
  };
  el.innerHTML = `${head("Heart health guide", "What your results mean, in plain language")}
    <div class="grid g3">${CLASSES.map((c) => `<div class="card learn-card"><span class="chip ${c === "NORM" ? "norm" : "pos"}">${esc(CLASS_NAMES[c])}</span><h3>${esc(PLAIN[c])}.</h3><p class="small ink2">${more[c]}</p></div>`).join("")}
      <div class="card learn-card"><span class="chip unclear">Unclear</span><h3>When the AI can't decide</h3><p class="small ink2">Some ECGs are borderline. Instead of guessing, CardioSentinel marks the result "unclear" and suggests you share it with your doctor.</p></div></div>
    <div class="grid g2" style="margin-top:18px">
      <div class="card"><div class="card-head"><h2>Warning signs: call emergency services</h2></div><div class="card-body small ink2"><p>• Chest pain or pressure lasting more than a few minutes</p><p>• Pain spreading to the arm, jaw, neck or back</p><p>• Sudden severe shortness of breath</p><p>• Fainting, or a very fast irregular heartbeat with dizziness</p><p>• Cold sweat or nausea together with chest discomfort</p></div></div>
      <div class="card"><div class="card-head"><h2>Looking after your heart</h2></div><div class="card-body small ink2"><p>• Know your blood pressure and cholesterol</p><p>• Stay active: about 150 minutes of moderate activity a week</p><p>• Don't smoke, and limit alcohol</p><p>• Take your medication as prescribed</p><p>• Keep a note of symptoms: patterns help your doctor</p></div></div>
    </div>
    <div class="callout" style="margin-top:18px">${icon("info")}<div class="small">CardioSentinel gives an AI pre-read of your ECG. It can't diagnose heart disease, rule out a heart attack, or predict cardiac arrest. Always talk to your doctor about symptoms.</div></div>`;
}

async function profile(el) {
  const o = await get("/api/me/overview");
  const p = o.patient;
  el.innerHTML = `${head("My profile", "Keep your details up to date")}
    <div class="grid g-1-2">
      <div class="card"><div class="card-body"><div class="row" style="margin-bottom:14px"><span class="avatar lg">${initials(p.name)}</span><div><h2>${esc(p.name)}</h2><div class="muted small">Member ID ${esc(p.mrn)}</div></div></div>
        <dl class="kv"><dt>Date of birth</dt><dd>${date(p.dob)}</dd><dt>Sex</dt><dd>${esc(p.sex || "–")}</dd><dt>Conditions</dt><dd>${esc(p.conditions || "–")}</dd><dt>My doctor</dt><dd>${esc(p.clinician_name || "–")}</dd></dl></div></div>
      <div class="card"><div class="card-head"><h2>Contact & health details</h2></div><div class="card-body"><form id="pf"><div class="grid g2" style="gap:0 14px">
        <div class="field"><label>Phone</label><input class="input" name="phone" value="${esc(p.phone || "")}"></div>
        <div class="field"><label>Email</label><input class="input" type="email" name="email" value="${esc(p.email || "")}"></div>
        <div class="field" style="grid-column:1/-1"><label>Address</label><input class="input" name="address" value="${esc(p.address || "")}"></div>
        <div class="field" style="grid-column:1/-1"><label>Emergency contact</label><input class="input" name="emergency_contact" value="${esc(p.emergency_contact || "")}"></div>
        <div class="field"><label>Current medications</label><input class="input" name="medications" value="${esc(p.medications || "")}"></div>
        <div class="field"><label>Allergies</label><input class="input" name="allergies" value="${esc(p.allergies || "")}"></div></div>
        <button class="btn primary">${icon("check")}Save changes</button></form></div></div>
    </div>`;
  $("#pf").addEventListener("submit", async (ev) => { ev.preventDefault(); await patch("/api/me/profile", Object.fromEntries(new FormData(ev.target).entries())); toast("Profile updated"); });
}

export const PATIENT_ROUTES = [
  { path: "/home", title: "Home", render: home },
  { path: "/check", title: "Check my heart", render: check },
  { path: "/results", title: "My results", render: results },
  { path: "/results/:id", title: "Result", render: result, nav: "#/results" },
  { path: "/journal", title: "How I feel", render: journal },
  { path: "/share", title: "Share with my doctor", render: share, nav: "#/doctor" },
  { path: "/doctor", title: "My doctor", render: doctor },
  { path: "/doctor/:id", title: "Shared", render: submission, nav: "#/doctor" },
  { path: "/appointments", title: "Appointments", render: appointments },
  { path: "/learn", title: "Heart health guide", render: learn },
  { path: "/profile", title: "My profile", render: profile },
];
