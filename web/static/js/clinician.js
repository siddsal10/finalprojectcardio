// Doctor workspace: what patients share, ECGs to review, patients, messages, appointments.
import { api, get, patch, post } from "./api.js";
import { mountEcg } from "./ecg.js";
import { mountHeart } from "./heart3d.js";
import { pipelineShell, playPipeline } from "./pipeline.js";
import { bindThumbOpen, downloadCsv, thumbsHtml } from "./patient.js";
import {
  $, CLASSES, CLASS_NAMES, LEVELS, ago, date, dateTime, esc, findingChips, findingList, icon, initials, loading, modal,
  moodFace, perClassFromPred, pictureNotes, statusBadge, toast,
} from "./ui.js";

const shell = () => import("./app.js");
const refresh = () => shell().then((m) => m.refreshCounts());
const query = () => new URLSearchParams(location.hash.split("?")[1] || "");
const head = (title, sub = "", actions = "", crumbs = "") => `
  <div class="page-head"><div>${crumbs ? `<div class="crumbs">${crumbs}</div>` : ""}<h1>${title}</h1>${sub ? `<p>${sub}</p>` : ""}</div>
  ${actions ? `<div class="row">${actions}</div>` : ""}</div>`;
const rowLinks = (el) => el.querySelectorAll("tr[data-href]").forEach((tr) => tr.addEventListener("click", () => (location.hash = tr.dataset.href)));

/** Overall level of a prediction, same wording as the patient sees. */
function levelOfPred(pred) {
  if (CLASSES.some((c) => c !== "NORM" && pred.triage[c] === "positive")) return "see_doctor";
  if (pred.indeterminate && pred.indeterminate.length) return "check";
  return "reassuring";
}
const levelBadge = (pred) => { const l = LEVELS[levelOfPred(pred)]; return `<span class="badge ${l.tone}">${icon(l.icon)}${l.short}</span>`; };
const unclearChips = (pred) => (pred.indeterminate && pred.indeterminate.length
  ? pred.indeterminate.map((c) => `<span class="chip unclear">${esc(CLASS_NAMES[c])}</span>`).join(" ")
  : `<span class="muted small">none</span>`);

// ======================================================================= patient shares (inbox)
async function inbox(el) {
  const rows = await get("/api/submissions");
  const fresh = rows.filter((r) => r.status === "sent").length;
  el.innerHTML = `${head("Patient shares", fresh ? `${fresh} new since you last looked` : "Everything patients chose to send you")}
    <div class="card">${rows.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>Patient</th><th>Sent</th><th>Message</th><th>Includes</th><th>Heart check</th><th>Status</th></tr></thead><tbody>
      ${rows.map((s) => `<tr class="click" data-href="#/inbox/${s.id}" style="${s.status === "sent" ? "font-weight:600" : ""}"><td><div class="row" style="flex-wrap:nowrap"><span class="avatar">${initials(s.patient_name)}</span><div><b>${esc(s.patient_name)}</b><div class="mono muted">${esc(s.mrn)}</div></div></div></td>
        <td>${ago(s.created_at)}</td><td class="small" style="max-width:320px">${esc((s.message || "").slice(0, 110)) || '<span class="muted">No message</span>'}</td>
        <td class="small">${[s.ecg_id ? `${icon("activity")} check` : "", s.note_ids.length ? `${icon("smile")} ${s.note_ids.length}` : "", s.n_attachments ? `${icon("paperclip")} ${s.n_attachments}` : ""].filter(Boolean).join(" &nbsp; ") || "–"}</td>
        <td>${s.ecg ? levelBadge(s.ecg.prediction) : "–"}</td><td>${statusBadge(s.status)}</td></tr>`).join("")}
    </tbody></table></div>` : `<div class="empty">${icon("inbox")}<p>Nothing shared yet. When a patient shares a heart check, notes or photos, it appears here.</p></div>`}</div>`;
  rowLinks(el);
}

async function share(el, { id }) {
  const s = await get(`/api/submissions/${id}`);
  const p = s.patient;
  el.innerHTML = `${head(`Shared by ${esc(p.name)}`, `${dateTime(s.created_at)} · ${esc(p.mrn)} · ${p.age ?? "–"} y · ${esc(p.sex || "")}`,
    `<a class="btn" href="#/patients/${p.id}">${icon("user")}Patient record</a>${statusBadge(s.status)}`, `<a href="#/inbox">Patient shares</a> / ${esc(p.name)}`)}
    <div class="grid g-2-1">
      <div class="stack">
        ${s.message ? `<div class="card"><div class="card-head"><h2>${icon("message")} Message</h2></div><div class="card-body"><p style="white-space:pre-wrap;font-size:15px">${esc(s.message)}</p></div></div>` : ""}
        ${s.ecg ? `<div class="card"><div class="card-head"><div><h2>${icon("activity")} Heart check · ${LEVELS[levelOfPred(s.ecg.prediction)].title}</h2><p>${dateTime(s.ecg.recorded_at)}${s.ecg.heart_rate ? ` · ${Math.round(s.ecg.heart_rate)} bpm` : ""} · what the patient saw</p></div><a class="btn sm primary" href="#/ecg/${s.ecg.id}">Open full ECG & sign off ${icon("arrowRight")}</a></div>
          <div class="card-body flush" style="margin-top:8px">${findingList(perClassFromPred(s.ecg.prediction))}</div></div>` : ""}
        ${s.notes.length ? `<div class="card"><div class="card-head"><h2>${icon("smile")} How they've been feeling</h2></div><div style="margin-top:12px">${s.notes.map((n) => `<div class="timeline-item"><div class="face">${moodFace(n.feeling)}</div><div><b>${dateTime(n.created_at)}</b>
          <div class="picks" style="margin:6px 0">${n.symptoms.map((x) => `<span class="chip unclear">${esc(x)}</span>`).join("")}${n.severity !== null ? `<span class="chip">Strength ${n.severity}/10</span>` : ""}${n.started ? `<span class="chip">Started: ${esc(n.started)}</span>` : ""}</div>${n.body ? `<p class="ink2">${esc(n.body)}</p>` : ""}</div></div>`).join("")}</div></div>` : ""}
        ${s.attachments.length ? `<div class="card"><div class="card-head"><div><h2>${icon("paperclip")} Photos & documents</h2><p>Click to open</p></div></div><div class="card-body"><div class="thumbs" id="thumbs"></div></div></div>` : ""}
      </div>
      <div class="stack">
        <div class="card"><div class="card-head"><h2>${s.reply ? "Your reply" : "Reply to patient"}</h2></div><div class="card-body">
          ${s.reply ? `<p style="white-space:pre-wrap">${esc(s.reply)}</p><p class="small muted">${esc(s.replied_by_name || "")} · ${dateTime(s.replied_at)}</p><div class="divider"></div><p class="small muted">Send another reply:</p>` : ""}
          <textarea class="input" id="reply" placeholder="e.g. Thanks for sharing. Your ECG looks reassuring; let's discuss the palpitations at your visit on Monday."></textarea>
          <button class="btn primary" id="send-reply" style="margin-top:10px">${icon("send")}Send reply</button>
          <p class="tiny muted" style="margin-top:8px">The patient sees your reply here and in their messages.</p></div></div>
        <div class="card"><div class="card-body"><h3 style="margin-bottom:8px">${esc(p.name)}</h3><dl class="kv small"><dt>Conditions</dt><dd>${esc(p.conditions || "–")}</dd><dt>Medications</dt><dd>${esc(p.medications || "–")}</dd><dt>Allergies</dt><dd>${esc(p.allergies || "–")}</dd><dt>Phone</dt><dd>${esc(p.phone || "–")}</dd></dl></div></div>
      </div>
    </div>`;
  if (s.attachments.length) { $("#thumbs").innerHTML = await thumbsHtml(s.attachments, false); bindThumbOpen(el); }
  $("#send-reply").addEventListener("click", async () => {
    const body = $("#reply").value.trim();
    if (!body) return toast("Write a reply first", true);
    await post(`/api/submissions/${id}/reply`, { body }); toast("Reply sent"); refresh(); share(el, { id });
  });
  refresh();
}

// ======================================================================= ECGs to review
async function queue(el) {
  const rows = await get("/api/ecgs?status=needs_review");
  el.innerHTML = `${head("ECGs to review", "ECGs where at least one finding is unclear. Most unclear findings first.")}
    <div class="card">${rows.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>Patient</th><th>Age / sex</th><th>Recorded</th><th>Source</th><th>Overall</th><th>Unclear findings</th></tr></thead><tbody>
      ${rows.map((e) => `<tr class="click" data-href="#/ecg/${e.id}"><td><b>${esc(e.patient_name)}</b><div class="mono muted">${esc(e.mrn)}</div></td><td>${e.patient_age ?? "–"} · ${esc(e.sex || "")}</td><td>${dateTime(e.recorded_at)}</td>
        <td>${String(e.source).startsWith("Self-check") ? `<span class="badge info">${icon("share")}Shared by patient</span>` : `<span class="badge neutral">Clinic</span>`}</td><td>${levelBadge(e.prediction)}</td><td>${unclearChips(e.prediction)}</td></tr>`).join("")}
    </tbody></table></div>` : `<div class="empty">${icon("checkCircle")}<p>Nothing to review.</p></div>`}</div>`;
  rowLinks(el);
}

// ======================================================================= patients
async function patients(el) {
  el.innerHTML = `${head("Patients", "", `<button class="btn primary" id="new-patient">${icon("plus")}New patient</button>`)}
    <div class="card"><div class="card-head" style="padding-bottom:14px"><input class="input" id="p-q" placeholder="Filter by name or ID" style="max-width:300px"><span class="muted small" id="p-count"></span></div><div id="p-table">${loading()}</div></div>`;
  let q = "";
  const load = async () => {
    const rows = await get(`/api/patients?q=${encodeURIComponent(q)}`);
    $("#p-count").textContent = `${rows.length} patient${rows.length === 1 ? "" : "s"}`;
    $("#p-table").innerHTML = rows.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>Patient</th><th>ID</th><th>Age</th><th>Conditions</th><th>Last ECG</th><th>Overall</th><th>Status</th></tr></thead><tbody>
      ${rows.map((p) => `<tr class="click" data-href="#/patients/${p.id}"><td><div class="row" style="flex-wrap:nowrap"><span class="avatar">${initials(p.name)}</span><b>${esc(p.name)}</b></div></td><td class="mono">${esc(p.mrn)}</td><td class="num">${p.age ?? "–"}</td><td class="small ink2">${esc(p.conditions || "–")}</td>
        <td>${p.last_ecg ? ago(p.last_ecg.recorded_at) : "–"}</td><td>${p.last_ecg ? levelBadge(p.last_ecg.prediction) : "–"}</td><td>${p.last_ecg ? statusBadge(p.last_ecg.status) : `<span class="muted">No ECG</span>`}</td></tr>`).join("")}
      </tbody></table></div>` : `<div class="empty">No patients match.</div>`;
    rowLinks($("#p-table"));
  };
  let t;
  $("#p-q").addEventListener("input", (e) => { clearTimeout(t); t = setTimeout(() => { q = e.target.value; load(); }, 200); });
  $("#new-patient").addEventListener("click", newPatientModal);
  load();
}

function newPatientModal() {
  modal({
    title: "New patient",
    body: `<form id="np"><div class="grid g2" style="gap:0 14px">
      <div class="field" style="grid-column:1/-1"><label>Full name</label><input class="input" name="name" required></div>
      <div class="field"><label>Date of birth</label><input class="input" type="date" name="dob"></div>
      <div class="field"><label>Sex</label><select class="input" name="sex"><option value="">–</option><option>F</option><option>M</option></select></div>
      <div class="field"><label>Phone</label><input class="input" name="phone"></div>
      <div class="field"><label>Email</label><input class="input" type="email" name="email"></div>
      <div class="field" style="grid-column:1/-1"><label>Known conditions</label><input class="input" name="conditions"></div>
      <div class="field"><label>Medications</label><input class="input" name="medications"></div>
      <div class="field"><label>Allergies</label><input class="input" name="allergies"></div></div></form>`,
    foot: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="np-save">Create patient</button>`,
    onMount: (m, close) => m.querySelector("#np-save").addEventListener("click", async () => {
      const f = m.querySelector("#np");
      if (!f.reportValidity()) return;
      const p = await post("/api/patients", Object.fromEntries(new FormData(f).entries()));
      close(); toast(`${p.name} added`); location.hash = `#/patients/${p.id}`;
    }),
  });
}

async function patientDetail(el, { id }) {
  const p = await get(`/api/patients/${id}`);
  el.innerHTML = `
    ${head(esc(p.name), `${esc(p.mrn)} · ${p.age ?? "–"} years · ${esc(p.sex || "–")}`,
      `<a class="btn" href="#/messages/${p.id}">${icon("message")}Message</a><a class="btn" href="#/appointments?patient=${p.id}">${icon("calendar")}Book</a><a class="btn primary" href="#/analyze?patient=${p.id}">${icon("upload")}Add ECG</a>`,
      `<a href="#/patients">Patients</a> / ${esc(p.name)}`)}
    <div class="grid g-1-2">
      <div class="stack">
        <div class="card"><div class="card-body"><div class="row" style="margin-bottom:14px"><span class="avatar lg">${initials(p.name)}</span><div><h2>${esc(p.name)}</h2><div class="muted small">Born ${date(p.dob)}</div></div></div>
          <dl class="kv"><dt>Conditions</dt><dd>${esc(p.conditions || "–")}</dd><dt>Medications</dt><dd>${esc(p.medications || "–")}</dd><dt>Allergies</dt><dd>${esc(p.allergies || "–")}</dd>
            <dt>Phone</dt><dd>${esc(p.phone || "–")}</dd><dt>Email</dt><dd>${esc(p.email || "–")}</dd><dt>Emergency</dt><dd>${esc(p.emergency_contact || "–")}</dd></dl></div></div>
        <div class="card"><div class="card-head"><h2>Shared by patient</h2></div><div class="list" style="margin-top:8px">${p.submissions.length ? p.submissions.map((s) => `<a class="list-item" href="#/inbox/${s.id}"><div class="grow"><div class="title">${esc(s.message || "Shared items")}</div><div class="small muted">${dateTime(s.created_at)}</div></div>${statusBadge(s.status)}</a>`).join("") : `<div class="empty small">Nothing shared yet.</div>`}</div></div>
        <div class="card"><div class="card-head"><h2>Symptom notes (shared)</h2></div><div style="margin-top:10px">${p.notes.length ? p.notes.map((n) => `<div class="timeline-item"><div class="face">${moodFace(n.feeling)}</div><div><b class="small">${dateTime(n.created_at)}</b><div class="picks" style="margin:4px 0">${n.symptoms.map((x) => `<span class="chip unclear">${esc(x)}</span>`).join("")}</div>${n.body ? `<p class="small ink2">${esc(n.body)}</p>` : ""}</div></div>`).join("") : `<div class="empty small">No shared notes.</div>`}</div></div>
      </div>
      <div class="stack">
        <div class="card"><div class="card-head"><div><h2>ECGs</h2><p>${p.ecgs.length} recording${p.ecgs.length === 1 ? "" : "s"}</p></div></div>
          <div class="card-body flush">${p.ecgs.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>Recorded</th><th>Source</th><th>Overall</th><th>Unclear</th><th>Status</th><th>Released</th></tr></thead><tbody>
          ${p.ecgs.map((e) => `<tr class="click" data-href="#/ecg/${e.id}"><td>${dateTime(e.recorded_at)}</td><td class="small">${String(e.source).startsWith("Self-check") ? "Shared by patient" : "Clinic"}</td><td>${levelBadge(e.prediction)}</td><td>${unclearChips(e.prediction)}</td><td>${statusBadge(e.status)}</td><td>${e.released ? "Yes" : "No"}</td></tr>`).join("")}
          </tbody></table></div>` : `<div class="empty">No ECG on file. <a href="#/analyze?patient=${p.id}">Add one</a>.</div>`}</div></div>
        <div class="card"><div class="card-head"><h2>Appointments</h2></div><div class="list" style="margin-top:8px">${p.appointments.length ? p.appointments.map((a) => `<div class="list-item"><div class="grow"><div class="title">${esc(a.kind)}</div><div class="muted small">${dateTime(a.starts_at)} · ${esc(a.location || "")}</div></div>${statusBadge(a.status)}</div>`).join("") : `<div class="empty small">None.</div>`}</div></div>
      </div>
    </div>`;
  rowLinks(el);
}

// ======================================================================= ECG detail + sign-off
async function ecgDetail(el, { id }) {
  const e = await get(`/api/ecgs/${id}`);
  const p = e.patient, pred = e.prediction, rev = e.review;
  const lv = LEVELS[levelOfPred(pred)];
  el.innerHTML = `
    ${head("ECG", `${esc(p.name)} · ${esc(p.mrn)} · ${p.age ?? "–"} y · ${esc(p.sex || "")} · ${dateTime(e.recorded_at)}`,
      `${statusBadge(e.status)}<button class="btn" id="csv">${icon("download")}Signal (CSV)</button><a class="btn" href="#/report/${e.id}">${icon("printer")}Print</a>`,
      `<a href="#/patients/${p.id}">${esc(p.name)}</a> / ECG`)}
    <div class="card" style="margin-bottom:18px"><div class="card-head" style="padding-bottom:12px"><div><h2>12-lead ECG</h2><p>${e.heart_rate ? `${Math.round(e.heart_rate)} bpm · ` : ""}${esc(e.indication || "")}</p></div></div><div id="ecg-view"></div></div>
    ${pictureNotes(pred)}
    <div class="grid g-2-1">
      <div class="card"><div class="card-head"><div><h2>AI pre-read · ${lv.title}</h2><p>${pred.indeterminate.length ? `Unclear: ${pred.indeterminate.map((c) => esc(CLASS_NAMES[c])).join(", ")}` : "All five findings decided with high confidence"}</p></div>${levelBadge(pred)}</div>
        <div class="card-body flush" style="margin-top:8px">${findingList(perClassFromPred(pred))}</div></div>
      <div class="card" style="grid-column:1/-1"><div class="card-head"><div><h2>Where on the heart</h2><p>Typical region of each AI finding (illustration, not imaging)</p></div></div><div class="card-body" id="heart3d"></div></div>
      <div class="card"><div class="card-head"><div><h2>${rev ? "Your review" : "Sign off"}</h2><p>${rev ? `${esc(rev.clinician_name)} · ${dateTime(rev.created_at)}` : "Confirm or correct the findings, then release to the patient."}</p></div></div>
        <div class="card-body" id="review-box"></div></div>
    </div>`;
  mountEcg($("#ecg-view"), await get(`/api/ecgs/${id}/signal`));
  mountHeart($("#heart3d"), { perClass: perClassFromPred(pred), bpm: e.heart_rate || 72 });
  $("#csv").addEventListener("click", () => downloadCsv(id));
  const box = $("#review-box");
  const init = rev ? rev.final_labels : pred.decisions;
  box.innerHTML = `
    ${rev ? `<div style="margin-bottom:12px">${findingChips(CLASSES.filter((c) => rev.final_labels[c]))}</div>${rev.notes ? `<p class="small"><b>Notes:</b> ${esc(rev.notes)}</p>` : ""}<div class="divider"></div>` : ""}
    <div class="field"><label>Findings present</label><div class="label-toggle">${CLASSES.map((c) => `<label><input type="checkbox" value="${c}" ${init[c] ? "checked" : ""}><span>${esc(CLASS_NAMES[c])}</span></label>`).join("")}</div></div>
    <div class="field"><label>Clinical notes (internal)</label><textarea class="input" id="rv-notes">${esc(rev?.notes || "")}</textarea></div>
    <div class="field"><label>Message for the patient</label><textarea class="input" id="rv-sum" placeholder="Plain language. Leave empty for the standard summary.">${esc(rev?.patient_summary || "")}</textarea></div>
    <label class="check small" style="margin-bottom:14px"><input type="checkbox" id="rv-release" checked> Show the result to the patient</label>
    <button class="btn primary" id="rv-save">${icon("check")}${rev ? "Update review" : "Sign & save"}</button>`;
  $("#rv-save").addEventListener("click", async () => {
    const final = Object.fromEntries(CLASSES.map((c) => [c, box.querySelector(`input[value=${c}]`).checked ? 1 : 0]));
    await post(`/api/ecgs/${id}/review`, { final_labels: final, notes: $("#rv-notes").value, patient_summary: $("#rv-sum").value || null, release: $("#rv-release").checked });
    toast("Review saved"); refresh(); ecgDetail(el, { id });
  });
}

async function report(el, { id }) {
  const e = await get(`/api/ecgs/${id}`);
  const p = e.patient, pred = e.prediction, rev = e.review;
  el.innerHTML = `${head("ECG report", "", `<a class="btn no-print" href="#/ecg/${e.id}">${icon("arrowLeft")}Back</a><button class="btn primary no-print" onclick="window.print()">${icon("printer")}Print / PDF</button>`)}
    <div class="card"><div class="card-body">
      <div class="row between"><div class="row"><div class="brand-mark"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><path d="M2 13h4.5l1.8-4.5 3.4 9 2.6-6.2 1.7 1.7H22"/></svg></div><div><h2>CardioSentinel ECG report</h2><div class="muted small">${dateTime(new Date().toISOString())}</div></div></div>${statusBadge(e.status)}</div>
      <div class="divider"></div>
      <div class="grid g2"><dl class="kv"><dt>Patient</dt><dd><b>${esc(p.name)}</b></dd><dt>ID</dt><dd class="mono">${esc(p.mrn)}</dd><dt>Date of birth</dt><dd>${date(p.dob)}</dd></dl>
        <dl class="kv"><dt>Recorded</dt><dd>${dateTime(e.recorded_at)}</dd><dt>Heart rate</dt><dd>${e.heart_rate ? Math.round(e.heart_rate) + " bpm" : "–"}</dd><dt>Signed</dt><dd>${rev ? `${esc(rev.clinician_name)}, ${date(rev.created_at)}` : "Not yet reviewed"}</dd></dl></div>
      <div class="divider"></div><h3>Conclusion</h3><div style="margin:8px 0">${findingChips(e.findings)}</div><p class="ink2">${esc(rev?.notes || "")}</p>
    </div><div id="rep-ecg"></div><div class="card-body flush">${findingList(perClassFromPred(pred))}</div></div>`;
  mountEcg($("#rep-ecg"), await get(`/api/ecgs/${id}/signal`), { toolbar: false });
}

// ======================================================================= add an ECG
async function analyze(el) {
  const pre = query().get("patient") || "";
  const [pts, samples] = await Promise.all([get("/api/patients"), get("/api/samples").catch(() => [])]);
  el.innerHTML = `${head("Add an ECG", "Upload a 12-lead recording for a patient.")}
    <div class="grid g-1-2">
      <div class="card" style="align-self:start"><div class="card-body">
        <div class="field"><label>Patient</label><select class="input" id="pt"><option value="">Preview only, don't save</option>${pts.map((p) => `<option value="${p.id}" ${String(p.id) === pre ? "selected" : ""}>${esc(p.name)} · ${esc(p.mrn)}</option>`).join("")}</select></div>
        <div class="dropzone" id="dz">${icon("upload")}<p style="margin:8px 0 2px"><b>Drop ECG files here</b> or click</p><p class="muted small">Photo / scan / PDF of a printed 12-lead ECG, or CSV, NPY, WFDB (.hea + .dat)</p><input type="file" id="file" multiple hidden accept=".npy,.csv,.txt,.hea,.dat,.png,.jpg,.jpeg,.webp,.pdf"></div>
        <div id="file-list" class="small" style="margin-top:10px"></div>
        <div class="field" style="margin-top:12px"><label>Sampling rate (Hz)</label><input class="input" id="fs" type="number" value="100"></div>
        <button class="btn primary" id="run" style="width:100%;justify-content:center">${icon("heart")}Analyze</button>
        ${samples.length ? `<div class="divider"></div><div class="label" style="margin-bottom:8px">Example recordings</div><div class="picks">${samples.map((s, i) => `<button class="pick" data-sample="${s.ecg_id}">Example ${i + 1}</button>`).join("")}</div>` : ""}
      </div></div>
      <div id="result"><div class="card"><div class="empty">${icon("activity")}<p>The result will appear here.</p></div></div></div>
    </div>`;
  let files = [];
  const dz = $("#dz"), input = $("#file");
  const show = () => { $("#file-list").innerHTML = files.map((f) => `${icon("file")} ${esc(f.name)}`).join("<br>"); };
  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => { files = [...input.files]; show(); });
  dz.addEventListener("dragover", (ev) => { ev.preventDefault(); dz.classList.add("drag"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", (ev) => { ev.preventDefault(); dz.classList.remove("drag"); files = [...ev.dataTransfer.files]; show(); });
  const run = async (fn, label) => {
    pipelineShell($("#result"), label);
    try {
      const r = await fn();
      if (r.prediction.pipeline) {
        await new Promise((resolve) => playPipeline($("#result"), r.prediction.pipeline,
          { perClass: perClassFromPred(r.prediction), level: levelOfPred(r.prediction), onDone: resolve, doneLabel: "Show the ECG" }));
      }
      const lv = LEVELS[levelOfPred(r.prediction)];
      $("#result").innerHTML = `<div class="stack"><div class="card"><div class="card-head"><div><h2>${lv.title}</h2><p>${r.heart_rate ? `${Math.round(r.heart_rate)} bpm · ` : ""}${r.ecg_id ? "Saved to the patient's record" : "Preview only"}</p></div>${r.ecg_id ? `<a class="btn sm primary" href="#/ecg/${r.ecg_id}">Open ${icon("arrowRight")}</a>` : ""}</div>
        <div class="card-body flush" style="margin-top:8px">${findingList(perClassFromPred(r.prediction))}</div></div><div class="card"><div id="res-ecg"></div></div></div>`;
      mountEcg($("#res-ecg"), r.signal);
      refresh();
    } catch (ex) { $("#result").innerHTML = `<div class="callout crit">${icon("alert")}<div>${esc(ex.message)}</div></div>`; }
  };
  $("#run").addEventListener("click", () => {
    if (!files.length) return toast("Choose a file first", true);
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    form.append("fs", $("#fs").value);
    if ($("#pt").value) form.append("patient_id", $("#pt").value);
    run(() => api("/api/analyze", { method: "POST", form }), files.map((f) => f.name).join(" + "));
  });
  el.querySelectorAll("[data-sample]").forEach((b) => b.addEventListener("click", () => {
    const pid = $("#pt").value;
    run(() => post(`/api/analyze/sample/${b.dataset.sample}${pid ? `?patient_id=${pid}` : ""}`), "Example recording");
  }));
}

// ======================================================================= messages
async function messages(el, { pid } = {}) {
  const threads = await get("/api/messages");
  const sel = pid || (threads[0] && threads[0].patient.id);
  el.innerHTML = `${head("Messages")}
    <div class="card chat"><div class="chat-list">${threads.map((t) => `<a href="#/messages/${t.patient.id}" class="${String(t.patient.id) === String(sel) ? "active" : ""}">
      <div class="row between" style="flex-wrap:nowrap"><b>${esc(t.patient.name)}</b><span class="tiny muted">${ago(t.last.created_at)}</span></div>
      <div class="small muted" style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${t.unread ? `<span class="badge crit" style="margin-right:6px">${t.unread}</span>` : ""}${esc(t.last.body)}</div></a>`).join("")}</div>
      <div class="chat-pane" id="pane">${sel ? loading() : `<div class="empty">No conversations yet.</div>`}</div></div>`;
  if (!sel) return;
  const me = (await shell()).app.user;
  const t = await get(`/api/messages/${sel}`);
  const draw = (msgs) => {
    $("#pane").innerHTML = `<div class="card-head" style="padding-bottom:12px;border-bottom:1px solid var(--border)"><div><h2>${esc(t.patient.name)}</h2><p><a href="#/patients/${t.patient.id}">Open record</a></p></div></div>
      <div class="chat-msgs" id="msgs">${msgs.map((m) => `<div class="bubble ${m.sender_id === me.id ? "me" : "them"}">${esc(m.body)}<div class="meta">${esc(m.sender_name)} · ${dateTime(m.created_at)}</div></div>`).join("")}</div>
      <form class="chat-input" id="send"><textarea class="input" placeholder="Write a message…" required></textarea><button class="btn primary">${icon("send")}Send</button></form>`;
    const box = $("#msgs"); box.scrollTop = box.scrollHeight;
    $("#send").addEventListener("submit", async (ev) => { ev.preventDefault(); const ta = $("#send textarea"); if (!ta.value.trim()) return; draw(await post(`/api/messages/${sel}`, { body: ta.value })); });
  };
  draw(t.messages);
  refresh();
}

// ======================================================================= appointments
async function appointments(el) {
  const rows = await get("/api/appointments");
  const now = new Date().toISOString();
  const up = rows.filter((a) => a.starts_at >= now && a.status === "scheduled");
  const reqs = rows.filter((a) => a.status === "requested");
  const past = rows.filter((a) => a.starts_at < now).reverse();
  const table = (list, actions = false) => (list.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>When</th><th>Patient</th><th>Type</th><th>Status</th>${actions ? "<th></th>" : ""}</tr></thead><tbody>
    ${list.map((a) => `<tr><td class="num">${dateTime(a.starts_at)}</td><td><a href="#/patients/${a.patient_id}"><b>${esc(a.patient_name)}</b></a></td><td>${esc(a.kind)}${a.notes ? `<div class="muted small">${esc(a.notes)}</div>` : ""}</td><td>${statusBadge(a.status)}</td>
      ${actions ? `<td><div class="row"><button class="btn sm primary" data-appt="${a.id}" data-status="scheduled">Confirm</button><button class="btn sm" data-appt="${a.id}" data-status="cancelled">Decline</button></div></td>` : ""}</tr>`).join("")}
    </tbody></table></div>` : `<div class="empty small">Nothing here.</div>`);
  el.innerHTML = `${head("Appointments", "", `<button class="btn primary" id="book">${icon("plus")}Book</button>`)}
    ${reqs.length ? `<div class="card" style="margin-bottom:18px"><div class="card-head"><h2>Requests from patients</h2></div><div class="card-body flush">${table(reqs, true)}</div></div>` : ""}
    <div class="card" style="margin-bottom:18px"><div class="card-head"><h2>Upcoming</h2></div><div class="card-body flush">${table(up)}</div></div>
    <div class="card"><div class="card-head"><h2>Past</h2></div><div class="card-body flush">${table(past.slice(0, 30))}</div></div>`;
  el.querySelectorAll("[data-appt]").forEach((b) => b.addEventListener("click", async () => { await patch(`/api/appointments/${b.dataset.appt}`, { status: b.dataset.status }); toast("Updated"); appointments(el); }));
  const open = async () => {
    const pts = await get("/api/patients");
    const d = new Date(Date.now() + 3 * 86400000); d.setHours(10, 0, 0, 0);
    modal({
      title: "Book appointment",
      body: `<form id="bk"><div class="field"><label>Patient</label><select class="input" name="patient_id">${pts.map((p) => `<option value="${p.id}" ${String(p.id) === query().get("patient") ? "selected" : ""}>${esc(p.name)}</option>`).join("")}</select></div>
        <div class="grid g2" style="gap:0 12px"><div class="field"><label>Date & time</label><input class="input" type="datetime-local" name="starts_at" value="${d.toISOString().slice(0, 16)}"></div>
        <div class="field"><label>Type</label><select class="input" name="kind"><option>Cardiology consultation</option><option>ECG follow-up</option><option>Echocardiogram</option><option>Holter monitor</option><option>Medication review</option></select></div></div>
        <div class="field"><label>Location</label><input class="input" name="location" value="Cardiology Clinic"></div><div class="field"><label>Notes</label><textarea class="input" name="notes"></textarea></div></form>`,
      foot: `<button class="btn" data-close>Cancel</button><button class="btn primary" id="bk-save">Book</button>`,
      onMount: (m, close) => m.querySelector("#bk-save").addEventListener("click", async () => {
        const body = Object.fromEntries(new FormData(m.querySelector("#bk")).entries());
        body.patient_id = Number(body.patient_id); body.starts_at = new Date(body.starts_at).toISOString();
        await post("/api/appointments", body); close(); toast("Booked"); appointments(el);
      }),
    });
  };
  $("#book").addEventListener("click", open);
  if (query().get("patient")) open();
}

// ======================================================================= settings
async function settings(el) {
  const u = (await shell()).app.user;
  el.innerHTML = `${head("Settings")}
    <div class="card" style="max-width:560px"><div class="card-body"><div class="row" style="margin-bottom:14px"><span class="avatar lg">${initials(u.name)}</span><div><b>${esc(u.name)}</b><div class="muted small">${esc(u.title || "")} · ${esc(u.email)}</div></div></div>
      <div class="field"><label>Appearance</label><div class="seg" id="theme-seg"><button data-v="light">Light</button><button data-v="dark">Dark</button><button data-v="">System</button></div></div></div></div>`;
  const cur = document.documentElement.dataset.theme || "";
  el.querySelectorAll("#theme-seg button").forEach((b) => {
    b.classList.toggle("active", b.dataset.v === cur);
    b.addEventListener("click", () => {
      if (b.dataset.v) document.documentElement.dataset.theme = b.dataset.v; else delete document.documentElement.dataset.theme;
      try { b.dataset.v ? localStorage.setItem("cs-theme", b.dataset.v) : localStorage.removeItem("cs-theme"); } catch (x) {}
      window.dispatchEvent(new Event("cs-theme"));
      el.querySelectorAll("#theme-seg button").forEach((x) => x.classList.toggle("active", x === b));
    });
  });
}

export const CLINICIAN_ROUTES = [
  { path: "/inbox", title: "Patient shares", render: inbox },
  { path: "/inbox/:id", title: "Shared by patient", render: share, nav: "#/inbox" },
  { path: "/queue", title: "ECGs to review", render: queue },
  { path: "/patients", title: "Patients", render: patients },
  { path: "/patients/:id", title: "Patient", render: patientDetail, nav: "#/patients" },
  { path: "/ecg/:id", title: "ECG", render: ecgDetail, nav: "#/queue" },
  { path: "/report/:id", title: "Report", render: report, nav: "#/queue" },
  { path: "/analyze", title: "Add an ECG", render: analyze },
  { path: "/messages", title: "Messages", render: messages },
  { path: "/messages/:pid", title: "Messages", render: messages, nav: "#/messages" },
  { path: "/appointments", title: "Appointments", render: appointments },
  { path: "/settings", title: "Settings", render: settings },
];
