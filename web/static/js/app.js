// CardioSentinel SPA: public site, sign-in / sign-up, app shell (sidebar + top bar), hash router.
import { get, hasToken, post, setToken } from "./api.js";
import { CLINICIAN_ROUTES } from "./clinician.js";
import { PATIENT_ROUTES } from "./patient.js";
import { bpmPulse, mountMonitor } from "./monitor.js";
import { $, LOGO, errorBox, esc, icon, initials, loading } from "./ui.js";

export const app = { user: null, system: null, counts: {} };
const root = document.getElementById("app");

const NAV = {
  patient: [
    { href: "#/home", icon: "heart", text: "Home" },
    { href: "#/check", icon: "activity", text: "Check my heart" },
    { href: "#/results", icon: "file", text: "My results" },
    { href: "#/ask", icon: "mic", text: "Ask about my heart" },
    { href: "#/journal", icon: "smile", text: "How I feel" },
    { href: "#/doctor", icon: "stethoscope", text: "My doctor", count: "doctor" },
    { href: "#/appointments", icon: "calendar", text: "Appointments" },
    { label: "More" },
    { href: "#/learn", icon: "book", text: "Heart health guide" },
    { href: "#/profile", icon: "user", text: "My profile" },
  ],
  clinician: [
    { href: "#/inbox", icon: "inbox", text: "Patient shares", count: "new_submissions" },
    { href: "#/queue", icon: "activity", text: "ECGs to review", count: "needs_review" },
    { href: "#/patients", icon: "users", text: "Patients" },
    { href: "#/messages", icon: "message", text: "Messages", count: "unread_messages" },
    { href: "#/appointments", icon: "calendar", text: "Appointments" },
    { label: "More" },
    { href: "#/analyze", icon: "upload", text: "Add an ECG" },
    { href: "#/settings", icon: "settings", text: "Settings" },
  ],
};
const HOME = { patient: "#/home", clinician: "#/inbox" };

// ---------------------------------------------------------------- theme
function theme() {
  return document.documentElement.dataset.theme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
}
export function toggleTheme() {
  const next = theme() === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("cs-theme", next); } catch (e) {}
  window.dispatchEvent(new Event("cs-theme"));
  const b = $("#theme-btn"); if (b) b.innerHTML = icon(next === "dark" ? "sun" : "moon");
}

// ---------------------------------------------------------------- public site
const siteNav = () => `<nav class="site-nav">
  <a class="brand" href="#/" style="padding:0"><div class="brand-mark">${LOGO}</div><div><div class="brand-name">CardioSentinel</div></div></a>
  <div class="grow"></div>
  <button class="icon-btn" id="theme-btn" aria-label="Toggle dark mode">${icon(theme() === "dark" ? "sun" : "moon")}</button>
  <a class="btn ghost" href="#/login">Sign in</a><a class="btn primary" href="#/signup">Get started</a></nav>`;


function renderLanding() {
  root.innerHTML = `<div class="site">${siteNav()}
    <section class="hero">
      <div>
        <span class="hero-eyebrow">12-lead ECG · file, photo or scan</span>
        <h1>Four models.<br>One <em>gated</em> read.</h1>
        <p class="lead">Upload your ECG, or just a photo of the paper printout, and get a clear heart check in seconds: five findings, each found, not found or unclear, and a 3-D heart that shows where they come from.</p>
        <div class="row"><a class="btn brand lg" href="#/signup">Check my heart</a><a class="btn lg" href="#how">How it works</a></div>
        <div class="stat-row"><div><b>5</b>Findings checked</div><div><b>4</b>AI readers</div><div><b>0.94 / 0.91</b>Precision / recall on auto-reports</div></div>
      </div>
      <div class="hero-card">
        <div class="hero-status good" style="box-shadow:none"><div class="eyebrow">Your heart check · today</div><h2>No warning signs found</h2><p>Your ECG looks typical for the five things we check.</p>
          <div class="hero-meta"><div><b>68 bpm</b>Heart rate</div><div><b>5 / 5</b>Checks clear</div></div></div>
        <div class="monitor" style="margin-top:14px"><div class="monitor-top"><span>Heart rhythm · illustration</span><span class="bpm" id="hero-bpm">${icon("heart")}<span>72</span></span></div><div id="hero-mon"></div></div>
        <div class="row between" style="padding:6px 6px 0"><span class="small muted">Shared with Dr. Rao</span><span class="badge good">${icon("message")}Doctor replied</span></div>
      </div>
    </section>
    <section class="section" id="how">
      <span class="hero-eyebrow">How it works</span>
      <h2 class="big">Three steps. No jargon.</h2><p class="sub">From a file or a photo of your printout to a result you can share.</p>
      <div class="grid g3 steps-line">
        ${[["camera", "info", "Add your ECG", "Upload the file from your clinic or device, or photograph the paper printout. We trace the waveform and show you the tracing."],
          ["activity", "crit", "Get your heart check", "Four AI readers vote and a gate combines them. Five findings, each found, not found or unclear, with a 3-D heart showing where."],
          ["share", "good", "Share with your doctor", "You decide what to send: the check, how you've been feeling, photos or documents. Your doctor replies here."]]
          .map(([i, t, h, p]) => `<div class="card feature"><div class="kpi-icon" style="background:var(--${t}-soft);color:var(--${t})">${icon(i)}</div><h3>${h}</h3><p>${p}</p></div>`).join("")}
      </div>
    </section>
    <section class="section" style="padding-top:10px">
      <span class="hero-eyebrow">What we check</span>
      <h2 class="big">Five warning signs.</h2><p class="sub">Patterns that can show up on a 12-lead ECG, plus how you feel.</p>
      <div class="grid g3">
        ${[["heart", "good", "Normal ECG pattern", "Whether your heart's electrical activity looks typical."],
          ["alert", "crit", "Signs of a heart attack", "Patterns that can appear during or after a heart attack, even an old one that went unnoticed."],
          ["activity", "warn", "ST/T-wave changes", "Changes in how the heart recovers between beats, which can be linked to reduced blood flow."],
          ["zap", "info", "Conduction problems", "Delays or blocks in the heart's electrical wiring."],
          ["layers", "warn", "Thickened heart muscle", "Signs that a heart wall may be thicker than usual, often linked to high blood pressure."],
          ["smile", "good", "How you feel", "Log symptoms like palpitations or breathlessness, so your doctor sees the whole picture."]]
          .map(([i, t, h, p]) => `<div class="card feature"><div class="kpi-icon" style="background:var(--${t}-soft);color:var(--${t})">${icon(i)}</div><h3>${h}</h3><p>${p}</p></div>`).join("")}
      </div>
    </section>
    <section class="section" style="padding-top:10px">
      <div class="band"><div><h2>Your data, your choice.</h2><p>Your heart checks and notes stay private. Nothing reaches your doctor until you press "Share".</p></div><a class="btn brand lg" href="#/signup">Create my free account</a></div>
    </section>
    <footer class="site-foot">CardioSentinel gives an AI pre-read of your ECG. It is not a diagnosis and does not replace your doctor. It cannot rule out a heart attack or predict cardiac arrest. <b>If you have chest pain, fainting or severe breathlessness, call your local emergency number now.</b></footer>
  </div>`;
  $("#theme-btn").addEventListener("click", toggleTheme);
  mountMonitor($("#hero-mon"), { bpm: 72, color: "#c8102e", height: 96, onBeat: bpmPulse($("#hero-bpm")) });
}

function renderAuth(mode, err = "") {
  const signup = mode === "signup";
  root.innerHTML = `<div class="site">${siteNav()}<div class="auth-wrap"><form class="card auth-card" id="auth-form" novalidate>
    <h1>${signup ? "Create your account" : "Welcome back"}</h1>
    <p class="muted" style="margin:6px 0 20px">${signup ? "Free, private, and ready in under a minute." : "Sign in to see your heart checks."}</p>
    ${err ? `<div class="callout crit" style="margin-bottom:16px">${icon("alert")}<div>${esc(err)}</div></div>` : ""}
    ${signup ? `<div class="field"><label for="f-name">Full name</label><input class="input" id="f-name" autocomplete="name" required></div>` : ""}
    <div class="field"><label for="f-email">Email</label><input class="input" id="f-email" type="email" autocomplete="username" required></div>
    <div class="field"><label for="f-pw">Password</label><input class="input" id="f-pw" type="password" autocomplete="${signup ? "new-password" : "current-password"}" required>${signup ? `<span class="tiny muted">At least 8 characters</span>` : ""}</div>
    ${signup ? `<div class="grid g2" style="gap:0 12px"><div class="field"><label for="f-dob">Date of birth (optional)</label><input class="input" id="f-dob" type="date"></div>
      <div class="field"><label for="f-sex">Sex (optional)</label><select class="input" id="f-sex"><option value="">Prefer not to say</option><option value="F">Female</option><option value="M">Male</option></select></div></div>
      <label class="check small" style="margin-bottom:16px"><input type="checkbox" id="f-ok" required> I understand CardioSentinel gives an AI pre-read, not a diagnosis.</label>` : ""}
    <button class="btn primary lg" style="width:100%;justify-content:center">${signup ? "Create account" : "Sign in"}</button>
    <p class="small muted" style="text-align:center;margin-top:16px">${signup ? `Already have an account? <a href="#/login">Sign in</a>` : `New here? <a href="#/signup">Create an account</a>`}</p>
    <div class="divider"></div>
    <div class="row between small"><span class="muted">Just looking around?</span><button type="button" class="btn sm" id="demo-patient">${icon("user")}Try the demo</button></div>
    <p class="tiny muted" style="margin-top:12px">Doctors sign in here with their clinic account (demo: <a href="#" id="demo-doc">clinician@cardiosentinel.demo</a>).</p>
  </form></div></div>`;
  $("#theme-btn").addEventListener("click", toggleTheme);
  const go = async (path, body) => {
    try {
      const r = await post(path, body);
      setToken(r.token); app.user = r.user;
      location.hash = HOME[r.user.role];
      renderShell(); route();
    } catch (ex) { renderAuth(mode, ex.message); }
  };
  $("#demo-patient").addEventListener("click", () => go("/api/auth/login", { email: "patient@cardiosentinel.demo", password: "demo1234" }));
  $("#demo-doc").addEventListener("click", (e) => { e.preventDefault(); go("/api/auth/login", { email: "clinician@cardiosentinel.demo", password: "demo1234" }); });
  $("#auth-form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (signup) {
      if (!$("#f-ok").checked) return renderAuth(mode, "Please confirm that you understand this is not a diagnosis.");
      go("/api/auth/signup", { name: $("#f-name").value, email: $("#f-email").value, password: $("#f-pw").value, dob: $("#f-dob").value || null, sex: $("#f-sex").value || null });
    } else go("/api/auth/login", { email: $("#f-email").value, password: $("#f-pw").value });
  });
}

async function logout() {
  try { await post("/api/auth/logout"); } catch (e) {}
  setToken(null);
  app.user = null;
  location.hash = "#/";
  renderPublic();
}

// ---------------------------------------------------------------- shell
function renderShell() {
  const u = app.user;
  const nav = NAV[u.role].map((n) => n.label
    ? `<div class="nav-label">${n.label}</div>`
    : `<a href="${n.href}" data-nav="${n.href}">${icon(n.icon)}<span>${n.text}</span>${n.count ? `<span class="count hidden" data-count="${n.count}"></span>` : ""}</a>`).join("");
  root.innerHTML = `
  <div class="shell" id="shell">
    <aside class="sidebar" aria-label="Main navigation">
      <a class="brand" href="${HOME[u.role]}"><div class="brand-mark">${LOGO}</div><div><div class="brand-name">CardioSentinel</div><div class="brand-sub">${u.role === "clinician" ? "Doctor workspace" : "Your heart, checked"}</div></div></a>
      ${u.role === "patient" ? `<div class="nav-cta"><a class="btn brand" href="#/check">${icon("activity")}New heart check</a></div>` : ""}
      <nav class="nav">${nav}</nav>
      <div class="sidebar-foot">${u.role === "patient"
        ? `<div class="sos">${icon("phone")}<div><b>Emergency?</b><br>Chest pain, fainting or severe breathlessness: call your local emergency number now.</div></div>`
        : `<div class="small muted">${icon("lock")} Patients choose what they share with you.</div>`}</div>
    </aside>
    <div class="main">
      <header class="topbar">
        <button class="icon-btn menu-btn" id="menu-btn" aria-label="Open menu">${icon("menu")}</button>
        ${u.role === "clinician" ? `<div class="search">${icon("search")}<input id="search" placeholder="Search patients…" autocomplete="off" aria-label="Search patients"><div class="search-results hidden" id="search-results"></div></div>` : `<div></div>`}
        <div class="top-actions">
          <button class="icon-btn" id="theme-btn" aria-label="Toggle dark mode">${icon(theme() === "dark" ? "sun" : "moon")}</button>
          <button class="icon-btn" id="bell-btn" aria-label="Notifications">${icon("bell")}<span class="badge-dot hidden" id="bell-dot"></span></button>
          <button class="user-menu" id="user-btn"><span class="avatar">${initials(u.name)}</span><span class="who" style="text-align:left"><b style="display:block;font-size:13.5px">${esc(u.name)}</b><span class="muted tiny">${u.role === "clinician" ? esc(u.title || "Doctor") : "My account"}</span></span></button>
        </div>
      </header>
      <main class="content" id="view"></main>
      <footer class="disclaimer">CardioSentinel gives an AI pre-read of ECGs. It is not a diagnosis and does not replace a doctor. In an emergency, call your local emergency number.</footer>
    </div>
  </div>`;
  $("#theme-btn").addEventListener("click", toggleTheme);
  $("#menu-btn").addEventListener("click", () => $("#shell").classList.toggle("nav-open"));
  $("#shell").addEventListener("click", (e) => { if (e.target.closest(".nav a, .nav-cta a")) $("#shell").classList.remove("nav-open"); });
  $("#user-btn").addEventListener("click", (e) => { e.stopPropagation(); userMenu(); });
  $("#bell-btn").addEventListener("click", (e) => { e.stopPropagation(); notifications(); });
  if (u.role === "clinician") bindSearch();
  refreshCounts();
}

function closeDropdowns() { document.querySelectorAll(".dropdown").forEach((d) => d.remove()); }
document.addEventListener("click", (e) => { if (!e.target.closest(".dropdown")) closeDropdowns(); });

function userMenu() {
  closeDropdowns();
  const d = document.createElement("div");
  d.className = "dropdown";
  d.innerHTML = `<div style="padding:8px 10px 10px"><b>${esc(app.user.name)}</b><div class="muted small">${esc(app.user.email)}</div></div>
    <a href="${app.user.role === "clinician" ? "#/settings" : "#/profile"}">${icon("user")} ${app.user.role === "clinician" ? "Settings" : "My profile"}</a>
    <button id="dd-theme">${icon("moon")} Light / dark mode</button>
    <button id="dd-logout">${icon("logout")} Sign out</button>`;
  document.body.appendChild(d);
  d.querySelector("#dd-theme").addEventListener("click", toggleTheme);
  d.querySelector("#dd-logout").addEventListener("click", logout);
}

function notifications() {
  closeDropdowns();
  const d = document.createElement("div");
  d.className = "dropdown";
  d.style.right = "88px";
  d.style.width = "330px";
  const c = app.counts;
  const items = app.user.role === "clinician" ? [
    c.new_submissions ? ["#/inbox", "inbox", `${c.new_submissions} new share${c.new_submissions > 1 ? "s" : ""} from patients`] : null,
    c.needs_review ? ["#/queue", "activity", `${c.needs_review} ECG${c.needs_review > 1 ? "s" : ""} to review`] : null,
    c.unread_messages ? ["#/messages", "message", `${c.unread_messages} unread message${c.unread_messages > 1 ? "s" : ""}`] : null,
  ] : [
    c.unread_messages ? ["#/doctor", "message", `${c.unread_messages} new message${c.unread_messages > 1 ? "s" : ""} from your doctor`] : null,
    c.pending ? ["#/results", "clock", `${c.pending} clinic ECG result${c.pending > 1 ? "s" : ""} on the way`] : null,
  ];
  const list = items.filter(Boolean);
  d.innerHTML = `<div style="padding:8px 10px"><b>Notifications</b></div>` + (list.length
    ? list.map(([h, i, t]) => `<a href="${h}">${icon(i)} ${esc(t)}</a>`).join("")
    : `<div class="muted small" style="padding:8px 10px 12px">You're all caught up.</div>`);
  document.body.appendChild(d);
}

export async function refreshCounts() {
  try {
    if (app.user.role === "clinician") {
      const d = await get("/api/dashboard");
      app.counts = { needs_review: d.counts.needs_review || 0, unread_messages: d.unread_messages, new_submissions: d.new_submissions };
    } else {
      const o = await get("/api/me/overview");
      app.counts = { unread_messages: o.unread_messages, pending: o.pending, doctor: o.unread_messages };
    }
  } catch (e) { return; }
  document.querySelectorAll("[data-count]").forEach((el) => {
    const v = app.counts[el.dataset.count] || 0;
    el.textContent = v;
    el.classList.toggle("hidden", !v);
  });
  const dot = $("#bell-dot");
  if (dot) dot.classList.toggle("hidden", !Object.entries(app.counts).some(([k, v]) => k !== "doctor" && v));
}

function bindSearch() {
  const input = $("#search"), box = $("#search-results");
  let t;
  input.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(async () => {
      const q = input.value.trim();
      if (!q) return box.classList.add("hidden");
      const rows = await get(`/api/patients?q=${encodeURIComponent(q)}`);
      box.innerHTML = rows.length ? rows.slice(0, 8).map((p) => `<a href="#/patients/${p.id}"><span><b>${esc(p.name)}</b> <span class="muted small">${p.age ?? ""} ${p.sex ?? ""}</span></span><span class="mono muted">${esc(p.mrn)}</span></a>`).join("")
        : `<div class="empty small">No patients match “${esc(q)}”.</div>`;
      box.classList.remove("hidden");
    }, 150);
  });
  box.addEventListener("click", () => { box.classList.add("hidden"); input.value = ""; });
  input.addEventListener("blur", () => setTimeout(() => box.classList.add("hidden"), 200));
}

// ---------------------------------------------------------------- router
function match(routes, hash) {
  const path = (hash.replace(/^#/, "") || "/").split("?")[0];
  for (const r of routes) {
    const keys = [];
    const re = new RegExp("^" + r.path.replace(/:(\w+)/g, (_, k) => { keys.push(k); return "([^/]+)"; }) + "$");
    const m = path.match(re);
    if (m) return { r, params: Object.fromEntries(keys.map((k, i) => [k, decodeURIComponent(m[i + 1])])) };
  }
  return null;
}
export const query = () => new URLSearchParams(location.hash.split("?")[1] || "");

function renderPublic() {
  const h = location.hash.split("?")[0];
  if (h === "#/login") renderAuth("login");
  else if (h === "#/signup") renderAuth("signup");
  else renderLanding();
  window.scrollTo(0, 0);
}

async function route() {
  if (!app.user) return renderPublic();
  const routes = app.user.role === "clinician" ? CLINICIAN_ROUTES : PATIENT_ROUTES;
  const found = match(routes, location.hash);
  if (!found) { location.hash = HOME[app.user.role]; return; }
  let view = $("#view");
  if (!view) { renderShell(); view = $("#view"); }
  const base = location.hash.split("?")[0];
  document.querySelectorAll("[data-nav]").forEach((a) => {
    const h = a.dataset.nav;
    a.classList.toggle("active", base === h || base.startsWith(h + "/") || (!!found.r.nav && found.r.nav === h));
  });
  view.innerHTML = loading();
  window.scrollTo(0, 0);
  document.title = `${found.r.title || "Home"} · CardioSentinel`;
  try {
    await found.r.render(view, found.params);
  } catch (e) {
    if (e.status === 401) return logout();
    view.innerHTML = errorBox(e);
  }
}

window.addEventListener("hashchange", route);

async function boot() {
  try { app.system = await get("/api/system"); } catch (e) {
    root.innerHTML = `<div class="content">${errorBox(new Error("Cannot reach the CardioSentinel server. Start it with: python web/server.py"))}</div>`;
    return;
  }
  if (hasToken()) {
    try { app.user = await get("/api/auth/me"); } catch (e) { setToken(null); }
  }
  if (!app.user) return renderPublic();
  renderShell();
  if (!match(app.user.role === "clinician" ? CLINICIAN_ROUTES : PATIENT_ROUTES, location.hash)) location.hash = HOME[app.user.role];
  route();
}

boot();
