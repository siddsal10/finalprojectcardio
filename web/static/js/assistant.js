// "Ask CardioSentinel": a voice + text assistant that answers from the patient's own records (RAG on
// the server, see web/assistant.py). Speech recognition and speech synthesis are the browser's own
// Web Speech API, so voice works in Chrome / Edge / Safari; elsewhere the text box still works.
import { get, post } from "./api.js";
import { esc, icon, toast } from "./ui.js";

const KIND_ICON = { check: "activity", journal: "smile", message: "message", appointment: "calendar", share: "share", profile: "user", review: "stethoscope" };

export async function assistantPage(el) {
  const info = await get("/api/me/assistant").catch(() => ({ mode: "offline", suggestions: [] }));
  const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
  const canSpeak = "speechSynthesis" in window;
  let speakOn = canSpeak, history = [], busy = false, rec = null;

  el.innerHTML = `
    <div class="page-head"><div><span class="eyebrow">Voice assistant</span><h1>Ask about your heart</h1>
      <p class="muted">Ask out loud or type. Answers come only from your own heart checks, journal, messages and appointments, with the records they used.</p></div></div>
    <div class="grid g-2-1">
      <div class="card ask-card"><div class="ask-log" id="ask-log" aria-live="polite">
          <div class="ask-msg bot"><div class="ask-bubble">Hi! I can tell you about your heart checks, what your doctor said, how you've been feeling and your appointments. What would you like to know?</div></div></div>
        <div class="ask-suggest" id="ask-suggest">${(info.suggestions || []).map((s) => `<button class="pick" data-q="${esc(s)}">${esc(s)}</button>`).join("")}</div>
        <form class="ask-bar" id="ask-form">
          <button type="button" class="mic ${Rec ? "" : "off"}" id="ask-mic" aria-label="Hold to talk" title="${Rec ? "Hold to talk (or click to start / stop)" : "Voice input is not supported in this browser"}">${icon("mic")}<span class="mic-ring"></span></button>
          <input class="input" id="ask-q" placeholder="${Rec ? "Tap the mic and ask, or type here…" : "Type your question…"}" autocomplete="off">
          <button class="btn brand" id="ask-send">${icon("arrowRight")}</button>
        </form></div>
      <div class="stack">
        <div class="card"><div class="card-body">
          <h3 style="margin-bottom:8px">How it answers</h3>
          <p class="small ink2">Your question is matched against your own records, and the closest ones are given to the AI with the rule to answer only from them and say which it used. Nothing outside your account is searched.</p>
          <div class="row" style="margin-top:10px;gap:8px"><span class="chip">${info.mode === "openrouter" ? `${icon("cpu")}AI: ${esc(info.model || "OpenRouter")}` : `${icon("file")}Offline: reads your records back`}</span>
            ${canSpeak ? `<label class="chip" style="cursor:pointer"><input type="checkbox" id="ask-voice" checked style="margin:0 4px 0 0">Read answers aloud</label>` : ""}</div></div></div>
        <div class="callout warn">${icon("alert")}<div class="small">Not a doctor and not for emergencies. If you have chest pain, fainting or severe breathlessness, call your local emergency number now.</div></div>
      </div>
    </div>`;

  const log = el.querySelector("#ask-log"), q = el.querySelector("#ask-q"), mic = el.querySelector("#ask-mic");
  const add = (who, html) => {
    const d = document.createElement("div");
    d.className = `ask-msg ${who}`;
    d.innerHTML = html;
    log.appendChild(d);
    log.scrollTop = log.scrollHeight;
    return d;
  };
  const speak = (text) => {
    if (!speakOn || !canSpeak) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text.replace(/\[[^\]]+\]/g, ""));   // don't read citation ids aloud
    u.rate = 1.0;
    speechSynthesis.speak(u);
  };

  async function ask(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    busy = true;
    q.value = "";
    add("me", `<div class="ask-bubble">${esc(text)}</div>`);
    const wait = add("bot", `<div class="ask-bubble typing"><i></i><i></i><i></i></div>`);
    try {
      const r = await post("/api/me/assistant", { question: text, history });
      history = [...history, { role: "user", content: text }, { role: "assistant", content: r.answer }].slice(-12);
      const cited = new Set((r.answer.match(/\[([^\]]+)\]/g) || []).map((m) => m.slice(1, -1).toLowerCase()));
      const srcs = r.sources.filter((s) => !cited.size || cited.has(s.id.toLowerCase()));
      wait.innerHTML = `${r.urgent ? `<div class="callout crit" style="margin-bottom:8px">${icon("phone")}<div class="small"><b>If this is happening now, call your local emergency number.</b></div></div>` : ""}
        <div class="ask-bubble">${esc(r.answer).replace(/\[([^\]]+)\]/g, '<span class="cite">$1</span>')}</div>
        ${srcs.length ? `<div class="ask-sources">${srcs.map((s) => `<details class="src"><summary>${icon(KIND_ICON[s.kind] || "file")}${esc(s.title)}${s.date ? ` · ${esc(s.date)}` : ""}</summary><p class="small">${esc(s.text)}</p></details>`).join("")}</div>` : ""}`;
      speak(r.answer);
    } catch (ex) {
      wait.innerHTML = `<div class="ask-bubble">Sorry, I couldn't answer that just now (${esc(ex.message)}).</div>`;
    } finally {
      busy = false;
      log.scrollTop = log.scrollHeight;
    }
  }

  el.querySelector("#ask-form").addEventListener("submit", (ev) => { ev.preventDefault(); ask(q.value); });
  el.querySelectorAll("#ask-suggest [data-q]").forEach((b) => b.addEventListener("click", () => ask(b.dataset.q)));
  const vt = el.querySelector("#ask-voice");
  if (vt) vt.addEventListener("change", () => { speakOn = vt.checked; if (!speakOn) speechSynthesis.cancel(); });

  // ---- voice input: click to start/stop, or hold to talk
  if (Rec) {
    let listening = false, heard = "";
    const start = () => {
      if (listening) return;
      if (canSpeak) speechSynthesis.cancel();
      rec = new Rec();
      rec.lang = navigator.language || "en-US";
      rec.interimResults = true;
      rec.continuous = false;
      heard = "";
      rec.onresult = (e) => { heard = [...e.results].map((r) => r[0].transcript).join(" "); q.value = heard; };
      rec.onerror = (e) => { if (e.error !== "no-speech" && e.error !== "aborted") toast(`Microphone: ${e.error}`, true); };
      rec.onend = () => { listening = false; mic.classList.remove("on"); if (heard.trim()) ask(heard); };
      rec.start();
      listening = true;
      mic.classList.add("on");
    };
    const stop = () => { if (listening && rec) rec.stop(); };
    let downAt = 0;
    mic.addEventListener("pointerdown", () => { downAt = Date.now(); start(); });
    mic.addEventListener("pointerup", () => { if (Date.now() - downAt > 450) stop(); });   // held: release to send; tapped: stops by itself after a pause
  } else {
    mic.addEventListener("click", () => toast("Voice input isn't supported in this browser. Type your question instead.", true));
  }
  return () => { if (rec) try { rec.abort(); } catch (e) {} if (canSpeak) speechSynthesis.cancel(); };
}
