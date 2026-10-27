"""Heart-health assistant: answers a patient's questions from their own records (retrieval-augmented).

1. The server turns everything the patient can see (heart checks and their findings, doctor reviews,
   symptom-journal entries, messages, appointments, what they shared and the replies, profile)
   into short passages, each with an id such as "check 12" or "journal 4".
2. retrieve() ranks the passages against the question with TF-IDF (word 1-2 grams) and keeps the
   best few, plus the latest heart check so "how is my heart?" always has context.
3. answer() asks an LLM through OpenRouter (OPENROUTER_API_KEY; model OPENROUTER_MODEL, default
   "openrouter/auto") to answer only from those passages and to cite them. Without a key (or if the
   call fails) it falls back to an extractive answer: the best-matching passages read out in plain
   sentences. Either way the cited passages are returned for display.

The browser handles voice: speech recognition for the question, speech synthesis for the answer.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

log = logging.getLogger("cardiosentinel.assistant")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TOP_K = 6

SYSTEM = """You are the CardioSentinel heart-health assistant, talking with a patient about their own records.

Answer only from the patient's records given in <records>. Each record starts with an id in square
brackets, such as [check 12] or [journal 4]; cite the ids you used in square brackets. If the records
do not answer the question, say so plainly and suggest asking their doctor. Use everyday words; the
answer is read aloud, so keep it to 2-5 short sentences with no lists, tables or markdown.

The heart checks are AI pre-reads of an ECG, not a diagnosis. Never diagnose, never change or suggest
medication, and never tell the patient a finding is nothing to worry about when it is marked found or
unclear; point them to their doctor instead. If the question mentions chest pain, fainting, severe
breathlessness or other emergency symptoms, start by telling them to call their local emergency number."""

EMERGENCY = re.compile(r"chest pain|chest pressure|faint|passed out|can'?t breathe|short(ness)? of breath|"
                       r"heart attack|collapse", re.I)


@dataclass
class Passage:
    id: str          # "check 12", "journal 4", ...
    kind: str        # check | review | journal | message | appointment | share | profile
    date: str        # ISO date (may be empty for the profile)
    title: str
    text: str


def retrieve(question: str, passages: list[Passage], k: int = TOP_K) -> list[Passage]:
    """Best-matching passages for the question (TF-IDF cosine), plus the latest heart check."""
    if not passages:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [f"{p.title}. {p.text}" for p in passages]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    try:
        m = vec.fit_transform(docs + [question])
    except ValueError:  # empty vocabulary (e.g. a question made only of stop words)
        return sorted(passages, key=lambda p: p.date, reverse=True)[:k]
    scores = (m[:-1] @ m[-1].T).toarray().ravel()
    recent = re.search(r"\b(latest|last|recent|newest|today|now|current)\b", question, re.I)
    order = sorted(range(len(passages)), key=lambda i: (scores[i] + (0.15 if recent else 0) * _recency(passages, i)),
                   reverse=True)
    picked = [passages[i] for i in order[:k] if scores[i] > 0 or recent]
    checks = sorted((p for p in passages if p.kind == "check"), key=lambda p: p.date, reverse=True)
    if checks and checks[0] not in picked:
        picked = picked[: k - 1] + [checks[0]]
    profile = next((p for p in passages if p.kind == "profile"), None)
    if profile and profile not in picked and re.search(r"\b(medic|allerg|condition|age|weight|height)", question, re.I):
        picked.append(profile)
    return picked


def _recency(passages: list[Passage], i: int) -> float:
    dates = sorted({p.date for p in passages if p.date})
    if not passages[i].date or not dates:
        return 0.0
    return (dates.index(passages[i].date) + 1) / len(dates)


def llm_config() -> dict | None:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        return None
    return {"key": key, "model": os.environ.get("OPENROUTER_MODEL", "openrouter/auto").strip() or "openrouter/auto"}


def answer(question: str, passages: list[Passage], history: list[dict] | None = None) -> dict:
    """Answer + the passages it was based on. mode: 'openrouter' or 'offline'."""
    found = retrieve(question, passages)
    sources = [{"id": p.id, "kind": p.kind, "date": p.date[:10], "title": p.title, "text": p.text} for p in found]
    urgent = bool(EMERGENCY.search(question))
    cfg = llm_config()
    if cfg is not None:
        try:
            text, model = _ask_llm(cfg, question, found, history or [])
            if text:
                return {"answer": text, "sources": sources, "mode": "openrouter", "model": model, "urgent": urgent}
        except Exception as e:  # API down, bad key, network: fall back rather than fail the user
            log.warning("assistant: OpenRouter call failed (%s); answering offline", e)
    return {"answer": _offline(question, found, urgent), "sources": sources, "mode": "offline", "urgent": urgent}


def _ask_llm(cfg: dict, question: str, found: list[Passage], history: list[dict]) -> tuple[str, str]:
    import httpx

    records = "\n".join(f"[{p.id}] {p.date[:10]} {p.title}: {p.text}" for p in found) or "(no records)"
    msgs = [{"role": "system", "content": SYSTEM}]
    for h in history[-6:]:  # short memory of the spoken conversation
        if h.get("role") in ("user", "assistant") and h.get("content"):
            msgs.append({"role": h["role"], "content": str(h["content"])[:1500]})
    msgs.append({"role": "user", "content": f"<records>\n{records}\n</records>\n\nQuestion: {question}"})
    r = httpx.post(OPENROUTER_URL, timeout=45.0,
                   headers={"Authorization": f"Bearer {cfg['key']}", "X-Title": "CardioSentinel"},
                   json={"model": cfg["model"], "messages": msgs, "max_tokens": 600})
    r.raise_for_status()
    data = r.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return text.strip(), data.get("model", cfg["model"])


def _offline(question: str, found: list[Passage], urgent: bool) -> str:
    lead = ("If you have chest pain, fainting or severe breathlessness right now, call your local emergency number. "
            if urgent else "")
    if not found:
        return lead + "I couldn't find anything in your records about that. Your doctor can help with this question."
    best = found[:3]
    parts = [f"{p.title} on {_spoken_date(p.date)}: {p.text}" if p.date else f"{p.title}: {p.text}" for p in best]
    return lead + "Here is what your records say. " + " ".join(parts) + \
        " This is an AI pre-read, not a diagnosis; your doctor can explain what it means for you."


def _spoken_date(iso: str) -> str:
    from datetime import date

    try:
        d = date.fromisoformat(iso[:10])
        return d.strftime("%d %B %Y").lstrip("0")
    except ValueError:
        return iso[:10]
