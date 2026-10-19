# CardioSentinel web portal

A clinician and patient portal on top of the four-client ECG model and its gate.

```bash
pip install -r requirements.txt && pip install -e .
python web/server.py            # http://127.0.0.1:8000
python web/server.py --reseed   # delete the demo database and seed it again
```

Demo accounts (password `demo1234`):

| Portal | Email |
|---|---|
| Clinician | `clinician@cardiosentinel.demo` (also `dr.okafor@cardiosentinel.demo`) |
| Patient | `patient@cardiosentinel.demo` |

## Live vs demo mode

At start-up the server looks for `outputs/fourclient/seed42/{A,B,C,D}/model.pt` plus
`gate/combiners.pkl` and `gate/aggregate.json`:

- **Live:** found. Uploaded ECGs are run through the four clients, the selected gate and the
  triage rule. Demo patients are seeded with real PTB-XL fold-10 (test) recordings and real
  predictions, and the PTB-XL label is shown as a reference.
- **Demo:** not found. ECGs come from a synthetic 12-lead generator, predictions are simulated,
  and every screen says so.

After training, run `python web/server.py --reseed` so the demo patients use the real models.

## Pages

CardioSentinel is patient-first: people check their own ECG and decide what their doctor sees.

**Public:** a welcome page (how it works, what we check, privacy), sign in, and create account.

**Patient app:**
- **Home:** the latest heart check, a "how are you feeling today?" mood picker, quick actions, the
  doctor and the next appointment.
- **Check my heart:** upload an ECG file (CSV, NPY or WFDB) or use an example recording, optionally
  note how you feel. The result is ready in seconds.
- **Result:** an overall verdict ("No warning signs found", "Some results are unclear" or "Warning
  signs found"), and the five findings in plain language marked found / not found / unclear, each with
  the AI likelihood. Also the heart-rhythm strip and a one-tap "Share with my doctor". If signs of a
  heart attack are possible, an emergency warning is shown.
- **How I feel:** a symptom journal (mood, symptom checklist, strength 0–10, when it started, free text).
- **Share with my doctor:** pick a heart check, journal entries, photos or PDFs (e.g. a photo of a
  paper ECG, lab results) and a message. Nothing is shared until the patient presses Send.
- **My doctor:** what was shared (Sent → Seen → Replied, with the doctor's reply) and messages.
- Appointments, a heart health guide and the profile.

**Doctor workspace:**
- **Patient shares:** an inbox of what patients sent, with the message, the heart check, symptom
  notes and attachments, plus a reply box.
- **ECGs to review**, patients, the 12-lead ECG viewer with sign-off, messages, appointments, and
  adding an ECG for a patient.

Self-checks are private: doctors only see a check, a note or a file after the patient shares it
(enforced by the API and covered by `tests/test_web.py`).

**What it does not do:** the model detects five ECG patterns (normal ECG, signs of myocardial
infarction, ST/T changes, conduction disturbances, hypertrophy). It does not predict cardiac arrest:
PTB-XL has no outcome follow-up to learn that from. The site says so on the welcome page, results
pages and guide. Photos of paper ECGs can be shared with the doctor but are not analysed automatically.

## Files

```
server.py        FastAPI app: auth, clinician + patient API, static frontend
inference.py     loads the clients, gate and triage rule; parses uploads
demo.py          seed data: real PTB-XL test records or the synthetic ECG generator
store.py         SQLite schema and helpers (web/data/, git-ignored)
static/          index.html, css/app.css, js/{app,patient,clinician,ecg,ui,api}.js
```

The database lives in `web/data/` (override with `CARDIOSENTINEL_DATA`). This is a research
prototype: passwords are hashed (PBKDF2) and sessions are bearer tokens. It is not hardened for
real patient data.
