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
pages and guide.

## Photos, scans and PDFs of paper ECGs

"Check my heart" and "Add an ECG" also take up to 4 pictures (PNG, JPG, WEBP) or a PDF of one
printed 12-lead ECG. Each picture is digitized (`src/cardiomoe/images/`): a trace segmenter marks the
waveform, the digitizer straightens the page, finds the rows and turns the tracing into a (12, 1000)
signal at 25 mm/s and 10 mm/mV. The image clients A-D (the signal clients fine-tuned on real ECG
pictures) and their gate and triage then read it. Setup (after the signal pipeline):

```bash
python scripts/fetch_ecg_images.py --folds 1 2 3 4 5 6 7 8 9 10   # ECG-Image-Database, PTB-XL part
python -m cardiomoe.images.segnet                                 # trace segmenter (synthetic printouts)
python -m cardiomoe.images.real --folds 1 2 3 4 5 6 7 8 9         # digitize the real pictures
python -m cardiomoe.images.finetune --seeds 42                    # image clients + gate + triage
python -m cardiomoe.images.evaluate --set real-test               # real fold-10 pictures, once
```

What the photo reader offers:

- **Any capture:** phone photos, flatbed scans (colour or grayscale), clinic PDFs, photos of a monitor,
  red, pink, green, blue or gray grids, curled, creased or stained pages.
- **Trained and tested on real pictures:** fine-tuned on real scans and photos of PTB-XL printouts
  (ECG-Image-Database) and tested on pictures of records it never saw, with results per capture type.
- **Same safety rule as digital ECGs:** a finding is reported automatically only in the band where
  precision, recall and accuracy reached at least 0.90 on validation pictures; anything else is
  marked unclear and referred to a cardiologist.
- **Shows its work:** the tracing is drawn on the user's own photo, so anyone can see whether the
  waveform was followed.
- **Picture quality check:** every tracing gets a quality score. Below the threshold nothing is
  auto-reported and the user gets retake advice, instead of a confident answer from a bad photo.
- **Several pictures, one answer:** up to 4 pictures of the same ECG are combined, and their agreement
  is reported; pictures that cannot be traced are skipped and listed.
- **Lead coverage:** says which leads were printed for 2.5 s and which for 10 s.
- **Digitized signal to keep:** patients and doctors can download the 12-lead signal as CSV (unprinted
  stretches left empty) and the doctor sees it in the 12-lead viewer.
- **Private and local:** everything runs on the server's CPU; no picture leaves it.
- **Explains each step:** the animated walkthrough shows the traced photo, the four readers' votes, the
  combination and the confidence check.
- **Safe by default on photos:** on real fold-10 photos the auto-report bands missed the 0.90 targets, so every
  photo finding is referred to a doctor (`images.auto_report: false` in `configs/images.yaml`) until validated.

## 3-D heart

Every result shows a beating 3-D heart (three.js from a CDN; drag to turn) that lights up the region each finding
usually comes from: heart-attack signs on the left-ventricle front wall and coronary arteries, ST/T changes across the
ventricles, conduction problems along the SA node / AV node / bundle branches, hypertrophy as a thicker left-ventricle
wall. It says it is an illustration, not an image of the patient's heart.

## Voice assistant ("Ask about my heart")

Patients ask out loud (browser speech recognition) or type, and hear the answer (speech synthesis). `assistant.py`
retrieves the closest of the patient's own records (heart checks, doctor reviews, journal, messages, appointments,
shares, profile; TF-IDF) and an LLM answers only from them, citing the records used. The LLM runs through OpenRouter:

```bash
set OPENROUTER_API_KEY=sk-or-...          # PowerShell: $env:OPENROUTER_API_KEY="sk-or-..."
set OPENROUTER_MODEL=openrouter/auto      # optional: any OpenRouter model id
python web/server.py
```

Without a key it answers offline by reading the matching records back. It never sees other patients' data, flags
emergency words with "call your local emergency number" and does not diagnose.

## Files

```
server.py        FastAPI app: auth, clinician + patient API, static frontend
inference.py     loads the signal and image clients, gates and triage rules; parses uploads, reads ECG pictures
demo.py          seed data: real PTB-XL test records or the synthetic ECG generator
store.py         SQLite schema and helpers (web/data/, git-ignored)
static/          index.html, css/app.css, js/{app,patient,clinician,ecg,ui,api}.js
```

The database lives in `web/data/` (override with `CARDIOSENTINEL_DATA`). This is a research
prototype: passwords are hashed (PBKDF2) and sessions are bearer tokens. It is not hardened for
real patient data.
