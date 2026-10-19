"""Seed data for the CardioSentinel portal.

With prepared PTB-XL data and trained models, demo patients get real fold-10 (test) recordings
and real model predictions; the PTB-XL label is stored as the reference label. Without them, a
synthetic 12-lead generator is used and predictions are simulated (flagged as demo everywhere).
All names, contact details and appointments are fictional.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from cardiomoe.data.labels import SUPERCLASSES
from cardiomoe.data.signal import bandpass
from cardiomoe.fourclient.train import CLIENTS

from inference import sample_meta
from store import DB, dumps, hash_pw, heart_rate, now, save_signal

log = logging.getLogger(__name__)

DEMO_PASSWORD = "demo1234"
CLINICIANS = [
    ("clinician@cardiosentinel.demo", "Dr. Aanya Rao", "Cardiologist"),
    ("dr.okafor@cardiosentinel.demo", "Dr. Chidi Okafor", "Electrophysiologist"),
]
NAMES = [
    "Maya Fernandes", "Arjun Mehta", "Lena Fischer", "Tomás Alvarez", "Priya Nair", "Jonas Berg",
    "Sofia Rossi", "Kenji Watanabe", "Amara Okoye", "Daniel Cohen", "Isabel Moreau", "Rahul Iyer",
    "Hannah Schmidt", "Mateo García", "Aisha Khan", "Lukas Novak", "Chen Wei", "Grace Mensah",
    "Oliver Hughes", "Fatima Zahra", "Noah Andersson", "Elena Petrova", "Samuel Adeyemi", "Yuki Tanaka",
    "Leila Haddad", "Marco Bianchi", "Ananya Sharma", "Pedro Costa", "Zara Ahmed", "Erik Johansson",
    "Nadia Kowalski", "Ravi Krishnan", "Clara Dubois", "Kwame Asante", "Ingrid Larsen", "Vikram Singh",
]
CONDITIONS = ["Hypertension", "Type 2 diabetes", "Hyperlipidaemia", "Previous MI", "Atrial fibrillation (history)",
              "Chronic kidney disease", "Smoker", "Family history of CAD", "Asthma", "None known"]
MEDS = ["Aspirin 75 mg", "Atorvastatin 40 mg", "Ramipril 5 mg", "Bisoprolol 2.5 mg", "Metformin 500 mg",
        "Amlodipine 5 mg", "Apixaban 5 mg", "None"]
INDICATIONS = ["Routine check-up", "Chest pain", "Palpitations", "Shortness of breath", "Pre-operative assessment",
               "Follow-up after medication change", "Dizziness", "Hypertension review"]


# --------------------------------------------------------------------------- synthetic ECG
# 3-D lead vectors (x: patient's left, y: inferior, z: anterior); limb leads in the frontal plane.
def _lead_vectors() -> np.ndarray:
    limb = [0, 60, 120, -150, -30, 90]
    v = [(np.cos(np.radians(a)), np.sin(np.radians(a)), 0.0) for a in limb]
    v += [(-0.25, 0.1, 0.96), (0.0, 0.1, 1.0), (0.35, 0.15, 0.92), (0.6, 0.2, 0.77), (0.85, 0.2, 0.48), (0.97, 0.2, 0.1)]
    return np.array(v)


LEAD_V = _lead_vectors()


def synthetic_ecg(labels: list[str], rng: np.random.Generator, fs: int = 100, seconds: int = 10) -> np.ndarray:
    """Dipole-model 12-lead ECG (12, fs*seconds) in mV with class-specific morphology."""
    n = fs * seconds
    t = np.arange(n) / fs
    hr = rng.uniform(55, 95)
    rr = 60 / hr
    beats = np.arange(rng.uniform(0, rr), seconds + rr, rr)
    beats = beats + rng.normal(0, 0.02, len(beats))
    qrs_w = 0.012 if "CD" in labels else 0.007
    waves = {  # name: (offset s, width s, direction, amplitude mV)
        "P": (-0.16, 0.025, (0.6, 0.8, 0.1), 0.15),
        "Q": (-0.025, qrs_w, (-0.3, -0.2, 0.6), 0.15),
        "R": (0.0, qrs_w, (0.6, 0.75, -0.3), 1.3),
        "S": (0.03 + (0.02 if "CD" in labels else 0), qrs_w, (-0.4, -0.3, -0.5), 0.45),
        "T": (0.28, 0.05, (0.5, 0.6, 0.3), 0.32),
    }
    if "HYP" in labels:
        waves["R"] = (0.0, qrs_w, (0.75, 0.55, -0.45), 2.4)
        waves["S"] = (0.03, qrs_w, (-0.3, -0.2, -0.9), 1.2)
        waves["T"] = (0.28, 0.06, (-0.5, -0.2, 0.2), 0.2)  # lateral strain pattern
    if "STTC" in labels:
        waves["T"] = (0.28, 0.055, (-0.5, -0.6, -0.3), 0.25)
        waves["ST"] = (0.16, 0.06, (-0.6, -0.5, -0.4), 0.12)
    if "MI" in labels:
        waves["Qmi"] = (-0.02, qrs_w * 1.6, (0.0, -1.0, 0.1), 0.5)  # inferior Q waves
        waves["ST"] = (0.15, 0.07, (0.1, 0.9, 0.2), 0.18)
    if "CD" in labels:
        waves["R2"] = (0.06, qrs_w, (-0.6, 0.0, 0.8), 0.6)  # RSR' in V1
    vec = np.zeros((3, n))
    for b in beats:
        for off, w, d, a in waves.values():
            vec += np.outer(np.array(d) / np.linalg.norm(d), a * np.exp(-0.5 * ((t - b - off) / w) ** 2))
    sig = LEAD_V @ vec
    sig += 0.1 * np.sin(2 * np.pi * rng.uniform(0.1, 0.3) * t + rng.uniform(0, 6))[None]  # baseline wander
    sig += rng.normal(0, 0.02, sig.shape)
    return bandpass(sig).astype(np.float32)


def simulated_prediction(labels: list[str], rng: np.random.Generator, th: np.ndarray, triage=None):
    """Plausible client probabilities around the true label (demo mode only)."""
    from inference import package

    y = np.array([c in labels for c in SUPERCLASSES], float)
    difficulty = rng.uniform(0.3, 2.2)
    base = 1 / (1 + np.exp(-((2 * y - 1) * rng.uniform(1.0, 3.5) / difficulty + rng.normal(0, 0.6, 5))))
    client_p = np.clip(base[None] + rng.normal(0, 0.06 * difficulty, (4, 5)), 0.01, 0.99)
    out = package(client_p.mean(0), client_p, np.full(4, 0.25), th, triage)
    out["demo"] = True
    return out


# --------------------------------------------------------------------------- seeding
def _real_records(engine, n: int, rng: np.random.Generator):
    """Fold-10 PTB-XL records with a mix of classes (requires prepared data + trained models)."""
    import pandas as pd

    from cardiomoe.data import ptbxl
    from cardiomoe.utils import load_config, repo_path

    cfg = load_config()
    meta = ptbxl.load_meta(cfg, False)
    raw = pd.read_csv(repo_path(cfg["paths"]["ptbxl"]) / "ptbxl_database.csv", index_col="ecg_id")
    sigs = ptbxl.load_signal_array(cfg, False)
    test = meta[meta.strat_fold == 10]
    picks = []
    for c in SUPERCLASSES:  # a few of each class, then fill randomly
        idx = test.index[test[f"y_{c}"] == 1].tolist()
        picks += list(rng.choice(idx, size=min(len(idx), n // 6), replace=False))
    rest = [i for i in test.index if i not in set(picks)]
    picks += list(rng.choice(rest, size=max(0, n - len(picks)), replace=False))
    for key in picks[:n]:
        row = test.loc[key]
        labels = [c for c in SUPERCLASSES if row[f"y_{c}"] == 1]
        sig = np.asarray(sigs[row["row"]], dtype=np.float32)
        yield {"labels": labels, "signal": sig, "prediction": engine.predict(sig, sample_meta(sig, "Example recording")), "source": "PTB-XL fold 10",
               "source_ref": f"ecg_id {key}", "age": int(min(np.nan_to_num(raw.at[key, "age"], nan=60), 89)),
               "sex": "F" if raw.at[key, "sex"] == 1 else "M"}


def _synthetic_records(n: int, rng: np.random.Generator, th, triage):
    combos = [["NORM"], ["NORM"], ["NORM"], ["MI"], ["STTC"], ["CD"], ["HYP"], ["MI", "STTC"], ["CD", "HYP"],
              ["STTC", "HYP"], ["MI", "CD"]]
    for i in range(n):
        labels = combos[i % len(combos)]
        sig = synthetic_ecg(labels, rng)
        yield {"labels": labels, "signal": sig, "prediction": simulated_prediction(labels, rng, th, triage),
               "source": "Synthetic demo", "source_ref": None, "age": int(rng.integers(28, 86)),
               "sex": "F" if rng.random() < 0.5 else "M"}


def seed(db: DB, engine, n_patients: int = 36, seed_: int = 7) -> str:
    rng = np.random.default_rng(seed_)
    clin_ids = []
    for email, name, title in CLINICIANS:
        h, s = hash_pw(DEMO_PASSWORD)
        clin_ids.append(db.run("INSERT INTO users (email, pw_hash, salt, role, name, title, created_at) "
                               "VALUES (?,?,?,?,?,?,?)", (email, h, s, "clinician", name, title, now())))

    mode = "real"
    records = []
    if engine.available:
        try:
            records = list(_real_records(engine, n_patients * 2, rng))
        except Exception as e:  # data not prepared on this machine
            log.warning("real PTB-XL seeding failed (%s); using synthetic ECGs", e)
    if not records:
        mode = "synthetic"
        th = engine.th if engine.available else np.array([0.42, 0.62, 0.62, 0.64, 0.64])
        records = list(_synthetic_records(n_patients * 2, rng, th, engine.triage if engine.available else None))

    today = datetime.now(timezone.utc).replace(microsecond=0)
    local = datetime.now().astimezone().replace(microsecond=0)
    rec_iter = iter(records)
    for i, name in enumerate(NAMES[:n_patients]):
        first = next(rec_iter)
        age, sex = first["age"], first["sex"]
        dob = (today - timedelta(days=int(age * 365.25 + rng.integers(0, 365)))).date().isoformat()
        conds = list(rng.choice(CONDITIONS[:-1], size=int(rng.integers(0, 3)), replace=False)) or ["None known"]
        meds = list(rng.choice(MEDS[:-1], size=int(rng.integers(0, 3)), replace=False)) or ["None"]
        slug = name.lower().replace(" ", ".").encode("ascii", "ignore").decode()
        pid = db.run(
            "INSERT INTO patients (mrn, name, dob, sex, height_cm, weight_kg, phone, email, address, conditions, "
            "medications, allergies, emergency_contact, clinician_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"CS-{100231 + i * 37}", name, dob, sex, float(rng.normal(176 if sex == "M" else 163, 7)).__round__(0),
             float(rng.normal(82 if sex == "M" else 68, 11)).__round__(0), f"+1 555 01{i:02d} {rng.integers(1000, 9999)}",
             f"{slug}@example.com", f"{rng.integers(10, 99)} Demo Street, Springfield",
             ", ".join(conds), ", ".join(meds), "Penicillin" if rng.random() < 0.15 else "None known",
             f"Family contact, +1 555 02{i:02d} {rng.integers(1000, 9999)}", clin_ids[i % 2], now()))
        n_ecg = 1 if i % 3 else 2
        recs = [first] + [next(rec_iter) for _ in range(n_ecg - 1)]
        for j, r in enumerate(recs):
            when = today - timedelta(days=int(rng.integers(0, 6) if j == 0 else rng.integers(40, 200)),
                                     hours=int(rng.integers(0, 10)), minutes=int(rng.integers(0, 59)))
            pred = r["prediction"]
            older = j > 0 or (i > 0 and rng.random() < 0.35)  # demo patient keeps one pending ECG
            status = "reviewed" if older else ("auto_reported" if pred["auto_report"] else "needs_review")
            eid = db.run(
                "INSERT INTO ecgs (patient_id, recorded_at, source, source_ref, device, indication, prediction, "
                "reference_labels, status, released, demo, heart_rate, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (pid, when.isoformat(), r["source"], r["source_ref"], "12-lead resting ECG, 100 Hz, 10 s",
                 str(rng.choice(INDICATIONS)), dumps(pred), dumps(r["labels"]), status,
                 int(status == "reviewed"), int(mode == "synthetic"), heart_rate(r["signal"]), when.isoformat()))
            save_signal(eid, r["signal"])
            if status == "reviewed":
                final = {c: int(c in r["labels"]) for c in SUPERCLASSES}
                agrees = int(all(final[c] == pred["decisions"][c] for c in SUPERCLASSES))
                db.run("INSERT INTO reviews (ecg_id, clinician_id, final_labels, agrees_with_ai, notes, patient_summary, "
                       "created_at) VALUES (?,?,?,?,?,?,?)",
                       (eid, clin_ids[i % 2], dumps(final), agrees,
                        "Reviewed against AI pre-read." + ("" if agrees else " AI decision corrected."),
                        patient_summary(final), (when + timedelta(hours=6)).isoformat()))
        # appointments + messages
        nxt = (local + timedelta(days=int(rng.integers(2, 40)))).replace(hour=int(rng.integers(9, 17)))
        db.run("INSERT INTO appointments (patient_id, clinician_id, starts_at, kind, location, status, notes, created_at) "
               "VALUES (?,?,?,?,?,?,?,?)", (pid, clin_ids[i % 2], nxt.replace(minute=0, second=0).isoformat(),
                                            str(rng.choice(["ECG follow-up", "Cardiology consultation", "Echocardiogram",
                                                            "Medication review"])),
                                            "Cardiology Clinic, Room " + str(rng.integers(101, 120)), "scheduled", "", now()))
        past = (local - timedelta(days=int(rng.integers(30, 120)))).replace(hour=int(rng.integers(9, 17)))
        db.run("INSERT INTO appointments (patient_id, clinician_id, starts_at, kind, location, status, notes, created_at) "
               "VALUES (?,?,?,?,?,?,?,?)", (pid, clin_ids[i % 2], past.replace(minute=30, second=0).isoformat(),
                                            "Resting 12-lead ECG", "ECG Lab, Ground floor", "completed", "", now()))
        db.run("INSERT INTO messages (patient_id, sender_id, body, read_by_patient, read_by_clinician, created_at) "
               "VALUES (?,?,?,?,?,?)", (pid, clin_ids[i % 2],
                                        f"Hello {name.split()[0]}, your recent ECG has been received. We will let you "
                                        "know here once a cardiologist has reviewed it.", int(i % 2), 1,
                                        (today - timedelta(days=1)).isoformat()))
        if i == 0:
            h, s = hash_pw(DEMO_PASSWORD)
            uid = db.run("INSERT INTO users (email, pw_hash, salt, role, name, title, patient_id, created_at) "
                         "VALUES (?,?,?,?,?,?,?,?)", ("patient@cardiosentinel.demo", h, s, "patient", name, None, pid, now()))
            db.run("INSERT INTO messages (patient_id, sender_id, body, read_by_patient, read_by_clinician, created_at) "
                   "VALUES (?,?,?,?,?,?)", (pid, uid, "Thank you. Should I keep taking my medication as usual before "
                                            "the follow-up?", 1, 0, (today - timedelta(hours=20)).isoformat()))
    _seed_patient_activity(db, rng, list(rec_iter), clin_ids, today)
    db.audit(None, "seed", f"{n_patients} demo patients, {mode} ECGs")
    return mode


def _self_check(db, pid, r, when, shared):
    pred = r["prediction"]
    eid = db.run(
        "INSERT INTO ecgs (patient_id, recorded_at, source, source_ref, device, indication, prediction, reference_labels, "
        "status, released, demo, heart_rate, created_at, shared) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (pid, when.isoformat(), "Self-check", "Uploaded ECG", "Uploaded by patient", "Self-check", dumps(pred), None,
         "auto_reported" if pred["auto_report"] else "needs_review", 0, int(bool(pred.get("demo"))),
         heart_rate(r["signal"]), when.isoformat(), int(shared)))
    save_signal(eid, r["signal"])
    return eid


def _note(db, pid, when, feeling, symptoms, severity, started, body, shared):
    return db.run("INSERT INTO notes (patient_id, feeling, symptoms, severity, started, body, shared, created_at) "
                  "VALUES (?,?,?,?,?,?,?,?)", (pid, feeling, dumps(symptoms), severity, started, body, int(shared),
                                               when.isoformat()))


def _seed_patient_activity(db, rng, extra, clin_ids, today):
    """Journal entries, heart checks and shares, so both portals have something to show."""
    if len(extra) < 6:
        return
    demo = db.one("SELECT patient_id, id FROM users WHERE email = 'patient@cardiosentinel.demo'")
    pid, uid = demo["patient_id"], demo["id"]
    # earlier share, already answered by the doctor
    t0 = today - timedelta(days=9)
    n1 = _note(db, pid, t0 - timedelta(hours=3), 2, ["Palpitations", "Tiredness"], 6, "A few days ago",
               "My heart sometimes races in the evening, especially after coffee.", True)
    e1 = _self_check(db, pid, extra[0], t0, True)
    sid = db.run("INSERT INTO submissions (patient_id, ecg_id, message, note_ids, status, reply, replied_by, replied_at, created_at) "
                 "VALUES (?,?,?,?,?,?,?,?,?)", (pid, e1, "Could you look at this? I've been feeling my heart racing.",
                                               dumps([n1]), "replied",
                                               "Thanks for sharing. I've reviewed your ECG and notes. Please cut down on "
                                               "coffee for two weeks and keep logging how you feel. We'll talk at your follow-up.",
                                               clin_ids[0], (t0 + timedelta(hours=20)).isoformat(), t0.isoformat()))
    db.run("INSERT INTO messages (patient_id, sender_id, body, read_by_patient, read_by_clinician, created_at) VALUES (?,?,?,?,?,?)",
           (pid, clin_ids[0], "Re: what you sent: thanks for sharing. Please cut down on coffee for two weeks and keep "
                              "logging how you feel.", 1, 1, (t0 + timedelta(hours=20)).isoformat()))
    # recent private activity
    _note(db, pid, today - timedelta(days=2), 3, ["Shortness of breath"], 4, "Yesterday",
          "A bit breathless climbing the stairs to my flat.", False)
    _note(db, pid, today - timedelta(hours=5), 4, [], None, None, "Feeling better today, went for a 20 minute walk.", False)
    _self_check(db, pid, extra[1], today - timedelta(days=1), False)
    # new shares from other patients waiting for the doctor
    msgs = ["I did a heart check after feeling dizzy this morning. Should I be worried?",
            "Chest felt tight while gardening yesterday. Sharing my check and notes.",
            "Uploaded my ECG from the pharmacy kiosk, plus how I've been feeling this week."]
    sym = [["Dizziness or light-headedness"], ["Chest pain or pressure", "Sweating"], ["Tiredness", "Swollen ankles"]]
    others = db.q("SELECT id FROM patients WHERE id != ? ORDER BY id LIMIT 3", (pid,))
    for k, o in enumerate(others):
        t = today - timedelta(hours=int(rng.integers(2, 40)))
        n = _note(db, o["id"], t - timedelta(hours=1), 2, sym[k], int(rng.integers(3, 8)), "Earlier today", "", True)
        e = _self_check(db, o["id"], extra[2 + k], t, True)
        db.run("INSERT INTO submissions (patient_id, ecg_id, message, note_ids, status, created_at) VALUES (?,?,?,?,?,?)",
               (o["id"], e, msgs[k], dumps([n]), "sent", t.isoformat()))


def patient_summary(final: dict) -> str:
    """Plain-language summary for the patient portal."""
    found = [c for c in SUPERCLASSES if final.get(c) and c != "NORM"]
    if not found:
        return ("Your ECG looks normal. Your heart's electrical activity is within the expected range. "
                "Keep following the advice of your care team.")
    from inference import CLASS_NAMES

    names = ", ".join(CLASS_NAMES[c].lower() for c in found)
    return (f"Your ECG shows signs consistent with: {names}. This does not by itself confirm a diagnosis. "
            "Your cardiologist will discuss what it means for you and whether further tests are needed.")


__all__ = ["seed", "synthetic_ecg", "patient_summary", "DEMO_PASSWORD", "CLIENTS"]
