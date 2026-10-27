"""CardioSentinel web portal: FastAPI backend + static single-page frontend.

    python web/server.py [--port 8000] [--reseed]

Opens http://127.0.0.1:8000. Demo accounts (password demo1234):
    clinician@cardiosentinel.demo   (clinician portal)
    patient@cardiosentinel.demo     (patient portal)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import numpy as np  # noqa: E402
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

import store  # noqa: E402
from cardiomoe.data.labels import SUPERCLASSES  # noqa: E402
from cardiomoe.data.signal import LEADS  # noqa: E402
from demo import patient_summary, seed  # noqa: E402
from inference import CLASS_NAMES, Engine, is_image, parse_upload, sample_meta  # noqa: E402
from store import dumps, loads, now  # noqa: E402

log = logging.getLogger("cardiosentinel")
app = FastAPI(title="CardioSentinel", version="1.0")
db: store.DB = None  # type: ignore[assignment]
engine: Engine = None  # type: ignore[assignment]
MODE = {"seed": "unknown"}


# --------------------------------------------------------------------------- auth
def current_user(authorization: str = Header(default="")) -> dict:
    token = authorization.removeprefix("Bearer ").strip()
    u = db.one("SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ?", (token,)) if token else None
    if not u:
        raise HTTPException(401, "not signed in")
    return u


def clinician(u: dict = Depends(current_user)) -> dict:
    if u["role"] != "clinician":
        raise HTTPException(403, "clinician access only")
    return u


def patient_user(u: dict = Depends(current_user)) -> dict:
    if u["role"] != "patient":
        raise HTTPException(403, "patient access only")
    return u


def public_user(u: dict) -> dict:
    return {k: u[k] for k in ("id", "email", "role", "name", "title", "patient_id")}


class Login(BaseModel):
    email: str
    password: str


@app.post("/api/auth/login")
def login(body: Login):
    u = db.one("SELECT * FROM users WHERE lower(email) = lower(?)", (body.email.strip(),))
    if not u or not store.check_pw(body.password, u["pw_hash"], u["salt"]):
        raise HTTPException(401, "Incorrect email or password")
    import secrets

    token = secrets.token_urlsafe(32)
    db.run("INSERT INTO sessions (token, user_id, created_at) VALUES (?,?,?)", (token, u["id"], now()))
    db.audit(u["id"], "login")
    return {"token": token, "user": public_user(u)}


class Signup(BaseModel):
    name: str
    email: str
    password: str
    dob: str | None = None
    sex: str | None = None


@app.post("/api/auth/signup")
def signup(body: Signup):
    """Patient self-registration. New patients are linked to the first clinician (demo setup)."""
    import re
    import secrets

    email = body.email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(400, "Please enter a valid email address.")
    if len(body.password) < 8:
        raise HTTPException(400, "Your password needs at least 8 characters.")
    if not body.name.strip():
        raise HTTPException(400, "Please enter your name.")
    if db.one("SELECT id FROM users WHERE lower(email) = ?", (email,)):
        raise HTTPException(400, "An account with this email already exists. Try signing in.")
    doc = db.one("SELECT id FROM users WHERE role = 'clinician' ORDER BY id LIMIT 1")
    n = db.one("SELECT COUNT(*) AS n FROM patients")["n"]
    pid = db.run("INSERT INTO patients (mrn, name, dob, sex, email, clinician_id, created_at) VALUES (?,?,?,?,?,?,?)",
                 (f"CS-{300001 + n}", body.name.strip(), body.dob or None, body.sex or None, email,
                  doc["id"] if doc else None, now()))
    h, salt = store.hash_pw(body.password)
    uid = db.run("INSERT INTO users (email, pw_hash, salt, role, name, patient_id, created_at) VALUES (?,?,?,?,?,?,?)",
                 (email, h, salt, "patient", body.name.strip(), pid, now()))
    token = secrets.token_urlsafe(32)
    db.run("INSERT INTO sessions (token, user_id, created_at) VALUES (?,?,?)", (token, uid, now()))
    db.audit(uid, "signup")
    return {"token": token, "user": public_user(db.one("SELECT * FROM users WHERE id = ?", (uid,)))}


@app.post("/api/auth/logout")
def logout(authorization: str = Header(default="")):
    db.run("DELETE FROM sessions WHERE token = ?", (authorization.removeprefix("Bearer ").strip(),))
    return {"ok": True}


@app.get("/api/auth/me")
def me(u: dict = Depends(current_user)):
    return public_user(u)


@app.get("/api/system")
def system():
    return {"engine": engine.info(), "seed_mode": MODE["seed"], "classes": SUPERCLASSES, "class_names": CLASS_NAMES,
            "leads": LEADS, "demo_accounts": [
                {"role": "clinician", "email": "clinician@cardiosentinel.demo", "password": "demo1234"},
                {"role": "patient", "email": "patient@cardiosentinel.demo", "password": "demo1234"}]}


# --------------------------------------------------------------------------- helpers
def ecg_row(e: dict, full: bool = False) -> dict:
    out = dict(e)
    out["prediction"] = loads(e["prediction"])
    out["reference_labels"] = loads(e["reference_labels"])
    rev = db.one("SELECT r.*, u.name AS clinician_name FROM reviews r JOIN users u ON u.id = r.clinician_id "
                 "WHERE r.ecg_id = ? ORDER BY r.id DESC LIMIT 1", (e["id"],))
    if rev:
        rev["final_labels"] = loads(rev["final_labels"])
    out["review"] = rev
    out["findings"] = findings(out)
    return out


def findings(e: dict) -> list[str]:
    labels = e["review"]["final_labels"] if e.get("review") else e["prediction"]["decisions"]
    return [c for c in SUPERCLASSES if labels.get(c)]


def age(dob: str | None) -> int | None:
    if not dob:
        return None
    from datetime import date

    d = date.fromisoformat(dob)
    t = date.today()
    return t.year - d.year - ((t.month, t.day) < (d.month, d.day))


def patient_row(p: dict) -> dict:
    out = dict(p)
    out["age"] = age(p["dob"])
    last = db.one("SELECT * FROM ecgs WHERE patient_id = ? AND shared = 1 ORDER BY recorded_at DESC LIMIT 1", (p["id"],))
    out["last_ecg"] = ecg_row(last) if last else None
    out["n_ecgs"] = db.one("SELECT COUNT(*) AS n FROM ecgs WHERE patient_id = ? AND shared = 1", (p["id"],))["n"]
    c = db.one("SELECT name FROM users WHERE id = ?", (p["clinician_id"],))
    out["clinician_name"] = c["name"] if c else None
    return out


def get_patient(pid: int) -> dict:
    p = db.one("SELECT * FROM patients WHERE id = ?", (pid,))
    if not p:
        raise HTTPException(404, "patient not found")
    return p


def get_ecg(eid: int) -> dict:
    e = db.one("SELECT * FROM ecgs WHERE id = ?", (eid,))
    if not e:
        raise HTTPException(404, "ECG not found")
    return e


def is_self_check(e: dict) -> bool:
    return str(e["source"]).startswith("Self-check")


def visible_ecg(eid: int, u: dict) -> dict:
    """Clinicians see shared ECGs; patients see their own released ECGs and their own self-checks."""
    e = get_ecg(eid)
    if u["role"] == "clinician":
        ok = bool(e["shared"])
    else:
        ok = e["patient_id"] == u["patient_id"] and (bool(e["released"]) or is_self_check(e))
    if not ok:
        raise HTTPException(404, "ECG not found")
    return e


# --------------------------------------------------------------------------- clinician API
@app.get("/api/dashboard")
def dashboard(u: dict = Depends(clinician)):
    counts = {r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) AS n FROM ecgs WHERE shared = 1 GROUP BY status")}
    total = sum(counts.values())
    recent = [ecg_row(e) | {"patient_name": e["patient_name"]} for e in db.q(
        "SELECT e.*, p.name AS patient_name FROM ecgs e JOIN patients p ON p.id = e.patient_id "
        "WHERE e.shared = 1 ORDER BY e.recorded_at DESC LIMIT 8")]
    queue = [ecg_row(e) | {"patient_name": e["patient_name"]} for e in db.q(
        "SELECT e.*, p.name AS patient_name FROM ecgs e JOIN patients p ON p.id = e.patient_id "
        "WHERE e.status = 'needs_review' AND e.shared = 1 ORDER BY e.recorded_at DESC LIMIT 6")]
    dist = {c: 0 for c in SUPERCLASSES}
    agree = db.one("SELECT AVG(agrees_with_ai) AS a, COUNT(*) AS n FROM reviews")
    for e in db.q("SELECT * FROM ecgs WHERE shared = 1"):
        for c in findings(ecg_row(e)):
            dist[c] += 1
    appts = db.q("SELECT a.*, p.name AS patient_name FROM appointments a JOIN patients p ON p.id = a.patient_id "
                 "WHERE a.status = 'scheduled' AND a.starts_at >= ? ORDER BY a.starts_at LIMIT 6", (now(),))
    unread = db.one("SELECT COUNT(*) AS n FROM messages m JOIN users u ON u.id = m.sender_id "
                    "WHERE u.role = 'patient' AND m.read_by_clinician = 0")["n"]
    return {"counts": counts, "total": total, "n_patients": db.one("SELECT COUNT(*) AS n FROM patients")["n"],
            "recent": recent, "queue": queue, "class_distribution": dist, "appointments": appts,
            "agreement": agree, "unread_messages": unread,
            "new_submissions": db.one("SELECT COUNT(*) AS n FROM submissions WHERE status = 'sent'")["n"],
            }


@app.get("/api/patients")
def patients(q: str = "", status: str = "", u: dict = Depends(clinician)):
    rows = db.q("SELECT * FROM patients WHERE name LIKE ? OR mrn LIKE ? ORDER BY name", (f"%{q}%", f"%{q}%"))
    out = [patient_row(p) for p in rows]
    if status:
        out = [p for p in out if p["last_ecg"] and p["last_ecg"]["status"] == status]
    return out


class NewPatient(BaseModel):
    name: str
    dob: str | None = None
    sex: str | None = None
    phone: str | None = None
    email: str | None = None
    conditions: str | None = None
    medications: str | None = None
    allergies: str | None = None


@app.post("/api/patients")
def create_patient(body: NewPatient, u: dict = Depends(clinician)):
    n = db.one("SELECT COUNT(*) AS n FROM patients")["n"]
    pid = db.run("INSERT INTO patients (mrn, name, dob, sex, phone, email, conditions, medications, allergies, "
                 "clinician_id, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 (f"CS-{200001 + n}", body.name, body.dob, body.sex, body.phone, body.email, body.conditions,
                  body.medications, body.allergies, u["id"], now()))
    db.audit(u["id"], "create_patient", body.name)
    return patient_row(get_patient(pid))


@app.get("/api/patients/{pid}")
def patient_detail(pid: int, u: dict = Depends(clinician)):
    p = patient_row(get_patient(pid))
    p["ecgs"] = [ecg_row(e) for e in db.q(
        "SELECT * FROM ecgs WHERE patient_id = ? AND shared = 1 ORDER BY recorded_at DESC", (pid,))]
    p["appointments"] = db.q("SELECT * FROM appointments WHERE patient_id = ? ORDER BY starts_at DESC", (pid,))
    p["notes"] = [note_row(n) for n in db.q(
        "SELECT * FROM notes WHERE patient_id = ? AND shared = 1 ORDER BY created_at DESC", (pid,))]
    p["submissions"] = [submission_row(x) for x in db.q(
        "SELECT * FROM submissions WHERE patient_id = ? ORDER BY created_at DESC", (pid,))]
    return p


@app.get("/api/ecgs")
def ecgs(status: str = "", u: dict = Depends(clinician)):
    sql = ("SELECT e.*, p.name AS patient_name, p.mrn, p.dob, p.sex FROM ecgs e JOIN patients p ON p.id = e.patient_id"
           " WHERE e.shared = 1" + (" AND e.status = ?" if status else "") + " ORDER BY e.recorded_at DESC")
    out = []
    for e in db.q(sql, (status,) if status else ()):
        r = ecg_row(e)
        r["patient_age"] = age(e["dob"])
        out.append(r)
    if status == "needs_review":  # most indeterminate findings first
        out.sort(key=lambda r: -len(r["prediction"].get("indeterminate", [])))
    return out


@app.get("/api/ecgs/{eid}")
def ecg_detail(eid: int, u: dict = Depends(current_user)):
    e = visible_ecg(eid, u)
    r = ecg_row(e)
    r["patient"] = patient_row(get_patient(e["patient_id"]))
    if u["role"] == "patient":
        r = patient_view(r)
    return r


@app.get("/api/ecgs/{eid}/signal")
def ecg_signal(eid: int, u: dict = Depends(current_user)):
    visible_ecg(eid, u)
    sig = store.load_signal(eid)
    return {"fs": 100, "leads": LEADS, "units": "mV", "data": np.round(sig, 3).tolist()}


@app.get("/api/ecgs/{eid}/signal.csv")
def ecg_signal_csv(eid: int, u: dict = Depends(current_user)):
    """The 12-lead signal as CSV (time in s + one column per lead, mV). For a picture this is the
    digitized signal; leads are empty where they were not printed."""
    from fastapi.responses import Response

    e = visible_ecg(eid, u)
    sig = store.load_signal(eid)
    pl = (loads(e.get("prediction")) or {}).get("pipeline") or {}
    printed = ((pl.get("input") or {}).get("image") or {}).get("seconds_per_lead")
    lines = ["time_s," + ",".join(LEADS)]
    for t in range(sig.shape[1]):
        row = []
        for i, l in enumerate(LEADS):
            visible = printed is None or _printed(l, t, printed)
            row.append(f"{sig[i, t]:.3f}" if visible else "")
        lines.append(f"{t / 100:.2f}," + ",".join(row))
    return Response("\n".join(lines) + "\n", media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="ecg_{eid}.csv"'})


def _printed(lead: str, t: int, seconds: dict) -> bool:
    """Was sample t of this lead on the page? 10 s leads: always; 2.5 s leads: their column of the 3x4 grid."""
    from cardiomoe.images.layouts import GRID_3x4

    if seconds.get(lead, 10) >= 9.9:
        return True
    col = next((r.index(lead) for r in GRID_3x4 if lead in r), None)
    return col is not None and col * 250 <= t < (col + 1) * 250


class Review(BaseModel):
    final_labels: dict[str, int]
    notes: str = ""
    patient_summary: str | None = None
    release: bool = True


@app.post("/api/ecgs/{eid}/review")
def review(eid: int, body: Review, u: dict = Depends(clinician)):
    e = ecg_row(visible_ecg(eid, u))
    final = {c: int(bool(body.final_labels.get(c, 0))) for c in SUPERCLASSES}
    agrees = int(all(final[c] == e["prediction"]["decisions"][c] for c in SUPERCLASSES))
    db.run("INSERT INTO reviews (ecg_id, clinician_id, final_labels, agrees_with_ai, notes, patient_summary, created_at) "
           "VALUES (?,?,?,?,?,?,?)", (eid, u["id"], dumps(final), agrees, body.notes,
                                      body.patient_summary or patient_summary(final), now()))
    db.run("UPDATE ecgs SET status = 'reviewed', released = ? WHERE id = ?", (int(body.release), eid))
    db.audit(u["id"], "review", f"ECG {eid}: {'agrees' if agrees else 'overrides'} AI")
    return ecg_row(get_ecg(eid))


@app.post("/api/ecgs/{eid}/release")
def release(eid: int, u: dict = Depends(clinician)):
    e = ecg_row(visible_ecg(eid, u))
    if e["status"] == "needs_review":
        raise HTTPException(400, "an ECG referred for review must be reviewed before release")
    db.run("UPDATE ecgs SET released = 1 WHERE id = ?", (eid,))
    db.audit(u["id"], "release", f"ECG {eid}")
    return ecg_row(get_ecg(eid))


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...), fs: int = Form(100), patient_id: int | None = Form(None),
                  indication: str = Form(""), u: dict = Depends(clinician)):
    if not engine.available:
        raise HTTPException(503, f"Live analysis needs the trained models: {engine.reason}. "
                                 "Run the training pipeline (see web/README.md).")
    data = {f.filename: await f.read() for f in files}
    if any(is_image(n) for n in data):
        sig, pred = _analyze_image(data, "Could not read the ECG image")
        return _store_prediction(sig, pred, patient_id, indication, "Photo / scan", files[0].filename, u)
    try:
        sig, meta = parse_upload(data, fs)
    except Exception as ex:
        raise HTTPException(400, f"Could not read the ECG: {ex}")
    return _store_prediction(sig, engine.predict(sig, meta), patient_id, indication, "Upload", files[0].filename, u)


MAX_PICTURES = 4


def _analyze_image(data: dict, what: str):
    """Photos / scans / PDFs of one printed ECG -> (digitized signal, prediction)."""
    from cardiomoe.images.digitize import DigitizeError

    if not engine.image_available:
        raise HTTPException(503, f"Reading ECG photos needs the image models: {engine.image_reason}. "
                                 "Run the image pipeline (see web/README.md).")
    imgs = [n for n in data if is_image(n)]
    if len(imgs) != len(data):
        raise HTTPException(400, "Upload either pictures of a printed ECG or a digital ECG file, not both.")
    if len(imgs) > MAX_PICTURES:
        raise HTTPException(400, f"Upload at most {MAX_PICTURES} pictures of the same ECG.")
    try:
        return engine.predict_images([(n, data[n]) for n in imgs])
    except DigitizeError as ex:
        raise HTTPException(400, f"{what}: {ex}. Take the photo straight on, with the whole ECG page in the frame "
                                 "and no shadows, or upload a scan or PDF.")


@app.get("/api/samples")
def samples(u: dict = Depends(clinician)):
    """PTB-XL fold-10 records that can be analysed with one click (needs prepared data)."""
    return _sample_list()


def _sample_list() -> list[dict]:
    try:
        from cardiomoe.data import ptbxl
        from cardiomoe.utils import load_config

        meta = ptbxl.load_meta(load_config(), False)
    except Exception:
        return []
    test = meta[meta.strat_fold == 10]
    rng = np.random.default_rng(3)
    out = []
    for c in SUPERCLASSES:
        idx = test.index[(test[f"y_{c}"] == 1) & (test[[f"y_{k}" for k in SUPERCLASSES]].sum(1) == 1)]
        for k in rng.choice(idx, size=min(2, len(idx)), replace=False):
            out.append({"ecg_id": int(k), "label": c})
    return out


@app.post("/api/analyze/sample/{ecg_id}")
def analyze_sample(ecg_id: int, patient_id: int | None = None, u: dict = Depends(clinician)):
    if not engine.available:
        raise HTTPException(503, f"Live analysis needs the trained models: {engine.reason}.")
    sig, ref = _sample_signal(ecg_id)
    return _store_prediction(sig, engine.predict(sig, sample_meta(sig, f"Example recording {ecg_id}")), patient_id, "Sample record", "PTB-XL fold 10",
                             f"ecg_id {ecg_id}", u, ref)


def _sample_signal(ecg_id: int):
    from cardiomoe.data import ptbxl
    from cardiomoe.utils import load_config

    cfg = load_config()
    meta = ptbxl.load_meta(cfg, False)
    if ecg_id not in meta.index or meta.at[ecg_id, "strat_fold"] != 10:
        raise HTTPException(404, "sample must be a PTB-XL fold-10 record")
    sig = np.asarray(ptbxl.load_signal_array(cfg, False)[meta.at[ecg_id, "row"]], dtype=np.float32)
    return sig, [c for c in SUPERCLASSES if meta.at[ecg_id, f"y_{c}"] == 1]


def _store_prediction(sig, pred, patient_id, indication, source, ref, u, reference=None):
    result = {"prediction": pred, "signal": {"fs": 100, "leads": LEADS, "data": np.round(sig, 3).tolist()},
              "heart_rate": store.heart_rate(sig), "reference_labels": reference}
    if patient_id:
        get_patient(patient_id)
        status = "auto_reported" if pred["auto_report"] else "needs_review"
        eid = db.run("INSERT INTO ecgs (patient_id, recorded_at, source, source_ref, device, indication, prediction, "
                     "reference_labels, status, released, demo, heart_rate, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                     (patient_id, now(), source, ref, "12-lead ECG", indication or "Clinician upload", dumps(pred),
                      dumps(reference), status, 0, 0, result["heart_rate"], now()))
        store.save_signal(eid, sig)
        db.audit(u["id"], "analyze", f"ECG {eid} for patient {patient_id}: {status}")
        result["ecg_id"] = eid
        result["status"] = status
    return result


@app.get("/api/audit")
def audit(u: dict = Depends(clinician)):
    return db.q("SELECT a.*, u.name AS user_name FROM audit a LEFT JOIN users u ON u.id = a.user_id "
                "ORDER BY a.id DESC LIMIT 100")


# --------------------------------------------------------------------------- messages & appointments
def _thread(pid: int) -> list[dict]:
    return db.q("SELECT m.*, u.name AS sender_name, u.role AS sender_role FROM messages m "
                "JOIN users u ON u.id = m.sender_id WHERE m.patient_id = ? ORDER BY m.created_at", (pid,))


@app.get("/api/messages")
def message_threads(u: dict = Depends(clinician)):
    out = []
    for p in db.q("SELECT DISTINCT p.* FROM patients p JOIN messages m ON m.patient_id = p.id ORDER BY p.name"):
        t = _thread(p["id"])
        out.append({"patient": {"id": p["id"], "name": p["name"], "mrn": p["mrn"]}, "last": t[-1],
                    "unread": sum(1 for m in t if m["sender_role"] == "patient" and not m["read_by_clinician"])})
    out.sort(key=lambda r: r["last"]["created_at"], reverse=True)
    return out


@app.get("/api/messages/{pid}")
def message_thread(pid: int, u: dict = Depends(clinician)):
    db.run("UPDATE messages SET read_by_clinician = 1 WHERE patient_id = ?", (pid,))
    return {"patient": patient_row(get_patient(pid)), "messages": _thread(pid)}


class Msg(BaseModel):
    body: str


@app.post("/api/messages/{pid}")
def send_message(pid: int, body: Msg, u: dict = Depends(clinician)):
    get_patient(pid)
    if not body.body.strip():
        raise HTTPException(400, "empty message")
    db.run("INSERT INTO messages (patient_id, sender_id, body, read_by_patient, read_by_clinician, created_at) "
           "VALUES (?,?,?,?,?,?)", (pid, u["id"], body.body.strip(), 0, 1, now()))
    return _thread(pid)


@app.get("/api/appointments")
def appointments(u: dict = Depends(clinician)):
    return db.q("SELECT a.*, p.name AS patient_name, p.mrn FROM appointments a JOIN patients p ON p.id = a.patient_id "
                "ORDER BY a.starts_at")


class Appt(BaseModel):
    patient_id: int | None = None
    starts_at: str
    kind: str
    location: str | None = "Cardiology Clinic"
    notes: str | None = ""


@app.post("/api/appointments")
def create_appointment(body: Appt, u: dict = Depends(clinician)):
    get_patient(body.patient_id)
    db.run("INSERT INTO appointments (patient_id, clinician_id, starts_at, kind, location, status, notes, created_at) "
           "VALUES (?,?,?,?,?,?,?,?)", (body.patient_id, u["id"], body.starts_at, body.kind, body.location,
                                        "scheduled", body.notes, now()))
    return {"ok": True}


class ApptStatus(BaseModel):
    status: str


@app.patch("/api/appointments/{aid}")
def update_appointment(aid: int, body: ApptStatus, u: dict = Depends(current_user)):
    a = db.one("SELECT * FROM appointments WHERE id = ?", (aid,))
    if not a or (u["role"] == "patient" and a["patient_id"] != u["patient_id"]):
        raise HTTPException(404, "appointment not found")
    allowed = {"scheduled", "completed", "cancelled", "requested"} if u["role"] == "clinician" else {"cancelled"}
    if body.status not in allowed:
        raise HTTPException(400, "status not allowed")
    db.run("UPDATE appointments SET status = ? WHERE id = ?", (body.status, aid))
    return {"ok": True}


# --------------------------------------------------------------------------- patient API
def ai_for_patient(pred: dict) -> dict:
    """The AI pre-read in patient terms: per finding 'found' / 'not_found' / 'unclear' plus advice."""
    word = {"positive": "found", "negative": "not_found", "indeterminate": "unclear"}
    per = {c: {"result": word[pred["triage"][c]], "probability": pred["probs"][c]} for c in SUPERCLASSES}
    abnormal = [c for c in SUPERCLASSES if c != "NORM" and pred["triage"][c] == "positive"]
    unclear = [c for c in SUPERCLASSES if pred["triage"][c] == "indeterminate"]
    mi = pred["triage"]["MI"] == "positive" or (pred["triage"]["MI"] == "indeterminate" and pred["suggested"]["MI"])
    if abnormal:
        level, advice = "see_doctor", "The AI found signs that your doctor should look at. We recommend sending this ECG to your doctor."
    elif unclear:
        level, advice = "check", "Part of this result is unclear to the AI. We recommend sending it to your doctor so a cardiologist can check it."
    else:
        level, advice = "reassuring", "The AI did not find signs of the heart conditions it looks for. Talk to your doctor if you have symptoms."
    return {"per_class": per, "abnormal": abnormal, "unclear": unclear, "level": level, "advice": advice,
            "urgent": bool(mi)}


def patient_view(e: dict) -> dict:
    """What a patient sees: the clinician's result when there is one, else (for self-checks) the AI pre-read."""
    rev = e.get("review")
    self_check = is_self_check(e)
    labels = rev["final_labels"] if rev else e["prediction"]["decisions"]
    found = [c for c in SUPERCLASSES if labels.get(c)]
    sub_ = db.one("SELECT id, status FROM submissions WHERE ecg_id = ? ORDER BY id DESC LIMIT 1", (e["id"],))
    return {"id": e["id"], "recorded_at": e["recorded_at"], "indication": e["indication"], "device": e["device"],
            "heart_rate": e["heart_rate"], "status": e["status"], "findings": found,
            "summary": (rev or {}).get("patient_summary") or patient_summary(labels),
            "reviewed_by": rev["clinician_name"] if rev else None, "reviewed_at": rev["created_at"] if rev else None,
            "self_check": self_check, "shared": bool(e["shared"]),
            "submission": sub_, "ai": ai_for_patient(e["prediction"]) if self_check else None,
            "pipeline": e["prediction"].get("pipeline") if self_check else None,
            "image_quality": e["prediction"].get("image_quality") if self_check else None,
            "image_agreement": e["prediction"].get("image_agreement") if self_check else None,
            "image_auto_report": e["prediction"].get("image_auto_report") if self_check else None,
            "demo": e["demo"]}


@app.get("/api/me/overview")
def my_overview(u: dict = Depends(patient_user)):
    p = patient_row(get_patient(u["patient_id"]))
    released = [patient_view(ecg_row(e)) for e in db.q(
        "SELECT * FROM ecgs WHERE patient_id = ? AND (released = 1 OR source LIKE 'Self-check%') "
        "ORDER BY recorded_at DESC", (u["patient_id"],))]
    pending = db.one("SELECT COUNT(*) AS n FROM ecgs WHERE patient_id = ? AND released = 0 "
                     "AND source NOT LIKE 'Self-check%'", (u["patient_id"],))["n"]
    nxt = db.one("SELECT a.*, u.name AS clinician_name FROM appointments a LEFT JOIN users u ON u.id = a.clinician_id "
                 "WHERE a.patient_id = ? AND a.status IN ('scheduled', 'requested') AND a.starts_at >= ? "
                 "ORDER BY a.starts_at LIMIT 1", (u["patient_id"], now()))
    unread = db.one("SELECT COUNT(*) AS n FROM messages m JOIN users s ON s.id = m.sender_id "
                    "WHERE m.patient_id = ? AND s.role = 'clinician' AND m.read_by_patient = 0", (u["patient_id"],))["n"]
    p.pop("last_ecg", None)
    notes = [note_row(n) for n in db.q("SELECT * FROM notes WHERE patient_id = ? ORDER BY created_at DESC LIMIT 3",
                                         (u["patient_id"],))]
    replies = db.one("SELECT COUNT(*) AS n FROM submissions WHERE patient_id = ? AND status = 'replied'",
                     (u["patient_id"],))["n"]
    return {"patient": p, "latest": released[0] if released else None, "n_released": len(released),
            "pending": pending, "next_appointment": nxt, "unread_messages": unread, "recent_notes": notes,
            "n_replies": replies}


@app.get("/api/me/ecgs")
def my_ecgs(u: dict = Depends(patient_user)):
    rows = db.q("SELECT * FROM ecgs WHERE patient_id = ? ORDER BY recorded_at DESC", (u["patient_id"],))
    out = []
    for e in rows:
        if e["released"] or is_self_check(e):
            out.append(patient_view(ecg_row(e)) | {"released": True})
        else:
            out.append({"id": e["id"], "recorded_at": e["recorded_at"], "indication": e["indication"],
                        "released": False, "status": "pending"})
    return out


@app.get("/api/me/appointments")
def my_appointments(u: dict = Depends(patient_user)):
    return db.q("SELECT a.*, u.name AS clinician_name FROM appointments a LEFT JOIN users u ON u.id = a.clinician_id "
                "WHERE a.patient_id = ? ORDER BY a.starts_at DESC", (u["patient_id"],))


class ApptRequest(BaseModel):
    starts_at: str
    kind: str
    notes: str | None = ""


@app.post("/api/me/appointments")
def request_appointment(body: ApptRequest, u: dict = Depends(patient_user)):
    p = get_patient(u["patient_id"])
    db.run("INSERT INTO appointments (patient_id, clinician_id, starts_at, kind, location, status, notes, created_at) "
           "VALUES (?,?,?,?,?,?,?,?)", (p["id"], p["clinician_id"], body.starts_at, body.kind, "To be confirmed",
                                        "requested", body.notes, now()))
    return {"ok": True}


@app.get("/api/me/messages")
def my_messages(u: dict = Depends(patient_user)):
    db.run("UPDATE messages SET read_by_patient = 1 WHERE patient_id = ?", (u["patient_id"],))
    return _thread(u["patient_id"])


@app.post("/api/me/messages")
def my_send(body: Msg, u: dict = Depends(patient_user)):
    if not body.body.strip():
        raise HTTPException(400, "empty message")
    db.run("INSERT INTO messages (patient_id, sender_id, body, read_by_patient, read_by_clinician, created_at) "
           "VALUES (?,?,?,?,?,?)", (u["patient_id"], u["id"], body.body.strip(), 1, 0, now()))
    return _thread(u["patient_id"])


class Profile(BaseModel):
    phone: str | None = None
    email: str | None = None
    address: str | None = None
    emergency_contact: str | None = None
    medications: str | None = None
    allergies: str | None = None


@app.patch("/api/me/profile")
def my_profile(body: Profile, u: dict = Depends(patient_user)):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    if fields:
        db.run(f"UPDATE patients SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
               (*fields.values(), u["patient_id"]))
        db.audit(u["id"], "update_profile", ", ".join(fields))
    p = patient_row(get_patient(u["patient_id"]))
    p.pop("last_ecg", None)
    return p


# --------------------------------------------------------------------------- patient: self-check, journal, attachments, send to doctor
@app.get("/api/me/samples")
def my_samples(u: dict = Depends(patient_user)):
    """Example recordings for patients without their own ECG file (labels are not shown)."""
    return [{"ecg_id": s["ecg_id"]} for s in _sample_list()][:6]


def _self_check(sig, pred, u, source_ref: str, indication: str) -> dict:
    status = "auto_reported" if pred["auto_report"] else "needs_review"
    eid = db.run("INSERT INTO ecgs (patient_id, recorded_at, source, source_ref, device, indication, prediction, "
                 "reference_labels, status, released, demo, heart_rate, created_at, shared) "
                 "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                 (u["patient_id"], now(), "Self-check", source_ref, "Uploaded by patient", indication or "Self-check",
                  dumps(pred), None, status, 0, 0, store.heart_rate(sig), now(), 0))
    store.save_signal(eid, sig)
    db.audit(u["id"], "self_check", f"ECG {eid}")
    return patient_view(ecg_row(get_ecg(eid)))


@app.post("/api/me/analyze")
async def my_analyze(files: list[UploadFile] = File(...), fs: int = Form(100), indication: str = Form(""),
                     u: dict = Depends(patient_user)):
    if not engine.available:
        raise HTTPException(503, "ECG checks are not available right now.")
    data = {f.filename: await f.read() for f in files}
    if any(is_image(n) for n in data):
        sig, pred = _analyze_image(data, "We could not read the ECG in this picture")
        return _self_check(sig, pred, u, files[0].filename, indication)
    try:
        sig, meta = parse_upload(data, fs)
    except Exception as ex:
        raise HTTPException(400, f"We could not read this file as a 12-lead ECG ({ex}).")
    return _self_check(sig, engine.predict(sig, meta), u, files[0].filename, indication)


@app.post("/api/me/analyze/sample/{ecg_id}")
def my_analyze_sample(ecg_id: int, u: dict = Depends(patient_user)):
    if not engine.available:
        raise HTTPException(503, "ECG checks are not available right now.")
    sig, _ = _sample_signal(ecg_id)
    return _self_check(sig, engine.predict(sig, sample_meta(sig, "Example recording")), u, f"Example recording {ecg_id}", "Example recording")


# --------------------------------------------------------------------------- voice assistant (RAG over the patient's records)
FEELING_WORDS = {1: "very bad", 2: "bad", 3: "okay", 4: "good", 5: "very good"}
RESULT_WORDS = {"positive": "found", "negative": "not found", "indeterminate": "unclear"}


def patient_passages(pid: int) -> list:
    """Everything the patient can see, as short passages for retrieval (same visibility as the patient API)."""
    from assistant import Passage

    out = []
    p = get_patient(pid)
    prof = [f"{p['name']}", f"age {age(p['dob'])}" if p["dob"] else "", p["sex"] or "",
            f"conditions: {p['conditions']}" if p["conditions"] else "", f"medications: {p['medications']}" if p["medications"] else "",
            f"allergies: {p['allergies']}" if p["allergies"] else ""]
    doc = db.one("SELECT name FROM users WHERE id = ?", (p["clinician_id"],))
    if doc:
        prof.append(f"doctor: {doc['name']}")
    out.append(Passage("profile", "profile", "", "Your profile", ", ".join(x for x in prof if x)))
    for e in db.q("SELECT * FROM ecgs WHERE patient_id = ? ORDER BY recorded_at", (pid,)):
        if not (e["released"] or is_self_check(e)):
            continue
        v = patient_view(ecg_row(e))
        if v["ai"]:
            res = "; ".join(f"{CLASS_NAMES[c]} {RESULT_WORDS[t]} (AI likelihood {round(100 * pr['probs'][c])}%)"
                            for c, t in pr["triage"].items()) if (pr := loads(e["prediction"])) else ""
            text = f"AI heart check: {v['ai']['advice']} Results: {res}."
        else:
            text = f"Result: {v['summary']}"
        if v["heart_rate"]:
            text += f" Heart rate about {round(v['heart_rate'])} beats per minute."
        if v["reviewed_by"]:
            text += f" Reviewed by {v['reviewed_by']}: findings {', '.join(CLASS_NAMES[c] for c in v['findings']) or 'none'}. {v['summary']}"
        if v["self_check"]:
            text += " Shared with your doctor." if v["shared"] else " Not shared with your doctor yet."
        out.append(Passage(f"check {e['id']}", "check", e["recorded_at"], "Heart check" if v["self_check"] else "Clinic ECG", text))
    for n in db.q("SELECT * FROM notes WHERE patient_id = ? ORDER BY created_at", (pid,)):
        sym = ", ".join(loads(n["symptoms"]) or []) or "no symptoms ticked"
        bits = [f"felt {FEELING_WORDS.get(n['feeling'], 'not recorded')}", f"symptoms: {sym}"]
        if n["severity"] is not None:
            bits.append(f"strength {n['severity']} out of 10")
        if n["started"]:
            bits.append(f"started {n['started']}")
        if n["body"]:
            bits.append(f'note: "{n["body"]}"')
        out.append(Passage(f"journal {n['id']}", "journal", n["created_at"], "Symptom journal", "; ".join(bits) + "."))
    for m in db.q("SELECT m.*, u.name AS sender, u.role FROM messages m JOIN users u ON u.id = m.sender_id "
                  "WHERE m.patient_id = ? ORDER BY m.created_at", (pid,)):
        who = "You wrote" if m["role"] == "patient" else f"{m['sender']} wrote"
        out.append(Passage(f"message {m['id']}", "message", m["created_at"], "Message", f'{who}: "{m["body"]}"'))
    for a in db.q("SELECT * FROM appointments WHERE patient_id = ? ORDER BY starts_at", (pid,)):
        out.append(Passage(f"appointment {a['id']}", "appointment", a["starts_at"], "Appointment",
                           f"{a['kind']} on {a['starts_at'][:16].replace('T', ' at ')}, {a['location'] or 'location not set'}, status {a['status']}."
                           + (f" Notes: {a['notes']}" if a["notes"] else "")))
    for x in db.q("SELECT s.*, u.name AS replier FROM submissions s LEFT JOIN users u ON u.id = s.replied_by "
                  "WHERE s.patient_id = ? ORDER BY s.created_at", (pid,)):
        text = f"You shared {'heart check ' + str(x['ecg_id']) if x['ecg_id'] else 'notes'} with your doctor"
        text += f' with the message "{x["message"]}"' if x["message"] else ""
        text += f". Status: {x['status']}."
        if x["reply"]:
            text += f' {x["replier"] or "Your doctor"} replied: "{x["reply"]}"'
        out.append(Passage(f"share {x['id']}", "share", x["created_at"], "Shared with doctor", text))
    return out


class Ask(BaseModel):
    question: str
    history: list[dict] = []


@app.post("/api/me/assistant")
def my_assistant(body: Ask, u: dict = Depends(patient_user)):
    """Voice / chat assistant: answers from the patient's own records only."""
    import assistant

    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Ask a question first.")
    if len(q) > 1000:
        raise HTTPException(400, "Please keep the question under 1000 characters.")
    res = assistant.answer(q, patient_passages(u["patient_id"]), body.history)
    db.audit(u["id"], "assistant", f"{res['mode']}: {len(res['sources'])} records")
    return res


@app.get("/api/me/assistant")
def my_assistant_info(u: dict = Depends(patient_user)):
    import assistant

    cfg = assistant.llm_config()
    return {"mode": "openrouter" if cfg else "offline", "model": cfg["model"] if cfg else None,
            "suggestions": ["How was my last heart check?", "What did my doctor say?", "When is my next appointment?",
                            "How have I been feeling this week?", "Was anything unclear in my results?"]}


SYMPTOMS = ["Chest pain or pressure", "Palpitations", "Shortness of breath", "Dizziness or light-headedness",
            "Fainting", "Tiredness", "Swollen ankles", "Sweating", "Nausea", "None"]


def note_row(n: dict) -> dict:
    out = dict(n)
    out["symptoms"] = loads(n["symptoms"]) or []
    return out


class Note(BaseModel):
    feeling: int | None = None          # 1 (very bad) .. 5 (very good)
    symptoms: list[str] = []
    severity: int | None = None         # 0..10
    started: str | None = None
    body: str = ""


@app.get("/api/me/notes")
def my_notes(u: dict = Depends(patient_user)):
    return {"symptom_options": SYMPTOMS, "notes": [note_row(n) for n in db.q(
        "SELECT * FROM notes WHERE patient_id = ? ORDER BY created_at DESC", (u["patient_id"],))]}


@app.post("/api/me/notes")
def add_note(body: Note, u: dict = Depends(patient_user)):
    if not body.body.strip() and not body.symptoms and body.feeling is None:
        raise HTTPException(400, "write how you feel or pick a symptom")
    sym = [x for x in body.symptoms if x in SYMPTOMS]
    nid = db.run("INSERT INTO notes (patient_id, feeling, symptoms, severity, started, body, shared, created_at) "
                 "VALUES (?,?,?,?,?,?,?,?)", (u["patient_id"], body.feeling, dumps(sym),
                                              None if body.severity is None else max(0, min(10, body.severity)),
                                              body.started, body.body.strip(), 0, now()))
    return note_row(db.one("SELECT * FROM notes WHERE id = ?", (nid,)))


@app.delete("/api/me/notes/{nid}")
def delete_note(nid: int, u: dict = Depends(patient_user)):
    n = db.one("SELECT * FROM notes WHERE id = ? AND patient_id = ?", (nid, u["patient_id"]))
    if not n:
        raise HTTPException(404, "note not found")
    if n["shared"]:
        raise HTTPException(400, "this note was already sent to your doctor")
    db.run("DELETE FROM notes WHERE id = ?", (nid,))
    return {"ok": True}


ALLOWED = {".png": ("image/png", b"\x89PNG"), ".jpg": ("image/jpeg", b"\xff\xd8\xff"),
           ".jpeg": ("image/jpeg", b"\xff\xd8\xff"), ".gif": ("image/gif", b"GIF8"),
           ".webp": ("image/webp", b"RIFF"), ".pdf": ("application/pdf", b"%PDF")}
MAX_ATTACHMENT = 10 * 1024 * 1024


@app.post("/api/me/attachments")
async def upload_attachment(file: UploadFile = File(...), description: str = Form(""),
                            u: dict = Depends(patient_user)):
    import uuid

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED:
        raise HTTPException(400, "Please attach a photo (JPG, PNG, WEBP, GIF) or a PDF.")
    data = await file.read()
    if len(data) > MAX_ATTACHMENT:
        raise HTTPException(400, "Files can be at most 10 MB.")
    mime, magic = ALLOWED[ext]
    if not data.startswith(magic) or (ext == ".webp" and data[8:12] != b"WEBP"):
        raise HTTPException(400, "This file does not look like a valid image or PDF.")
    stored = f"{uuid.uuid4().hex}{ext}"
    (store.ATTACH_DIR / stored).write_bytes(data)
    aid = db.run("INSERT INTO attachments (patient_id, filename, mime, size, stored_as, description, created_at) "
                 "VALUES (?,?,?,?,?,?,?)", (u["patient_id"], Path(file.filename).name[:120], mime, len(data), stored,
                                            description[:300], now()))
    return attachment_row(db.one("SELECT * FROM attachments WHERE id = ?", (aid,)))


def attachment_row(a: dict) -> dict:
    return {k: a[k] for k in ("id", "filename", "mime", "size", "description", "created_at", "submission_id")}


@app.delete("/api/me/attachments/{aid}")
def delete_attachment(aid: int, u: dict = Depends(patient_user)):
    a = db.one("SELECT * FROM attachments WHERE id = ? AND patient_id = ?", (aid, u["patient_id"]))
    if not a or a["submission_id"]:
        raise HTTPException(404, "attachment not found or already sent")
    (store.ATTACH_DIR / a["stored_as"]).unlink(missing_ok=True)
    db.run("DELETE FROM attachments WHERE id = ?", (aid,))
    return {"ok": True}


@app.get("/api/attachments/{aid}")
def get_attachment(aid: int, u: dict = Depends(current_user)):
    a = db.one("SELECT * FROM attachments WHERE id = ?", (aid,))
    ok = a and ((u["role"] == "patient" and a["patient_id"] == u["patient_id"])
                or (u["role"] == "clinician" and a["submission_id"]))
    if not ok:
        raise HTTPException(404, "attachment not found")
    return FileResponse(store.ATTACH_DIR / a["stored_as"], media_type=a["mime"],
                        headers={"X-Content-Type-Options": "nosniff",
                                 "Content-Disposition": f'inline; filename="{a["filename"]}"'})


def submission_row(x: dict, full: bool = False) -> dict:
    out = dict(x)
    out["note_ids"] = loads(x["note_ids"]) or []
    out["n_attachments"] = db.one("SELECT COUNT(*) AS n FROM attachments WHERE submission_id = ?", (x["id"],))["n"]
    r = db.one("SELECT name FROM users WHERE id = ?", (x["replied_by"],)) if x["replied_by"] else None
    out["replied_by_name"] = r["name"] if r else None
    if full:
        out["notes"] = [note_row(n) for n in db.q(
            f"SELECT * FROM notes WHERE id IN ({','.join('?' * len(out['note_ids']))})", out["note_ids"])] if out["note_ids"] else []
        out["attachments"] = [attachment_row(a) for a in db.q("SELECT * FROM attachments WHERE submission_id = ?", (x["id"],))]
    return out


class Submission(BaseModel):
    ecg_id: int | None = None
    note_ids: list[int] = []
    attachment_ids: list[int] = []
    message: str = ""


@app.post("/api/me/submissions")
def send_to_doctor(body: Submission, u: dict = Depends(patient_user)):
    pid = u["patient_id"]
    if not (body.ecg_id or body.note_ids or body.attachment_ids or body.message.strip()):
        raise HTTPException(400, "add an ECG, a note, a file or a message")
    if body.ecg_id:
        e = visible_ecg(body.ecg_id, u)
        if not is_self_check(e) and not e["shared"]:
            raise HTTPException(400, "this ECG cannot be sent")
    notes = [n["id"] for n in db.q(f"SELECT id FROM notes WHERE patient_id = ? AND id IN ({','.join('?' * len(body.note_ids))})",
                                   (pid, *body.note_ids))] if body.note_ids else []
    atts = [a["id"] for a in db.q(f"SELECT id FROM attachments WHERE patient_id = ? AND submission_id IS NULL "
                                  f"AND id IN ({','.join('?' * len(body.attachment_ids))})", (pid, *body.attachment_ids))] if body.attachment_ids else []
    sid = db.run("INSERT INTO submissions (patient_id, ecg_id, message, note_ids, status, created_at) VALUES (?,?,?,?,?,?)",
                 (pid, body.ecg_id, body.message.strip(), dumps(notes), "sent", now()))
    if body.ecg_id:
        db.run("UPDATE ecgs SET shared = 1 WHERE id = ?", (body.ecg_id,))  # enters the clinician's queue
    for n in notes:
        db.run("UPDATE notes SET shared = 1 WHERE id = ?", (n,))
    for a in atts:
        db.run("UPDATE attachments SET submission_id = ? WHERE id = ?", (sid, a))
    db.audit(u["id"], "send_to_doctor", f"submission {sid}: ECG {body.ecg_id}, {len(notes)} notes, {len(atts)} files")
    return submission_row(db.one("SELECT * FROM submissions WHERE id = ?", (sid,)), full=True)


@app.get("/api/me/submissions")
def my_submissions(u: dict = Depends(patient_user)):
    return [submission_row(x) for x in db.q("SELECT * FROM submissions WHERE patient_id = ? ORDER BY created_at DESC",
                                            (u["patient_id"],))]


@app.get("/api/me/submissions/{sid}")
def my_submission(sid: int, u: dict = Depends(patient_user)):
    x = db.one("SELECT * FROM submissions WHERE id = ? AND patient_id = ?", (sid, u["patient_id"]))
    if not x:
        raise HTTPException(404, "not found")
    out = submission_row(x, full=True)
    out["ecg"] = patient_view(ecg_row(get_ecg(x["ecg_id"]))) if x["ecg_id"] else None
    return out


@app.get("/api/me/uploads")
def my_uploads(u: dict = Depends(patient_user)):
    """Files attached but not sent yet."""
    return [attachment_row(a) for a in db.q("SELECT * FROM attachments WHERE patient_id = ? AND submission_id IS NULL "
                                            "ORDER BY created_at DESC", (u["patient_id"],))]


# --------------------------------------------------------------------------- clinician: patient submissions
@app.get("/api/submissions")
def submissions(u: dict = Depends(clinician)):
    rows = db.q("SELECT s.*, p.name AS patient_name, p.mrn FROM submissions s JOIN patients p ON p.id = s.patient_id "
                "ORDER BY CASE s.status WHEN 'sent' THEN 0 WHEN 'seen' THEN 1 ELSE 2 END, s.created_at DESC")
    out = []
    for x in rows:
        r = submission_row(x)
        r["patient_name"], r["mrn"] = x["patient_name"], x["mrn"]
        if x["ecg_id"]:
            r["ecg"] = ecg_row(get_ecg(x["ecg_id"]))
        out.append(r)
    return out


@app.get("/api/submissions/{sid}")
def submission_detail(sid: int, u: dict = Depends(clinician)):
    x = db.one("SELECT * FROM submissions WHERE id = ?", (sid,))
    if not x:
        raise HTTPException(404, "not found")
    if x["status"] == "sent":
        db.run("UPDATE submissions SET status = 'seen' WHERE id = ?", (sid,))
        x["status"] = "seen"
    out = submission_row(x, full=True)
    out["patient"] = patient_row(get_patient(x["patient_id"]))
    out["ecg"] = ecg_row(get_ecg(x["ecg_id"])) if x["ecg_id"] else None
    return out


@app.post("/api/submissions/{sid}/reply")
def reply_submission(sid: int, body: Msg, u: dict = Depends(clinician)):
    x = db.one("SELECT * FROM submissions WHERE id = ?", (sid,))
    if not x:
        raise HTTPException(404, "not found")
    if not body.body.strip():
        raise HTTPException(400, "empty reply")
    db.run("UPDATE submissions SET status = 'replied', reply = ?, replied_by = ?, replied_at = ? WHERE id = ?",
           (body.body.strip(), u["id"], now(), sid))
    db.run("INSERT INTO messages (patient_id, sender_id, body, read_by_patient, read_by_clinician, created_at) "
           "VALUES (?,?,?,?,?,?)", (x["patient_id"], u["id"], f"Re: what you sent on {x['created_at'][:10]}: "
                                    + body.body.strip(), 0, 1, now()))
    db.audit(u["id"], "reply_submission", f"submission {sid}")
    return submission_detail(sid, u)


# --------------------------------------------------------------------------- static frontend
@app.exception_handler(HTTPException)
async def http_error(_, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


def init(reseed: bool = False) -> None:
    global db, engine
    if reseed and store.DB_PATH.exists():
        store.DB_PATH.unlink()
        for f in store.SIGNAL_DIR.glob("*.npy"):
            f.unlink()
        for f in store.ATTACH_DIR.glob("*"):
            f.unlink()
    engine = Engine()
    log.info("model engine: %s", engine.info() if engine.available else engine.reason)
    db = store.DB()
    if db.is_empty():
        MODE["seed"] = seed(db, engine)
        log.info("seeded demo data (%s ECGs)", MODE["seed"])
    else:
        MODE["seed"] = "synthetic" if db.one("SELECT MAX(demo) AS d FROM ecgs")["d"] else "real"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reseed", action="store_true", help="delete the demo database and seed it again")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init(args.reseed)
    import uvicorn

    print(f"CardioSentinel running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
