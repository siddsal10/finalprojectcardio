"""SQLite storage for the CardioSentinel portal (users, patients, ECGs, reviews, messages, ...)."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATA_DIR = Path(os.environ.get("CARDIOSENTINEL_DATA", Path(__file__).parent / "data"))
DB_PATH = DATA_DIR / "cardiosentinel.db"
SIGNAL_DIR = DATA_DIR / "ecgs"
ATTACH_DIR = DATA_DIR / "attachments"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL, salt TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('clinician', 'patient')), name TEXT NOT NULL, title TEXT,
    patient_id INTEGER REFERENCES patients(id), created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id), created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY, mrn TEXT UNIQUE NOT NULL, name TEXT NOT NULL, dob TEXT, sex TEXT,
    height_cm REAL, weight_kg REAL, phone TEXT, email TEXT, address TEXT, conditions TEXT,
    medications TEXT, allergies TEXT, emergency_contact TEXT, clinician_id INTEGER REFERENCES users(id),
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ecgs (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id), recorded_at TEXT NOT NULL,
    source TEXT NOT NULL, source_ref TEXT, device TEXT, indication TEXT, prediction TEXT NOT NULL,
    reference_labels TEXT, status TEXT NOT NULL CHECK (status IN ('auto_reported', 'needs_review', 'reviewed')),
    released INTEGER NOT NULL DEFAULT 0, demo INTEGER NOT NULL DEFAULT 0, heart_rate REAL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY, ecg_id INTEGER NOT NULL REFERENCES ecgs(id), clinician_id INTEGER NOT NULL REFERENCES users(id),
    final_labels TEXT NOT NULL, agrees_with_ai INTEGER NOT NULL, notes TEXT, patient_summary TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS appointments (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id), clinician_id INTEGER REFERENCES users(id),
    starts_at TEXT NOT NULL, kind TEXT NOT NULL, location TEXT, status TEXT NOT NULL, notes TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id), sender_id INTEGER NOT NULL REFERENCES users(id),
    body TEXT NOT NULL, read_by_patient INTEGER NOT NULL DEFAULT 0, read_by_clinician INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id), feeling INTEGER,
    symptoms TEXT, severity INTEGER, started TEXT, body TEXT, shared INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id), ecg_id INTEGER REFERENCES ecgs(id),
    message TEXT, note_ids TEXT, status TEXT NOT NULL CHECK (status IN ('sent', 'seen', 'replied')),
    reply TEXT, replied_by INTEGER REFERENCES users(id), replied_at TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS attachments (
    id INTEGER PRIMARY KEY, patient_id INTEGER NOT NULL REFERENCES patients(id), submission_id INTEGER REFERENCES submissions(id),
    filename TEXT NOT NULL, mime TEXT NOT NULL, size INTEGER NOT NULL, stored_as TEXT NOT NULL, description TEXT,
    created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY, user_id INTEGER, action TEXT NOT NULL, detail TEXT, created_at TEXT NOT NULL);
"""

_lock = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SIGNAL_DIR.mkdir(parents=True, exist_ok=True)
    ATTACH_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    # ECGs a patient checked themselves stay private (shared = 0) until they send them to their doctor
    if "shared" not in {r[1] for r in con.execute("PRAGMA table_info(ecgs)")}:
        con.execute("ALTER TABLE ecgs ADD COLUMN shared INTEGER NOT NULL DEFAULT 1")
        con.commit()
    return con


class DB:
    def __init__(self):
        self.con = connect()

    def q(self, sql: str, args=()) -> list[dict]:
        with _lock:
            return [dict(r) for r in self.con.execute(sql, args).fetchall()]

    def one(self, sql: str, args=()) -> dict | None:
        rows = self.q(sql, args)
        return rows[0] if rows else None

    def run(self, sql: str, args=()) -> int:
        with _lock:
            cur = self.con.execute(sql, args)
            self.con.commit()
            return cur.lastrowid

    def audit(self, user_id: int | None, action: str, detail: str = "") -> None:
        self.run("INSERT INTO audit (user_id, action, detail, created_at) VALUES (?,?,?,?)",
                 (user_id, action, detail, now()))

    def is_empty(self) -> bool:
        return self.one("SELECT COUNT(*) AS n FROM users")["n"] == 0


# --------------------------------------------------------------------------- auth
def hash_pw(pw: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 120_000).hex(), salt


def check_pw(pw: str, pw_hash: str, salt: str) -> bool:
    return secrets.compare_digest(hash_pw(pw, salt)[0], pw_hash)


# --------------------------------------------------------------------------- signals
def save_signal(ecg_id: int, sig: np.ndarray) -> None:
    np.save(SIGNAL_DIR / f"{ecg_id}.npy", sig.astype(np.float16))


def load_signal(ecg_id: int) -> np.ndarray:
    return np.load(SIGNAL_DIR / f"{ecg_id}.npy").astype(np.float32)


def heart_rate(sig: np.ndarray, fs: int = 100) -> float | None:
    """Rough heart rate from lead II R peaks (for display only)."""
    from scipy.signal import find_peaks

    x = sig[1] - np.median(sig[1])
    if np.abs(x.min()) > np.abs(x.max()):
        x = -x
    peaks, _ = find_peaks(x, height=0.5 * x.max(), distance=int(0.3 * fs))
    if len(peaks) < 2:
        return None
    return float(round(60 * fs / np.median(np.diff(peaks))))


def dumps(x) -> str:
    return json.dumps(x)


def loads(x):
    return json.loads(x) if x else None
