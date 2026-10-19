"""CardioSentinel inference: the four trained clients + the selected gate + the triage rule.

The engine loads outputs/fourclient/seed{S}/{A,B,C,D}/model.pt, gate/combiners.pkl and
gate/aggregate.json (selected combiner, fold-9 thresholds, triage rule). If they are missing
the engine reports `available = False` and the website runs in demo mode.
"""
from __future__ import annotations

import io
import json
import time
import pickle
import tempfile
from pathlib import Path

import numpy as np
import torch

from cardiomoe.data.labels import SUPERCLASSES
from cardiomoe.data.signal import LEADS, N_SAMPLES, bandpass, clean, fix_length, normalize, resample
from cardiomoe.fourclient.models import ARCH_NAMES
from cardiomoe.fourclient.train import CLIENTS, fc_config, load_client, run_dir
from cardiomoe.fourclient.triage import Triage

CLASS_NAMES = {
    "NORM": "Normal ECG",
    "MI": "Myocardial infarction",
    "STTC": "ST/T change",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}


class Engine:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.cfg = fc_config()
        self.available = False
        self.reason = ""
        try:
            self._load()
            self.available = True
        except FileNotFoundError as e:
            self.reason = f"Trained models not found ({Path(e.filename).name if e.filename else e})"

    def _load(self) -> None:
        d = run_dir(self.cfg, self.seed, False)
        self.agg = json.loads((d / "gate" / "aggregate.json").read_text())
        with open(d / "gate" / "combiners.pkl", "rb") as f:
            self.combiners = pickle.load(f)
        self.clients = {c: load_client(self.cfg, c, self.seed, False, "cpu") for c in CLIENTS}
        self.selected = self.agg["selected"]
        self.th = np.array(self.agg["thresholds"][self.selected])
        self.triage = Triage.from_dict(self.agg["triage"]) if "triage" in self.agg else None

    def info(self) -> dict:
        if not self.available:
            return {"available": False, "reason": self.reason}
        fc = self.cfg["fourclient"]
        return {"available": True, "seed": self.seed, "selected": self.selected,
                "thresholds": dict(zip(SUPERCLASSES, self.th.tolist())),
                "triage_modes": {c: b["mode"] for c, b in self.triage.bands.items()} if self.triage else None,
                "clients": {c: {"arch": ARCH_NAMES[fc["clients"][c]["arch"]], "folds": fc["partition"][c]}
                            for c in CLIENTS}}

    @torch.no_grad()
    def predict(self, filtered: np.ndarray, meta: dict | None = None) -> dict:
        """filtered: (12, 1000) band-passed signal in mV (the stored PTB-XL format).

        `meta` (from parse_upload / sample_meta) describes how the file was read; the result then
        carries a `pipeline` record of every step, shown to users as an animated walkthrough.
        """
        t0 = time.perf_counter()
        x = torch.from_numpy(normalize(filtered.astype(np.float32)))[None]
        t_norm = time.perf_counter() - t0
        P, readers = [], []
        fc = self.cfg["fourclient"]
        for c in CLIENTS:
            t = time.perf_counter()
            m = self.clients[c]
            P.append(torch.sigmoid(m({"signal": x}).logits / m.temperature))
            readers.append({"key": c, "arch": ARCH_NAMES[fc["clients"][c]["arch"]], "role": READER_ROLES[c],
                            "ms": round(1000 * (time.perf_counter() - t), 1),
                            "probs": dict(zip(SUPERCLASSES, map(float, P[-1][0])))})
        P = torch.stack(P, dim=1).float()  # [1, 4, 5]
        t = time.perf_counter()
        comb = self.combiners[self.selected]
        p = comb.predict(P).numpy()[0]
        w = comb.weights(P).numpy()[0]
        out = package(p, P.numpy()[0], w, self.th, self.triage)
        t_gate = time.perf_counter() - t
        meta = meta or {}
        pre = (meta.get("t_read_ms") or 0) + (meta.get("t_clean_ms") or 0)
        out["pipeline"] = {
            "input": {k: v for k, v in meta.items() if k not in ("raw_ii", "t_read_ms", "t_clean_ms", "steps")},
            "read_ms": meta.get("t_read_ms"),
            "clean": {"raw_ii": meta.get("raw_ii"), "clean_ii": _thin(filtered[1]), "ms": meta.get("t_clean_ms"),
                      "steps": meta.get("steps", CLEAN_STEPS)},
            "scale_ms": round(1000 * t_norm, 2),
            "readers": readers,
            "combine": {"weights": dict(zip(CLIENTS, map(float, w))), "ms": round(1000 * t_gate, 2)},
            "total_ms": round(1000 * (time.perf_counter() - t0) + pre, 1),
        }
        return out


READER_ROLES = {"A": "Looks for local wave shapes", "B": "Looks for deep, layered patterns",
                "C": "Follows the rhythm beat by beat", "D": "Combines shapes and rhythm"}
CLEAN_STEPS = ["Put the 12 leads in standard order", "Resampled to 100 samples per second",
               "Trimmed or padded to exactly 10 seconds", "Removed baseline drift and noise (0.5-40 Hz filter)",
               "Scaled each lead to a common range"]


def _thin(x, n: int = 500) -> list:
    """Downsample a 1-D trace for display."""
    x = np.asarray(x, dtype=np.float64)
    idx = np.linspace(0, len(x) - 1, min(n, len(x))).astype(int)
    return np.round(x[idx], 3).tolist()


DEMO_BANDS = {  # used only in demo mode (no fitted triage): close to the fitted seed-42 bands
    "NORM": {"mode": "automated", "lo": 0.30, "hi": 0.80}, "MI": {"mode": "automated", "lo": 0.25, "hi": 0.88},
    "STTC": {"mode": "rule_out", "lo": 0.33, "hi": None}, "CD": {"mode": "automated", "lo": 0.20, "hi": 0.85},
    "HYP": {"mode": "rule_out", "lo": 0.22, "hi": None}}


def package(p, client_p, w, th, triage) -> dict:
    """Common JSON shape for a prediction (real or demo).

    `suggested` uses the fold-9 F1 thresholds (the gate's plain decision). `triage` says, per class,
    whether the finding is auto-reported as positive/negative or indeterminate (refer). `decisions`
    is the triage decision where one was made and the suggestion otherwise.
    """
    triage = triage or Triage(DEMO_BANDS)
    sug = (p >= th).astype(int)
    tri = triage.decide(p[None])[0]
    status = {c: ("positive" if t == 1 else "negative" if t == 0 else "indeterminate") for c, t in zip(SUPERCLASSES, tri)}
    dec = np.where(tri >= 0, tri, sug)
    ind = [c for c in SUPERCLASSES if status[c] == "indeterminate"]
    margin = {c: float(abs(p[i] - th[i])) for i, c in enumerate(SUPERCLASSES)}
    return {
        "probs": dict(zip(SUPERCLASSES, map(float, p))),
        "thresholds": dict(zip(SUPERCLASSES, map(float, th))),
        "suggested": dict(zip(SUPERCLASSES, map(int, sug))),
        "decisions": dict(zip(SUPERCLASSES, map(int, dec))),
        "triage": status,
        "bands": {c: {k: triage.bands[c][k] for k in ("mode", "lo", "hi")} for c in SUPERCLASSES},
        "indeterminate": ind,
        "auto_report": not ind,
        "client_probs": {c: dict(zip(SUPERCLASSES, map(float, client_p[i]))) for i, c in enumerate(CLIENTS)},
        "weights": dict(zip(CLIENTS, map(float, w))),
        "least_certain": min(ind or SUPERCLASSES, key=margin.get),
    }


# --------------------------------------------------------------------------- uploads
def preprocess_any(sig: np.ndarray, fs: int) -> tuple[np.ndarray, dict]:
    """(12, T) or (T, 12) raw signal in mV at fs Hz -> (12, 1000) filtered at 100 Hz, plus what was done."""
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim != 2 or 12 not in sig.shape:
        raise ValueError(f"expected a 12-lead array, got shape {sig.shape}")
    if sig.shape[0] != 12:
        sig = sig.T
    t = time.perf_counter()
    sig, n_bad = clean(sig)
    raw_ii = sig[1, : int(10 * fs)]
    out = bandpass(fix_length(resample(sig, int(fs)), N_SAMPLES)).astype(np.float32)
    meta = {"fs_in": int(fs), "n_samples_in": int(sig.shape[1]), "duration_s": round(sig.shape[1] / fs, 2),
            "n_leads": 12, "leads": LEADS, "bad_values_fixed": int(n_bad), "raw_ii": _thin(raw_ii),
            "t_clean_ms": round(1000 * (time.perf_counter() - t), 1)}
    return out, meta


def parse_upload(files: dict, fs: int = 100) -> tuple[np.ndarray, dict]:
    """Accepts .npy, .csv (12 columns, optional header with lead names) or a WFDB .hea + .dat pair.

    Returns the filtered (12, 1000) signal and a description of how the file was read and cleaned.
    """
    t = time.perf_counter()
    names = {n.lower(): n for n in files}
    hea = next((n for n in names if n.endswith(".hea")), None)
    if hea:
        import wfdb

        base = Path(hea).stem
        dat = next((n for n in names if n.endswith(".dat")), None)
        if dat is None:
            raise ValueError("a WFDB record needs both the .hea and the .dat file")
        with tempfile.TemporaryDirectory() as td:
            for n in (hea, dat):
                (Path(td) / (base + Path(n).suffix)).write_bytes(files[names[n]])
            rec = wfdb.rdrecord(str(Path(td) / base))
        from cardiomoe.data.signal import reorder_leads

        sig, fmt, fs, name = reorder_leads(rec.p_signal.T, rec.sig_name), "WFDB record", rec.fs, names[hea]
    else:
        key = next(iter(names))
        raw, name = files[names[key]], names[key]
        if key.endswith(".npy"):
            sig, fmt = np.load(io.BytesIO(raw), allow_pickle=False), "NumPy array"
        elif key.endswith(".csv") or key.endswith(".txt"):
            text = raw.decode("utf-8", errors="replace").strip().splitlines()
            first = [c.strip().strip('"') for c in text[0].replace(";", ",").split(",")]
            header = not _is_number(first[0])
            rows = [[float(v) for v in line.replace(";", ",").split(",")] for line in text[int(header):] if line.strip()]
            arr = np.array(rows)
            if header and len(first) == arr.shape[1] and arr.shape[1] >= 12:
                idx = {h.upper(): i for i, h in enumerate(first)}
                if all(l.upper() in idx for l in LEADS):
                    arr = arr[:, [idx[l.upper()] for l in LEADS]]
            if arr.shape[1] > 12 and arr.shape[0] != 12:
                arr = arr[:, -12:]  # e.g. a leading time column
            sig, fmt = arr, "CSV table"
        else:
            raise ValueError("unsupported file type: upload .npy, .csv or a WFDB .hea + .dat pair")
    t_read = time.perf_counter() - t
    out, meta = preprocess_any(sig, fs)
    meta.update({"filename": name, "format": fmt, "size_bytes": sum(len(b) for b in files.values()),
                 "t_read_ms": round(1000 * t_read, 1)})
    return out, meta


def sample_meta(sig: np.ndarray, label: str) -> dict:
    """Pipeline description for an example recording (already stored cleaned at 100 Hz)."""
    return {"filename": label, "format": "Example recording", "fs_in": 100, "n_samples_in": int(sig.shape[1]),
            "duration_s": round(sig.shape[1] / 100, 2), "n_leads": 12, "leads": LEADS, "bad_values_fixed": 0,
            "raw_ii": None, "t_read_ms": 0.0, "t_clean_ms": 0.0,
            "steps": ["Already in standard lead order", "Already 100 samples per second, 10 seconds",
                      "Already filtered (0.5-40 Hz)", "Scaled each lead to a common range"]}


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
