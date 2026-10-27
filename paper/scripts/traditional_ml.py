"""Traditional machine-learning baselines on hand-crafted ECG features (Table IV, Section IV-C).

    python paper/scripts/traditional_ml.py [--seeds 42 43 44] [--debug]

Features are computed from the filtered, un-normalised signal in mV (data/processed/ptbxl_signals.npy).
Training on folds 1-8 (pooled: these baselines are centralised), probability calibration and per-class
thresholds on fold 9, a single evaluation on fold 10 with the same metric code as the deep models
(cardiomoe.fourclient.metrics.full_report). Writes paper/scripts/traditional_ml.json.
--debug: debug data cache, one seed, small models, output under paper/scripts/debug/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from cardiomoe.data import ptbxl
from cardiomoe.data.datasets import YCOLS
from cardiomoe.eval.metrics import tune_thresholds
from cardiomoe.fourclient.metrics import full_report
from cardiomoe.utils import load_config

ROOT = Path(__file__).resolve().parents[2]
FS = 100
BANDS = [(0.5, 5), (5, 15), (15, 40)]


def features(x: np.ndarray) -> np.ndarray:
    """(12, 1000) mV -> 1-D feature vector: per-lead statistics, spectra, beat amplitudes, rhythm."""
    f = []
    f += list(x.std(1)) + list(x.min(1)) + list(x.max(1)) + list(np.ptp(x, 1))
    f += list(skew(x, axis=1)) + list(kurtosis(x, axis=1)) + list(np.abs(np.diff(x, axis=1)).mean(1))
    fr, pxx = welch(x, fs=FS, nperseg=256, axis=1)
    tot = pxx.sum(1) + 1e-12
    for lo, hi in BANDS:
        f += list(pxx[:, (fr >= lo) & (fr < hi)].sum(1) / tot)
    # R peaks on lead II (index 1): rhythm and per-lead QRS amplitudes around them
    ii = x[1]
    peaks, _ = find_peaks(np.abs(ii), distance=int(0.3 * FS), height=0.5 * np.abs(ii).max())
    peaks = peaks[(peaks > 10) & (peaks < x.shape[1] - 30)]
    if len(peaks) >= 2:
        rr = np.diff(peaks) / FS
        f += [60 / rr.mean(), rr.std(), np.sqrt(np.mean(np.diff(rr) ** 2)) if len(rr) > 1 else 0.0, len(peaks)]
        r_amp = np.median(np.stack([x[:, p - 5:p + 6].max(1) for p in peaks]), 0)
        s_amp = np.median(np.stack([x[:, p:p + 11].min(1) for p in peaks]), 0)
        st = np.median(np.stack([x[:, p + 8:p + 12].mean(1) for p in peaks]), 0)  # ~80-120 ms after R
        t_amp = np.median(np.stack([x[:, p + 15:p + 30].max(1) for p in peaks]), 0)
    else:
        f += [0.0, 0.0, 0.0, float(len(peaks))]
        r_amp = s_amp = st = t_amp = np.zeros(12)
    f += list(r_amp) + list(s_amp) + list(st) + list(t_amp)
    return np.nan_to_num(np.asarray(f, dtype=np.float32))


def models(seed: int, debug: bool) -> dict:
    n = 30 if debug else 300
    return {
        "Logistic Regression": lambda: LogisticRegression(C=1.0, max_iter=3000),
        "SVM (RBF)": lambda: SVC(C=1.0, kernel="rbf", gamma="scale"),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=n, min_samples_leaf=2, n_jobs=-1,
                                                        random_state=seed),
        "KNN": lambda: KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1),
        "XGBoost": lambda: XGBClassifier(n_estimators=n, max_depth=6, learning_rate=0.1, subsample=0.8,
                                         colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0),
    }


def score(m, X) -> np.ndarray:
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)


def local(args, meta, F, Y, split, sc, X, y):
    """Each model trained on one client's two folds only, as the deep clients are (seed 42)."""
    from cardiomoe.fourclient.train import fc_config

    part = fc_config()["fourclient"]["partition"]
    out = {}
    for name in models(0, args.debug):
        out[name] = {}
        for cl, folds in part.items():
            m_tr = meta["strat_fold"].isin(folds).to_numpy()
            Xtr, ytr = sc.transform(F[m_tr]), Y[m_tr]
            pv, pt = np.zeros_like(y["val"]), np.zeros_like(y["test"])
            for c in range(Y.shape[1]):
                m = models(42, args.debug)[name]().fit(Xtr, ytr[:, c])
                sv, st = score(m, X["val"]), score(m, X["test"])
                cal = LogisticRegression().fit(sv[:, None], y["val"][:, c])
                pv[:, c], pt[:, c] = cal.predict_proba(sv[:, None])[:, 1], cal.predict_proba(st[:, None])[:, 1]
            th = tune_thresholds(y["val"], pv)
            out[name][cl] = full_report(y["test"], pt, th)["summary"]
            print(name, cl, round(out[name][cl]["macro_auroc"], 3), round(out[name][cl]["macro_f1"], 3), flush=True)
    path = ROOT / "paper" / ("scripts/debug/traditional_ml_local.json" if args.debug else "scripts/traditional_ml_local.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--local", action="store_true",
                    help="train on each client's own folds instead of pooled folds 1-8 (seed 42); report per client")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    seeds = args.seeds[:1] if args.debug else args.seeds
    cfg = load_config()
    meta = ptbxl.load_meta(cfg, args.debug)
    sig = ptbxl.load_signal_array(cfg, args.debug)
    t0 = time.time()
    F = np.stack([features(np.asarray(sig[r], dtype=np.float32)) for r in meta["row"].to_numpy()])
    print(f"{F.shape[1]} features for {len(F)} records in {time.time() - t0:.0f}s", flush=True)
    Y = meta[YCOLS].to_numpy(np.float32)
    split = meta["split"].to_numpy()
    sc = StandardScaler().fit(F[split == "train"])
    X = {s: sc.transform(F[split == s]) for s in ("train", "val", "test")}
    y = {s: Y[split == s] for s in ("train", "val", "test")}
    if args.local:
        local(args, meta, F, Y, split, sc, X, y)
        return

    out = {"n_features": int(F.shape[1]), "results": {}}
    deterministic = {"Logistic Regression", "SVM (RBF)", "KNN"}
    for name in models(0, args.debug):
        runs = []
        for seed in (seeds[:1] if name in deterministic else seeds):
            t0 = time.time()
            pv, pt = np.zeros_like(y["val"]), np.zeros_like(y["test"])
            for c in range(Y.shape[1]):
                m = models(seed, args.debug)[name]().fit(X["train"], y["train"][:, c])
                sv, st = score(m, X["val"]), score(m, X["test"])
                # Platt scaling on fold 9 so every model yields comparable probabilities for the threshold grid
                cal = LogisticRegression().fit(sv[:, None], y["val"][:, c])
                pv[:, c], pt[:, c] = cal.predict_proba(sv[:, None])[:, 1], cal.predict_proba(st[:, None])[:, 1]
            th = tune_thresholds(y["val"], pv)
            rep = full_report(y["test"], pt, th)
            runs.append({"seed": seed, "sec": round(time.time() - t0, 1), **rep["summary"]})
            print(name, seed, {k: round(v, 3) for k, v in rep["summary"].items()}, flush=True)
        if name in deterministic:  # identical for every seed
            runs = [dict(runs[0], seed=s) for s in seeds]
        out["results"][name] = runs
    path = ROOT / "paper" / ("scripts/debug/traditional_ml.json" if args.debug else "scripts/traditional_ml.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
