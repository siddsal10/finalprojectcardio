"""Evaluate the image pipeline: picture of a printed ECG -> digitizer -> image clients -> gate.

    python -m cardiomoe.images.evaluate --set real-dev            # real scans/photos, PTB-XL fold 9 (development)
    python -m cardiomoe.images.evaluate --set synth               # digitized synthetic printouts, fold 10
    python -m cardiomoe.images.evaluate --set real-test           # real scans/photos, fold 10: once (--rerun to repeat)

Real images are ECG-Image-Database's PTB-XL part (scripts/fetch_ecg_images.py): each record printed,
scanned, photographed with phones, photographed off a monitor, stained, ... Labels are the PTB-XL
labels of the same ecg_id; the WFDB signal shipped with each record gives the digitizer's accuracy.

Compared on the same images (thresholds, weights and triage bands all fixed on fold 9 beforehand):
- image gate / image clients A-D (fine-tuned on digitized printouts, finetune.py)
- signal gate on the digitized signal (the original models, no fine-tuning): what fine-tuning adds
- signal gate on the recorded PTB-XL signal of the same record: the reference without any paper

Each record has ~10 images, so confidence intervals resample records, not images. Images the
digitizer rejects are counted (coverage) and left out of the classification metrics.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import pickle

import numpy as np
import pandas as pd
import torch

from cardiomoe.data.datasets import YCOLS, load_meta
from cardiomoe.data.signal import normalize
from cardiomoe.eval.metrics import macro_auroc
from cardiomoe.fourclient.metrics import full_report
from cardiomoe.fourclient.train import CLIENTS, fc_config, load_client, run_dir
from cardiomoe.fourclient.triage import Triage, report
from cardiomoe.images.build import img_config
from cardiomoe.images.layouts import masked_normalize
from cardiomoe.images.real import digitize_real
from cardiomoe.utils import repo_path

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- models
class Stack:
    """Four calibrated clients + gate + thresholds + triage from one output directory."""

    def __init__(self, cfg: dict, seed: int):
        d = run_dir(cfg, seed, False)
        self.agg = json.loads((d / "gate" / "aggregate.json").read_text())
        with open(d / "gate" / "combiners.pkl", "rb") as f:
            self.comb = pickle.load(f)
        self.clients = [load_client(cfg, c, seed, False, "cpu") for c in CLIENTS]
        self.sel = self.agg["selected"]
        self.triage = Triage.from_dict(self.agg["triage"]) if "triage" in self.agg else None

    @torch.no_grad()
    def __call__(self, X: np.ndarray, bs: int = 128) -> dict:
        P = []
        for m in self.clients:
            out = [torch.sigmoid(m({"signal": torch.from_numpy(X[i:i + bs])}).logits / m.temperature)
                   for i in range(0, len(X), bs)]
            P.append(torch.cat(out))
        P = torch.stack(P, 1).float()
        probs = {c: P[:, i].numpy() for i, c in enumerate(CLIENTS)}
        probs["gate"] = self.comb[self.sel].predict(P).numpy()
        return probs

    def th(self, name: str) -> np.ndarray:
        return np.array(self.agg["thresholds"][self.sel if name == "gate" else name])


def cluster_ci(y, p, groups, n: int = 1000, seed: int = 42):
    """95% CI of macro AUROC, resampling records (all images of a drawn record are kept)."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in ug}
    vals = []
    for _ in range(n):
        i = np.concatenate([idx_of[g] for g in rng.choice(ug, len(ug))])
        v = macro_auroc(y[i], p[i])
        if np.isfinite(v):
            vals.append(v)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def run(cfg: dict, seed: int, which: str, workers: int, limit: int | None, n_boot: int) -> dict:
    sig_cfg = fc_config()
    meta = load_meta(cfg)
    if which == "synth":
        z = np.load(repo_path(cfg["images"]["synth_dir"]) / f"synth_fold{cfg['fourclient']['test_fold']}.npz")
        d = {k: z[k] for k in z.files}
        d["variant"] = d["capture"]
    else:
        fold = cfg["images"]["real"]["dev_fold" if which == "real-dev" else "test_fold"]
        rc = cfg["images"]["real"]
        d = digitize_real(cfg, fold, workers, limit, rc["val_images"] if which == "real-dev" else rc["test_images"])
    ok = d["ok"].astype(bool)
    ids = d["ecg_id"]
    y_all = meta.loc[ids, YCOLS].to_numpy(np.float32)
    names = cfg["images"]["real"]["variants"]
    var_label = np.array([names.get(str(v), str(v)) for v in d["variant"]])

    img, sigs = Stack(cfg, seed), Stack(sig_cfg, seed)
    X = np.stack([masked_normalize(s.astype(np.float32), m) for s, m in zip(d["signal"][ok], d["mask"][ok])])
    y = y_all[ok]
    groups = ids[ok]
    out_img, out_sig_dig = img(X), sigs(X)
    # reference: the recorded PTB-XL signal of each image's record
    raw = np.load(repo_path(sig_cfg["paths"]["processed"]) / "ptbxl_signals.npy", mmap_mode="r")
    Xref = np.stack([normalize(np.asarray(raw[r], dtype=np.float32)) for r in meta.loc[groups, "row"]])
    out_ref = sigs(Xref)

    rows, reports = [], {}
    entries = [("Image gate (fine-tuned)", out_img["gate"], img.th("gate"))]
    entries += [(f"Image client {c}", out_img[c], img.th(c)) for c in CLIENTS]
    entries += [("Signal gate on digitized signal (no fine-tuning)", out_sig_dig["gate"], sigs.th("gate")),
                ("Signal gate on recorded signal (reference)", out_ref["gate"], sigs.th("gate"))]
    for name, p, th in entries:
        r = full_report(y, p, th)
        lo, hi = cluster_ci(y, p, groups, n_boot, seed)
        s = r["summary"]
        rows.append({"model": name, "macro_auroc": s["macro_auroc"], "ci_low": lo, "ci_high": hi,
                     "macro_auprc": s["macro_auprc"], "macro_f1": s["macro_f1"], "macro_precision": s["macro_precision"],
                     "macro_recall": s["macro_recall"], "labelwise_accuracy": s["labelwise_accuracy"],
                     "exact_match": s["exact_match_accuracy"]})
        reports[name] = r
    tri = report(y, out_img["gate"], img.triage) if img.triage else None

    # per capture type: digitizer success / accuracy and gate AUROC
    per_var = []
    for v in sorted(set(var_label)):
        sel_all = var_label == v
        sel = sel_all[ok]
        row = {"capture": v, "images": int(sel_all.sum()), "digitized": float(ok[sel_all].mean()),
               "median_r": float(np.nanmedian(d["r"][sel_all & ok])) if (sel_all & ok).any() else float("nan")}
        if sel.sum() > 10 and len(np.unique(y[sel], axis=0)) > 1:
            row["auroc_image_gate"] = macro_auroc(y[sel], out_img["gate"][sel])
            row["auroc_signal_gate_digitized"] = macro_auroc(y[sel], out_sig_dig["gate"][sel])
            row["auroc_reference"] = macro_auroc(y[sel], out_ref["gate"][sel])
        per_var.append(row)
    errors = pd.Series(d["error"][~ok]).str.split(";").str[0].str.replace(r"\d+", "#", regex=True).value_counts()
    return {"set": which, "seed": seed, "n_images": int(len(ok)), "n_records": int(len(np.unique(ids))),
            "coverage": float(ok.mean()), "median_r": float(np.nanmedian(d["r"][ok])) if ok.any() else None,
            "models": rows, "per_capture": per_var, "per_class_image_gate": reports["Image gate (fine-tuned)"]["per_class"],
            "triage_image_gate": tri, "failures": errors.head(10).to_dict(), "selected_combiner": img.sel}


def markdown(res: dict) -> str:
    L = [f"### Image pipeline, `{res['set']}` (seed {res['seed']})", "",
         f"{res['n_images']} images of {res['n_records']} records; digitized {res['coverage']:.1%}; "
         f"median per-lead correlation with the recorded signal {res['median_r']:.3f}. Gate: {res['selected_combiner']}.", "",
         "| Model | Macro AUROC (95% CI, records) | Macro AUPRC | Macro F1 | Precision | Recall | Label-wise acc. | Exact match |",
         "|---|---|---|---|---|---|---|---|"]
    for r in res["models"]:
        L.append(f"| {r['model']} | {r['macro_auroc']:.3f} ({r['ci_low']:.3f}-{r['ci_high']:.3f}) | {r['macro_auprc']:.3f} | "
                 f"{r['macro_f1']:.3f} | {r['macro_precision']:.3f} | {r['macro_recall']:.3f} | "
                 f"{r['labelwise_accuracy']:.3f} | {r['exact_match']:.3f} |")
    L += ["", "| Capture | Images | Digitized | Median r | AUROC image gate | AUROC signal gate on digitized | AUROC reference |",
          "|---|---|---|---|---|---|---|"]
    f = lambda v: "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.3f}"
    for r in res["per_capture"]:
        L.append(f"| {r['capture']} | {r['images']} | {r['digitized']:.1%} | {f(r['median_r'])} | {f(r.get('auroc_image_gate'))} | "
                 f"{f(r.get('auroc_signal_gate_digitized'))} | {f(r.get('auroc_reference'))} |")
    if res["triage_image_gate"]:
        s = res["triage_image_gate"]["summary"]
        L += ["", f"Triage on the image gate: {s.get('finding_coverage', float('nan')):.1%} of findings auto-decided, "
                  f"accuracy {s.get('finding_accuracy', float('nan')):.3f}; {s.get('ecg_fully_automated', float('nan')):.1%} of images fully automatic."]
    if res["failures"]:
        L += ["", "Digitizer failures: " + "; ".join(f"{k} ({v})" for k, v in res["failures"].items())]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["real-dev", "synth", "real-test"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--debug", action="store_true", help="first 5 records, no marker, 100 bootstrap draws")
    ap.add_argument("--rerun", action="store_true", help="allow re-evaluating fold 10 after a previous run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = img_config()
    base = repo_path(cfg["fourclient"]["out_dir"])
    marker = base / f"test_evaluated_{args.set}.json"
    on_test = args.set in ("synth", "real-test")
    if on_test and marker.exists() and not args.debug and not args.rerun:
        raise SystemExit(f"fold 10 ({args.set}) was already evaluated ({marker.read_text().strip()}); pass --rerun to repeat")
    tables = base / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        res = run(cfg, seed, args.set, args.workers, 5 if args.debug else None, 100 if args.debug else 1000)
        tag = f"{args.set}{'_debug' if args.debug else ''}_seed{seed}"
        (tables / f"eval_{tag}.json").write_text(json.dumps(res, indent=2, default=float))
        md = markdown(res)
        (tables / f"eval_{tag}.md").write_text(md + "\n")
        print(md)
    if on_test and not args.debug:
        marker.write_text(json.dumps({"evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                                      "seeds": args.seeds}) + "\n")


if __name__ == "__main__":
    main()
