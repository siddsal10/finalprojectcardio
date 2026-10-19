"""Per-finding confidence triage (rule-in / rule-out) on top of the selected gate.

    python -m cardiomoe.fourclient.triage --seeds 42 43 44 [--debug] [--rerun]

Every superclass decision of the gate is either reported automatically or marked
indeterminate and referred to a cardiologist:

    p_c >= hi_c          -> positive, auto-reported
    p_c <= lo_c          -> negative, auto-reported
    lo_c < p_c < hi_c    -> indeterminate, referred

The bands are fitted per class on the gate's patient-grouped CV (out-of-fold) predictions on
fold 9. Each class gets one of three modes:

  automated  the widest band whose auto-decided findings reach precision, recall and accuracy
             >= target + margin, with at least `min_positive_auto` of the class's true positives
             reported automatically (so a target cannot be met by referring every positive)
  rule_out   if no band qualifies: only confident negatives are auto-reported (lo_c chosen so
             NPV >= rule_out.min_npv and at most rule_out.max_missed of the positives are
             ruled out); every possible positive is referred
  review     if neither qualifies: every decision of this class is referred

An ECG is fully auto-reported only when all five decisions are made automatically. Coverage is
always reported next to the metrics. Fold 10 is only used to apply the frozen bands: this module
reads the test probabilities saved by evaluate.py (test_preds.pt), so nothing is re-run or tuned.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging

import numpy as np

from cardiomoe.data.labels import SUPERCLASSES

log = logging.getLogger(__name__)
INDETERMINATE = -1


# --------------------------------------------------------------------------- fitting
def _band_stats(yc: np.ndarray, pc: np.ndarray, lo: float, hi: float) -> dict:
    pos, neg = pc >= hi, pc <= lo
    tp, fp = int((pos & (yc == 1)).sum()), int((pos & (yc == 0)).sum())
    fn, tn = int((neg & (yc == 1)).sum()), int((neg & (yc == 0)).sum())
    n_pos = int((yc == 1).sum())
    dec = tp + fp + fn + tn
    div = lambda a, b: a / b if b else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "coverage": dec / len(yc),
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn), "accuracy": div(tp + tn, dec),
            "npv": div(tn, tn + fn), "specificity": div(tn, tn + fp),
            "positive_auto": div(tp, n_pos), "positive_ruled_out": div(fn, n_pos)}


def fit_class(yc: np.ndarray, pc: np.ndarray, tcfg: dict) -> dict:
    """Band for one class (see module docstring). Returns {mode, lo, hi, fold9: stats}."""
    t = {k: v + tcfg.get("margin", 0.0) for k, v in tcfg["targets"].items()}
    grid = np.unique(np.quantile(pc, np.linspace(0, 1, tcfg.get("grid", 201))))
    best = None
    for i, lo in enumerate(grid):
        for hi in grid[i + 1:]:
            s = _band_stats(yc, pc, lo, hi)
            if not s["tp"] or s["positive_auto"] < tcfg.get("min_positive_auto", 0.10):
                continue
            if all(s[k] >= v for k, v in t.items()) and (best is None or s["coverage"] > best[2]["coverage"]):
                best = (float(lo), float(hi), s)
    if best:
        return {"mode": "automated", "lo": best[0], "hi": best[1], "fold9": best[2]}
    ro = tcfg.get("rule_out", {})
    best = None
    for lo in grid:  # rule-out only: no auto positives (hi above every probability)
        s = _band_stats(yc, pc, lo, np.inf)
        if s["tn"] and s["npv"] >= ro.get("min_npv", 0.95) and s["positive_ruled_out"] <= ro.get("max_missed", 0.05):
            if best is None or s["coverage"] > best[1]["coverage"]:
                best = (float(lo), s)
    if best:
        return {"mode": "rule_out", "lo": best[0], "hi": None, "fold9": best[1]}
    return {"mode": "review", "lo": None, "hi": None, "fold9": None}


def fit_triage(y: np.ndarray, p_oof: np.ndarray, tcfg: dict) -> dict:
    """Per-class bands from fold-9 out-of-fold gate probabilities."""
    bands = {c: fit_class(y[:, i], p_oof[:, i], tcfg) for i, c in enumerate(SUPERCLASSES)}
    rule = Triage(bands)
    return {"bands": bands, "fold9_cv": report(y, p_oof, rule), "config": tcfg}


# --------------------------------------------------------------------------- applying
class Triage:
    def __init__(self, bands: dict):
        self.bands = bands

    def decide(self, p: np.ndarray) -> np.ndarray:
        """[N, 5] probabilities -> [N, 5] in {1, 0, -1 (indeterminate)}."""
        out = np.full(p.shape, INDETERMINATE, dtype=int)
        for i, c in enumerate(SUPERCLASSES):
            b = self.bands[c]
            if b["lo"] is not None:
                out[p[:, i] <= b["lo"], i] = 0
            if b["hi"] is not None:
                out[p[:, i] >= b["hi"], i] = 1
        return out

    @classmethod
    def from_dict(cls, d: dict) -> "Triage":
        return cls(d["bands"])


def report(y: np.ndarray, p: np.ndarray, rule: Triage) -> dict:
    """Metrics on auto-decided findings, per class and overall, plus coverage."""
    dec = rule.decide(p)
    per_class = {}
    for i, c in enumerate(SUPERCLASSES):
        b = rule.bands[c]
        s = _band_stats(y[:, i], p[:, i], b["lo"] if b["lo"] is not None else -np.inf,
                        b["hi"] if b["hi"] is not None else np.inf)
        per_class[c] = {"mode": b["mode"], **s}
    auto = [c for c in SUPERCLASSES if rule.bands[c]["mode"] == "automated"]
    decided = dec != INDETERMINATE
    full = decided.all(1)
    macro = lambda k: float(np.mean([per_class[c][k] for c in auto])) if auto else float("nan")
    summary = {
        "automated_classes": auto,
        "rule_out_classes": [c for c in SUPERCLASSES if rule.bands[c]["mode"] == "rule_out"],
        "review_classes": [c for c in SUPERCLASSES if rule.bands[c]["mode"] == "review"],
        "macro_precision_automated": macro("precision"), "macro_recall_automated": macro("recall"),
        "macro_accuracy_automated": macro("accuracy"), "macro_coverage_automated": macro("coverage"),
        "finding_coverage": float(decided.mean()),
        "finding_accuracy": float((dec[decided] == y[decided]).mean()) if decided.any() else float("nan"),
        "ecg_fully_automated": float(full.mean()),
        "ecg_fully_automated_exact_match": float((dec[full] == y[full]).all(1).mean()) if full.any() else float("nan"),
        "n": int(len(y)),
    }
    return {"summary": summary, "per_class": per_class}


# --------------------------------------------------------------------------- CLI: fit on fold 9, apply on fold 10
def _oof(cfg: dict, seed: int, debug: bool, selected: str):
    """Out-of-fold fold-9 predictions of the selected combiner (same CV as aggregate.py)."""
    import torch
    from sklearn.model_selection import GroupKFold

    from cardiomoe.fourclient.aggregate import load_val, make

    keys, P, y, groups = load_val(cfg, seed, debug)
    n_splits = min(cfg["fourclient"]["gate"]["cv_folds"], len(np.unique(groups)))
    oof = torch.zeros_like(y)
    for tr, va in GroupKFold(n_splits).split(P, y.numpy(), groups):
        oof[va] = make(selected, cfg, seed).fit(P[tr], y[tr], groups[tr]).predict(P[va])
    return y.numpy(), oof.numpy()


def fit_seed(cfg: dict, seed: int, debug: bool) -> dict:
    """Adds a 'triage' entry to gate/aggregate.json; the fitted combiners are not touched."""
    from cardiomoe.fourclient.train import run_dir

    path = run_dir(cfg, seed, debug) / "gate" / "aggregate.json"
    agg = json.loads(path.read_text())
    y, p = _oof(cfg, seed, debug, agg["selected"])
    agg["triage"] = fit_triage(y, p, cfg["fourclient"]["triage"])
    path.write_text(json.dumps(agg, indent=2))
    return agg["triage"]


def test_seed(cfg: dict, seed: int, debug: bool) -> dict:
    import torch

    from cardiomoe.fourclient.train import run_dir

    d = run_dir(cfg, seed, debug)
    agg = json.loads((d / "gate" / "aggregate.json").read_text())
    t = torch.load(d / "test_preds.pt", weights_only=False)
    sel = agg["selected"]
    y = t["y"].numpy()
    p = np.asarray(t["probs"][sel])
    return {"seed": seed, "selected": sel, **report(y, p, Triage.from_dict(agg["triage"])),
            "bands": agg["triage"]["bands"]}


def markdown(rows: list[dict]) -> str:
    pm = lambda v: f"{np.mean(v):.3f}" + (f" ± {np.std(v, ddof=1):.3f}" if len(v) > 1 else "")
    lines = ["| Class | Mode (seeds) | Auto-decided findings | Precision | Recall | Accuracy | NPV | "
             "True positives auto-reported |", "|" + "---|" * 8]
    for c in SUPERCLASSES:
        pcs = [r["per_class"][c] for r in rows]
        auto = [x for x in pcs if x["mode"] == "automated"]  # precision/recall only exist for automated seeds
        modes = ", ".join(f"{m} ×{sum(x['mode'] == m for x in pcs)}" for m in ("automated", "rule_out", "review")
                          if any(x["mode"] == m for x in pcs))
        f = lambda k, xs=pcs: pm([x[k] for x in xs]) if xs and all(np.isfinite(x[k]) for x in xs) else "–"
        lines.append(f"| {c} | {modes} | {f('coverage')} | {f('precision', auto)} | {f('recall', auto)} | "
                     f"{f('accuracy')} | {f('npv')} | {f('positive_auto', auto)} |")
    s = [r["summary"] for r in rows]
    lines += ["", f"Automated classes: macro precision {pm([x['macro_precision_automated'] for x in s])}, "
              f"macro recall {pm([x['macro_recall_automated'] for x in s])}, "
              f"macro accuracy {pm([x['macro_accuracy_automated'] for x in s])}.",
              f"All auto-decided findings: {pm([x['finding_coverage'] for x in s])} of findings, accuracy "
              f"{pm([x['finding_accuracy'] for x in s])}. ECGs fully auto-reported: "
              f"{pm([x['ecg_fully_automated'] for x in s])}."]
    return "\n".join(lines)


def main() -> None:
    from cardiomoe.fourclient.train import fc_config
    from cardiomoe.utils import repo_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--fit-only", action="store_true", help="fit the bands on fold 9; do not touch fold 10")
    ap.add_argument("--rerun", action="store_true", help="allow applying the bands to fold 10 again")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = fc_config()
    for s in args.seeds:
        tri = fit_seed(cfg, s, args.debug)
        log.info("seed %d bands: %s", s, {c: (b["mode"], b["lo"], b["hi"]) for c, b in tri["bands"].items()})
        log.info("seed %d fold-9 CV: %s", s, json.dumps(tri["fold9_cv"]["summary"]))
    if args.fit_only:
        return
    base = repo_path(cfg["fourclient"]["out_dir"]) / ("debug" if args.debug else "")
    marker = base / "triage_evaluated.json"
    if marker.exists() and not args.debug and not args.rerun:
        raise SystemExit(f"triage was already applied to fold 10 ({marker.read_text().strip()}); pass --rerun")
    rows = [test_seed(cfg, s, args.debug) for s in args.seeds]
    tables = base / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "triage.json").write_text(json.dumps(rows, indent=2))
    md = markdown(rows)
    (tables / "triage.md").write_text(md + "\n", encoding="utf-8")
    if not args.debug:
        marker.write_text(json.dumps({"evaluated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                                      "seeds": args.seeds}) + "\n")
    print(md)


if __name__ == "__main__":
    main()
