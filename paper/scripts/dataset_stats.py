"""Dataset statistics for Section IV-A, computed from the PTB-XL metadata only (no signals).

    python paper/scripts/dataset_stats.py [--debug]

Uses the same label pipeline as training (cardiomoe.data.ptbxl.build_meta). Writes
paper/tables/dataset.tex, paper/figures/class_distribution.pdf and paper/scripts/dataset_stats.json.
--debug uses 200 random records and writes under paper/scripts/debug/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cardiomoe.data import ptbxl
from cardiomoe.data.labels import SUPERCLASSES
from cardiomoe.utils import load_config, repo_path

ROOT = Path(__file__).resolve().parents[2]
PAL = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = [("Train (folds 1--8)", "train"), ("Validation (fold 9)", "val"), ("Test (fold 10)", "test")]
CLIENT_FOLDS = {"A": [1, 2], "B": [3, 4], "C": [5, 6], "D": [7, 8]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    out = ROOT / "paper" / ("scripts/debug" if args.debug else "")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    db, scp = ptbxl.load_database(repo_path(cfg["paths"]["ptbxl"]))
    n_raw, p_raw = len(db), db.patient_id.nunique()
    meta = ptbxl.build_meta(db, scp)
    if args.debug:
        meta = meta.sample(n=200, random_state=42)
    db = db.loc[meta.index]
    ycols = [f"y_{c}" for c in SUPERCLASSES]
    age = db.age.where(db.age < 300)

    st = {"n_raw": n_raw, "patients_raw": int(p_raw), "n_kept": len(meta), "patients_kept": int(meta.patient_id.nunique()),
          "n_dropped": n_raw - len(meta), "label_cardinality": float(meta[ycols].sum(1).mean()),
          "multi_label_frac": float((meta[ycols].sum(1) > 1).mean()),
          "age_mean": float(age.mean()), "age_sd": float(age.std()), "female_frac": float((db.sex == 1).mean()),
          "n_diag_statements": int((scp.diagnostic == 1).sum())}
    for name, sp in GROUPS:
        m = meta[meta.split == sp]
        st[sp] = {"records": len(m), "patients": int(m.patient_id.nunique()),
                  **{c: int(m[f"y_{c}"].sum()) for c in SUPERCLASSES}}
    for c, f in CLIENT_FOLDS.items():
        m = meta[meta.strat_fold.isin(f)]
        st[f"client_{c}"] = {"records": len(m), "patients": int(m.patient_id.nunique()),
                             **{k: int(m[f"y_{k}"].sum()) for k in SUPERCLASSES}}
    (out / ("dataset_stats.json" if args.debug else "scripts/dataset_stats.json")).write_text(json.dumps(st, indent=1))

    # Table: per split and per client, records, patients, positives per class (prevalence %)
    lines = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r"Partition & Folds & Records & Patients & NORM & MI & STTC & CD & HYP \\", r"\midrule"]
    for c, f in CLIENT_FOLDS.items():
        s = st[f"client_{c}"]
        lines.append(f"Client {c} & {f[0]}, {f[1]} & {s['records']:,} & {s['patients']:,} & "
                     + " & ".join(f"{s[k]:,}" for k in SUPERCLASSES) + r" \\")
    lines.append(r"\midrule")
    for (name, sp), folds in zip(GROUPS, ["1--8", "9", "10"]):
        s = st[sp]
        lines.append(f"{name.split(' (')[0]} & {folds} & {s['records']:,} & {s['patients']:,} & "
                     + " & ".join(f"{s[k]:,} ({100 * s[k] / s['records']:.1f})"
                                  for k in SUPERCLASSES) + r" \\")
    tot = {k: sum(st[sp][k] for _, sp in GROUPS) for k in SUPERCLASSES}
    lines.append(r"\midrule")
    lines.append(f"Total & 1--10 & {st['n_kept']:,} & {st['patients_kept']:,} & "
                 + " & ".join(f"{tot[k]:,} ({100 * tot[k] / st['n_kept']:.1f})" for k in SUPERCLASSES) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tables" / "dataset.tex").write_text("\n".join(lines).replace(",", "{,}").replace("{,} ", ", ") + "\n")

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "savefig.bbox": "tight"})
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    x = np.arange(len(SUPERCLASSES))
    w = 0.27
    for j, (name, sp) in enumerate(GROUPS):
        ax.bar(x + (j - 1) * w, [st[sp][k] for k in SUPERCLASSES], w * 0.92, color=PAL[j], label=name)
    ax.set_xticks(x, SUPERCLASSES)
    ax.set_ylabel("Records with label")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(out / "figures" / "class_distribution.pdf")
    print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
