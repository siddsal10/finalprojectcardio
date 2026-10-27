"""Tables and figure for the added baselines: traditional ML, activation/optimizer ablations, FL frameworks.

    python paper/scripts/make_baseline_tables.py [--debug]

Reads paper/scripts/{traditional_ml,ablation,fl_*}.json (written by traditional_ml.py, ablate.py and
fl_baselines.py) and the committed outputs/fourclient results. Writes paper/tables/{trad_rows,act,opt,
fedcomp}.tex and paper/figures/fl_convergence.pdf (--debug: under paper/scripts/debug/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "fourclient"
SCR = ROOT / "paper" / "scripts"
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED, GRID = "#52514e", "#e4e3df"
FL = [("fedavg", "FedAvg", "mcmahan2017fedavg"), ("fedprox", "FedProx ($\\mu=0.01$)", "li2020fedprox"),
      ("scaffold", "SCAFFOLD", "karimireddy2020scaffold"), ("fednova", "FedNova", "wang2020fednova")]


def ms(v, scale=1.0, nd=3):
    v = np.asarray(v, float) * scale
    return f"{v.mean():.{nd}f} $\\pm$ {v.std():.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    base = ROOT / "paper" / ("scripts/debug" if args.debug else "")
    src = SCR / "debug" if args.debug else SCR
    (base / "tables").mkdir(parents=True, exist_ok=True)
    (base / "figures").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- traditional ML rows (Table main)
    tm = json.loads((src / "traditional_ml.json").read_text())
    lines = []
    for name, runs in tm["results"].items():
        cells = [ms([r[k] for r in runs], 100, 1) for k in ["labelwise_accuracy", "macro_precision", "macro_recall",
                                                             "macro_f1"]]
        cells += [ms([r["macro_auroc"] for r in runs]), ms([r["macro_auprc"] for r in runs])]
        lines.append(f"{name} & -- & " + " & ".join(cells) + r" \\")
    (base / "tables" / "trad_rows.tex").write_text("\n".join(lines) + "\n")

    # pooled vs local (one client's folds) training of the same classifiers
    loc = json.loads((src / "traditional_ml_local.json").read_text())
    t = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Local (one client, mean of A--D)} & \multicolumn{2}{c}{Pooled (folds 1--8)} \\",
         r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
         r"Model & AUROC & F1 (\%) & AUROC & F1 (\%) \\", r"\midrule"]
    for name, runs in tm["results"].items():
        la = [v["macro_auroc"] for v in loc[name].values()]
        lf = [v["macro_f1"] for v in loc[name].values()]
        t.append(f"{name} & {ms(la)} & {ms(lf, 100, 1)} & {ms([r['macro_auroc'] for r in runs])} & "
                 f"{ms([r['macro_f1'] for r in runs], 100, 1)}" + r" \\")
    t.append(r"\midrule")
    for c, arch in [("C", "BiLSTM"), ("D", "CNN-LSTM")]:
        s = [json.loads((OUT / f"seed{sd}" / "test_results.json").read_text())["reports"][c]["summary"] for sd in (42, 43, 44)]
        t.append(f"Deep client {c} ({arch}) & {ms([x['macro_auroc'] for x in s])} & "
                 f"{ms([x['macro_f1'] for x in s], 100, 1)} & -- & --" + r" \\")
    t += [r"\bottomrule", r"\end{tabular}"]
    (base / "tables" / "trad_local.tex").write_text("\n".join(t) + "\n")

    # ---------------------------------------------------------------- activation / optimizer ablations
    ab = json.loads((src / "ablation.json").read_text())
    h = [json.loads(l) for l in (OUT / "seed42" / "D" / "history.jsonl").read_text().splitlines()]
    b = int(np.argmax([r["val_macro_auroc"] for r in h]))
    ref = {"best_val_macro_auroc": max(r["val_macro_auroc"] for r in h), "best_epoch": b + 1,
           "epochs_run": len(h), "train_loss_at_best": h[b]["train_loss"], "sec_per_epoch": np.mean([r["sec"] for r in h])}
    names = {"relu": "ReLU (used)", "leakyrelu": "LeakyReLU", "elu": "ELU", "gelu": "GELU", "tanh": "Tanh",
             "adamw": "AdamW (used)", "adam": "Adam \\cite{kingma2015adam}", "sgd": "SGD (momentum 0.9)",
             "rmsprop": "RMSprop (momentum 0.9)"}
    for kind, refname, order in [("activation", "relu", ["relu", "leakyrelu", "elu", "gelu", "tanh"]),
                                 ("optimizer", "adamw", ["adamw", "adam", "sgd", "rmsprop"])]:
        rows = {refname: ref, **{r["value"]: r for r in ab if r["kind"] == kind}}
        top = max(r["best_val_macro_auroc"] for v, r in rows.items() if v in order)
        t = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r"%s & Fold-9 AUROC & Best / run epochs & Train loss \\" % kind.capitalize(), r"\midrule"]
        for v in order:
            if v not in rows:
                t.append(f"{names[v]} & \\todo{{}} & \\todo{{}} & \\todo{{}}" + r" \\")
                continue
            r = rows[v]
            a = f"{r['best_val_macro_auroc']:.3f}"
            a = f"\\textbf{{{a}}}" if abs(r["best_val_macro_auroc"] - top) < 1e-12 else a
            t.append(f"{names[v]} & {a} & {r['best_epoch']} / {r['epochs_run']} & {r['train_loss_at_best']:.3f}"
                     + r" \\")
        t += [r"\bottomrule", r"\end{tabular}"]
        (base / "tables" / ("act.tex" if kind == "activation" else "opt.tex")).write_text("\n".join(t) + "\n")

    # ---------------------------------------------------------------- FL frameworks (seed 42 throughout)
    res42 = json.loads((OUT / "seed42" / "test_results.json").read_text())["reports"]
    pooled = next(r for r in json.loads((OUT / "tables" / "baseline_comparison.json").read_text()) if r["seed"] == 42)
    meta_all = json.loads((OUT / "seed42" / "ALL" / "meta.json").read_text())

    def row(name, rounds, same, s, f9):
        return (f"{name} & {rounds} & {same} & {f9} & {s['macro_auroc']:.3f} & {100 * s['macro_f1']:.1f} & "
                f"{100 * s['labelwise_accuracy']:.1f}" + r" \\")

    agg42 = json.loads((OUT / "seed42" / "gate" / "aggregate.json").read_text())
    t = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"Method & Rounds & Same arch. & Fold-9 AUROC & Test AUROC & F1 (\%) & Acc. (\%) \\", r"\midrule",
         row("Local only (best client, D)", 0, "--", res42["D"]["summary"], f"{agg42['client_fold9_auroc']['D']:.3f}")]
    fl = {}
    for key, name, cite in FL:
        p = src / f"fl_{key}.json"
        if not p.exists():
            t.append(f"{name} \\cite{{{cite}}} & \\todo{{}} & yes & \\todo{{}} & \\todo{{}} & \\todo{{}} & \\todo{{}}" + r" \\")
            continue
        fl[key] = json.loads(p.read_text())
        d = fl[key]
        t.append(row(f"{name} \\cite{{{cite}}}", f"{d['best_round']}/{d['rounds']}", "yes", d["test"],
                     f"{d['best_val_macro_auroc']:.3f}"))
    t.append(row(r"\textbf{Proposed (one-shot gate)}", 1, "no", res42["equal"]["summary"],
                 f"{agg42['cv']['equal']['cv_macro_auroc']:.3f}\\textsuperscript{{a}}"))
    t += [r"\midrule", row("Centralised (pooled data)", "--", "--", pooled["reports"]["baseline"],
                           f"{meta_all['best_val_macro_auroc']:.3f}"), r"\bottomrule", r"\end{tabular}"]
    (base / "tables" / "fedcomp.tex").write_text("\n".join(t) + "\n")

    if fl:
        plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                             "ytick.color": MUTED, "savefig.bbox": "tight", "lines.linewidth": 1.4})
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
        for i, (key, name, _) in enumerate(FL):
            if key not in fl:
                continue
            hh = fl[key]["history"]
            rr = [x["round"] for x in hh]
            lab = name.replace("$\\mu=0.01$", "μ=0.01")
            axs[0].plot(rr, [x["train_loss"] for x in hh], color=PAL[i], label=lab)
            axs[1].plot(rr, [x["val_macro_auroc"] for x in hh], color=PAL[i], label=lab)
        axs[1].axhline(agg42["client_fold9_auroc"]["D"], color=MUTED, ls=":", lw=0.9)
        axs[1].text(1, agg42["client_fold9_auroc"]["D"] + 0.002, "local client D", fontsize=6.5, color=MUTED)
        axs[1].axhline(meta_all["best_val_macro_auroc"], color=MUTED, ls="--", lw=0.9)
        axs[1].text(1, meta_all["best_val_macro_auroc"] + 0.002, "centralised", fontsize=6.5, color=MUTED)
        axs[0].set(xlabel="Communication round", ylabel="Weighted BCE (train)", ylim=(0.4, 2.4),
                   title="(a) Training loss")
        axs[1].set(xlabel="Communication round", ylabel="Fold-9 macro AUROC", title="(b) Global model on fold 9")
        for ax in axs:
            ax.grid(color=GRID, lw=0.6)
        axs[0].legend(frameon=False, loc="upper right", ncol=2, fontsize=6.5)
        fig.savefig(base / "figures" / "fl_convergence.pdf")
    print("ok")


if __name__ == "__main__":
    main()
