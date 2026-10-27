"""Build every results table and figure of the paper from the committed outputs.

    python paper/scripts/make_results.py [--debug]

Reads only outputs/fourclient/ (no data, no model re-run, no new fold-10 evaluation) and writes
paper/tables/*.tex, paper/figures/*.pdf and paper/scripts/numbers.json. --debug writes to a
scratch copy under paper/scripts/debug/ so the paper's files are not touched.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "fourclient"
SEEDS = [42, 43, 44]
CLIENTS = ["A", "B", "C", "D"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
ARCH = {"A": "1D CNN", "B": "1D ResNet", "C": "BiLSTM", "D": "CNN-LSTM", "ALL": "CNN-LSTM (pooled)"}
# Reference categorical palette (dataviz skill, light mode), fixed order.
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def load_json(p):
    return json.loads(Path(p).read_text())


def ms(vals, nd=3):
    v = np.asarray(vals, float)
    return f"{v.mean():.{nd}f} $\\pm$ {v.std(ddof=0):.{nd}f}"


def summary(seed, model):
    return load_json(OUT / f"seed{seed}" / "test_results.json")["reports"][model]["summary"]


def per_class(seed, model):
    return load_json(OUT / f"seed{seed}" / "test_results.json")["reports"][model]["per_class"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="write to paper/scripts/debug/ instead")
    args = ap.parse_args()
    base = ROOT / "paper" / ("scripts/debug" if args.debug else "")
    tab, fig = base / "tables", base / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    nums = {}

    # ------------------------------------------------------------------ main comparison
    baseline = {r["seed"]: r for r in load_json(OUT / "tables" / "baseline_comparison.json")}
    keys = ["labelwise_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auroc", "macro_auprc",
            "exact_match_accuracy"]
    rows = {}
    for m in CLIENTS + ["static", "dynamic", "equal"]:
        rows[m] = {k: [summary(s, m)[k] for s in SEEDS] for k in keys}
    nums["baseline_raw_keys"] = list(baseline[42].keys())
    nparams = {c: load_json(OUT / "seed42" / c / "meta.json")["n_params"] for c in CLIENTS + ["ALL"]}
    nums["n_params"] = nparams
    nums["main"] = {m: {k: float(np.mean(v)) for k, v in r.items()} for m, r in rows.items()}
    nums["main_sd"] = {m: {k: float(np.std(v)) for k, v in r.items()} for m, r in rows.items()}

    # ------------------------------------------------------------------ fold-9 selection numbers
    agg = {s: load_json(OUT / f"seed{s}" / "gate" / "aggregate.json") for s in SEEDS}
    nums["cv"] = {c: {k: float(np.mean([agg[s]["cv"][c][k] for s in SEEDS])) for k in ["cv_macro_auroc", "cv_bce"]}
                  for c in ["equal", "static", "dynamic"]}
    nums["static_w"] = {c: float(np.mean([agg[s]["static_weights"][c] for s in SEEDS])) for c in CLIENTS}
    nums["static_w_seed"] = {s: agg[s]["static_weights"] for s in SEEDS}
    nums["client_fold9"] = {c: [agg[s]["client_fold9_auroc"][c] for s in SEEDS] for c in CLIENTS}

    # ------------------------------------------------------------------ training histories
    hist, meta = {}, {}
    for s in SEEDS:
        for c in CLIENTS + ["ALL"]:
            d = OUT / f"seed{s}" / c
            hist[(s, c)] = [json.loads(l) for l in (d / "history.jsonl").read_text().splitlines()]
            meta[(s, c)] = load_json(d / "meta.json")
    tr = {}
    for c in CLIENTS + ["ALL"]:
        best_ep = [int(np.argmax([h["val_macro_auroc"] for h in hist[(s, c)]])) + 1 for s in SEEDS]
        tr[c] = dict(
            n_train=meta[(42, c)]["n_train"], params=meta[(42, c)]["n_params"],
            epochs=[meta[(s, c)]["epochs_run"] for s in SEEDS], best_epoch=best_ep,
            loss_at_best=[hist[(s, c)][b - 1]["train_loss"] for s, b in zip(SEEDS, best_ep)],
            first_loss=[hist[(s, c)][0]["train_loss"] for s in SEEDS],
            val=[meta[(s, c)]["best_val_macro_auroc"] for s in SEEDS],
            T=[meta[(s, c)]["temperature"] for s in SEEDS],
            sec=float(np.mean([h["sec"] for s in SEEDS for h in hist[(s, c)]])),
        )
    nums["training"] = tr

    # ------------------------------------------------------------------ baseline (pooled data)
    braw = load_json(OUT / "tables" / "baseline_comparison.json")
    nums["baseline_json_sample"] = json.dumps(braw[0])[:400]

    # ------------------------------------------------------------------ subsets + Shapley values
    sub = pd.read_csv(OUT / "tables" / "client_subsets.csv", header=[0, 1], index_col=0)
    sub = sub.iloc[1:] if sub.index[0] != sub.index[0] or str(sub.index[0]) == "subset" else sub
    sub.index = sub.index.astype(str)
    # columns: (mean, fold10), (mean, fold9), (std, fold10), (std, fold9)
    cols = list(sub.columns)
    m10 = sub[[c for c in cols if c[0] == "mean" and c[1] == "fold10"][0]].astype(float)
    m9 = sub[[c for c in cols if c[0] == "mean" and c[1] == "fold9"][0]].astype(float)
    s10 = sub[[c for c in cols if c[0] == "std" and c[1] == "fold10"][0]].astype(float)
    shap = {}
    for split, ser in [("fold9", m9), ("fold10", m10)]:
        v = {frozenset(k): float(x) for k, x in ser.items()}
        v[frozenset()] = 0.5  # an empty ensemble ranks at chance
        phi = {}
        for c in CLIENTS:
            tot = 0.0
            others = [o for o in CLIENTS if o != c]
            for r in range(len(others) + 1):
                for S in itertools.combinations(others, r):
                    S = frozenset(S)
                    w = math.factorial(len(S)) * math.factorial(3 - len(S)) / math.factorial(4)
                    tot += w * (v[S | {c}] - v[S])
            phi[c] = tot
        shap[split] = phi
    nums["shapley"] = shap
    nums["subsets"] = {k: {"fold9": float(m9[k]), "fold10": float(m10[k]), "sd10": float(s10[k])} for k in m10.index}

    # ------------------------------------------------------------------ per-class (3-seed means) + supports
    t42 = torch.load(OUT / "seed42" / "test_preds.pt", weights_only=False)
    y = t42["y"].numpy()
    support = y.sum(0).astype(int)
    nums["support"] = dict(zip(CLASSES, support.tolist()))
    nums["n_test"] = int(len(y))
    pcm = {}
    for m in CLIENTS + ["equal", "static", "dynamic"]:
        pcm[m] = {cl: {k: float(np.mean([per_class(s, m)[cl][k] for s in SEEDS]))
                       for k in ["auroc", "precision", "recall", "specificity", "f1", "accuracy"]} for cl in CLASSES}
    pcsd = {cl: {k: float(np.std([per_class(s, "equal")[cl][k] for s in SEEDS]))
                 for k in ["auroc", "precision", "recall", "specificity", "f1"]} for cl in CLASSES}
    nums["per_class_mean"] = pcm

    braw = load_json(OUT / "tables" / "baseline_comparison.json")

    # ------------------------------------------------------------------ TABLE: main comparison
    def row(name, m, params):
        r = rows[m]
        cells = [ms([100 * x for x in r[k]], 1) for k in ["labelwise_accuracy", "macro_precision", "macro_recall",
                                                          "macro_f1"]]
        cells += [ms(r["macro_auroc"]), ms(r["macro_auprc"])]
        return f"{name} & {params} & " + " & ".join(cells) + r" \\"

    def prm(n):
        return f"{n / 1e6:.2f}M"

    lines = [
        r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
        r"Model & Params & Acc. (\%) & Prec. (\%) & Rec. (\%) & F1 (\%) & AUROC & AUPRC \\", r"\midrule",
        r"\multicolumn{8}{@{}l}{\emph{Traditional machine learning (172 hand-crafted features, pooled folds 1--8)}}\\",
    ]
    trad = tab / "trad_rows.tex"  # written by make_baseline_tables.py
    if trad.exists():
        lines += trad.read_text().splitlines()
    else:
        for t in ["Logistic Regression", "SVM (RBF)", "Random Forest", "KNN", "XGBoost"]:
            lines.append(f"{t} & -- & " + " & ".join([r"\todo{}"] * 6) + r" \\")
    lines += [r"\midrule", r"\multicolumn{8}{@{}l}{\emph{Single deep clients (25\% of training data each)}}\\"]
    for c in CLIENTS:
        lines.append(row(f"Client {c}: {ARCH[c]}", c, prm(nparams[c])))
    lines += [r"\midrule", r"\multicolumn{8}{@{}l}{\emph{Aggregation of the four clients}}\\"]
    lines.append(row("Static learned weights", "static", "4"))
    lines.append(row("Dynamic gate", "dynamic", r"$\approx$1k"))
    lines.append(row(r"\textbf{Proposed (equal-weight gate)}", "equal", prm(sum(nparams[c] for c in CLIENTS))))
    lines += [r"\midrule", r"\multicolumn{8}{@{}l}{\emph{Centralised reference (data pooled, not federated)}}\\"]
    rows["pooled"] = {k: [b["reports"]["baseline"][k] for b in braw] for k in keys}
    lines.append(row("CNN-LSTM on folds 1--8", "pooled", prm(nparams["ALL"])))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tab / "main_comparison.tex").write_text("\n".join(lines) + "\n")

    # pooled baseline numbers (from baseline_comparison.csv: mean/sd over seeds)
    bc = pd.read_csv(OUT / "tables" / "baseline_comparison.csv", index_col=0)
    nums["pooled"] = bc.to_dict(orient="index")

    # ------------------------------------------------------------------ TABLE: single client vs aggregation (per class AUROC)
    lines = [r"\begin{tabular}{@{}lccccc|c@{}}", r"\toprule",
             r"Model & NORM & MI & STTC & CD & HYP & Macro \\", r"\midrule"]
    for m, name in [(c, f"Client {c} ({ARCH[c]})") for c in CLIENTS] + [("equal", r"\textbf{Proposed}")]:
        vals = [pcm[m][cl]["auroc"] for cl in CLASSES]
        best = [max(pcm[x][cl]["auroc"] for x in CLIENTS + ["equal"]) for cl in CLASSES]
        cells = [(f"\\textbf{{{v:.3f}}}" if abs(v - b) < 1e-9 else f"{v:.3f}") for v, b in zip(vals, best)]
        lines.append(f"{name} & " + " & ".join(cells) + f" & {nums['main'][m]['macro_auroc']:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tab / "client_vs_gate.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ TABLE: client-subset ablation
    order = sorted(m10.index, key=lambda k: (-len(k), -m9[k]))
    lines = [r"\begin{tabular}{@{}clccc@{}}", r"\toprule",
             r"\#Clients & Subset & Fold-9 CV AUROC & Test AUROC & $\Delta$ vs.\ ABCD \\", r"\midrule"]
    prev = None
    for k in order:
        if prev is not None and len(k) != prev:
            lines.append(r"\midrule")
        prev = len(k)
        name = r"\textbf{ABCD}" if k == "ABCD" else k
        lines.append(f"{len(k)} & {name} & {m9[k]:.3f} & {m10[k]:.3f} $\\pm$ {s10[k]:.3f} & "
                     f"{m10[k] - m10['ABCD']:+.3f}".replace("-0.000", "0.000").replace("+0.000", "0.000") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tab / "subsets.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Client & Architecture & $\phi$ (fold 9) & $\phi$ (test) & Static weight $w_i$ \\", r"\midrule"]
    for c in CLIENTS:
        lines.append(f"{c} & {ARCH[c]} & {shap['fold9'][c]:.4f} & {shap['fold10'][c]:.4f} & {nums['static_w'][c]:.3f}"
                     + r" \\")
    lines += [r"\midrule",
              f"Sum & -- & {sum(shap['fold9'].values()):.4f} & {sum(shap['fold10'].values()):.4f} & 1.000" + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    (tab / "shapley.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ TABLE: fusion strategies
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Fusion strategy & CV AUROC & CV BCE & Test AUROC & Test F1 (\%) & Exact match (\%) \\", r"\midrule"]
    for m, name in [("equal", r"\textbf{Equal weights (selected)}"), ("static", "Static learned weights"),
                    ("dynamic", "Dynamic per-patient gate")]:
        r = rows[m]
        lines.append(f"{name} & {nums['cv'][m]['cv_macro_auroc']:.3f} & {nums['cv'][m]['cv_bce']:.3f} & "
                     f"{ms(r['macro_auroc'])} & {ms([100 * x for x in r['macro_f1']], 1)} & "
                     f"{ms([100 * x for x in r['exact_match_accuracy']], 1)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tab / "fusion.tex").write_text("\n".join(lines) + "\n")

    # ------------------------------------------------------------------ TABLE: client training summary
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r"Model & $n_{\text{train}}$ & Epochs run & Best epoch & Train loss & Fold-9 AUROC & $T_i$ \\",
             r"\midrule"]
    for c in CLIENTS + ["ALL"]:
        t = tr[c]
        name = f"Client {c}" if c != "ALL" else "Pooled (folds 1--8)"
        lines.append(f"{name} & {t['n_train']:,} & {np.mean(t['epochs']):.1f} & {np.mean(t['best_epoch']):.1f} & "
                     f"{ms(t['loss_at_best'])} & {ms(t['val'])} & {ms(t['T'], 2)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (tab / "client_training.tex").write_text("\n".join(lines).replace(",", "{,}") + "\n")

    # ------------------------------------------------------------------ TABLE: class-wise performance
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Class & Precision & Recall & F1-score & AUROC & Support \\", r"\midrule"]
    for cl in CLASSES:
        p = pcm["equal"][cl]
        lines.append(f"{cl} & {p['precision']:.3f} & {p['recall']:.3f} & {p['f1']:.3f} & {p['auroc']:.3f} & "
                     f"{nums['support'][cl]}" + r" \\")
    w = support / support.sum()
    mac = {k: np.mean([pcm["equal"][cl][k] for cl in CLASSES]) for k in ["precision", "recall", "f1", "auroc"]}
    wtd = {k: float(np.sum(w * np.array([pcm["equal"][cl][k] for cl in CLASSES])))
           for k in ["precision", "recall", "f1", "auroc"]}
    nums["classwise_macro"], nums["classwise_weighted"] = mac, wtd
    lines += [r"\midrule",
              f"Macro avg. & {mac['precision']:.3f} & {mac['recall']:.3f} & {mac['f1']:.3f} & {mac['auroc']:.3f} & "
              f"{support.sum()}" + r" \\",
              f"Weighted avg. & {wtd['precision']:.3f} & {wtd['recall']:.3f} & {wtd['f1']:.3f} & {wtd['auroc']:.3f} & "
              f"{support.sum()}" + r" \\", r"\bottomrule", r"\end{tabular}"]
    (tab / "classwise.tex").write_text("\n".join(lines) + "\n")
    nums["classwise_sd"] = pcsd

    # ------------------------------------------------------------------ FIG: training curves (seed 42)
    fig_, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for i, c in enumerate(CLIENTS):
        h = hist[(42, c)]
        ep = [r["epoch"] for r in h]
        axs[0].plot(ep, [r["val_macro_auroc"] for r in h], color=PAL[i], label=f"{c}: {ARCH[c]}")
        axs[1].plot(ep, [r["train_loss"] for r in h], color=PAL[i], label=f"{c}: {ARCH[c]}")
        b = int(np.argmax([r["val_macro_auroc"] for r in h]))
        axs[0].plot(ep[b], h[b]["val_macro_auroc"], "o", ms=4, color=PAL[i], mec="white", mew=1)
    axs[0].set(xlabel="Local epoch", ylabel="Fold-9 macro AUROC", title="(a) Validation macro AUROC per client")
    axs[1].set(xlabel="Local epoch", ylabel="Weighted BCE (train)", title="(b) Training loss per client")
    axs[1].legend(frameon=False, loc="upper right")
    fig_.savefig(fig / "client_curves.pdf")
    plt.close(fig_)

    # ------------------------------------------------------------------ FIG: aggregation as clients are added (fold 9 / test)
    fig_, ax = plt.subplots(figsize=(3.4, 2.3))
    for j, (split, ser, lab) in enumerate([("fold9", m9, "Fold-9 CV"), ("fold10", m10, "Test (fold 10)")]):
        xs, best, mean = [], [], []
        for n in range(1, 5):
            ks = [k for k in ser.index if len(k) == n]
            xs.append(n)
            best.append(max(ser[k] for k in ks))
            mean.append(np.mean([ser[k] for k in ks]))
        ax.plot(xs, mean, "-o", color=PAL[j], ms=4, mec="white", mew=1, label=f"{lab}, mean over subsets")
        ax.plot(xs, best, "--", color=PAL[j], lw=1.0, label=f"{lab}, best subset")
    ax.set(xlabel="Number of aggregated clients", ylabel="Macro AUROC", xticks=[1, 2, 3, 4])
    ax.legend(frameon=False, loc="lower right")
    fig_.savefig(fig / "server_convergence.pdf")
    plt.close(fig_)

    # ------------------------------------------------------------------ FIG: ROC curves (seed 42, selected gate)
    pe = t42["probs"]["equal"].numpy() if torch.is_tensor(t42["probs"]["equal"]) else np.asarray(t42["probs"]["equal"])
    fig_, ax = plt.subplots(figsize=(3.4, 3.0))
    for i, cl in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y[:, i], pe[:, i])
        ax.plot(fpr, tpr, color=PAL[i], label=f"{cl} (AUC {per_class(42, 'equal')[cl]['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8, ls=":")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(frameon=False, loc="lower right")
    fig_.savefig(fig / "roc.pdf")
    plt.close(fig_)

    # ------------------------------------------------------------------ FIG: confusion matrices (seed 42, selected gate)
    pc = pd.read_csv(OUT / "tables" / "per_class.csv")
    pc = pc[pc.model == "equal"].set_index("class")
    fig_, axs = plt.subplots(1, 5, figsize=(7.0, 1.7))
    for ax, cl in zip(axs, CLASSES):
        r = pc.loc[cl]
        cm = np.array([[r.tn, r.fp], [r.fn, r.tp]], float)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=matplotlib.colors.ListedColormap(plt.cm.Blues(np.linspace(0, 0.6, 256))), vmin=0, vmax=1)
        for (a, b), v in np.ndenumerate(cm):
            ax.text(b, a, f"{int(v)}\n({norm[a, b]:.2f})", ha="center", va="center", fontsize=6.5, color=INK)
        ax.set_title(cl)
        ax.set_xticks([0, 1], ["Pred. 0", "Pred. 1"])
        ax.set_yticks([0, 1], ["True 0", "True 1"] if cl == "NORM" else ["", ""])
        ax.grid(False)
    fig_.savefig(fig / "confusion.pdf")
    plt.close(fig_)
    nums["confusion_seed42"] = {cl: {k: int(pc.loc[cl][k]) for k in ["tp", "fp", "fn", "tn"]} for cl in CLASSES}
    nums["thresholds_seed42"] = {cl: float(pc.loc[cl]["threshold"]) for cl in CLASSES}

    # ------------------------------------------------------------------ FIG: model comparison bars
    fig_, ax = plt.subplots(figsize=(3.4, 2.2))
    models = CLIENTS + ["dynamic", "static", "equal"]
    labels = ["A", "B", "C", "D", "Dyn.", "Static", "Equal"]
    mean = [nums["main"][m]["macro_auroc"] for m in models]
    sd = [nums["main_sd"][m]["macro_auroc"] for m in models]
    cols = [PAL[0]] * 4 + [PAL[1]] * 3
    ax.bar(range(len(models)), mean, yerr=sd, color=cols, width=0.7, capsize=2, error_kw=dict(lw=0.8))
    ax.axhline(float(bc.loc["baseline", "macro_auroc"]), color=MUTED, ls="--", lw=0.9)
    ax.text(len(models) - 0.5, float(bc.loc["baseline", "macro_auroc"]) + 0.001, "pooled CNN-LSTM", ha="right",
            va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(range(len(models)), labels)
    ax.set(ylim=(0.87, 0.915), ylabel="Test macro AUROC")
    ax.grid(axis="x", visible=False)
    fig_.savefig(fig / "model_comparison.pdf")
    plt.close(fig_)

    # ------------------------------------------------------------------ FIG: aggregation weights
    fig_, ax = plt.subplots(figsize=(3.4, 2.1))
    width = 0.25
    for j, s in enumerate(SEEDS):
        ax.bar(np.arange(4) + (j - 1) * width, [agg[s]["static_weights"][c] for c in CLIENTS], width * 0.92,
               color=PAL[j], label=f"seed {s}")
    ax.axhline(0.25, color=MUTED, ls="--", lw=0.9)
    ax.text(0.5, 0.26, "equal weight 1/4", ha="center", fontsize=6.5, color=MUTED)
    ax.set_xticks(range(4), [f"{c}\n{ARCH[c]}" for c in CLIENTS])
    ax.set(ylabel="Learned static weight $w_i$")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig_.savefig(fig / "weights.pdf")
    plt.close(fig_)

    (ROOT / "paper" / "scripts" / ("debug" if args.debug else "")).mkdir(parents=True, exist_ok=True)
    (base / ("numbers.json" if args.debug else "scripts/numbers.json")).write_text(
        json.dumps(nums, indent=1, default=float))
    print("wrote", tab, fig)


if __name__ == "__main__":
    main()
