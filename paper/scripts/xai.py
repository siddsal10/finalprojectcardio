"""Explainability of the selected (equal-weight) gate with Integrated Gradients (Section IV-G).

    python paper/scripts/xai.py [--n 400] [--seed 42] [--device cpu] [--debug]

Attributions are computed for the gate output P(y_c|x) = 1/4 sum_i sigmoid(z_ic / T_i) with respect to
the normalised 12-lead input, using the trained seed-42 clients. Only fold-9 (validation) records are
used, so the test fold is not touched. Writes paper/figures/xai_{leads,local}.pdf,
paper/tables/xai_deletion.tex and paper/scripts/xai.json. --debug: 16 records, 4 IG steps, CPU,
outputs under paper/scripts/debug/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cardiomoe.data.datasets import SignalDataset, keys_for
from cardiomoe.data.labels import SUPERCLASSES
from cardiomoe.data.signal import LEADS
from cardiomoe.fourclient.train import CLIENTS, fc_config, load_client
from cardiomoe.utils import get_device, set_seed

ROOT = Path(__file__).resolve().parents[2]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


class Gate(torch.nn.Module):
    """Equal-weight mixture of the four calibrated clients, as a differentiable function of the signal."""

    def __init__(self, clients):
        super().__init__()
        self.clients = torch.nn.ModuleList(clients)

    def forward(self, x):
        ps = [torch.sigmoid(m.head(m.encode({"signal": x})) / m.temperature) for m in self.clients]
        return torch.stack(ps, 0).mean(0)


def integrated_gradients(f, x, cls, steps):
    """IG with an all-zero baseline (= every lead at its per-recording mean after z-scoring)."""
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps):
        xi = (a * x).detach().requires_grad_(True)
        out = f(xi)[:, cls].sum()
        (g,) = torch.autograd.grad(out, xi)
        total += g
    return x * total / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="fold-9 records used for the global analysis")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="override the config device (e.g. cpu on an 8 GB Mac)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.debug:
        args.n, args.steps = 16, 4
    set_seed(args.seed)
    base = ROOT / "paper" / ("scripts/debug" if args.debug else "")
    (base / "figures").mkdir(parents=True, exist_ok=True)
    (base / "tables").mkdir(parents=True, exist_ok=True)

    cfg = fc_config()
    device = get_device(args.device or cfg["device"], args.debug)
    clients = [load_client(cfg, c, args.seed, False) for c in CLIENTS]
    gate = Gate(clients).to(device).eval()
    torch.backends.cudnn.enabled = False  # LSTM backward in eval mode

    ds = SignalDataset(cfg, keys_for(cfg, [cfg["fourclient"]["val_fold"]], args.debug), debug=args.debug)
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(ds), min(args.n, len(ds)), replace=False))
    X = torch.stack([ds[i]["signal"] for i in idx]).to(device)
    Y = torch.from_numpy(ds.y[idx])
    with torch.no_grad():
        P = torch.cat([gate(X[i:i + 32]) for i in range(0, len(X), 32)]).cpu()

    # ---------------------------------------------------------------- global: lead importance per class
    lead_imp = np.zeros((len(SUPERCLASSES), 12))
    attr_cache = {}
    cache = base / "scripts" / f"xai_attr_seed{args.seed}_n{len(idx)}_s{args.steps}.pt" if not args.debug \
        else base / "xai_attr.pt"
    saved = torch.load(cache) if cache.exists() else {}  # IG is slow on CPU; plots can be redone from the cache
    for c, name in enumerate(SUPERCLASSES):
        pos = torch.where(Y[:, c] == 1)[0]
        if len(pos) == 0:
            continue
        if name in saved:
            attr_cache[name] = (pos, saved[name])
            imp = saved[name].abs().sum(-1)
            lead_imp[c] = (imp / imp.sum(1, keepdim=True)).mean(0).numpy()
            continue
        parts = []
        for i in range(0, len(pos), 16):
            t0 = time.time()
            parts.append(integrated_gradients(gate, X[pos[i:i + 16]], c, args.steps).detach().cpu())
            print(f"{name}: batch {i // 16 + 1}/{-(-len(pos) // 16)} in {time.time() - t0:.0f}s", flush=True)
        A = torch.cat(parts)
        attr_cache[name] = (pos, A)
        saved[name] = A
        torch.save(saved, cache)
        imp = A.abs().sum(-1)  # [n, 12]
        lead_imp[c] = (imp / imp.sum(1, keepdim=True)).mean(0).numpy()
        print(name, len(pos), np.round(lead_imp[c], 3), flush=True)

    # ---------------------------------------------------------------- faithfulness: delete top-k leads vs random
    rows = []
    for c, name in enumerate(SUPERCLASSES):
        if name not in attr_cache:
            continue
        pos, A = attr_cache[name]
        x = X[pos]
        p0 = P[pos, c]
        rank = A.abs().sum(-1).argsort(1, descending=True)
        res = {"class": name, "n": int(len(pos)), "p0": float(p0.mean())}
        for k in (1, 3):
            xt, xr = x.clone(), x.clone()
            for j in range(len(pos)):
                xt[j, rank[j, :k]] = 0.0
                xr[j, torch.from_numpy(rng.choice(12, k, replace=False))] = 0.0
            with torch.no_grad():
                res[f"top{k}"] = float(gate(xt)[:, c].cpu().mean())
                res[f"rand{k}"] = float(gate(xr)[:, c].cpu().mean())
        rows.append(res)
    lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
             r" & & & \multicolumn{2}{c}{Top-$k$ IG leads removed} & \multicolumn{2}{c}{Random $k$ leads removed} \\",
             r"\cmidrule(lr){4-5}\cmidrule(l){6-7}",
             r"Class & $n$ & Mean $p$ (intact) & $k=1$ & $k=3$ & $k=1$ & $k=3$ \\", r"\midrule"]
    for r in rows:
        lines.append(f"{r['class']} & {r['n']} & {r['p0']:.3f} & {r['top1']:.3f} & {r['top3']:.3f} & "
                     f"{r['rand1']:.3f} & {r['rand3']:.3f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (base / "tables" / "xai_deletion.tex").write_text("\n".join(lines) + "\n")

    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "savefig.bbox": "tight", "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    im = ax.imshow(lead_imp, cmap=matplotlib.colors.ListedColormap(plt.cm.Blues(np.linspace(0, 0.6, 256))),
                   aspect="auto", vmin=0)
    ax.set_xticks(range(12), LEADS, fontsize=7, rotation=90)
    ax.set_yticks(range(5), SUPERCLASSES, fontsize=7)
    for (a, b), v in np.ndenumerate(lead_imp):
        ax.text(b, a, f"{100 * v:.0f}", ha="center", va="center", fontsize=5.5, color=INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("Share of |IG| (%)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ticks = cb.get_ticks()
    cb.set_ticks(ticks, labels=[f"{100 * t:.0f}" for t in ticks])
    fig.savefig(base / "figures" / "xai_leads.pdf")
    plt.close(fig)

    # ---------------------------------------------------------------- local: the most confident true MI and STTC cases
    local = {}
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
    for ax, name in zip(axs, ["MI", "STTC"]):
        c = SUPERCLASSES.index(name)
        pos, A = attr_cache[name]
        j = int(P[pos, c].argmax())
        x, a = X[pos[j]].cpu().numpy(), A[j].numpy()
        top = np.argsort(-np.abs(a).sum(-1))[:3]
        key = ds.keys[idx[int(pos[j])]]
        local[name] = {"ecg_id": int(key), "p": float(P[pos[j], c]), "top_leads": [LEADS[t] for t in top],
                       "probs": {k: float(v) for k, v in zip(SUPERCLASSES, P[pos[j]].tolist())},
                       "labels": [k for k, v in zip(SUPERCLASSES, Y[pos[j]].tolist()) if v == 1]}
        t = np.arange(x.shape[-1]) / 100.0
        scale = np.abs(a[top]).max()
        for r, l in enumerate(top):
            off = -4.0 * r
            ax.plot(t, x[l] / max(np.abs(x[l]).max(), 1e-6) * 1.6 + off, color=INK, lw=0.6)
            sal = np.abs(a[l]) / scale
            sal = np.convolve(sal, np.ones(9) / 9, mode="same")
            ax.scatter(t, np.full_like(t, off - 2.1), c=sal, cmap="Oranges", s=2, vmin=0, vmax=sal.max(),
                       marker="s", linewidths=0)
            ax.text(-0.15, off, LEADS[l], ha="right", va="center", fontsize=7)
        ax.set_yticks([])
        for s in ("left", "top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"({'a' if name == 'MI' else 'b'}) ECG {key}: labels {', '.join(local[name]['labels'])}; "
                     f"P({name}) = {local[name]['p']:.2f}; three leads with largest |IG|", fontsize=7.5)
    axs[-1].set_xlabel("Time (s)")
    fig.savefig(base / "figures" / "xai_local.pdf")
    plt.close(fig)

    out = {"n_records": int(len(idx)), "steps": args.steps, "seed": args.seed,
           "positives": {k: int(len(v[0])) for k, v in attr_cache.items()},
           "lead_importance": {k: dict(zip(LEADS, map(float, lead_imp[i]))) for i, k in enumerate(SUPERCLASSES)},
           "deletion": rows, "local": local}
    (base / ("xai.json" if args.debug else "scripts/xai.json")).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["deletion"], indent=1))


if __name__ == "__main__":
    main()
