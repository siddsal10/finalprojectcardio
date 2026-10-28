"""Mean inter-lead Pearson correlation over the training folds 1-8 (Fig. lead_corr, Section III-C).

    python paper/scripts/lead_corr.py [--debug]

Needs data/processed/ptbxl_signals.npy (python -m cardiomoe.data.ptbxl --prepare). Signals are
normalised exactly as at training time. --debug uses the debug cache and writes under paper/scripts/debug/.
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
from cardiomoe.data.signal import LEADS, normalize
from cardiomoe.utils import load_config

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    base = ROOT / "paper" / ("scripts/debug" if args.debug else "")
    (base / "figures").mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    meta = ptbxl.load_meta(cfg, args.debug)
    sig = ptbxl.load_signal_array(cfg, args.debug)
    rows = meta.loc[meta.split == "train", "row"].to_numpy()
    acc, n = np.zeros((12, 12)), 0
    for r in rows:
        x = normalize(np.asarray(sig[r], dtype=np.float32))
        if (x.std(-1) > 0).all():
            acc += np.corrcoef(x)
            n += 1
    corr = acc / n
    (base / ("lead_corr.json" if args.debug else "scripts/lead_corr.json")).write_text(
        json.dumps({"n_records": n, "corr": corr.round(4).tolist()}))

    plt.rcParams.update({"font.family": "serif", "font.size": 7, "savefig.bbox": "tight", "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    # diverging: polarity of correlation; ends truncated so all labels can stay dark (white text reads as hidden text)
    cmap = matplotlib.colors.ListedColormap(plt.cm.RdBu_r(np.linspace(0.18, 0.82, 256)))
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(12), LEADS, rotation=90)
    ax.set_yticks(range(12), LEADS)
    for (a, b), v in np.ndenumerate(corr):
        ax.text(b, a, "0" if abs(v) < 0.05 else f"{v:.1f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=4.8,
                color="#0b0b0b")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)
    fig.savefig(base / "figures" / "lead_corr.pdf")
    print("records", n)


if __name__ == "__main__":
    main()
