"""Activation-function and optimizer ablations for Client D (CNN-LSTM), seed 42 (Tables act/opt).

    python paper/scripts/ablate.py --kind activation --value gelu
    python paper/scripts/ablate.py --kind optimizer --value sgd      [--debug]

Same data (folds 7-8), recipe and early stopping (fold-9 macro AUROC, patience 8) as Client D; only the
activation of the convolutional blocks and the head, or the optimizer, is changed. The learning rate
(OneCycle, max 1e-3) is kept for every optimizer. Only fold 9 is used. Appends to
paper/scripts/ablation.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cardiomoe.data.datasets import SignalDataset, keys_for
from cardiomoe.fourclient.models import SignalClient
from cardiomoe.fourclient.train import fc_config
from cardiomoe.train import common
from cardiomoe.utils import get_device, set_seed

ROOT = Path(__file__).resolve().parents[2]
ACT = {"relu": nn.ReLU, "leakyrelu": lambda: nn.LeakyReLU(0.01), "elu": nn.ELU, "gelu": nn.GELU, "tanh": nn.Tanh}
OPT = {
    "adamw": lambda ps, t: torch.optim.AdamW(ps, lr=t["lr"], weight_decay=t["weight_decay"]),
    "adam": lambda ps, t: torch.optim.Adam(ps, lr=t["lr"]),
    "sgd": lambda ps, t: torch.optim.SGD(ps, lr=t["lr"], momentum=0.9, weight_decay=t["weight_decay"]),
    "rmsprop": lambda ps, t: torch.optim.RMSprop(ps, lr=t["lr"], momentum=0.9, weight_decay=t["weight_decay"]),
}


def swap_relu(module: nn.Module, make) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, make())
        else:
            swap_relu(child, make)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["activation", "optimizer"])
    ap.add_argument("--value", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)
    cfg = fc_config()
    fc = cfg["fourclient"]
    tcfg = dict(fc["train"], num_workers=0)
    device = get_device(cfg["device"], args.debug)
    model = SignalClient("cnn_lstm", fc["embed_dim"], fc["dropout"])
    if args.kind == "activation":
        swap_relu(model, ACT[args.value])
    else:
        make = OPT[args.value]
        common.make_optimizer = lambda m, t: make([p for p in m.parameters() if p.requires_grad], t)
    tr = SignalDataset(cfg, keys_for(cfg, fc["partition"]["D"], args.debug), train=True, debug=args.debug)
    va = SignalDataset(cfg, keys_for(cfg, [fc["val_fold"]], args.debug), debug=args.debug)
    _, info = common.train_model(model, tr, va, tcfg, device, args.debug)
    h = info["history"]
    best = int(np.argmax([r["val_macro_auroc"] for r in h]))
    rec = {"kind": args.kind, "value": args.value, "seed": args.seed, "best_val_macro_auroc": info["best_val_macro_auroc"],
           "best_epoch": best + 1, "epochs_run": len(h), "sec_per_epoch": float(np.mean([r["sec"] for r in h])),
           "train_loss_at_best": h[best]["train_loss"], "n_params": sum(p.numel() for p in model.parameters())}
    print(json.dumps(rec), flush=True)
    path = ROOT / "paper" / ("scripts/debug" if args.debug else "scripts") / "ablation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    allr = json.loads(path.read_text()) if path.exists() else []
    allr = [r for r in allr if not (r["kind"] == rec["kind"] and r["value"] == rec["value"] and r["seed"] == rec["seed"])]
    path.write_text(json.dumps(allr + [rec], indent=1))


if __name__ == "__main__":
    main()
