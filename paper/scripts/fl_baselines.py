"""Parameter-averaging FL baselines on the same four client partitions (Table fedcomp, Section IV-E3).

    python paper/scripts/fl_baselines.py --method fedavg|fedprox|scaffold|fednova [--rounds 20] [--seed 42] [--debug]

All methods train one shared CNN-LSTM (Client D's architecture) on clients A-D (folds 1-2, 3-4, 5-6, 7-8)
with 1 local epoch per round, local AdamW (lr 1e-3, wd 1e-2, re-created every round), batch 128, the
client's own pos_weight, train-only augmentation and gradient clipping 1.0, i.e. the recipe of the
clients. Aggregation weights are n_k / n. The round with the best fold-9 macro AUROC is kept, a
temperature and per-class thresholds are fitted on fold 9, and fold 10 is evaluated once.
    FedProx  : local loss + mu/2 ||w - w_global||^2, mu = 0.01
    SCAFFOLD : gradient correction c - c_k; control variates updated with option II
    FedNova  : client updates normalised by their number of local steps
Writes paper/scripts/fl_<method>.json (per-round history and test report).
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from cardiomoe.clients.base import masked_bce
from cardiomoe.data.datasets import SignalDataset, keys_for, make_loader, to_device
from cardiomoe.eval.metrics import macro_auroc, tune_thresholds
from cardiomoe.fourclient.metrics import full_report
from cardiomoe.fourclient.models import SignalClient
from cardiomoe.fourclient.train import CLIENTS, fc_config
from cardiomoe.fusion.calibration import fit_temperature
from cardiomoe.train.common import pos_weight_from, predict
from cardiomoe.utils import get_device, set_seed

ROOT = Path(__file__).resolve().parents[2]
MU = 0.01


def average(states: list[dict], w: np.ndarray) -> dict:
    out = {}
    for k, v in states[0].items():
        if v.dtype.is_floating_point:
            out[k] = sum(wi * s[k] for wi, s in zip(w, states))
        else:  # BatchNorm num_batches_tracked
            out[k] = v.clone()
    return out


def local_train(model, loader, pw, device, method, glob=None, c=None, ck=None):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    model.train()
    tot, n, steps = 0.0, 0, 0
    for batch in loader:
        out = model(to_device(batch, device))
        loss = masked_bce(out.logits.float(), batch["y"].to(device), pw)
        if method == "fedprox":
            loss = loss + MU / 2 * sum(((p - glob[k]) ** 2).sum() for k, p in model.named_parameters())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if method == "scaffold":
            for k, p in model.named_parameters():
                p.grad += c[k] - ck[k]
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += loss.item() * len(batch["y"])
        n += len(batch["y"])
        steps += 1
    return tot / n, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["fedavg", "fedprox", "scaffold", "fednova"])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.debug:
        args.rounds = 2
    set_seed(args.seed)
    cfg = fc_config()
    fc = cfg["fourclient"]
    device = get_device(args.device or cfg["device"], args.debug)
    bs = 32 if args.debug else fc["train"]["batch_size"]
    data = {k: SignalDataset(cfg, keys_for(cfg, fc["partition"][k], args.debug), train=True, debug=args.debug)
            for k in CLIENTS}
    loaders = {k: make_loader(d, bs, True, 0) for k, d in data.items()}
    pws = {k: pos_weight_from(d.y).to(device) for k, d in data.items()}
    n = np.array([len(data[k]) for k in CLIENTS], float)
    w = n / n.sum()
    val = SignalDataset(cfg, keys_for(cfg, [fc["val_fold"]], args.debug), debug=args.debug)
    test = SignalDataset(cfg, keys_for(cfg, [fc["test_fold"]], args.debug), debug=args.debug)

    gm = SignalClient("cnn_lstm", fc["embed_dim"], fc["dropout"]).to(device)
    params = [k for k, _ in gm.named_parameters()]
    c = {k: torch.zeros_like(p) for k, p in gm.named_parameters()}
    ck = {cl: {k: torch.zeros_like(p) for k, p in gm.named_parameters()} for cl in CLIENTS}
    hist, best, best_state = [], -1.0, None
    for r in range(1, args.rounds + 1):
        t0 = time.time()
        g_state = copy.deepcopy(gm.state_dict())
        glob = {k: v.detach() for k, v in g_state.items() if k in params}
        states, losses, taus, new_ck = [], [], [], {}
        for cl in CLIENTS:
            lm = copy.deepcopy(gm)
            loss, tau = local_train(lm, loaders[cl], pws[cl], device, args.method, glob, c, ck[cl])
            s = {k: v.detach().clone() for k, v in lm.state_dict().items()}
            if args.method == "scaffold":  # option II: c_k+ = c_k - c + (x - y_k) / (K * lr)
                new_ck[cl] = {k: ck[cl][k] - c[k] + (glob[k] - s[k]) / (tau * 1e-3) for k in params}
            states.append(s)
            losses.append(loss)
            taus.append(tau)
        if args.method == "fednova":  # normalise each client's update by its local steps
            tau_eff = float(np.dot(w, taus))
            new = {}
            for k, v in g_state.items():
                if v.dtype.is_floating_point:
                    d = sum(wi * (v - s[k]) / t for wi, s, t in zip(w, states, taus))
                    new[k] = v - tau_eff * d
                else:
                    new[k] = v.clone()
        else:
            new = average(states, w)
        if args.method == "scaffold":
            for k in params:
                c[k] = c[k] + sum(wi * (new_ck[cl][k] - ck[cl][k]) for wi, cl in zip(w, CLIENTS))
            ck = new_ck
        gm.load_state_dict(new)
        pv = predict(gm, val, device, 128)
        auc = macro_auroc(pv["y"].numpy(), torch.sigmoid(pv["logits"]).numpy())
        vloss = float(masked_bce(pv["logits"], pv["y"]))
        rec = {"round": r, "train_loss": float(np.dot(w, losses)), "client_loss": dict(zip(CLIENTS, losses)),
               "val_macro_auroc": auc, "val_bce": vloss, "sec": round(time.time() - t0, 1)}
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if auc > best:
            best, best_state = auc, copy.deepcopy(gm.state_dict())

    gm.load_state_dict(best_state)
    pv = predict(gm, val, device, 128)
    T = fit_temperature(pv["logits"], pv["y"])
    p_val = torch.sigmoid(pv["logits"] / T).numpy()
    th = tune_thresholds(pv["y"].numpy(), p_val)
    pt = predict(gm, test, device, 128)
    rep = full_report(pt["y"].numpy(), torch.sigmoid(pt["logits"] / T).numpy(), th)
    out = {"method": args.method, "seed": args.seed, "rounds": args.rounds, "best_round": int(np.argmax(
        [h["val_macro_auroc"] for h in hist]) + 1), "best_val_macro_auroc": best, "temperature": T,
        "history": hist, "test": rep["summary"], "per_class": rep["per_class"]}
    path = ROOT / "paper" / ("scripts/debug" if args.debug else "scripts") / f"fl_{args.method}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print("TEST", {k: round(v, 4) for k, v in rep["summary"].items()}, flush=True)


if __name__ == "__main__":
    main()
