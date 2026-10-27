"""Image clients: fine-tune each signal client on digitized real ECG pictures of its own folds, then
fit the gate, thresholds and triage bands on the real fold-9 pictures exactly as for the signals.

    python scripts/fetch_ecg_images.py --folds 1 2 3 4 5 6 7 8 9 10   # real scans / photos
    python -m cardiomoe.images.real --folds 1 2 3 4 5 6 7 8 9          # digitize them once
    python -m cardiomoe.images.build --folds 1 2 3 4 5 6 7 8           # optional synthetic printouts
    python -m cardiomoe.images.finetune [--clients A B C D] [--seeds 42] [--debug]

A printout shows most leads for only 2.5 s, and digitizing adds its own noise, so the signal clients
see a different input than they were trained on. Each image client starts from the signal client of
the same seed and trains on the same PTB-XL folds (A: 1-2, B: 3-4, C: 5-6, D: 7-8):
- real pictures (ECG-Image-Database: scans, phone photos, monitor photos, stained pages) of the
  records of those folds, weighted `real_weight` times, plus
- synthetic printouts of every record of those folds (build.py), if built, for volume.
Early stopping, temperature, gate, thresholds and triage use the real fold-9 pictures only, so the
precision / recall / accuracy >= 0.90 triage targets are fitted on what users upload. Input:
per-lead z-score over the printed samples (layouts.masked_normalize), 0 elsewhere. Outputs go to
outputs/images/seed{s}/ in the same format as outputs/fourclient/, so aggregate.py and triage.py run
on them unchanged (a record's pictures share its patient, so their CV keeps them together).
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

from cardiomoe.data.datasets import YCOLS, load_meta
from cardiomoe.data.signal import augment as augment_signal
from cardiomoe.fourclient import aggregate, triage
from cardiomoe.fourclient.train import CLIENTS, build, fc_config, load_client, run_dir, spec
from cardiomoe.fusion.calibration import fit_temperature
from cardiomoe.images.build import img_config, synth_path
from cardiomoe.images.layouts import masked_normalize
from cardiomoe.images.real import digitize_real
from cardiomoe.train.common import predict, save_checkpoint, train_model
from cardiomoe.utils import get_device, set_seed

log = logging.getLogger(__name__)


KEYS = ("ecg_id", "signal", "mask")


def load_synth(cfg: dict, folds: list[int], debug: bool) -> dict | None:
    paths = [synth_path(cfg, f, debug) for f in folds]
    if not all(p.exists() for p in paths):
        return None
    parts = [np.load(p) for p in paths]
    keep = np.concatenate([p["ok"] for p in parts])
    return {k: np.concatenate([p[k] for p in parts])[keep] for k in KEYS}


def load_real(cfg: dict, folds: list[int], workers: int = 8, limit: int | None = None,
              per_fold: int | None = None) -> dict:
    parts = [digitize_real(cfg, f, workers, limit, per_fold) for f in folds]
    keep = np.concatenate([p["ok"] for p in parts]).astype(bool)
    return {k: np.concatenate([p[k] for p in parts])[keep] for k in KEYS}


class DigitizedDataset(Dataset):
    """Digitized pictures (pictures the digitizer rejected are left out). `parts` = [(dict, repeat)]."""

    def __init__(self, cfg: dict, parts: list[tuple[dict, int]], *, train: bool = False, debug: bool = False):
        parts = [(d, r) for d, r in parts if d is not None and len(d["ecg_id"])]
        cat = lambda k: np.concatenate([np.concatenate([d[k]] * r) for d, r in parts])
        ids = cat("ecg_id")
        self.keys = [int(k) for k in ids]
        self.signal, self.mask = cat("signal"), cat("mask")
        self.y = load_meta(cfg, debug).loc[self.keys, YCOLS].to_numpy(np.float32)
        self.train = train
        self._rng = None

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int) -> dict:
        m = self.mask[i]
        x = masked_normalize(self.signal[i].astype(np.float32), m)
        if self.train:
            if self._rng is None:
                self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            x = augment_signal(x, self._rng) * m   # unprinted samples stay 0
        return {"key": self.keys[i], "y": torch.from_numpy(self.y[i]), "signal": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))}


def train_one(cfg: dict, client: str, seed: int, debug: bool) -> dict:
    tcfg = cfg["images"]["finetune"]
    fc = cfg["fourclient"]
    set_seed(seed)
    device = get_device(cfg["device"], debug)
    folds = spec(cfg, client)[1]
    limit = 3 if debug else None
    rc = cfg["images"]["real"]
    real_tr = load_real(cfg, folds, limit=limit, per_fold=rc["train_images_per_fold"])
    synth_tr = load_synth(cfg, folds, debug) if cfg["images"]["use_synthetic"] else None
    train_ds = DigitizedDataset(cfg, [(real_tr, cfg["images"]["real_weight"]), (synth_tr, 1)], train=True, debug=debug)
    val_ds = DigitizedDataset(cfg, [(load_real(cfg, [fc["val_fold"]], limit=limit, per_fold=rc["val_images"]), 1)], debug=debug)
    n_real, n_synth = len(real_tr["ecg_id"]), (len(synth_tr["ecg_id"]) if synth_tr else 0)
    try:
        model = load_client(fc_config(), client, seed, debug, "cpu")
        init = "signal client"
    except FileNotFoundError:
        model = build(cfg, client)
        init = "random (no signal client found)"
    model.train()
    log.info("image client %s (%s): folds %s, train %d real pictures (x%d) + %d synthetic printouts, "
             "val %d real fold-9 pictures, init %s", client, model.arch, folds, n_real, cfg["images"]["real_weight"],
             n_synth, len(val_ds), init)
    out = run_dir(cfg, seed, debug) / client
    hist = out / "history.jsonl"
    hist.unlink(missing_ok=True)
    model, info = train_model(model, train_ds, val_ds, tcfg, device, debug, hist)
    val = predict(model, val_ds, device, tcfg["eval_batch_size"])
    T = fit_temperature(val["logits"], val["y"])
    model.temperature.fill_(T)
    model.cpu()
    save_checkpoint(model, out / "model.pt", client=client, arch=model.arch, folds=folds, seed=seed, temperature=T,
                    init=init, best_val_macro_auroc=info["best_val_macro_auroc"])
    torch.save({"keys": val["keys"], "probs": torch.sigmoid(val["logits"] / T), "y": val["y"]}, out / "val_preds.pt")
    meta = {"client": client, "arch": model.arch, "folds": folds, "n_train_real_pictures": n_real,
            "real_weight": cfg["images"]["real_weight"], "n_train_synthetic": n_synth, "n_val_real_pictures": len(val_ds), "init": init,
            "temperature": T, "best_val_macro_auroc": info["best_val_macro_auroc"], "epochs_run": len(info["history"])}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log.info("image client %s done: real fold-9 pictures macro AUROC %.4f, T=%.3f", client, info["best_val_macro_auroc"], T)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", nargs="+", default=CLIENTS, choices=CLIENTS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = img_config()
    for seed in args.seeds:
        for c in args.clients:
            if (run_dir(cfg, seed, args.debug) / c / "meta.json").exists() and not args.retrain:
                log.info("image client %s seed %d already trained, skipping", c, seed)
                continue
            print(json.dumps(train_one(cfg, c, seed, args.debug)))
        if all((run_dir(cfg, seed, args.debug) / c / "val_preds.pt").exists() for c in CLIENTS):
            agg = aggregate.run(cfg, seed, args.debug)          # gate + thresholds on real fold-9 pictures
            bands = triage.fit_seed(cfg, seed, args.debug)       # triage bands on real fold-9 pictures
            print(json.dumps({"seed": seed, "selected": agg["selected"], "cv": agg["cv"],
                              "triage_modes": {c: b["mode"] for c, b in bands["bands"].items()}}, indent=2))


if __name__ == "__main__":
    main()
