"""Real ECG images: ECG-Image-Database's PTB-XL part (scripts/fetch_ecg_images.py), digitized once.

    python -m cardiomoe.images.real --folds 1 2 3 4 5 6 7 8 9 [--workers 8]

Each PTB-XL record in it was printed and then scanned (colour / grayscale), photographed with phones,
photographed off a monitor, stained, ... (about 10 images per record), and ships with its recorded
12-lead signal. Every image is digitized with the current segmenter and cached per fold in
data/processed/images/real_fold{f}_seg{step}.npz: ecg_id, variant, file, ok, layout, r (mean per-lead
correlation with the recorded signal over the printed samples), quality (digitize._quality score),
error, sec, signal (N, 12, 1000) mV and mask (N, 12, 1000).

Folds 1-8 train the image clients (each client only its own folds), fold 9 validates and fits the
gate, thresholds and triage, fold 10 is the test (images/evaluate.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from cardiomoe.data.datasets import load_meta
from cardiomoe.images.build import img_config, mean_corr
from cardiomoe.utils import repo_path

log = logging.getLogger(__name__)


def real_records(cfg: dict, fold: int) -> list[Path]:
    meta = load_meta(cfg)
    root = repo_path(cfg["images"]["real"]["dir"])
    out = []
    for d in sorted(root.glob("D0_*")):
        k = int(d.name[3:])
        complete = (d / f"{d.name}.hea").exists() and (d / f"{d.name}.dat").exists()   # skip half-downloaded records
        if complete and k in meta.index and meta.loc[k, "strat_fold"] == fold:
            out.append(d)
    return out


def _digitize_file(path: str) -> dict:
    import wfdb

    from cardiomoe.data.signal import preprocess
    from cardiomoe.images.digitize import digitize, load_image

    p = Path(path)
    rid = p.parent.name
    t = time.time()
    out = {"ecg_id": int(rid[3:]), "variant": p.stem.rsplit("_", 1)[1], "fname": p.name}
    try:
        d = digitize(load_image(p.read_bytes(), p.name))
        rec = wfdb.rdrecord(str(p.parent / rid))
        truth, _ = preprocess(rec.p_signal, rec.fs, rec.sig_name)
        out.update(ok=True, signal=d.signal.astype(np.float16), mask=d.mask, layout=d.layout,
                   r=mean_corr(d.signal, truth, d.mask), quality=d.quality.get("score", 0.0), error="")
    except Exception as e:  # noqa: BLE001 - every failure is a counted outcome
        out.update(ok=False, signal=np.zeros((12, 1000), np.float16), mask=np.zeros((12, 1000), bool), layout="",
                   r=float("nan"), quality=0.0, error=str(e)[:200])
    out["sec"] = round(time.time() - t, 2)
    return out


def _init_worker() -> None:
    torch.set_num_threads(1)


def digitize_real(cfg: dict, fold: int, workers: int, limit: int | None = None, max_images: int | None = None) -> dict:
    """Digitize the fold's images once (all, or `max_images` drawn evenly over capture types with a
    fixed seed); cached per segmenter checkpoint."""
    from concurrent.futures import ProcessPoolExecutor

    seg = repo_path(cfg["images"]["segnet"])
    tag = int(torch.load(seg, map_location="cpu", weights_only=False)["step"]) if seg.exists() else "colour"
    cache = repo_path(cfg["images"]["synth_dir"]) / (f"real_fold{fold}_seg{tag}{'_n' + str(limit) if limit else ''}"
                                                     f"{'_max' + str(max_images) if max_images else ''}.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    variants = cfg["images"]["real"]["variants"]
    recs = real_records(cfg, fold)[:limit]
    files = [str(f) for d in recs for f in sorted(d.iterdir())
             if f.suffix.lower() in (".png", ".jpg", ".jpeg") and f.stem.rsplit("_", 1)[1] in variants]
    if max_images and len(files) > max_images:   # round-robin over capture types, random within each
        rng = np.random.default_rng([cfg["images"]["seed"], fold])
        by_var = {}
        for f in rng.permutation(files):
            by_var.setdefault(Path(f).stem.rsplit("_", 1)[1], []).append(str(f))
        pick, queues = [], [by_var[v] for v in sorted(by_var)]
        while len(pick) < max_images:
            for q in queues:
                if q and len(pick) < max_images:
                    pick.append(q.pop())
        files = sorted(pick)
    log.info("digitizing %d real images of %d fold-%d records", len(files), len(recs), fold)
    if not files:
        raise FileNotFoundError(f"no real images for fold {fold}: run scripts/fetch_ecg_images.py --folds {fold}")
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        res = list(ex.map(_digitize_file, files, chunksize=2))
    d = {k: np.array([r[k] for r in res]) for k in ("ecg_id", "variant", "fname", "ok", "layout", "r", "quality", "error", "sec")}
    d["signal"] = np.stack([r["signal"] for r in res])
    d["mask"] = np.stack([r["mask"] for r in res])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **d)
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--debug", action="store_true", help="first 3 records per fold")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = img_config()
    for f in args.folds:
        rc = cfg["images"]["real"]
        cap = rc["val_images"] if f == cfg["fourclient"]["val_fold"] else rc["test_images"]             if f == cfg["fourclient"]["test_fold"] else rc["train_images_per_fold"]
        d = digitize_real(cfg, f, args.workers, 3 if args.debug else None, cap)
        ok = d["ok"].astype(bool)
        print(json.dumps({"fold": f, "images": int(len(ok)), "records": int(len(np.unique(d["ecg_id"]))),
                          "digitized": float(ok.mean()) if len(ok) else None,
                          "median_r": float(np.nanmedian(d["r"][ok])) if ok.any() else None}))


if __name__ == "__main__":
    main()
