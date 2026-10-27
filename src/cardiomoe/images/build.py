"""Build the image clients' data: every PTB-XL record printed as a synthetic ECG (random style),
degraded like a scan / phone photo / monitor photo, and digitized back with the real digitizer.

    python -m cardiomoe.images.build [--folds 1 2 ... 10] [--workers 8] [--debug]

Writes data/processed/images/synth_fold{f}.npz per fold: ecg_id, signal (N, 12, 1000) float16 mV,
mask (N, 12, 1000) bool (printed samples), layout, capture, ok (digitized without error) and r
(mean per-lead correlation with the true signal over printed samples, a digitizer quality check).
The style and degradation of record k are drawn from rng(seed, k), so files are reproducible.
Fold 10 is built only for the synthetic test in images/evaluate.py.

Training folds (1-8) are traced from the printout's exact waveform mask instead of the segmenter's
(the segmenter costs ~8 s per page on one CPU core, ~5 h for all of them). The page geometry,
damage, layout and tracing noise are the same; only segmentation errors are missing. Folds 9 and
10 run the full pipeline (segmenter included), so early stopping, thresholds, the gate, triage
and every reported number see exactly what users' pictures go through.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from cardiomoe.utils import load_config, repo_path

log = logging.getLogger(__name__)


def img_config() -> dict:
    return load_config("fourclient", "images")


def synth_path(cfg: dict, fold: int, debug: bool):
    d = repo_path(cfg["images"]["synth_dir"]) / ("debug" if debug else "")
    d.mkdir(parents=True, exist_ok=True)
    return d / f"synth_fold{fold}.npz"


def mean_corr(est: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    cs = []
    for i in range(12):
        a, b = est[i, mask[i]], truth[i, mask[i]]
        cs.append(float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-6 and b.std() > 1e-6 else 0.0)
    return float(np.mean(cs))


_SIG = None


def _init(signals_path: str) -> None:
    global _SIG
    import torch

    torch.set_num_threads(1)
    _SIG = np.load(signals_path, mmap_mode="r")


def _one(args) -> tuple:
    ecg_id, row, seed, given_mask = args
    from cardiomoe.images.digitize import digitize
    from cardiomoe.images.synth import degrade, random_page

    rng = np.random.default_rng([seed, int(ecg_id)])
    truth = np.asarray(_SIG[row], dtype=np.float32)
    page = degrade(random_page(truth, rng), rng)
    capture = page.info.get("degrade", "clean")
    try:
        d = digitize(page.image, trace_mask=page.mask if given_mask else None)
        return ecg_id, d.signal.astype(np.float16), d.mask, d.layout, capture, True, mean_corr(d.signal, truth, d.mask)
    except Exception:
        z = np.zeros((12, 1000), np.float16)
        return ecg_id, z, np.zeros((12, 1000), bool), page.info["layout"], capture, False, float("nan")


def build_fold(cfg: dict, fold: int, workers: int, debug: bool) -> dict:
    from cardiomoe.data.datasets import load_meta
    from cardiomoe.data.ptbxl import processed_dir

    meta = load_meta(cfg, debug)
    m = meta[meta.strat_fold == fold]
    seed = cfg["images"]["seed"]
    given = fold not in (cfg["fourclient"]["val_fold"], cfg["fourclient"]["test_fold"])
    jobs = [(int(k), int(r), seed, given) for k, r in zip(m.index, m["row"])]
    sp = str(processed_dir(cfg, debug) / "ptbxl_signals.npy")
    t0 = time.time()
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(sp,)) as ex:
        res = []
        for i, r in enumerate(ex.map(_one, jobs, chunksize=4), 1):
            res.append(r)
            if i % 200 == 0:
                log.info("fold %d: %d/%d (%.1f min)", fold, i, len(jobs), (time.time() - t0) / 60)
    ids, sig, mask, layout, capture, ok, r = map(list, zip(*res))
    out = synth_path(cfg, fold, debug)
    np.savez_compressed(out, mask_source=np.array("printout" if given else "segmenter"), ecg_id=np.array(ids), signal=np.stack(sig), mask=np.stack(mask), layout=np.array(layout),
                        capture=np.array(capture), ok=np.array(ok), r=np.array(r))
    ok_ = np.array(ok)
    rr = np.array(r)[ok_]
    summary = {"fold": fold, "n": len(ids), "mask": "printout" if given else "segmenter", "ok": float(ok_.mean()), "median_r": float(np.median(rr)) if len(rr) else None,
               "minutes": round((time.time() - t0) / 60, 1)}
    log.info("%s", json.dumps(summary))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = img_config()
    for f in args.folds:
        if synth_path(cfg, f, args.debug).exists() and not args.rebuild:
            log.info("fold %d already built, skipping", f)
            continue
        print(json.dumps(build_fold(cfg, f, 2 if args.debug else args.workers, args.debug)))


if __name__ == "__main__":
    main()
