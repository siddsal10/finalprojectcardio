"""Trace segmenter: a small U-Net that marks waveform pixels on a photo / scan of a printed ECG.

    python -m cardiomoe.images.segnet [--steps 6000] [--debug]

Trained only on synthetic printouts (synth.py) of PTB-XL folds 1-8, drawn on the fly in worker
processes, so it never sees fold 9/10 signals or any real image. It removes everything that is not
the waveform (grid of any colour, text, calibration pulses, lead separators, stains, handwriting),
which makes the digitizer independent of grid colour and scan quality.

Inference works at a fixed scale: the image is resized so its long side is WORK_SIDE pixels.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset

from cardiomoe.utils import load_config, repo_path, set_seed

log = logging.getLogger(__name__)
WORK_SIDE = 1600
CROP = 224
MODEL_PATH = "outputs/images/segnet.pt"


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """U-Net that works at half resolution (CPU budget) plus a thin full-resolution path, so the
    output keeps pixel accuracy: stem (full) -> stride-2 conv -> U-Net (1/2 .. 1/16) -> fuse (full)."""

    def __init__(self, ch=(16, 32, 48, 64), stem: int = 8):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, stem, 3, padding=1, bias=False), nn.BatchNorm2d(stem), nn.ReLU(inplace=True))
        self.to_half = nn.Sequential(nn.Conv2d(stem, ch[0], 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(ch[0]),
                                     nn.ReLU(inplace=True))
        self.down = nn.ModuleList([_block(ch[0], ch[0])] + [_block(a, b) for a, b in zip(ch[:-1], ch[1:])])
        self.up = nn.ModuleList(_block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(len(ch) - 1)))
        self.fuse = nn.Sequential(nn.Conv2d(ch[0] + stem, stem, 3, padding=1, bias=False), nn.BatchNorm2d(stem),
                                  nn.ReLU(inplace=True), nn.Conv2d(stem, 1, 1))

    def forward(self, x):
        f = self.stem(x)
        h = self.to_half(f)
        skips = []
        for i, b in enumerate(self.down):
            h = b(h if i == 0 else F.max_pool2d(h, 2))
            skips.append(h)
        for b, s in zip(self.up, reversed(skips[:-1])):
            h = F.interpolate(h, size=s.shape[-2:], mode="bilinear", align_corners=False)
            h = b(torch.cat([h, s], 1))
        h = F.interpolate(h, size=f.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([h, f], 1))


def to_input(a: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> (1, 3, H, W) float, per-image contrast normalised."""
    x = torch.from_numpy(np.ascontiguousarray(a)).float().permute(2, 0, 1)[None] / 255.0
    return (x - x.mean()) / (x.std() + 1e-6)


# --------------------------------------------------------------------------- data
class PageStream(IterableDataset):
    """Endless stream of (crop, mask) pairs cut from freshly drawn synthetic printouts."""

    def __init__(self, rows: np.ndarray, signals_path: str, seed: int, crops_per_page: int = 10):
        self.rows, self.signals_path, self.seed, self.k = rows, signals_path, seed, crops_per_page

    def __iter__(self):
        from cardiomoe.images.synth import degrade, random_page

        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, wi.id if wi else 0, os.getpid()])
        sig = np.load(self.signals_path, mmap_mode="r")
        while True:
            page = degrade(random_page(np.asarray(sig[rng.choice(self.rows)], dtype=np.float32), rng), rng)
            a, m = _work_scale(page.image, page.mask, rng.uniform(0.8, 1.25))
            h, w = m.shape
            if h < CROP or w < CROP:
                continue
            ys, xs = np.nonzero(m)
            for j in range(self.k):
                if len(ys) and j % 3 != 2:      # 2 of 3 crops centred on a trace pixel
                    q = rng.integers(len(ys))
                    y0 = int(np.clip(ys[q] - rng.integers(0, CROP), 0, h - CROP))
                    x0 = int(np.clip(xs[q] - rng.integers(0, CROP), 0, w - CROP))
                else:
                    y0, x0 = int(rng.integers(0, h - CROP + 1)), int(rng.integers(0, w - CROP + 1))
                yield (to_input(a[y0:y0 + CROP, x0:x0 + CROP])[0],
                       torch.from_numpy(m[y0:y0 + CROP, x0:x0 + CROP].astype(np.float32))[None])


def _work_scale(a: np.ndarray, m: np.ndarray | None = None, jitter: float = 1.0):
    import cv2

    s = WORK_SIDE * jitter / max(a.shape[:2])
    size = (max(1, round(a.shape[1] * s)), max(1, round(a.shape[0] * s)))
    a = cv2.resize(a, size, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if m is not None:
        m = (cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_LINEAR) > 0.35).astype(np.uint8)
    return a, m


def loss_fn(logits, y):
    bce = F.binary_cross_entropy_with_logits(logits, y, pos_weight=torch.tensor(4.0))
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * y).sum() + 1) / (p.sum() + y.sum() + 1)
    return bce + dice


# --------------------------------------------------------------------------- inference
_MODEL = None


def load(path: str | None = None) -> UNet | None:
    global _MODEL
    if _MODEL is None:
        p = repo_path(path or MODEL_PATH)
        if not p.exists():
            return None
        ck = torch.load(p, map_location="cpu", weights_only=False)
        _MODEL = UNet(tuple(ck["channels"]), ck["stem"])
        _MODEL.load_state_dict(ck["state_dict"])
        _MODEL.eval()
    return _MODEL


@torch.no_grad()
def segment(a: np.ndarray, model: UNet | None = None) -> tuple[np.ndarray, np.ndarray]:
    """RGB image -> (image at working scale, trace probability map at the same scale)."""
    model = model or load()
    if model is None:
        raise FileNotFoundError(MODEL_PATH)
    a, _ = _work_scale(a)
    h, w = a.shape[:2]
    x = to_input(a)
    ph, pw = (-h) % 16, (-w) % 16
    x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    p = torch.sigmoid(model(x))[0, 0, :h, :w].numpy()
    return a, p


# --------------------------------------------------------------------------- training
def dice(p: np.ndarray, m: np.ndarray) -> float:
    b = p > 0.5
    return float(2 * (b & (m > 0)).sum() / max(b.sum() + (m > 0).sum(), 1))


def val_pages(rows, sig, n, seed):
    from cardiomoe.images.synth import degrade, random_page

    rng = np.random.default_rng(seed)
    out = []
    for r in rng.choice(rows, n, replace=False):
        pg = degrade(random_page(np.asarray(sig[r], dtype=np.float32), rng), rng)
        out.append(_work_scale(pg.image, pg.mask))
    return out


def train(steps: int, seed: int, debug: bool, workers: int, out_path: str) -> dict:
    set_seed(seed)
    cfg = load_config("fourclient")
    from cardiomoe.data.datasets import load_meta
    from cardiomoe.data.ptbxl import processed_dir

    meta = load_meta(cfg, debug)
    train_folds = sorted({f for v in cfg["fourclient"]["partition"].values() for f in v})
    rows = meta.loc[meta.strat_fold.isin(train_folds), "row"].to_numpy()
    vrows = meta.loc[meta.strat_fold == cfg["fourclient"]["val_fold"], "row"].to_numpy()
    sp = str(processed_dir(cfg, debug) / "ptbxl_signals.npy")
    sig = np.load(sp, mmap_mode="r")
    vals = val_pages(vrows, sig, 4 if debug else 24, seed + 1)
    bs = 8 if debug else 10
    dl = DataLoader(PageStream(rows, sp, seed), batch_size=bs, num_workers=workers, persistent_workers=workers > 0,
                    prefetch_factor=4 if workers else None)
    model = UNet()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    hist, t0, run = [], time.time(), 0.0
    out = repo_path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    every = 20 if debug else 500
    for step, (x, y) in enumerate(dl, 1):
        model.train()
        loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0:
            log.info("step %d loss %.4f (%.1f min)", step, run, (time.time() - t0) / 60)
        if step % every == 0 or step == steps:
            model.eval()
            d = float(np.mean([dice(segment(a, model)[1], m) for a, m in vals]))
            rec = {"step": step, "loss": round(run, 4), "val_dice": round(d, 4), "min": round((time.time() - t0) / 60, 1)}
            hist.append(rec)
            log.info("%s", json.dumps(rec))
            if d > best:
                best = d
                torch.save({"state_dict": model.state_dict(), "channels": [16, 32, 48, 64], "stem": 8, "step": step,
                            "val_dice": d, "work_side": WORK_SIDE, "seed": seed}, out)
        if step >= steps:
            break
    (out.parent / "segnet_history.jsonl").write_text("\n".join(json.dumps(h) for h in hist) + "\n")
    return {"best_val_dice": best, "steps": steps, "minutes": round((time.time() - t0) / 60, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - args.workers))   # the U-Net, not page drawing, is the bottleneck
    steps = 40 if args.debug else args.steps
    path = "outputs/images/debug/segnet.pt" if args.debug else MODEL_PATH
    print(json.dumps(train(steps, args.seed, args.debug, 2 if args.debug else args.workers, path)))


if __name__ == "__main__":
    main()
