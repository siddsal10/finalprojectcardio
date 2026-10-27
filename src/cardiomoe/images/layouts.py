"""Printed ECG layouts and the matching model input.

12x1           every lead for the full 10 s (all information is on the page)
3x4            the standard clinical printout: 3 rows x 4 columns of 2.5 s each, plus a 10 s lead-II rhythm strip
3x4_no_rhythm  the same without the rhythm strip

For the 3x4 layout a lead is only visible during its 2.5 s column, so the model input keeps that
window (z-scored on the window itself) and is zero elsewhere. Training (images/finetune.py) and inference
(digitize.py + web) build the input the same way through `masked_normalize`.
"""
from __future__ import annotations

import numpy as np

from cardiomoe.data.signal import LEADS, N_SAMPLES

FS = 100
GRID_3x4 = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
RHYTHM_LEAD = "II"
LAYOUTS = ("12x1", "3x4", "3x4_no_rhythm")


def visible_mask(layout: str, n: int = N_SAMPLES) -> np.ndarray:
    """(12, n) bool: which samples of each lead are printed."""
    if layout == "12x1":
        return np.ones((12, n), dtype=bool)
    if layout not in ("3x4", "3x4_no_rhythm"):
        raise ValueError(layout)
    m = np.zeros((12, n), dtype=bool)
    w = n // 4
    for row in GRID_3x4:
        for c, lead in enumerate(row):
            m[LEADS.index(lead), c * w:(c + 1) * w] = True
    if layout == "3x4":
        m[LEADS.index(RHYTHM_LEAD)] = True  # the rhythm strip shows lead II for 10 s
    return m


def masked_normalize(sig: np.ndarray, mask: np.ndarray, clip: float = 10.0, eps: float = 1e-6) -> np.ndarray:
    """Per-lead z-score over the visible samples only; invisible samples are 0."""
    out = np.zeros_like(sig, dtype=np.float32)
    for i in range(sig.shape[0]):
        v = mask[i]
        if v.sum() < 2:
            continue
        x = sig[i, v]
        sd = x.std()
        if sd > eps:
            out[i, v] = np.clip((x - x.mean()) / sd, -clip, clip)
    return out
