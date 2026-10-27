"""Render a (12, 1000) ECG in mV as a printed ECG on standard paper (25 mm/s, 10 mm/mV).

    img = render_ecg(sig, layout="3x4")          # PIL.Image, clean "scan / PDF export"
    img = photo_like(img, rng)                   # tilt, uneven light, blur, noise, JPEG

The renderer exists to test the digitizer against signals whose truth is known; it draws what a
cardiograph prints: pink grid (1 mm / 5 mm), a 1 mV calibration pulse per row, lead labels.
"""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cardiomoe.data.signal import LEADS
from cardiomoe.images.layouts import GRID_3x4, RHYTHM_LEAD

SPEED, GAIN = 25.0, 10.0          # mm/s, mm/mV
X0, WIDTH = 12.0, 250.0           # trace start and length (mm)
PAPER = (255, 249, 247)
MINOR, MAJOR = (247, 208, 204), (236, 164, 156)
INK = (20, 20, 24)


def geometry(layout: str) -> dict:
    if layout == "12x1":
        return {"rows": 12, "row_h": 20.0, "top": 9.0, "height": 9.0 + 12 * 20.0 + 6}
    return {"rows": 4, "row_h": 32.0, "top": 9.0, "height": 9.0 + 4 * 32.0 + 6}


@lru_cache(maxsize=4)
def _font(px: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def render_ecg(sig: np.ndarray, layout: str = "12x1", px_per_mm: float = 6.0, ss: int = 2) -> Image.Image:
    """Draw the printout. ss = supersampling factor for smooth lines."""
    g = geometry(layout)
    k = px_per_mm * ss
    W, H = int((X0 + WIDTH + 6) * k), int(g["height"] * k)
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)
    for mm in range(int(W / k) + 1):  # grid
        major = mm % 5 == 0
        d.line([(mm * k, 0), (mm * k, H)], fill=MAJOR if major else MINOR, width=max(1, int(ss * (1.4 if major else 0.7))))
    for mm in range(int(H / k) + 1):
        major = mm % 5 == 0
        d.line([(0, mm * k), (W, mm * k)], fill=MAJOR if major else MINOR, width=max(1, int(ss * (1.4 if major else 0.7))))
    font = _font(int(3.2 * k))
    d.text((X0 * k, 2.5 * k), "25 mm/s   10 mm/mV", fill=INK, font=_font(int(2.6 * k)))
    lw = max(1, int(round(0.32 * k)))
    n = sig.shape[1]
    t = np.arange(n) / 100.0

    def trace(lead: str, t0: float, t1: float, base: float, label: bool = True):
        i = LEADS.index(lead)
        a, b = int(round(t0 * 100)), int(round(t1 * 100))
        xs = (X0 + t[a:b] * SPEED) * k
        ys = (base - sig[i, a:b] * GAIN) * k
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=INK, width=lw, joint="curve")
        if label:
            d.text(((X0 + t0 * SPEED + 1.2) * k, (base - g["row_h"] * 0.44) * k), lead, fill=INK, font=font)

    def pulse(base: float):  # 1 mV calibration pulse, 5 mm wide, before the trace
        x = [3.0, 4.0, 4.0, 9.0, 9.0, 10.0]
        y = [base, base, base - GAIN, base - GAIN, base, base]
        d.line([(xx * k, yy * k) for xx, yy in zip(x, y)], fill=INK, width=lw)

    rows = g["rows"]
    for r in range(rows):
        base = g["top"] + (r + 0.55) * g["row_h"]
        pulse(base)
        if layout == "12x1":
            trace(LEADS[r], 0, 10, base)
        elif r < 3:
            for c, lead in enumerate(GRID_3x4[r]):
                trace(lead, c * 2.5, (c + 1) * 2.5, base)
                if c:  # lead-change tick
                    x = (X0 + c * 2.5 * SPEED) * k
                    d.line([(x, (base - 4) * k), (x, (base + 4) * k)], fill=INK, width=lw)
        else:
            trace(RHYTHM_LEAD, 0, 10, base)
    return img.resize((W // ss, H // ss), Image.LANCZOS)


def photo_like(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Simulate a phone photo / poor scan: tilt, scale, uneven light, colour cast, blur, noise, JPEG."""
    import cv2

    a = np.asarray(img).astype(np.float32)
    h, w = a.shape[:2]
    ang, sc = rng.uniform(-2.5, 2.5), rng.uniform(0.88, 0.98)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    a = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(235, 232, 228))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 1.0 - rng.uniform(0.05, 0.25) * ((xx / w) * rng.uniform(-1, 1) + (yy / h) * rng.uniform(-1, 1)) ** 2
    a = a * light[..., None] * rng.uniform(0.92, 1.04, size=3)[None, None]
    a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.5, 1.1))
    a = np.clip(a + rng.normal(0, 5, a.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", quality=int(rng.integers(55, 80)))
    buf.seek(0)
    return Image.open(buf).convert("RGB")
