"""Synthetic ECG printouts with a pixel-exact trace mask, for training the trace segmenter and for
building digitized training data for the image clients.

    page = random_page(sig, rng)          # Page(image RGB uint8, mask uint8 (1 = waveform ink), info)
    page = degrade(page, rng)             # scan / phone photo / monitor photo / stains / creases / ...

The styles cover both the renderer in render.py ("classic": separate calibration pulse, lead-change
ticks, labels above) and ECG-Image-Kit, which made ECG-Image-Database ("kit": pulse joined to the
trace, thick lead separators, labels below, dashed major lines, header + QR code). Grid colour, line
widths, resolution, fonts and layout are randomised. The mask holds only the waveform, never the
pulse, separators, grid or text, so the segmenter learns to drop those.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from cardiomoe.data.signal import LEADS
from cardiomoe.images.layouts import GRID_3x4, RHYTHM_LEAD
from cardiomoe.images.render import _font

SPEED, GAIN = 25.0, 10.0
GRIDS = [  # (minor, major) colours
    ((250, 205, 205), (240, 120, 120)), ((248, 190, 190), (232, 40, 40)), ((252, 222, 214), (236, 164, 156)),
    ((255, 200, 200), (255, 0, 0)), ((205, 236, 215), (95, 190, 125)), ((215, 228, 245), (110, 150, 210)),
    ((222, 222, 222), (150, 150, 150)), ((250, 222, 190), (235, 150, 80)), ((235, 235, 235), (60, 60, 60)),
]
WORDS = ["ID:", "Name:", "Age:", "Sex:", "Date:", "Weight:", "HR", "bpm", "PR", "QRS", "QT/QTc", "ms", "Male",
         "Female", "Sinus rhythm", "Normal ECG", "Unconfirmed", "Ward", "Room", "Dr.", "Height", "kg", "yrs"]


@dataclass
class Page:
    image: np.ndarray                 # (H, W, 3) uint8
    mask: np.ndarray                  # (H, W) uint8, 1 = waveform ink
    info: dict = field(default_factory=dict)


def _layout_rows(layout: str) -> list[list[tuple[str, float, float]]]:
    """Per printed row: [(lead, t0, t1)] in seconds."""
    if layout == "12x1":
        return [[(l, 0.0, 10.0)] for l in LEADS]
    rows = [[(lead, c * 2.5, (c + 1) * 2.5) for c, lead in enumerate(r)] for r in GRID_3x4]
    if layout == "3x4":
        rows.append([(RHYTHM_LEAD, 0.0, 10.0)])
    return rows


def random_page(sig: np.ndarray, rng: np.random.Generator, layout: str | None = None,
                px_per_mm: float | None = None) -> Page:
    """Draw (12, 1000) mV at 100 Hz as a printed ECG in a random style."""
    if layout is None:
        layout = rng.choice(["3x4", "3x4_no_rhythm", "12x1"], p=[0.7, 0.1, 0.2])
    kit = rng.random() < 0.6
    ppm = float(px_per_mm or rng.uniform(3.5, 7.0))
    ss = 2
    k = ppm * ss
    x0 = rng.uniform(4, 14)                      # calibration pulse start (mm)
    t_x0 = x0 + (5.0 if kit else rng.uniform(7, 9))   # trace start
    W_mm = t_x0 + 250 + rng.uniform(3, 25)
    H_mm = rng.uniform(185, 220) if layout != "12x1" else rng.uniform(200, 280)
    top = rng.uniform(12, 50) if rng.random() < 0.8 else rng.uniform(4, 10)
    bottom = rng.uniform(6, 25)
    rows = _layout_rows(layout)
    row_h = (H_mm - top - bottom) / len(rows)
    W, H = int(W_mm * k), int(H_mm * k)

    paper = tuple(int(v) for v in rng.integers(238, 256, 3)) if rng.random() < 0.5 else (255, 255, 255)
    img = Image.new("RGB", (W, H), paper)
    msk = Image.new("L", (W, H), 0)
    d, dm = ImageDraw.Draw(img), ImageDraw.Draw(msk)

    # grid
    if rng.random() < 0.95:
        minor, major = GRIDS[rng.integers(len(GRIDS))]
        jitter = rng.integers(-12, 13, 3)
        minor = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(minor, jitter))
        major = tuple(int(np.clip(c + j, 0, 255)) for c, j in zip(major, jitter))
        w_min, w_maj = max(1, int(round(rng.uniform(0.05, 0.12) * k))), max(1, int(round(rng.uniform(0.1, 0.25) * k)))
        dashed = kit and rng.random() < 0.7
        draw_minor = rng.random() < 0.9
        g0x, g0y = rng.uniform(0, 1), rng.uniform(0, 1)
        for axis, n in ((0, int(W_mm) + 2), (1, int(H_mm) + 2)):
            for mm in range(n):
                is_major = mm % 5 == 0
                if not is_major and not draw_minor:
                    continue
                p = (mm + (g0x if axis == 0 else g0y)) * k
                col, w = (major, w_maj) if is_major else (minor, w_min)
                if axis == 0:
                    if dashed and is_major:
                        for y in np.arange(0, H, 2 * k):
                            d.line([(p, y), (p, y + k)], fill=col, width=w)
                    else:
                        d.line([(p, 0), (p, H)], fill=col, width=w)
                else:
                    d.line([(0, p), (W, p)], fill=col, width=w)

    ink = (int(rng.integers(0, 60)),) * 3 if rng.random() < 0.85 else (20, 30, int(rng.integers(90, 160)))
    lw = max(1, int(round(rng.uniform(0.15, 0.45) * k)))
    lw_mask = max(lw, int(round(0.8 * ss)))
    font = _font(int(rng.uniform(2.6, 4.2) * k))
    small = _font(int(rng.uniform(2.0, 3.0) * k))

    # header text, QR-like block, footer
    if top > 12:
        for _ in range(int(rng.integers(2, 9))):
            txt = " ".join(rng.choice(WORDS, int(rng.integers(1, 4)))) + " " + str(int(rng.integers(0, 2000)))
            d.text((rng.uniform(2, W_mm * 0.8) * k, rng.uniform(1, top - 5) * k), txt, fill=ink, font=small)
        if kit and rng.random() < 0.7:
            q = rng.uniform(8, 13)
            qx, qy, n = (W_mm - q - rng.uniform(2, 6)) * k, rng.uniform(1, 3) * k, 21
            cells = rng.random((n, n)) < 0.5
            for i in range(n):
                for j in range(n):
                    if cells[i, j]:
                        d.rectangle([qx + j * q * k / n, qy + i * q * k / n, qx + (j + 1) * q * k / n, qy + (i + 1) * q * k / n], fill=(0, 0, 0))
    if rng.random() < 0.8:
        d.text(((t_x0 + rng.uniform(10, 60)) * k, (H_mm - bottom + rng.uniform(1, max(1.5, bottom - 5))) * k),
               "25mm/s    10mm/mV", fill=ink, font=small)

    t = np.arange(sig.shape[1]) / 100.0
    pulse_mode = rng.choice(["kit", "sep", "none"], p=[0.45, 0.45, 0.1]) if not kit else "kit"
    sep_mode = "bar" if kit else rng.choice(["tick", "none"], p=[0.7, 0.3])
    for r, segs in enumerate(rows):
        base = top + (r + rng.uniform(0.5, 0.65)) * row_h
        if pulse_mode == "kit":
            px = [t_x0 - 5, t_x0 - 5, t_x0, t_x0]
            d.line([(px[0] * k, base * k), (px[1] * k, (base - GAIN) * k), (px[2] * k, (base - GAIN) * k),
                    (px[3] * k, base * k)], fill=ink, width=lw)
        elif pulse_mode == "sep":
            xs_ = [x0 - 1, x0, x0, x0 + 5, x0 + 5, x0 + 6]
            ys_ = [base, base, base - GAIN, base - GAIN, base, base]
            d.line([(a * k, b * k) for a, b in zip(xs_, ys_)], fill=ink, width=lw)
        for c, (lead, t0, t1) in enumerate(segs):
            i = LEADS.index(lead)
            a, b = int(round(t0 * 100)), min(int(round(t1 * 100)), sig.shape[1])
            if c > 0 and b - a > 1:
                a0 = a - 1   # join the previous column so the line is continuous, as printers do
            else:
                a0 = a
            xs = (t_x0 + t[a0:b] * SPEED) * k
            ys = (base - sig[i, a0:b] * GAIN) * k
            pts = list(zip(xs.tolist(), ys.tolist()))
            d.line(pts, fill=ink, width=lw, joint="curve")
            dm.line(pts, fill=255, width=lw_mask, joint="curve")
            xl = t_x0 + t0 * SPEED
            if kit:
                d.text(((xl + rng.uniform(0.5, 2)) * k, (base + rng.uniform(3, 6)) * k), lead, fill=ink, font=font)
            else:
                d.text(((xl + 1.2) * k, (base - row_h * 0.45) * k), lead, fill=ink, font=font)
            if c > 0:
                if sep_mode == "bar":
                    d.line([(xl * k, (base - 3) * k), (xl * k, (base + 3) * k)], fill=ink, width=max(lw * 2, int(0.6 * k)))
                elif sep_mode == "tick":
                    d.line([(xl * k, (base - 4) * k), (xl * k, (base + 4) * k)], fill=ink, width=lw)
        if kit and layout == "3x4" and r == len(rows) - 1:
            d.text(((t_x0 + 1) * k, (base + 4) * k), RHYTHM_LEAD, fill=ink, font=font)

    img = img.resize((W // ss, H // ss), Image.LANCZOS)
    msk = msk.resize((W // ss, H // ss), Image.BILINEAR)
    info = {"layout": layout, "kit": bool(kit), "px_per_mm": ppm, "trace_x0_px": t_x0 * ppm,
            "trace_x1_px": (t_x0 + 250) * ppm}
    return Page(np.asarray(img).copy(), (np.asarray(msk) > 100).astype(np.uint8), info)


# --------------------------------------------------------------------------- degradations
def _blobs(shape, rng, n, smin, smax):
    h, w = shape
    a = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(a)
    for _ in range(n):
        cx, cy, rx, ry = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(smin, smax) * w, rng.uniform(smin, smax) * w
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=int(rng.integers(80, 256)))
    return np.asarray(a.filter(ImageFilter.GaussianBlur(rng.uniform(0.01, 0.04) * w))).astype(np.float32) / 255


def degrade(page: Page, rng: np.random.Generator, p_clean: float = 0.15) -> Page:
    """Make the printout look scanned, photographed or damaged. Geometry is applied to the mask too."""
    import cv2

    if rng.random() < p_clean:
        return page
    a = page.image.astype(np.float32)
    m = page.mask.astype(np.float32)
    h, w = m.shape
    info = dict(page.info)

    # paper damage: stains, creases, handwriting (handwriting is ink that is NOT in the mask)
    if rng.random() < 0.3:
        s = _blobs((h, w), rng, int(rng.integers(1, 5)), 0.03, 0.15)
        col = rng.choice([[0.75, 0.55, 0.35], [0.85, 0.75, 0.45], [0.7, 0.4, 0.45], [0.55, 0.55, 0.5]])
        a = a * (1 - s[..., None] * rng.uniform(0.2, 0.6) * (1 - np.array(col)[None, None]))
    if rng.random() < 0.25:
        for _ in range(int(rng.integers(1, 4))):
            p0, p1 = rng.uniform(0, 1, 2) * [w, h], rng.uniform(0, 1, 2) * [w, h]
            ln = np.zeros((h, w), np.float32)
            cv2.line(ln, tuple(int(v) for v in p0), tuple(int(v) for v in p1), 1.0, int(rng.integers(1, 4)))
            ln = cv2.GaussianBlur(ln, (0, 0), rng.uniform(1, 4))
            a = a * (1 - rng.uniform(0.05, 0.25) * ln[..., None])
    if rng.random() < 0.15:
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(img)
        col = [(30, 40, 140), (20, 20, 20), (150, 20, 20)][int(rng.integers(3))]
        for _ in range(int(rng.integers(1, 4))):
            x, y = rng.uniform(0.1, 0.8) * w, rng.uniform(0.05, 0.9) * h
            pts = [(x, y)]
            for _ in range(int(rng.integers(8, 30))):
                x += rng.normal(6, 4) * w / 1500
                y += rng.normal(0, 8) * w / 1500
                pts.append((x, y))
            d.line(pts, fill=col, width=max(1, int(w / 700)), joint="curve")
        a = np.asarray(img).astype(np.float32)

    mode = rng.choice(["scan", "photo", "monitor"], p=[0.45, 0.45, 0.1])
    info["degrade"] = str(mode)
    if mode == "monitor":  # screen: blue-ish tint, moire, lower contrast
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        f = rng.uniform(0.3, 1.2)
        moire = 1 + rng.uniform(0.02, 0.08) * np.sin(f * xx + 0.7 * f * yy)
        a = (a * rng.uniform(0.75, 0.95) + rng.uniform(10, 40)) * moire[..., None] * np.array([0.95, 0.98, 1.05])

    if mode in ("photo", "monitor"):
        # place the page on a background, curl it, then project with a random perspective
        pad = rng.uniform(0.02, 0.2)
        H2, W2 = int(h * (1 + 2 * pad)), int(w * (1 + 2 * pad))
        bg = np.empty((H2, W2, 3), np.float32)
        base_col = rng.uniform(30, 200, 3)
        bg[:] = base_col
        grain = cv2.resize(rng.normal(0, 1, (max(2, H2 // 40), max(2, W2 // 4))).astype(np.float32), (W2, H2))
        bg += grain[..., None] * rng.uniform(0, 25)
        oy, ox = (H2 - h) // 2, (W2 - w) // 2
        bg[oy:oy + h, ox:ox + w] = a
        mm = np.zeros((H2, W2), np.float32)
        mm[oy:oy + h, ox:ox + w] = m
        a, m, h, w = bg, mm, H2, W2
        # curl / bend: smooth vertical displacement varying along x (and a little horizontal)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        amp = rng.uniform(0, 0.025) * h
        ph = rng.uniform(0, np.pi)
        mapy = yy + amp * np.sin(np.pi * xx / w + ph) * (0.5 + yy / h)
        mapx = xx + rng.uniform(0, 0.01) * w * np.sin(np.pi * yy / h + rng.uniform(0, np.pi))
        a = cv2.remap(a, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = cv2.remap(m, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        j = rng.uniform(0, 0.07)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = (src + rng.uniform(-j, j, (4, 2)) * [w, h]).astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        a = cv2.warpPerspective(a, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpPerspective(m, M, (w, h))
        # light: gradient + soft shadow
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        light = 1 - rng.uniform(0, 0.35) * ((xx / w - rng.uniform()) ** 2 + (yy / h - rng.uniform()) ** 2)
        if rng.random() < 0.4:
            light *= 1 - rng.uniform(0.15, 0.45) * _blobs((h, w), rng, 1, 0.1, 0.4)
        a = a * light[..., None] * rng.uniform(0.85, 1.08, 3)[None, None]
    else:  # scan: small rotation, slight colour shift
        ang = rng.uniform(-3, 3)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        a = cv2.warpAffine(a, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        m = cv2.warpAffine(m, M, (w, h))
        a = a * rng.uniform(0.9, 1.05, 3)[None, None] + rng.uniform(-10, 10)

    if rng.random() < 0.25:  # grayscale scan / photo
        g = a.mean(2, keepdims=True)
        a = np.repeat(g, 3, axis=2)
    if rng.random() < 0.6:
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.3, 1.3))
    a = a + rng.normal(0, rng.uniform(0, 7), a.shape)
    a = np.clip(a, 0, 255).astype(np.uint8)
    # resolution change, then JPEG
    s = rng.uniform(0.75, 1.15)
    if abs(s - 1) > 0.02:
        a = cv2.resize(a, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        m = cv2.resize(m, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        info["px_per_mm"] = info["px_per_mm"] * s
    if rng.random() < 0.8:
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=int(rng.integers(35, 95)))
        a = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return Page(np.ascontiguousarray(a), (m > 0.4).astype(np.uint8), info)
