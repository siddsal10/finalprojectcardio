"""Digitize a printed 12-lead ECG (scan, PDF export, phone photo, photo of a monitor) into a (12, 1000) signal.

Supports the common layouts on standard ECG paper at 25 mm/s and 10 mm/mV:
    12x1           12 rows, every lead for 10 s
    3x4            3 rows of 4 leads x 2.5 s (I aVR V1 V4 / II aVL V2 V5 / III aVF V3 V6) + a lead-II rhythm strip
    3x4_no_rhythm  the same without the rhythm strip

Steps:
1. The trace segmenter (segnet.py, a U-Net trained on synthetic printouts) marks waveform pixels and
   drops grid (any colour), text, calibration pulses, separators, stains and handwriting. Without a
   trained segmenter a colour rule (dark, unsaturated ink on a red/pink grid) is used instead.
2. Straighten the page (projection-profile skew search on the trace mask).
3. Find the rows from the trace profile and track each baseline across vertical strips, so a curled
   or obliquely photographed page still works.
4. Follow each trace column by column (never crossing into a neighbouring row).
5. Each row spans 10 s (250 mm): its first and last trace columns fix the time axis and the scale
   (pixels per mm), so the grid does not have to be measured.
6. Split rows into leads by layout, resample to 100 Hz, remove the baseline per printed segment.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from cardiomoe.data.signal import LEADS
from cardiomoe.images.layouts import GRID_3x4, RHYTHM_LEAD, visible_mask

SPEED, GAIN = 25.0, 10.0
STRIP_MM = 250.0
N_STRIPS = 10


class DigitizeError(ValueError):
    pass


@dataclass
class Digitized:
    signal: np.ndarray            # (12, 1000) mV, 0 where not printed
    mask: np.ndarray              # (12, 1000) bool, printed samples
    layout: str
    px_per_mm: float
    skew_deg: float
    rows: int
    image: np.ndarray             # straightened RGB image at working scale (for the overlay)
    traces: list = field(default_factory=list)   # per row: (xs, ys) in image pixels
    notes: list = field(default_factory=list)
    method: str = "segnet"
    quality: dict = field(default_factory=dict)  # see _quality

    def info(self) -> dict:
        return {"layout": self.layout, "px_per_mm": round(self.px_per_mm, 2), "skew_deg": round(self.skew_deg, 2),
                "rows": self.rows, "method": self.method, "notes": self.notes,
                "quality": {k: round(v, 3) for k, v in self.quality.items()}}


# --------------------------------------------------------------------------- loading
def load_image(data: bytes, filename: str = "") -> np.ndarray:
    """Bytes of a PNG/JPG/WEBP or the first page of a PDF -> RGB uint8 array."""
    if filename.lower().endswith(".pdf") or data[:4] == b"%PDF":
        import fitz  # PyMuPDF

        doc = fitz.open(stream=data, filetype="pdf")
        if not len(doc):
            raise DigitizeError("the PDF has no pages")
        pix = doc[0].get_pixmap(dpi=200)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        try:
            img = Image.open(io.BytesIO(data))
        except Exception as e:
            raise DigitizeError("the file is not an image this tool can read") from e
        from PIL import ImageOps

        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))   # transparent PNG -> white paper
        img = ImageOps.exif_transpose(Image.alpha_composite(bg, img)).convert("RGB")
    return np.asarray(img)


# --------------------------------------------------------------------------- trace mask
def colour_ink(a: np.ndarray) -> np.ndarray:
    """Fallback without the segmenter: dark, colourless strokes on a coloured grid."""
    import cv2

    r_, g_, b_ = (a[..., i].astype(np.int16) for i in range(3))
    gray = ((r_ + g_ + b_) // 3).astype(np.uint8)
    sat = np.maximum(np.maximum(r_, g_), b_) - np.minimum(np.minimum(r_, g_), b_)
    bg = cv2.dilate(gray, np.ones((9, 9), np.uint8))
    dark = bg.astype(np.int16) - gray
    return ((dark > 45) & (sat < 110)).astype(np.float32)


def _skew(mask: np.ndarray) -> float:
    """Angle (deg) that makes horizontal structures horizontal: maximise row-profile energy."""
    import cv2

    small = cv2.resize(mask.astype(np.float32), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    h, w = small.shape

    def energy(ang):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        return float((cv2.warpAffine(small, M, (w, h)).sum(1) ** 2).sum())

    coarse = np.arange(-8, 8.01, 0.5)
    best = float(coarse[int(np.argmax([energy(x) for x in coarse]))])
    fine = np.arange(best - 0.45, best + 0.46, 0.05)
    return float(fine[int(np.argmax([energy(x) for x in fine]))])


def _clean(ink: np.ndarray, min_len: float) -> np.ndarray:
    """Drop specks: keep connected components at least min_len wide or tall."""
    import cv2

    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    keep = (np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT]) >= min_len)
    keep[0] = False
    return keep[lab]


def thin(m: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """Zhang-Suen thinning to a 1-pixel centre line. Tracing the centre line keeps QRS peaks at the
    right time; on a thick stroke the full-height run already shows at the stroke's leading edge.
    Works on the coordinates of ink pixels only (a few % of the page)."""
    m = np.pad(m.astype(bool), 1)
    ys, xs = np.nonzero(m)
    offs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]   # P2 .. P9
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            nb = [m[ys + dy, xs + dx] for dy, dx in offs]
            p2, p3, p4, p5, p6, p7, p8, p9 = nb
            b = np.sum(nb, axis=0)
            a = np.sum([(~nb[i]) & nb[(i + 1) % 8] for i in range(8)], axis=0)
            if step == 0:
                c = ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                c = ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            rm = (b >= 2) & (b <= 6) & (a == 1) & c
            if rm.any():
                m[ys[rm], xs[rm]] = False
                ys, xs = ys[~rm], xs[~rm]
                changed = True
        if not changed:
            break
    return m[1:-1, 1:-1]


# --------------------------------------------------------------------------- rows
def _rows(ink: np.ndarray, x_lo: int, x_hi: int, ppm: float, n_hint: int | None):
    """Baselines as (n_rows, W) arrays of y, tracked across vertical strips."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    h, w = ink.shape
    prof = gaussian_filter1d(ink[:, x_lo:x_hi].sum(1).astype(np.float64), sigma=1.0 * ppm)
    peaks, props = find_peaks(prof, distance=max(3, int(7 * ppm)), prominence=0.08 * prof.max())
    if not len(peaks):
        raise DigitizeError("no ECG traces found in the image")
    hts = prof[peaks]
    ref = np.median(np.sort(hts)[::-1][: min(4, len(hts))])
    peaks = peaks[hts >= 0.3 * ref]
    if n_hint and len(peaks) > n_hint:   # keep the strongest
        peaks = np.sort(peaks[np.argsort(prof[peaks])[::-1][:n_hint]])
    n = len(peaks)
    sp = float(np.median(np.diff(peaks))) if n > 1 else 30 * ppm
    # track each baseline strip by strip, outwards from the centre strip
    edges = np.linspace(x_lo, x_hi, N_STRIPS + 1).astype(int)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ys = np.zeros((n, N_STRIPS))
    profs = [gaussian_filter1d(ink[:, edges[s]:edges[s + 1]].sum(1).astype(np.float64), sigma=1.0 * ppm)
             for s in range(N_STRIPS)]
    mid = N_STRIPS // 2
    order = [mid] + [s for k in range(1, N_STRIPS) for s in (mid - k, mid + k) if 0 <= s < N_STRIPS]
    for s in order:
        if s == mid:
            prev = peaks.astype(np.float64)
        else:
            prev = ys[:, s + 1] if s < mid else ys[:, s - 1]
        for r in range(n):
            lo, hi = int(max(0, prev[r] - 0.3 * sp)), int(min(h, prev[r] + 0.3 * sp + 1))
            seg = profs[s][lo:hi]
            ys[r, s] = lo + int(np.argmax(seg)) if len(seg) and seg.max() > 0.1 * profs[s].max() else prev[r]
    # smooth across strips (a baseline is a smooth curve), then interpolate to every column
    from scipy.ndimage import median_filter

    ys = median_filter(ys, size=(1, 3), mode="nearest")
    xs = np.arange(w)
    base = np.stack([np.interp(xs, centres, ys[r]) for r in range(n)])
    return base, sp


def _follow(ink: np.ndarray, base: np.ndarray, sp: float, ppm: float):
    """Column-by-column tracing of every row. Returns (ys (n, W) with NaN where nothing matched)."""
    h, w = ink.shape
    n = len(base)
    sub = ink.T.astype(np.int8)                                   # [columns, rows]
    d_ = np.diff(sub, axis=1, prepend=0, append=0)
    st_c, st_r = np.nonzero(d_ == 1)
    en_c, en_r = np.nonzero(d_ == -1)
    runs_by_col = [[] for _ in range(w)]
    for c_, a_, b_ in zip(st_c.tolist(), st_r.tolist(), (en_r - 1).tolist()):
        runs_by_col[c_].append((a_, b_))
    tol = 0.45 * sp
    reset_after = max(3, int(0.15 * SPEED * ppm))
    lo_, hi_ = np.full((n, w), np.nan), np.full((n, w), np.nan)
    for r in range(n):
        up = (base[r] - base[r - 1]) if r > 0 else np.full(w, sp)
        dn = (base[r + 1] - base[r]) if r < n - 1 else np.full(w, sp)
        lo_lim = np.maximum(0.0, base[r] - 0.8 * up)
        hi_lim = np.minimum(h - 1.0, base[r] + 0.8 * dn)
        p_lo = p_hi = prev = float(base[r, 0])
        miss = 0
        for j in range(w):
            b0, l0, h0 = base[r, j], lo_lim[j], hi_lim[j]
            best, best_cost, best_gap = None, None, None
            for a_, b_ in runs_by_col[j]:
                r0_, r1_ = max(a_, l0), min(b_, h0)
                if r1_ < r0_ - 1:
                    continue
                gap_ = max(0.0, r0_ - p_hi, p_lo - r1_)
                cost_ = gap_ + 0.3 * abs(0.5 * (r0_ + r1_) - b0)
                if best_cost is None or cost_ < best_cost:
                    best, best_cost, best_gap = (r0_, r1_), cost_, gap_
            if best is None:
                continue
            if best_gap > tol:
                miss += 1
                if miss > reset_after:
                    p_lo = p_hi = prev = float(b0)
                    miss = 0
                continue
            miss = 0
            r0, r1 = float(best[0]), float(best[1])
            lo_[r, j], hi_[r, j] = r0, r1
            p_lo, p_hi, prev = r0, r1, 0.5 * (r0 + r1)
    # On the 1-pixel centre line a column crosses a straight stroke at the middle of its run. Only a
    # column at a peak or trough (its run reaches beyond both neighbours) takes the run's far end.
    out = 0.5 * (lo_ + hi_)
    lo_p, lo_n = np.roll(lo_, 1, 1), np.roll(lo_, -1, 1)
    hi_p, hi_n = np.roll(hi_, 1, 1), np.roll(hi_, -1, 1)
    with np.errstate(invalid="ignore"):
        top = (lo_ < np.fmin(lo_p, lo_n) - 1) & (hi_ - lo_ > 1)
        bot = (hi_ > np.fmax(hi_p, hi_n) + 1) & (hi_ - lo_ > 1)
    out[top & ~bot] = lo_[top & ~bot]
    out[bot & ~top] = hi_[bot & ~top]
    return out


def _extent(found: np.ndarray, win: int) -> tuple[int, int] | None:
    """First and last column of a row's trace (coverage > 50% in a sliding window)."""
    if not found.any():
        return None
    cov = np.convolve(found.astype(np.float64), np.ones(win) / win, mode="same")
    ok = np.flatnonzero(cov > 0.5)
    if not len(ok):
        return None
    idx = np.flatnonzero(found)
    a = idx[np.searchsorted(idx, ok[0] - win // 2)]
    b = idx[np.searchsorted(idx, ok[-1] + win // 2, side="right") - 1]
    return int(a), int(b)


def _detrend(seg: np.ndarray) -> np.ndarray:
    """Remove the baseline of one printed segment: a line through the medians of its two ends."""
    n = len(seg)
    q = max(2, n // 4)
    m0, m1 = np.median(seg[:q]), np.median(seg[-q:])
    line = m0 + (m1 - m0) * (np.arange(n) - (q - 1) / 2) / max(n - q, 1)
    return seg - line


def _quality(found, starts, ends, spans_raw, prob, ys) -> dict:
    """How much to trust a tracing, without knowing the true signal.

    coverage    share of each row's columns where the trace was actually followed (rest interpolated)
    span        agreement of the rows' lengths (all rows of a printout span the same 10 s)
    confidence  mean segmenter probability on the traced pixels
    score       their product, 0..1
    """
    cov = float(np.mean([f[a:b + 1].mean() for f, a, b in zip(found, starts, ends)]))
    span = float(max(0.0, 1.0 - np.std(spans_raw) / max(np.median(spans_raw), 1) * 10))
    h, w = prob.shape
    vals = []
    for f, y, a, b in zip(found, ys, starts, ends):
        xs = np.flatnonzero(f[a:b + 1]) + a
        yy = np.clip(np.round(y[xs]).astype(int), 0, h - 1)
        vals.append(prob[yy, xs])
    conf = float(np.mean(np.concatenate(vals))) if vals else 0.0
    return {"coverage": cov, "span": span, "confidence": conf, "score": cov * span * conf}


# --------------------------------------------------------------------------- main
def digitize(a: np.ndarray, layout_hint: str | None = None, use_segnet: bool = True,
             trace_mask: np.ndarray | None = None) -> Digitized:
    """trace_mask: a known waveform mask at the image's resolution (synthetic printouts), used
    instead of the segmenter."""
    import cv2

    notes = []
    method = "segnet"
    prob = None
    if trace_mask is not None:
        from cardiomoe.images.segnet import _work_scale

        a, m = _work_scale(a, trace_mask)
        prob, method = m.astype(np.float32), "given mask"
    elif use_segnet:
        from cardiomoe.images import segnet

        try:
            a, prob = segnet.segment(a)
        except FileNotFoundError:
            prob = None
    if prob is None:
        method = "colour"
        from cardiomoe.images.segnet import _work_scale

        a, _ = _work_scale(a)
        prob = colour_ink(a)
        notes.append("trace segmenter not trained: used the colour rule (red/pink grids only)")
    ink = prob > 0.5
    ang = _skew(ink)
    if abs(ang) > 0.05:
        hh, ww = ink.shape
        M = cv2.getRotationMatrix2D((ww / 2, hh / 2), ang, 1.0)
        a = cv2.warpAffine(a, M, (ww, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        prob = cv2.warpAffine(prob.astype(np.float32), M, (ww, hh), flags=cv2.INTER_LINEAR)
        ink = prob > 0.5
    h, w = ink.shape
    cols = np.flatnonzero(ink.sum(0) > 0)
    if len(cols) < 0.2 * w:
        raise DigitizeError("no ECG traces found in the image")
    x_lo, x_hi = int(np.percentile(cols, 1)), int(np.percentile(cols, 99)) + 1
    ppm0 = (x_hi - x_lo) / STRIP_MM
    ink = _clean(ink, 2.5 * ppm0)

    n_hint = {"12x1": 12, "3x4": 4, "3x4_no_rhythm": 3}.get(layout_hint or "")
    base, sp = _rows(ink, x_lo, x_hi, ppm0, n_hint)
    n_rows = len(base)
    if layout_hint:
        layout = layout_hint
    elif n_rows >= 10:
        layout = "12x1"
    elif n_rows == 4 or n_rows == 5:
        layout = "3x4"
    elif n_rows == 3:
        layout = "3x4_no_rhythm"
    else:
        raise DigitizeError(f"found {n_rows} rows of traces; supported layouts are 12x1 and 3x4 (+ rhythm strip)")
    want = {"12x1": 12, "3x4": 4, "3x4_no_rhythm": 3}[layout]
    if n_rows > want:
        keep = np.sort(np.argsort([ink[np.clip(b.astype(int), 0, h - 1), np.arange(w)].sum() for b in base])[::-1][:want])
        base = base[keep]
    elif n_rows < want:
        raise DigitizeError(f"expected {want} rows for a {layout} layout, found {n_rows}")

    ys = _follow(thin(ink), base, sp, ppm0)
    found = np.isfinite(ys)
    ext = [_extent(f, max(5, int(2 * ppm0))) for f in found]
    if any(e is None for e in ext):
        raise DigitizeError("could not follow every row of the trace")
    spans = np.array([e[1] - e[0] for e in ext], dtype=np.float64)
    starts, ends = np.array([e[0] for e in ext]), np.array([e[1] for e in ext])
    # Every row of a printout spans the same 10 s. Faint ends of a scanned / photographed trace get
    # lost far more often than blobs get added, so the reference length is the second-longest row
    # (the longest could carry a blob). A row more than 1.5% off is re-anchored on whichever of its
    # ends agrees better with the other rows.
    ref = float(np.sort(spans)[-2] if len(spans) >= 3 else spans.max())
    bad = np.abs(spans - ref) > 0.015 * ref
    if bad.any():
        notes.append(f"{int(bad.sum())} row(s) did not span the full strip; aligned them with the others")
        s_med, e_med = np.median(starts[~bad]) if (~bad).any() else np.median(starts), \
            np.median(ends[~bad]) if (~bad).any() else np.median(ends)
        for r in np.flatnonzero(bad):
            if abs(starts[r] - s_med) <= abs(ends[r] - e_med):
                ends[r] = min(w - 1, int(round(starts[r] + ref)))
            else:
                starts[r] = max(0, int(round(ends[r] - ref)))
    ppm = float(np.median((ends - starts) / STRIP_MM))
    quality = _quality(found, starts, ends, spans, prob, ys)

    t = np.arange(1000) / 100.0
    rows_mv, traces = [], []
    for r in range(len(base)):
        y, f = ys[r], found[r]
        if f.sum() < 0.3 * (ends[r] - starts[r]):
            raise DigitizeError(f"could not follow the trace in row {r + 1}")
        idx = np.flatnonzero(f)
        yi = np.interp(np.arange(w), idx, y[idx])
        xs = np.arange(starts[r], ends[r] + 1)
        traces.append((xs.tolist(), yi[xs].tolist()))
        t_px = (xs - starts[r]) / max(ends[r] - starts[r], 1) * 9.99   # last sample is at 9.99 s
        mv = (base[r, xs] - yi[xs]) / ((ends[r] - starts[r]) / STRIP_MM * GAIN)
        rows_mv.append(np.interp(t, t_px, mv))

    sig = np.zeros((12, 1000), dtype=np.float32)
    mask = visible_mask(layout)
    if layout == "12x1":
        for r in range(12):
            sig[r] = _detrend_long(rows_mv[r])
    else:
        for r in range(3):
            for c, lead in enumerate(GRID_3x4[r]):
                a0, a1 = c * 250, (c + 1) * 250
                seg = rows_mv[r][a0:a1].copy()
                edge = 3  # the line jumps between leads on the column borders
                if c:
                    seg[:edge] = seg[edge]
                if c < 3:
                    seg[-edge:] = seg[-edge - 1]
                sig[LEADS.index(lead), a0:a1] = _detrend(seg)
        if layout == "3x4":
            sig[LEADS.index(RHYTHM_LEAD)] = _detrend_long(rows_mv[3])
    return Digitized(sig, mask, layout, ppm, float(ang), int(len(base)), a, traces, notes, method, quality)


def _detrend_long(v: np.ndarray) -> np.ndarray:
    """10-s strip: the pipeline's 0.5-40 Hz band-pass (as for recorded signals)."""
    from cardiomoe.data.signal import bandpass

    return bandpass(v.astype(np.float64)).astype(np.float32)


def overlay_png(d: Digitized, max_w: int = 1400) -> bytes:
    """The straightened image with the traced signal drawn on top (for users to check the tracing)."""
    from PIL import ImageDraw

    img = Image.fromarray(d.image).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")
    colors = [(37, 99, 235, 230), (234, 88, 12, 230), (22, 163, 74, 230), (147, 51, 234, 230)]
    for i, (xs, ys) in enumerate(d.traces):
        dr.line(list(zip(xs[::2], ys[::2])), fill=colors[i % 4], width=2)
    if img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()
