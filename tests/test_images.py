"""Image pipeline: printout generator, layouts, thinning, digitizer, segmenter shapes (no trained models)."""
import io

import numpy as np
import pytest

pytest.importorskip("cv2")

from cardiomoe.data.signal import LEADS
from cardiomoe.images.digitize import DigitizeError, digitize, load_image, thin
from cardiomoe.images.layouts import masked_normalize, visible_mask
from cardiomoe.images.synth import degrade, random_page


def ecg_like(seed=0) -> np.ndarray:
    """(12, 1000) mV: a 75 bpm spike train with a T wave, different per lead."""
    rng = np.random.default_rng(seed)
    t = np.arange(1000) / 100.0
    beat = np.zeros(1000)
    for b in np.arange(0.4, 10, 0.8):
        beat += np.exp(-((t - b) / 0.012) ** 2) - 0.25 * np.exp(-((t - b - 0.03) / 0.01) ** 2)
        beat += 0.2 * np.exp(-((t - b - 0.28) / 0.05) ** 2)
    return np.stack([beat * rng.uniform(0.4, 1.5) * rng.choice([-1, 1]) for _ in range(12)]).astype(np.float32)


def test_visible_mask_layouts():
    m = visible_mask("3x4")
    assert m.shape == (12, 1000)
    assert m[LEADS.index("II")].all()                      # rhythm strip
    assert m[LEADS.index("V6")].sum() == 250 and m[LEADS.index("V6"), 750:].all()
    assert visible_mask("3x4_no_rhythm")[LEADS.index("II")].sum() == 250
    assert visible_mask("12x1").all()


def test_masked_normalize_zero_outside():
    sig = ecg_like()
    m = visible_mask("3x4")
    x = masked_normalize(sig, m)
    assert (x[~m] == 0).all()
    i = LEADS.index("V1")
    assert abs(x[i, m[i]].mean()) < 1e-5 and abs(x[i, m[i]].std() - 1) < 1e-3


def test_page_mask_marks_only_waveform():
    p = random_page(ecg_like(), np.random.default_rng(1), layout="3x4")
    assert p.image.shape[:2] == p.mask.shape
    assert 0.002 < p.mask.mean() < 0.1
    d = degrade(p, np.random.default_rng(2))
    assert d.image.shape[:2] == d.mask.shape and d.mask.any()


def test_thin_gives_one_pixel_lines():
    import cv2

    m = np.zeros((80, 200), np.uint8)
    cv2.line(m, (5, 40), (195, 30), 1, 5)
    t = thin(m > 0)
    assert t.sum() > 150 and t.sum() < m.sum() / 3
    blocks = t[:-1, :-1] & t[1:, :-1] & t[:-1, 1:] & t[1:, 1:]
    assert not blocks.any()


@pytest.mark.parametrize("layout", ["3x4", "12x1"])
def test_digitize_recovers_signal_from_known_mask(layout):
    sig = ecg_like(3)
    p = random_page(sig, np.random.default_rng(4), layout=layout)
    d = digitize(p.image, trace_mask=p.mask)
    assert d.layout == layout
    r = [np.corrcoef(d.signal[i, d.mask[i]], sig[i, d.mask[i]])[0, 1] for i in range(12)]
    assert np.median(r) > 0.8, r


def test_digitize_rejects_blank_page():
    blank = np.full((800, 1100, 3), 250, np.uint8)
    with pytest.raises(DigitizeError):
        digitize(blank, trace_mask=np.zeros((800, 1100), np.uint8))


def test_load_image_flattens_transparency():
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (40, 30), (0, 0, 0, 0)).save(buf, "PNG")
    a = load_image(buf.getvalue(), "x.png")
    assert a.shape == (30, 40, 3) and (a == 255).all()
    with pytest.raises(DigitizeError):
        load_image(b"not an image", "x.jpg")


def test_segnet_output_matches_input_size():
    import torch

    from cardiomoe.images.segnet import UNet

    m = UNet().eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 64, 96))
    assert y.shape == (1, 1, 64, 96)
