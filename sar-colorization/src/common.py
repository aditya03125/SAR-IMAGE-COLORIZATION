"""Shared paths and numpy-only helpers (no torch import here)."""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"


def stretch_u8(a, valid=None, k=2.5):
    """Per-scene normalisation in the spirit of SEN1-2: clip to mean +/- k*std, scale to 0..255.
    SEN1-2 describes its stretch only loosely, so this is an approximation; the Odisha
    fine-tuning step absorbs small differences."""
    a = np.asarray(a, dtype=np.float32)
    ok = np.isfinite(a)
    if valid is not None:
        ok &= valid
    if ok.sum() < 100:
        return np.zeros(a.shape, np.uint8)
    m, s = float(a[ok].mean()), float(a[ok].std())
    s = s if s > 1e-6 else 1.0
    lo, hi = m - k * s, m + k * s
    out = np.clip((np.where(ok, a, lo) - lo) / (hi - lo), 0.0, 1.0)
    out[~ok] = 0.0
    return (out * 255.0 + 0.5).astype(np.uint8)


def tile_positions(h, w, tile=256, overlap=32):
    """Padded size and top-left corners so that overlapping tiles cover the whole image."""
    stride = tile - overlap

    def pad(n):
        return tile if n <= tile else tile + int(np.ceil((n - tile) / stride)) * stride

    ph, pw = pad(h), pad(w)
    ys = range(0, ph - tile + 1, stride)
    xs = range(0, pw - tile + 1, stride)
    return ph, pw, [(y, x) for y in ys for x in xs]


def hann2d(t):
    """2-D blending window (never exactly zero)."""
    w = np.hanning(t + 2)[1:-1].astype(np.float32)
    return np.maximum(np.outer(w, w), 1e-3)
