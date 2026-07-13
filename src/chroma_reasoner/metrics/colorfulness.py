"""Hasler & Süsstrunk (2003) colorfulness metric.

CF = sigma_rgyb + 0.3 * mu_rgyb, computed on the rg / yb opponent axes.
Higher = more vivid. Gameable by saturation pumping, which is exactly why the
protocol pairs it with hue-invariant FID (see hue_invariant.py).
"""

from __future__ import annotations

import numpy as np


def colorfulness(img_rgb: np.ndarray) -> float:
    """img_rgb: HxWx3 uint8 or float RGB image."""
    img = img_rgb.astype(np.float64)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    sigma = np.hypot(rg.std(), yb.std())
    mu = np.hypot(rg.mean(), yb.mean())
    return float(sigma + 0.3 * mu)
