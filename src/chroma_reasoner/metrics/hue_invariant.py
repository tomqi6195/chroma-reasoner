"""Hue-invariant FID preprocessing (arXiv:2503.14974 §5.2.1).

Before computing FID, each colorized image I1 has its chromaticity scaled in
YUV space, F_alpha: {U,V} -> {alpha*U, alpha*V} (around the neutral point),
with alpha* = argmin_alpha |CF(F_alpha(I1)) - CF(I0)| where I0 is the ground
truth. This removes the "just saturate more" degree of freedom, so the FID
that remains reflects structure/overflow/artifact quality rather than hue or
vividness choices.

CF(alpha) is monotonically increasing in alpha (both sigma and mu of the
opponent axes scale with chroma), so a bisection search suffices.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .colorfulness import colorfulness


def scale_chroma_yuv(img_rgb: np.ndarray, alpha: float) -> np.ndarray:
    """Scale U,V around the neutral point (128 in 8-bit YUV) by alpha."""
    yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV).astype(np.float32)
    yuv[..., 1:] = (yuv[..., 1:] - 128.0) * alpha + 128.0
    yuv = np.clip(yuv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)


def match_colorfulness(pred_rgb: np.ndarray, target_cf: float,
                       lo: float = 0.0, hi: float = 4.0, iters: int = 25) -> tuple[np.ndarray, float]:
    """Find alpha* such that CF(scale_chroma_yuv(pred, alpha)) ~= target_cf.

    Returns (corrected image, alpha*). Bisection on the monotone CF(alpha).
    Clipping in YUV->RGB can flatten CF growth for extreme alpha; if even
    alpha=hi undershoots the target, alpha=hi is returned.
    """
    if colorfulness(scale_chroma_yuv(pred_rgb, hi)) < target_cf:
        return scale_chroma_yuv(pred_rgb, hi), hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if colorfulness(scale_chroma_yuv(pred_rgb, mid)) < target_cf:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    return scale_chroma_yuv(pred_rgb, alpha), alpha


def build_hue_corrected_dir(pred_dir: Path, gt_dir: Path, out_dir: Path) -> list[float]:
    """Write CF-matched copies of pred images; returns the alpha* per image.

    Files are paired by stem. GT images are resized to the prediction's size
    only for CF computation (CF is scale-sensitive only weakly, but we keep it
    consistent).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    alphas: list[float] = []
    preds = sorted(p for p in pred_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    gt_by_stem = {p.stem: p for p in gt_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    for pred_path in tqdm(preds, desc="HI-FID chroma matching"):
        gt_path = gt_by_stem.get(pred_path.stem)
        if gt_path is None:
            raise FileNotFoundError(f"no ground truth for {pred_path.stem}")
        pred = cv2.cvtColor(cv2.imread(str(pred_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(cv2.imread(str(gt_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        corrected, alpha = match_colorfulness(pred, colorfulness(gt))
        alphas.append(alpha)
        cv2.imwrite(str(out_dir / (pred_path.stem + ".png")),
                    cv2.cvtColor(corrected, cv2.COLOR_RGB2BGR))
    return alphas
