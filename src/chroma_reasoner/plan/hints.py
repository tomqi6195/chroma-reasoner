"""Convert (mask, resolved_colour) pairs into colorizer inputs.

Two consumers:

1. `render_naive` — deterministic Lab-paste: keep the input L channel, set
   each region's ab to the plan's resolved colour. No model, runs anywhere.
   This is the Phase-2 control: it proves masks+colours flow end to end,
   upper-bounds palette adherence (ΔE ≈ 0 by construction), and gives the
   diffusion colorizer a floor to beat on realism.

2. `make_hint_image` — Control Color's interface: a copy of the grayscale
   input with colour strokes painted on it (its `get_mask` recovers hinted
   pixels by comparing hint image to input). Strokes are painted inside an
   eroded mask so hints stay away from boundaries and don't bleed across
   edges.
"""

from __future__ import annotations

import numpy as np

from .colors import LabColor, lab_to_srgb
from .masks import erode_frac, region_key


def render_naive(gray_l8: np.ndarray, masks: dict[str, np.ndarray], plan: dict) -> np.ndarray:
    """Naive Lab-paste render.

    gray_l8: HxW uint8 — the Lab L channel in cv2's 0-255 scaling (what
    data/coco/gray/*.png stores). Returns HxWx3 RGB uint8.
    """
    import cv2

    h, w = gray_l8.shape
    lab = np.zeros((h, w, 3), dtype=np.uint8)
    lab[:, :, 0] = gray_l8
    lab[:, :, 1:] = 128  # neutral ab
    for region in plan["regions"]:
        key = region_key(region)
        mask = masks[key]
        colour = LabColor.from_plan(region["resolved_colour"])
        # cv2 8-bit Lab stores a,b offset by +128
        lab[:, :, 1][mask] = np.clip(round(colour.a) + 128, 0, 255)
        lab[:, :, 2][mask] = np.clip(round(colour.b) + 128, 0, 255)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_hint_image(gray_rgb: np.ndarray, masks: dict[str, np.ndarray], plan: dict,
                    erosion: float = 0.15) -> np.ndarray:
    """Control Color hint image: grayscale input + colour strokes.

    gray_rgb: HxWx3 uint8, the grayscale input replicated to 3 channels
    (must be the exact pixels the model receives as input_image, because
    Control Color detects hints by input/hint pixel comparison).
    Returns HxWx3 uint8.
    """
    hint = gray_rgb.copy()
    for region in plan["regions"]:
        key = region_key(region)
        core = erode_frac(masks[key], erosion)
        colour = LabColor.from_plan(region["resolved_colour"])
        r, g, b = (round(c * 255) for c in lab_to_srgb(colour))
        hint[core] = (r, g, b)
    return hint
