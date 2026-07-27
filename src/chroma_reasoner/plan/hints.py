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
from .masks import erode_frac, paint_order, region_key


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
    # large -> small: specific regions override the backgrounds containing them
    for region in paint_order(masks, plan):
        key = region_key(region)
        mask = masks[key]
        colour = LabColor.from_plan(region["resolved_colour"])
        # cv2 8-bit Lab stores a,b offset by +128
        lab[:, :, 1][mask] = np.clip(round(colour.a) + 128, 0, 255)
        lab[:, :, 2][mask] = np.clip(round(colour.b) + 128, 0, 255)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# Global modifiers describe whole-image rendering (film stock, era fade,
# overall mood) — effects that deliberately do NOT route through one object.
# Per-region hints cannot express them, so they are translated into diffusion
# prompt terms instead. Showcase finding: without this, everything the plan
# did not mask is filled by the colorizer's own prior, and on a 1940s photo
# with 20% mask coverage that meant neon cardigans (colourfulness 79 against
# DDColor's 1.8 on a monochrome original).
GLOBAL_PROMPT_TERMS: dict[tuple[str, str], tuple[str, str]] = {
    ("era", "1910s"): ("early autochrome-era photography, muted natural dyes, low saturation",
                       "vivid, neon, saturated modern colours"),
    ("era", "1940s"): ("1940s colour photography, muted utility palette, restrained saturation",
                       "vivid, neon, saturated modern colours"),
    ("era", "1950s"): ("1950s colour photography, soft pastel palette", "neon, harsh saturation"),
    ("era", "1970s"): ("1970s colour photography, avocado and harvest-gold palette, warm cast",
                       "cool blue tones, neon"),
    ("mood", "melancholic"): ("desaturated sombre palette, cool cast", "vivid, cheerful, saturated"),
    ("mood", "cheerful"): ("bright saturated palette, warm cast", "drab, desaturated, grey"),
    ("mood", "ominous"): ("crushed desaturated palette, cold cast", "bright, cheerful, saturated"),
    ("mood", "nostalgic"): ("faded warm print, gentle yellow cast", "clinical, cold, oversaturated"),
    ("weather", "overcast"): ("flat diffuse overcast light, low chroma", "harsh sunlight, vivid colours"),
    ("weather", "fog"): ("hazy low-contrast atmosphere, washed-out colour", "vivid, high contrast"),
    ("time_of_day", "golden_hour"): ("warm low golden sunlight", "cold blue light"),
    ("time_of_day", "night"): ("dim cool night lighting", "bright daylight"),
}


def global_prompt_terms(plan: dict) -> tuple[str, str]:
    """Translate the plan's `global` modifiers into (positive, negative) prompt
    fragments for a text-conditioned colorizer. Empty strings when the plan
    has no global block."""
    modifiers = (plan.get("global") or {}).get("modifiers", [])
    positive, negative = [], []
    for modifier in modifiers:
        terms = GLOBAL_PROMPT_TERMS.get((modifier["family"], modifier["value"]))
        if terms:
            positive.append(terms[0])
            negative.append(terms[1])
        elif modifier.get("effect"):
            positive.append(modifier["effect"])
    return ", ".join(positive), ", ".join(negative)


def make_hint_image(gray_rgb: np.ndarray, masks: dict[str, np.ndarray], plan: dict,
                    erosion: float = 0.15) -> np.ndarray:
    """Control Color hint image: grayscale input + colour strokes.

    gray_rgb: HxWx3 uint8, the grayscale input replicated to 3 channels
    (must be the exact pixels the model receives as input_image, because
    Control Color detects hints by input/hint pixel comparison).
    Returns HxWx3 uint8.
    """
    hint = gray_rgb.copy()
    for region in paint_order(masks, plan):
        key = region_key(region)
        core = erode_frac(masks[key], erosion)
        colour = LabColor.from_plan(region["resolved_colour"])
        r, g, b = (round(c * 255) for c in lab_to_srgb(colour))
        hint[core] = (r, g, b)
    return hint
