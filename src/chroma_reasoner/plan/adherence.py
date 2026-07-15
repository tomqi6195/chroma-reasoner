"""Palette adherence: did the output apply the planned colours to the planned
regions? The objective half of the Phase-5 evaluation protocol (roadmap §6),
built in Phase 2 because the manual pipeline needs it to debug hint-following.

Per region: median Lab of output pixels inside the mask, **ab-plane** CIE76
ΔE to the plan's resolved colour, pass/fail against tolerance_delta_e
(default 10). Median, not mean: shading gradients and small mask errors
shouldn't drag the realized colour estimate.

Why ab-only: colorization predicts chrominance; L is copied from the input
and no colorizer (naive paste, DDColor, Control Color) can change it. A plan's
resolved L is the author's *guess* at the region's lightness — scoring it
would punish the pipeline for authoring error (Phase-2 finding: the naive
paste, which adheres perfectly by construction, showed ΔE up to 88 on the
school-bus region purely from the L term). ΔL is still reported as authoring
feedback (large |ΔL| = the plan imagined a different lightness than the
image has, worth revisiting the colour choice).
"""

from __future__ import annotations

import math

import numpy as np

from .colors import LabColor, srgb_array_to_lab
from .masks import exclusive_masks, region_key

DEFAULT_TOLERANCE = 10.0


def region_adherence(output_rgb: np.ndarray, mask: np.ndarray, region: dict) -> dict:
    """output_rgb: HxWx3 uint8; mask: HxW bool."""
    target = LabColor.from_plan(region["resolved_colour"])
    pixels = output_rgb[mask].astype(np.float64) / 255.0
    if len(pixels) == 0:
        return {"region": region_key(region), "object": region["object"],
                "error": "empty mask", "pass": False}
    lab = srgb_array_to_lab(pixels)
    realized = LabColor(*np.median(lab, axis=0))
    tolerance = region.get("tolerance_delta_e", DEFAULT_TOLERANCE)
    de_ab = math.hypot(target.a - realized.a, target.b - realized.b)
    return {
        "region": region_key(region),
        "object": region["object"],
        "target_lab": [target.L, target.a, target.b],
        "realized_lab": [round(realized.L, 2), round(realized.a, 2), round(realized.b, 2)],
        "delta_e": round(de_ab, 2),
        "delta_L": round(realized.L - target.L, 2),
        "tolerance": tolerance,
        "pass": de_ab <= tolerance,
    }


def evaluate_adherence(output_rgb: np.ndarray, masks: dict[str, np.ndarray], plan: dict) -> dict:
    """Full-plan adherence report.

    Regions are scored on their *exclusive* pixels (mask minus smaller
    regions' masks) — mirroring the large->small paint order, so a background
    region isn't blamed for pixels that were handed to an object inside it.
    """
    excl = exclusive_masks(masks, plan)
    regions = [region_adherence(output_rgb, excl[region_key(r)], r) for r in plan["regions"]]
    des = [r["delta_e"] for r in regions if "delta_e" in r]
    return {
        "regions": regions,
        "mean_delta_e": round(float(np.mean(des)), 2) if des else None,
        "n_pass": sum(r["pass"] for r in regions),
        "n_regions": len(regions),
    }
