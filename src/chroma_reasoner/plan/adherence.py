"""Palette adherence: did the output apply the planned colours to the planned
regions? The objective half of the Phase-5 evaluation protocol (roadmap §6),
built in Phase 2 because the manual pipeline needs it to debug hint-following.

Per region: median Lab of output pixels inside the mask, CIE76 ΔE to the
plan's resolved colour, pass/fail against the region's tolerance_delta_e
(default 10). Median, not mean: shading gradients and small mask errors
shouldn't drag the realized colour estimate.
"""

from __future__ import annotations

import numpy as np

from .colors import LabColor, delta_e76, srgb_array_to_lab
from .masks import region_key

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
    de = delta_e76(target, realized)
    return {
        "region": region_key(region),
        "object": region["object"],
        "target_lab": [target.L, target.a, target.b],
        "realized_lab": [round(realized.L, 2), round(realized.a, 2), round(realized.b, 2)],
        "delta_e": round(de, 2),
        "tolerance": tolerance,
        "pass": de <= tolerance,
    }


def evaluate_adherence(output_rgb: np.ndarray, masks: dict[str, np.ndarray], plan: dict) -> dict:
    """Full-plan adherence report."""
    regions = [region_adherence(output_rgb, masks[region_key(r)], r) for r in plan["regions"]]
    des = [r["delta_e"] for r in regions if "delta_e" in r]
    return {
        "regions": regions,
        "mean_delta_e": round(float(np.mean(des)), 2) if des else None,
        "n_pass": sum(r["pass"] for r in regions),
        "n_regions": len(regions),
    }
