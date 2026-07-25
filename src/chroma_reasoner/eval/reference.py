"""Plan-vs-reference scoring: how close is each plan colour to the colour the
region ACTUALLY had in the original photograph?

This is the objective backbone of the KB-on/off ablation (roadmap §6): when
the context prompt describes the real scene, the original colours are a fair
proxy for intent, and any colour source (KB resolution, LLM guess, human
author) can be scored on the same masks with the same metric.

ΔE is ab-plane CIE76 on the masked median (matching adherence.py); ΔL is
reported for diagnostics. Images whose original is itself monochrome (the
1940s school photo) carry no chroma reference — detect via reference
colourfulness and exclude.
"""

from __future__ import annotations

import math

import numpy as np

from ..plan.colors import LabColor, srgb_array_to_lab
from ..plan.masks import exclusive_masks, region_key


def region_reference_lab(original_rgb: np.ndarray, mask: np.ndarray) -> LabColor:
    """Masked median Lab of the original colour image (uint8 RGB)."""
    pixels = original_rgb[mask].astype(np.float64) / 255.0
    lab = srgb_array_to_lab(pixels)
    return LabColor(*np.median(lab, axis=0))


def plan_vs_reference(plan: dict, masks: dict[str, np.ndarray],
                      original_rgb: np.ndarray) -> dict:
    """Score every region's planned colour against the original's colour."""
    excl = exclusive_masks(masks, plan)
    rows = []
    for region in plan["regions"]:
        key = region_key(region)
        mask = excl.get(key)
        if mask is None or not mask.any():
            rows.append({"region": key, "error": "no mask"})
            continue
        ref = region_reference_lab(original_rgb, mask)
        planned = LabColor.from_plan(region["resolved_colour"])
        planned_chroma = math.hypot(planned.a, planned.b)
        ref_chroma = math.hypot(ref.a, ref.b)
        rows.append({
            "region": key,
            "object": region["object"],
            "planned_ab": [round(planned.a, 1), round(planned.b, 1)],
            "reference_ab": [round(ref.a, 1), round(ref.b, 1)],
            "delta_e": round(math.hypot(planned.a - ref.a, planned.b - ref.b), 2),
            "reference_chroma": round(ref_chroma, 2),
            # Chroma deficit: how much colour the plan FAILED to commit to.
            # ΔE alone rewards conservative desaturation (the classic PSNR
            # bias, rediscovered in Phase 5 when a model collapsed to gray
            # and "won"); a gray guess for a colourful region must show up.
            "chroma_deficit": round(max(0.0, ref_chroma - planned_chroma), 2),
        })
    des = [r["delta_e"] for r in rows if "delta_e" in r]
    deficits = [r["chroma_deficit"] for r in rows if "chroma_deficit" in r]
    undercommitted = sum(1 for r in rows
                         if r.get("chroma_deficit", 0) > 10 and r.get("reference_chroma", 0) > 15)
    return {
        "regions": rows,
        "mean_delta_e": round(float(np.mean(des)), 2) if des else None,
        "mean_chroma_deficit": round(float(np.mean(deficits)), 2) if deficits else None,
        "n_undercommitted": undercommitted,
        "n_regions": len(rows),
    }
