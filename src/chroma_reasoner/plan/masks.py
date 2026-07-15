"""Mask conventions for the plan pipeline (Phase 2 lock).

Layout: masks/{image_id}/{region_key}.png — single-channel 8-bit PNG,
255 = inside the region, same HxW as the source image. region_key is the
region's `id` if present, else its `object` name. Grounded-SAM (Colab)
produces these; the renderer, hint generator, and adherence evaluator
consume them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def region_key(region: dict) -> str:
    return region.get("id") or region["object"]


def mask_path(masks_root: Path, image_id: str, region: dict) -> Path:
    return Path(masks_root) / image_id / f"{region_key(region)}.png"


def save_mask(mask: np.ndarray, masks_root: Path, image_id: str, region: dict) -> Path:
    """mask: HxW bool or uint8. Saved as 0/255 PNG."""
    path = mask_path(masks_root, image_id, region)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask.astype(np.uint8) > 0).astype(np.uint8) * 255)
    return path


def load_masks(masks_root: Path, image_id: str, plan: dict,
               shape: tuple[int, int] | None = None) -> dict[str, np.ndarray]:
    """Load one bool mask per region. Raises if any region's mask is missing.

    shape: optional (H, W) to assert against (catches image/mask mismatches).
    """
    masks: dict[str, np.ndarray] = {}
    for region in plan["regions"]:
        key = region_key(region)
        path = mask_path(masks_root, image_id, region)
        if not path.exists():
            raise FileNotFoundError(f"mask missing for region '{key}': {path}")
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise ValueError(f"unreadable mask: {path}")
        if shape is not None and m.shape != shape:
            raise ValueError(f"mask {path} shape {m.shape} != image shape {shape}")
        masks[key] = m > 127
    return masks


def paint_order(masks: dict[str, np.ndarray], plan: dict) -> list[dict]:
    """Regions sorted by mask area, largest first.

    Painting large->small makes specific objects override the broad
    backgrounds that swallow them (Phase-2 finding: "walls" masks contain the
    floor, a "mirror" mask contains the bus reflected in it). Broad first,
    specific last = specific wins.
    """
    return sorted(plan["regions"], key=lambda r: -int(masks[region_key(r)].sum()))


def exclusive_masks(masks: dict[str, np.ndarray], plan: dict) -> dict[str, np.ndarray]:
    """Each region's mask minus every strictly smaller region's mask.

    The evaluation-side counterpart of paint_order: a region's realized
    colour must be measured only on pixels that weren't handed to a more
    specific region. Falls back to the full mask if exclusion empties it.
    """
    order = paint_order(masks, plan)  # largest -> smallest
    out: dict[str, np.ndarray] = {}
    for i, region in enumerate(order):
        key = region_key(region)
        excl = masks[key].copy()
        for smaller in order[i + 1:]:
            excl &= ~masks[region_key(smaller)]
        out[key] = excl if excl.any() else masks[key]
    return out


def erode_frac(mask: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """Erode a bool mask by `frac` of its equivalent radius.

    Used when painting hints: keeping strokes away from region boundaries
    stops the hint from contaminating neighbouring regions when the
    colorizer diffuses it. Falls back to the original mask if erosion
    would erase it entirely.
    """
    area = int(mask.sum())
    if area == 0:
        return mask
    radius = max(1, int(np.sqrt(area / np.pi) * frac))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    eroded = cv2.erode(mask.astype(np.uint8), kernel) > 0
    return eroded if eroded.any() else mask
