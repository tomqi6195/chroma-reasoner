"""End-to-end test of the Phase 2 consumption path with synthetic data:
plan + masks -> naive render -> adherence. No model, no files needed.
"""

import numpy as np
import pytest

from chroma_reasoner.plan.adherence import evaluate_adherence
from chroma_reasoner.plan.hints import make_hint_image, render_naive
from chroma_reasoner.plan.masks import erode_frac


@pytest.fixture()
def synthetic():
    gray = np.full((80, 120), 128, dtype=np.uint8)  # mid-gray L channel
    left = np.zeros((80, 120), dtype=bool)
    left[10:70, 10:50] = True
    right = np.zeros((80, 120), dtype=bool)
    right[10:70, 70:110] = True
    masks = {"left": left, "right": right}
    plan = {
        "plan_version": "1.0",
        "regions": [
            {"id": "left", "object": "thing_a", "grounding_phrase": "left thing",
             "resolved_colour": {"space": "Lab", "L": 50, "a": 40, "b": 20},
             "confidence": 0.9, "rationale": "test"},
            {"id": "right", "object": "thing_b", "grounding_phrase": "right thing",
             "resolved_colour": {"space": "Lab", "L": 50, "a": -20, "b": -12},
             "tolerance_delta_e": 5,
             "confidence": 0.9, "rationale": "test"},
        ],
    }
    return gray, masks, plan


def test_naive_render_adheres(synthetic):
    gray, masks, plan = synthetic
    rgb = render_naive(gray, masks, plan)
    report = evaluate_adherence(rgb, masks, plan)
    # Naive paste keeps the input L (50.2 for gray 128) and pastes exact ab;
    # only cv2 8-bit quantization and sRGB gamut clipping remain.
    assert report["n_pass"] == 2, report
    assert report["mean_delta_e"] < 3, report


def test_naive_render_leaves_background_neutral(synthetic):
    gray, masks, plan = synthetic
    rgb = render_naive(gray, masks, plan).astype(int)
    background = ~(masks["left"] | masks["right"])
    spread = (rgb.max(axis=2) - rgb.min(axis=2))[background]
    assert spread.max() <= 2  # neutral gray: r==g==b up to rounding


def test_hint_image_only_touches_eroded_masks(synthetic):
    gray, masks, plan = synthetic
    gray_rgb = np.stack([gray] * 3, axis=-1)
    hint = make_hint_image(gray_rgb, masks, plan, erosion=0.2)
    changed = (hint != gray_rgb).any(axis=2)
    inside_any = masks["left"] | masks["right"]
    assert changed.any()
    assert not (changed & ~inside_any).any()  # strokes never leave the masks
    # erosion means strictly fewer painted pixels than mask pixels
    assert changed.sum() < inside_any.sum()


def test_erode_frac_never_erases():
    tiny = np.zeros((20, 20), dtype=bool)
    tiny[9:11, 9:11] = True  # 4-pixel region
    eroded = erode_frac(tiny, 0.9)
    assert eroded.any()


def test_adherence_flags_wrong_colour(synthetic):
    gray, masks, plan = synthetic
    rgb = render_naive(gray, masks, plan)
    # sabotage: claim region 'right' should have been strongly red
    plan["regions"][1]["resolved_colour"] = {"space": "Lab", "L": 50, "a": 60, "b": 40}
    report = evaluate_adherence(rgb, masks, plan)
    right = [r for r in report["regions"] if r["region"] == "right"][0]
    assert not right["pass"]
    assert right["delta_e"] > 50
