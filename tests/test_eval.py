"""Phase-5 eval tests: reference scoring on synthetic data, and the ablation
arm with a mock backend."""

import numpy as np
import pytest

from chroma_reasoner.eval import llm_color_plan, plan_vs_reference
from chroma_reasoner.plan import validate_plan
from chroma_reasoner.plan.colors import LabColor, lab_to_srgb


def _plan(a, b, tol=10):
    return {
        "plan_version": "1.0",
        "image_id": "synthetic",
        "prompt": "test scene",
        "regions": [{
            "id": "patch", "object": "thing", "grounding_phrase": "the patch",
            "modifiers": [],
            "resolved_colour": {"space": "Lab", "L": 50, "a": a, "b": b},
            "tolerance_delta_e": tol, "confidence": 0.8, "rationale": "t",
        }],
    }


def _scene(lab: LabColor):
    r, g, b = (round(c * 255) for c in lab_to_srgb(lab))
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[:] = (r, g, b)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    return img, {"patch": mask}


def test_reference_score_zero_when_plan_matches_reality():
    img, masks = _scene(LabColor(50, 20, -10))
    res = plan_vs_reference(_plan(20, -10), masks, img)
    assert res["mean_delta_e"] < 1.5


def test_reference_score_large_when_plan_wrong():
    img, masks = _scene(LabColor(50, 20, -10))
    res = plan_vs_reference(_plan(-30, 40), masks, img)
    assert res["mean_delta_e"] > 40
    row = res["regions"][0]
    assert row["reference_ab"] == pytest.approx([20, -10], abs=1.5)


def test_gray_guess_for_colourful_region_flagged_as_undercommitted():
    """The desaturation exploit: a gray guess scores moderate dE but must be
    caught by the chroma deficit (the PSNR-bias rediscovery)."""
    img, masks = _scene(LabColor(50, 30, 25))       # colourful reality
    res = plan_vs_reference(_plan(0, 0), masks, img)  # plan says gray
    row = res["regions"][0]
    assert row["chroma_deficit"] > 30
    assert res["n_undercommitted"] == 1


def test_gray_guess_for_gray_region_not_penalized():
    img, masks = _scene(LabColor(50, 1, 2))          # reality is near-neutral
    res = plan_vs_reference(_plan(0, 0), masks, img)
    assert res["regions"][0]["chroma_deficit"] < 4
    assert res["n_undercommitted"] == 0


class MockColorBackend:
    def __init__(self, colours):
        self.colours = colours
        self.kwargs = None

    def complete(self, system, messages, **kwargs):
        self.kwargs = kwargs
        return {"colours": self.colours}


def test_llm_color_plan_swaps_colours_and_keeps_L(tmp_path):
    import cv2

    img_path = tmp_path / "synthetic.png"
    cv2.imwrite(str(img_path), np.full((20, 20), 128, dtype=np.uint8))

    plan = _plan(16, 28)   # KB colour: warm brown
    backend = MockColorBackend([{"id": "patch", "hex": "#4060a0"}])  # model picks blue
    out = llm_color_plan(plan, backend, str(img_path))

    assert validate_plan(out) == []
    c = out["regions"][0]["resolved_colour"]
    assert c["L"] == 50                      # luminance held fixed
    assert c["b"] < 0                        # blue chroma from the model
    assert out["regions"][0]["base_prior"] is None
    assert "ABLATION" in out["regions"][0]["rationale"]
    # original plan untouched
    assert plan["regions"][0]["resolved_colour"]["a"] == 16
    # the ablation contract was passed through to the backend
    assert backend.kwargs["schema"]["required"] == ["colours"]
    assert "colours" in backend.kwargs["format_contract"]


def test_llm_color_plan_missing_region_raises(tmp_path):
    import cv2

    img_path = tmp_path / "synthetic.png"
    cv2.imwrite(str(img_path), np.full((20, 20), 128, dtype=np.uint8))
    with pytest.raises(ValueError):
        llm_color_plan(_plan(0, 0), MockColorBackend([]), str(img_path))
