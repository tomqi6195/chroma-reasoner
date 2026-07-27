"""Counterfactual protocol tests: condition application and the separation /
direction measurements."""

import pytest

from chroma_reasoner.eval.counterfactual import (CONDITIONS, apply_condition,
                                                 condition_variants, evaluate_contrast,
                                                 Contrast, format_contrast)
from chroma_reasoner.kb import load_kb
from chroma_reasoner.plan import validate_plan


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def _plan(objects_with_L):
    return {
        "plan_version": "1.0", "image_id": "img", "prompt": "",
        "regions": [{
            "id": obj, "object": obj, "grounding_phrase": f"the {obj}",
            "modifiers": [{"family": "weather", "value": "overcast", "effect": "x"}],
            "resolved_colour": {"space": "Lab", "L": L, "a": 10, "b": 10},
            "tolerance_delta_e": 10, "confidence": 0.8, "rationale": "t",
        } for obj, L in objects_with_L],
    }


def test_apply_condition_preserves_regions_and_luminance(kb):
    plan = _plan([("dress", 45), ("foliage", 40)])
    out = apply_condition(kb, plan, CONDITIONS["mood_melancholic"])

    assert validate_plan(out) == []
    assert [r["id"] for r in out["regions"]] == ["dress", "foliage"]
    assert [r["resolved_colour"]["L"] for r in out["regions"]] == [45, 40]
    assert out["prompt"] == CONDITIONS["mood_melancholic"].prompt
    # the condition's modifier is present, the unrelated one is preserved
    families = {m["family"] for m in out["regions"][0]["modifiers"]}
    assert families == {"weather", "mood"}


def test_conditions_do_not_stack(kb):
    """Applying a second condition of the same family replaces the first."""
    plan = _plan([("dress", 45)])
    melancholic = apply_condition(kb, plan, CONDITIONS["mood_melancholic"])
    cheerful = apply_condition(kb, melancholic, CONDITIONS["mood_cheerful"])
    moods = [m["value"] for m in cheerful["regions"][0]["modifiers"]
             if m["family"] == "mood"]
    assert moods == ["cheerful"]


def test_mood_contrast_separates_in_the_declared_direction(kb):
    plan = _plan([("dress", 45), ("wall_interior", 60), ("sky", 70)])
    variants = condition_variants(kb, plan, ["mood_melancholic", "mood_cheerful"])
    res = evaluate_contrast({"img": variants["mood_melancholic"]},
                            {"img": variants["mood_cheerful"]},
                            Contrast("mood_melancholic", "mood_cheerful",
                                     (("chroma", "lower"), ("warmth", "lower"))))
    assert res["n_active"] >= 2                       # mood applies broadly
    assert res["median_separation"] > 1
    for metric in ("chroma", "warmth"):
        d = res["directions"][metric]
        assert d["as_expected"] > d["against"], (metric, d)


def test_inapplicable_condition_leaves_colour_untouched(kb):
    """'Autumn does nothing to a car' - the region is inactive, not wrong."""
    plan = _plan([("car", 40)])
    variants = condition_variants(kb, plan, ["season_autumn", "season_summer"])
    res = evaluate_contrast({"img": variants["season_autumn"]},
                            {"img": variants["season_summer"]},
                            Contrast("season_autumn", "season_summer",
                                     (("redness", "higher"),)))
    assert res["n_regions"] == 1
    assert res["n_active"] == 0                       # no colour movement
    assert res["median_separation"] == 0


def test_autumn_reddens_foliage(kb):
    plan = _plan([("foliage", 45)])
    variants = condition_variants(kb, plan, ["season_autumn", "season_summer"])
    res = evaluate_contrast({"img": variants["season_autumn"]},
                            {"img": variants["season_summer"]},
                            Contrast("season_autumn", "season_summer",
                                     (("redness", "higher"),)))
    assert res["n_active"] == 1
    assert res["directions"]["redness"]["as_expected"] == 1


def test_unpaired_images_are_skipped(kb):
    plan = _plan([("dress", 45)])
    v = condition_variants(kb, plan, ["mood_melancholic", "mood_cheerful"])
    res = evaluate_contrast({"img_a": v["mood_melancholic"]},
                            {"img_b": v["mood_cheerful"]},
                            Contrast("mood_melancholic", "mood_cheerful",
                                     (("chroma", "lower"),)))
    assert res["n_regions"] == 0


def test_format_contrast_is_readable(kb):
    plan = _plan([("dress", 45)])
    v = condition_variants(kb, plan, ["mood_melancholic", "mood_cheerful"])
    res = evaluate_contrast({"img": v["mood_melancholic"]}, {"img": v["mood_cheerful"]},
                            Contrast("mood_melancholic", "mood_cheerful",
                                     (("chroma", "lower"),), "note here"))
    text = format_contrast(res)
    assert "mood_melancholic vs mood_cheerful" in text and "separation" in text
