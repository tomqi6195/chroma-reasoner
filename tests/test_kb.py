"""KB tests: does composition produce sane distributions? (roadmap §5 Phase 3:
'give it its own tests')."""

import math

import pytest

from chroma_reasoner.kb import load_kb, resolve
from chroma_reasoner.kb.engine import compose
from chroma_reasoner.kb.store import KBError
from chroma_reasoner.plan import validate_plan
from chroma_reasoner.plan.colors import LabColor, is_in_srgb_gamut


@pytest.fixture(scope="module")
def kb():
    return load_kb()


def _mod(family, value):
    return {"family": family, "value": value, "effect": ""}


def test_kb_loads_and_validates(kb):
    assert len(kb.objects) >= 15
    assert {"era", "season", "weather", "geography", "mood", "time_of_day"} <= set(kb.modifiers)


def test_weights_normalized_after_composition(kb):
    modes, *_ = compose(kb, "foliage", [_mod("season", "autumn"), _mod("mood", "melancholic")])
    assert sum(m["weight"] for m in modes) == pytest.approx(1.0)


def test_base_modes_in_gamut_at_midrange_L(kb):
    """Every base mode must be sRGB-feasible at the middle of its L_range."""
    for name, entry in kb.objects.items():
        for mode in entry["modes"]:
            lo, hi = mode.get("L_range", [0, 100])
            lab = LabColor((lo + hi) / 2, *mode["ab"])
            assert is_in_srgb_gamut(lab, tolerance=2.0), (name, mode["name"], lab)


def test_autumn_shifts_foliage_toward_orange(kb):
    """The roadmap's canonical season-object interaction."""
    base = resolve(kb, "foliage", [], measured_L=45)
    autumn = resolve(kb, "foliage", [_mod("season", "autumn")], measured_L=45)
    assert autumn.chosen_mode == "senescent"
    assert autumn.resolved.a > base.resolved.a   # toward red
    assert "season:autumn" in autumn.applied


def test_1910s_melancholic_dress_is_muted_and_cool(kb):
    """The Phase-1 canonical example, now derived from the KB instead of
    hand-picked: era mutes every chromatic mode, mood cools the result.
    (The resolved colour lands near-neutral slate — a cool shift on a neutral
    garment *adds* a little chroma, which is correct, so mutedness is asserted
    mode-by-mode on the chromatic modes.)"""
    base_modes, *_ = compose(kb, "dress", [])
    styled_modes, *_ = compose(kb, "dress", [_mod("era", "1910s"), _mod("mood", "melancholic")])
    base_c = {m["name"]: math.hypot(*m["ab"]) for m in base_modes}
    styled_c = {m["name"]: math.hypot(*m["ab"]) for m in styled_modes}
    for name in ("warm", "cool", "earth"):   # chromatic modes must all mute
        assert styled_c[name] < base_c[name], name

    styled = resolve(kb, "dress", [_mod("era", "1910s"), _mod("mood", "melancholic")],
                     measured_L=45)
    base = resolve(kb, "dress", [], measured_L=45)
    assert styled.resolved.b < base.resolved.b            # cooler
    assert math.hypot(styled.resolved.a, styled.resolved.b) < 15   # still muted overall


def test_autumn_does_nothing_to_a_car(kb):
    """'Autumn does nothing to a car and everything to foliage' — factor
    selection routes through objects (roadmap §1)."""
    plain = resolve(kb, "car", [], measured_L=40)
    autumn = resolve(kb, "car", [_mod("season", "autumn")], measured_L=40)
    assert autumn.applied == []
    assert "season:autumn" in autumn.skipped
    assert (autumn.resolved.a, autumn.resolved.b) == (plain.resolved.a, plain.resolved.b)


def test_1950s_car_gains_pastel_mode(kb):
    styled = resolve(kb, "car", [_mod("era", "1950s")], measured_L=70)
    names = {m["name"] for m in styled.modes}
    assert "pastel_two_tone" in names
    assert styled.chosen_mode == "pastel_two_tone"


def test_school_bus_chroma_clamps_at_low_L(kb):
    """Phase-2 finding as a KB behaviour: resolution at the dim mirror's
    luminance must return a feasible dark gold, not canonical bus yellow."""
    bright = resolve(kb, "school_bus", [], measured_L=75)
    dark = resolve(kb, "school_bus", [], measured_L=20)
    assert bright.resolved.b > 55
    assert dark.resolved.b < 35
    assert is_in_srgb_gamut(dark.resolved, tolerance=2.0)


def test_unknown_object_raises(kb):
    with pytest.raises(KBError):
        resolve(kb, "unicorn", [])


def test_unknown_modifier_raises(kb):
    with pytest.raises(KBError):
        resolve(kb, "dress", [_mod("era", "3020s")])


def test_alias_resolution(kb):
    res = resolve(kb, "t_shirt", [])
    assert res.object == "dress"


def test_resolution_emits_valid_plan_region(kb):
    mods = [_mod("era", "1940s"), _mod("mood", "melancholic")]
    res = resolve(kb, "jumper", mods, measured_L=30)
    region = res.to_region("the dark knitted jumper of the boy at the front",
                           [{**m, "effect": "muted"} for m in mods], region_id="boy_jumper")
    plan = {"plan_version": "1.0", "regions": [region]}
    assert validate_plan(plan) == [], validate_plan(plan)


def test_composition_order_matters(kb):
    """scale-then-shift != shift-then-scale; order sensitivity is intended
    and must stay stable (documented in docs/phase3.md)."""
    a = resolve(kb, "grass", [_mod("mood", "melancholic"), _mod("time_of_day", "golden_hour")],
                measured_L=50)
    b = resolve(kb, "grass", [_mod("time_of_day", "golden_hour"), _mod("mood", "melancholic")],
                measured_L=50)
    assert (a.resolved.a, a.resolved.b) != (b.resolved.a, b.resolved.b)
