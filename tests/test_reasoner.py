"""Reasoner pipeline tests with a mock backend — no API calls.

The mock returns canned selections, so these tests cover everything except
the model itself: prompt construction, content validation, the repair round,
KB resolution, and plan assembly/validation.
"""

import numpy as np
import pytest

from chroma_reasoner.kb import load_kb
from chroma_reasoner.plan import validate_plan
from chroma_reasoner.reasoner.planner import ReasonerError, reason_plan
from chroma_reasoner.reasoner.prompts import kb_vocabulary, system_prompt


@pytest.fixture(scope="module")
def kb():
    return load_kb()


@pytest.fixture()
def gray_png(tmp_path):
    import cv2

    path = tmp_path / "000000002299.png"
    cv2.imwrite(str(path), np.full((60, 80), 110, dtype=np.uint8))
    return path


class MockBackend:
    def __init__(self, *selections):
        self.selections = list(selections)
        self.calls: list[list[dict]] = []

    def complete(self, system, messages):
        self.calls.append(messages)
        return self.selections[len(self.calls) - 1]


GOOD_SELECTION = {
    "scene_summary": "A 1940s British school class photo in front of a stone wall, overcast.",
    "global_modifiers": [
        {"family": "era", "value": "1940s", "why": "period photograph"}
    ],
    "regions": [
        {"object": "stone_wall", "grounding_phrase": "the stone wall behind the children",
         "estimated_L": 45, "modifiers": [
             {"family": "geography", "value": "britain", "why": "British institutional stone"}],
         "confidence": 0.8, "rationale": "Weathered institutional stone."},
        {"object": "jumper", "grounding_phrase": "the dark jumper of the boy at front left",
         "estimated_L": 28, "modifiers": [
             {"family": "era", "value": "1940s", "why": "wartime knitwear"},
             {"family": "mood", "value": "nostalgic", "why": "aged photo feel"}],
         "confidence": 0.5, "rationale": "Luminance under-determines garment colour."},
        {"object": "pavement", "grounding_phrase": "the paved schoolyard ground",
         "estimated_L": 40, "modifiers": [
             {"family": "weather", "value": "overcast", "why": "flat light"}],
         "confidence": 0.85, "rationale": "Asphalt under overcast light."},
    ],
}


def test_happy_path_produces_valid_plan(kb, gray_png):
    backend = MockBackend(GOOD_SELECTION)
    plan = reason_plan(kb, backend, gray_png, user_prompt="1940s British school photo")
    assert validate_plan(plan) == []
    assert plan["image_id"] == "000000002299"
    assert len(plan["regions"]) == 3
    assert len(backend.calls) == 1
    # jumper is an alias of dress; colours came from the KB, luminance-conditioned
    jumper = plan["regions"][1]
    assert jumper["object"] == "dress"
    assert jumper["base_prior"] == "kb:dress"
    assert jumper["resolved_colour"]["L"] == 28
    # era 1940s + mood applied; global block preserved
    assert plan["global"]["modifiers"][0]["value"] == "1940s"


def test_repair_round_fixes_bad_selection(kb, gray_png):
    bad = {**GOOD_SELECTION,
           "regions": [{**GOOD_SELECTION["regions"][0], "object": "unicorn"}]}
    backend = MockBackend(bad, GOOD_SELECTION)
    plan = reason_plan(kb, backend, gray_png)
    assert validate_plan(plan) == []
    assert len(backend.calls) == 2
    # the repair turn shows the model its own output and the errors
    repair_turn = backend.calls[1]
    assert any("unicorn" in str(m.get("content")) for m in repair_turn)


def test_unrepaired_selection_raises_with_errors(kb, gray_png):
    bad = {**GOOD_SELECTION,
           "regions": [{**GOOD_SELECTION["regions"][0], "object": "unicorn",
                        "estimated_L": 400}]}
    backend = MockBackend(bad, bad)
    with pytest.raises(ReasonerError) as exc:
        reason_plan(kb, backend, gray_png)
    assert any("unicorn" in e for e in exc.value.errors)
    assert any("estimated_L" in e for e in exc.value.errors)


def test_broken_regions_dropped_when_enough_valid_remain(kb, gray_png):
    """Salvage policy: region-local failures (e.g. 'tennis_racket', a real
    live-run case) are dropped after the failed repair round instead of
    sinking the whole image, as long as >=2 valid regions survive."""
    bad = {**GOOD_SELECTION, "regions": GOOD_SELECTION["regions"] + [
        {**GOOD_SELECTION["regions"][0], "object": "tennis_racket"}]}
    backend = MockBackend(bad, bad)   # repair round also fails
    plan = reason_plan(kb, backend, gray_png)
    assert len(plan["regions"]) == 3           # the 3 valid ones survive
    assert len(backend.calls) == 2             # repair round was attempted
    objects = {r["object"] for r in plan["regions"]}
    assert "tennis_racket" not in objects


def test_salvage_refuses_when_too_few_valid(kb, gray_png):
    bad = {**GOOD_SELECTION, "regions": [
        GOOD_SELECTION["regions"][0],
        {**GOOD_SELECTION["regions"][1], "object": "unicorn"},
        {**GOOD_SELECTION["regions"][2], "object": "dragon"},
    ]}
    backend = MockBackend(bad, bad)
    with pytest.raises(ReasonerError):
        reason_plan(kb, backend, gray_png)


def test_duplicate_objects_get_unique_region_ids(kb, gray_png):
    sel = {**GOOD_SELECTION, "global_modifiers": [], "regions": [
        {**GOOD_SELECTION["regions"][1]},
        {**GOOD_SELECTION["regions"][1],
         "grounding_phrase": "the cardigan of the girl on the right"},
    ]}
    plan = reason_plan(kb, MockBackend(sel), gray_png)
    ids = [r["id"] for r in plan["regions"]]
    assert len(ids) == len(set(ids))


def test_duplicate_selections_are_deduped(kb, gray_png):
    """Verbatim duplicate regions (a real Qwen-7B failure mode) collapse to one."""
    sel = {**GOOD_SELECTION, "regions": [
        GOOD_SELECTION["regions"][0],
        {**GOOD_SELECTION["regions"][0]},   # exact duplicate
        GOOD_SELECTION["regions"][2],
    ]}
    plan = reason_plan(kb, MockBackend(sel), gray_png)
    assert len(plan["regions"]) == 2


def test_re_resolve_with_masks_uses_measured_L(kb, gray_png):
    """Colours re-resolve at the mask-measured luminance (ΔL~60 estimate
    errors observed from the 7B model)."""
    from chroma_reasoner.reasoner import re_resolve_with_masks

    sel = {**GOOD_SELECTION, "global_modifiers": [],
           "regions": [{**GOOD_SELECTION["regions"][0], "estimated_L": 20}]}
    plan = reason_plan(kb, MockBackend(sel), gray_png)
    assert plan["regions"][0]["resolved_colour"]["L"] == 20

    gray = np.full((40, 40), 204, dtype=np.uint8)   # true L ~ 80
    masks = {"stone_wall": np.ones((40, 40), dtype=bool)}
    plan = re_resolve_with_masks(kb, plan, gray, masks)
    assert plan["regions"][0]["resolved_colour"]["L"] == 80
    assert "re-resolved at mask-measured L=80" in plan["regions"][0]["rationale"]
    from chroma_reasoner.plan import validate_plan as vp
    assert vp(plan) == []


def test_system_prompt_embeds_kb_vocabulary(kb):
    text = system_prompt(kb)
    assert "school_bus" in text
    assert "era:1940s" in text
    assert "mood:melancholic" in text
    # deterministic (cache-friendly): two renders are byte-identical
    assert text == system_prompt(kb)


def test_vocabulary_lists_aliases(kb):
    vocab = kb_vocabulary(kb)
    assert "jumper" in vocab   # dress alias, needed for garment selection


def test_ungrounded_globals_are_dropped(kb, gray_png):
    """Global modifiers reshape the whole image via the colorizer prompt, so
    an invented one is the costliest hallucination available. Two rounds of
    prompt-tuning did not stop the 7B ("overcast day in an American town"
    -> era:1940s, mood:nostalgic), so groundedness is enforced here."""
    sel = {**GOOD_SELECTION, "global_modifiers": [
        {"family": "weather", "value": "overcast", "why": "stated"},
        {"family": "geography", "value": "usa", "why": "stated"},
        {"family": "era", "value": "1940s", "why": "a vintage car"},
        {"family": "mood", "value": "nostalgic", "why": "vibes"},
    ]}
    plan = reason_plan(kb, MockBackend(sel), gray_png,
                       user_prompt="overcast day in an American town")
    kept = {m["value"] for m in plan["global"]["modifiers"]}
    assert kept == {"overcast", "usa"}                 # prompt-supported only
    assert "era:1940s" in plan["global"]["rationale"]  # the drop is recorded
    assert "mood:nostalgic" in plan["global"]["rationale"]


def test_empty_prompt_yields_no_global_block(kb, gray_png):
    sel = {**GOOD_SELECTION, "global_modifiers": [
        {"family": "era", "value": "1940s", "why": "guessed"}]}
    plan = reason_plan(kb, MockBackend(sel), gray_png, user_prompt="")
    assert "global" not in plan


def test_grounding_matches_synonyms_not_just_literals(kb):
    from chroma_reasoner.reasoner.planner import ground_global_modifiers

    mods = [{"family": "geography", "value": "britain", "why": ""},
            {"family": "era", "value": "1940s", "why": ""},
            {"family": "geography", "value": "tropics", "why": ""}]
    kept, dropped = ground_global_modifiers(mods, "a wartime British schoolroom")
    assert {m["value"] for m in kept} == {"britain", "1940s"}   # 'British', 'wartime'
    assert dropped == ["geography:tropics"]


def test_grounding_declines_vague_references(kb):
    """Precision over recall: 'the war' does not pin a decade, so it must not
    licence era:1940s. A missed global costs one prompt term; an invented one
    restyles the whole image."""
    from chroma_reasoner.reasoner.planner import ground_global_modifiers

    kept, dropped = ground_global_modifiers(
        [{"family": "era", "value": "1940s", "why": ""}], "a schoolroom during the war")
    assert kept == [] and dropped == ["era:1940s"]


def test_region_modifiers_are_not_grounded_against_the_prompt(kb, gray_png):
    """Only globals are gated - per-region modifiers stay the reasoner's call,
    and the KB's applies_to already constrains them."""
    plan = reason_plan(kb, MockBackend({**GOOD_SELECTION, "global_modifiers": []}),
                       gray_png, user_prompt="")
    jumper = plan["regions"][1]
    assert {m["value"] for m in jumper["modifiers"]} == {"1940s", "nostalgic"}


def test_system_prompt_pushes_for_a_global_block(kb):
    """Showcase finding: the global block is the only lever over unmasked
    pixels, and only 2/23 plans carried one under the old 'use sparingly'
    wording."""
    text = system_prompt(kb)
    assert "global_modifiers" in text
    assert "only control over everything you do not select" in text
    assert "sparingly" not in text
    # ...but bounded: pushing for globals made the 7B invent era/season
    # (a "vintage car" in a mirror became era:1940s on a modern street scene)
    assert "At most 2 global modifiers" in text
    assert "Do not infer a period from a" in text
    assert "one old-looking car does not make a 1940s photograph" in text
