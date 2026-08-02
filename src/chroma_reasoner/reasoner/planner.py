"""The reasoner pipeline: image + context prompt -> validated colour plan.

Flow (roadmap §4.2):
  1. The backend LLM (with vision) reads the grayscale image and the user's
     abstract prompt, and emits a *selection*: objects, grounding phrases,
     operative modifiers, estimated luminance. Structured outputs guarantee
     the shape; this module checks the *content* against the KB.
  2. Every region is resolved locally: kb.resolve(object, modifiers,
     measured_L=estimated_L) -> gamut-feasible colour + tolerance + rationale.
  3. Selection errors (unknown object, unknown modifier, out-of-range values)
     are fed back to the LLM for ONE repair round; anything still broken
     raises with the full error list — silent degradation would hide reasoner
     failures the evaluation needs to see.
  4. The assembled plan is schema-validated before it is returned.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..kb.engine import resolve
from ..kb.store import KBError, KnowledgeBase
from ..plan.schema import assert_valid
from .backend import Backend, image_block
from .prompts import repair_message, system_prompt, user_message_text


class ReasonerError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("reasoner selection invalid after repair:\n"
                         + "\n".join(f"  - {e}" for e in errors))


def _check_selection(kb: KnowledgeBase, selection: dict) -> list[str]:
    """Content-level validation the JSON schema can't do. Returns error list."""
    errors = []
    regions = selection.get("regions", [])
    if not 1 <= len(regions) <= 8:
        errors.append(f"expected 1-8 regions, got {len(regions)}")
    for i, region in enumerate(regions):
        where = f"regions[{i}] ({region.get('object', '?')})"
        try:
            kb.object_entry(region["object"])
        except KBError as e:
            errors.append(f"{where}: {e}")
        for mod in region.get("modifiers", []):
            try:
                kb.modifier_entry(mod["family"], mod["value"])
            except KBError as e:
                errors.append(f"{where}: {e}")
        if not 0 <= region.get("estimated_L", -1) <= 100:
            errors.append(f"{where}: estimated_L must be 0-100, got {region.get('estimated_L')}")
        if not 0 <= region.get("confidence", -1) <= 1:
            errors.append(f"{where}: confidence must be 0-1, got {region.get('confidence')}")
    for mod in selection.get("global_modifiers", []):
        try:
            kb.modifier_entry(mod["family"], mod["value"])
        except KBError as e:
            errors.append(f"global_modifiers: {e}")
    return errors


# Prompt cues for global modifiers. A global modifier reshapes the WHOLE
# image (it becomes the colorizer's prompt terms), so an invented one is the
# most expensive kind of hallucination available to the reasoner. Two rounds
# of prompt-tuning failed to stop the 7B inventing era and mood — "overcast
# day in an American town" yielded era:1940s + mood:nostalgic — so
# groundedness is enforced here instead of asked for.
#
# Deliberate precision-over-recall trade: a global must be traceable to the
# user's words. Per-region modifiers are unaffected; they are cheap to get
# wrong and the KB's applies_to already contains them.
GLOBAL_PROMPT_CUES: dict[tuple[str, str], tuple[str, ...]] = {
    ("geography", "usa"): ("usa", "u.s.", "american", "america"),
    ("geography", "britain"): ("britain", "british", "england", "english", "uk"),
    ("geography", "tropics"): ("tropic", "jungle", "rainforest"),
    ("geography", "mediterranean"): ("mediterranean", "greek", "italian", "spanish"),
    ("era", "1910s"): ("1910", "edwardian", "wwi", "first world war"),
    ("era", "1940s"): ("1940", "forties", "wartime", "wwii", "second world war"),
    ("era", "1950s"): ("1950", "fifties"),
    ("era", "1970s"): ("1970", "seventies"),
    ("time_of_day", "golden_hour"): ("golden hour", "sunset", "sunrise", "dusk", "dawn"),
    ("time_of_day", "night"): ("night", "evening", "nocturnal", "after dark"),
}


def _global_cues(family: str, value: str) -> tuple[str, ...]:
    """Words in the user's prompt that would justify this global modifier."""
    explicit = GLOBAL_PROMPT_CUES.get((family, value))
    if explicit:
        return explicit
    # default: the value itself, minus any underscores ("melancholic",
    # "overcast", "summer"), which covers mood/season/weather directly
    return (value.replace("_", " "),)


def ground_global_modifiers(modifiers: list[dict], user_prompt: str) -> tuple[list[dict], list[str]]:
    """Keep only global modifiers the user's prompt actually supports.

    Returns (kept, dropped_labels).
    """
    text = (user_prompt or "").lower()
    kept, dropped = [], []
    for modifier in modifiers:
        cues = _global_cues(modifier["family"], modifier["value"])
        if text and any(cue in text for cue in cues):
            kept.append(modifier)
        else:
            dropped.append(f"{modifier['family']}:{modifier['value']}")
    return kept, dropped


def _dedupe_regions(regions: list[dict]) -> list[dict]:
    """Small open models sometimes emit the same region twice verbatim
    (Phase-4 finding). Keep the first of each (object, grounding_phrase)."""
    seen: set[tuple[str, str]] = set()
    out = []
    for region in regions:
        key = (region["object"], region["grounding_phrase"].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(region)
    return out


def _selection_to_plan(kb: KnowledgeBase, selection: dict, image_id: str,
                       user_prompt: str) -> dict:
    regions = []
    used_ids: set[str] = set()
    for region in _dedupe_regions(selection["regions"]):
        mods = [{"family": m["family"], "value": m["value"],
                 "effect": kb.modifier_entry(m["family"], m["value"]).get("note", m["why"])}
                for m in region["modifiers"]]
        res = resolve(kb, region["object"], mods, measured_L=region["estimated_L"])
        rid = base = res.object
        n = 2
        while rid in used_ids:
            rid = f"{base}_{n}"
            n += 1
        used_ids.add(rid)
        plan_region = res.to_region(
            grounding_phrase=region["grounding_phrase"],
            modifiers=mods,
            region_id=rid,
            confidence=round(float(region["confidence"]), 2),
        )
        # keep the LLM's evidence trail alongside the KB's provenance
        plan_region["rationale"] = f"{region['rationale']} | {plan_region['rationale']}"
        regions.append(plan_region)

    plan: dict = {"plan_version": "1.0", "image_id": image_id,
                  "prompt": user_prompt, "regions": regions}
    grounded, dropped = ground_global_modifiers(selection.get("global_modifiers", []),
                                                user_prompt)
    if grounded:
        rationale = selection.get("scene_summary", "")
        if dropped:
            rationale += f" [dropped ungrounded globals: {', '.join(dropped)}]"
        plan["global"] = {
            "modifiers": [{"family": m["family"], "value": m["value"],
                           "effect": kb.modifier_entry(m["family"], m["value"]).get("note", m["why"])}
                          for m in grounded],
            "rationale": rationale.strip(),
        }
    return assert_valid(plan)


def re_resolve_with_masks(kb: KnowledgeBase, plan: dict, gray_l8, masks: dict) -> dict:
    """Replace each region's colour with one resolved at the mask-measured L.

    Phase-4 finding: open-VLM luminance estimates err by up to ΔL 60, so
    colours must be re-resolved once real masks exist. gray_l8 is the L
    channel in cv2 0-255 scaling; masks maps region id/object -> bool array.
    Deterministic, no model involved.
    """
    import numpy as np

    for region in plan["regions"]:
        key = region.get("id") or region["object"]
        mask = masks.get(key)
        if mask is None or not mask.any():
            continue
        measured_L = float(np.median(gray_l8[mask])) * 100.0 / 255.0
        res = resolve(kb, region["object"], region["modifiers"], measured_L=measured_L)
        region["resolved_colour"] = res.resolved.to_plan()
        region["tolerance_delta_e"] = round(res.tolerance_delta_e, 1)
        region["rationale"] += f" | re-resolved at mask-measured L={measured_L:.0f}"
    return assert_valid(plan)


def reason_plan(kb: KnowledgeBase, backend: Backend, image_path: str | Path,
                user_prompt: str = "", image_id: str | None = None) -> dict:
    """Grayscale image + abstract prompt -> validated, KB-resolved colour plan."""
    image_path = str(image_path)
    image_id = image_id or Path(image_path).stem

    system = system_prompt(kb)
    messages = [{"role": "user",
                 "content": [image_block(image_path),
                             {"type": "text", "text": user_message_text(user_prompt)}]}]
    selection = backend.complete(system, messages)

    errors = _check_selection(kb, selection)
    if errors:
        # one repair round: show the LLM its own output and the errors
        messages.append({"role": "assistant", "content": json.dumps(selection)})
        messages.append({"role": "user", "content": repair_message(errors)})
        selection = backend.complete(system, messages)
        errors = _check_selection(kb, selection)
        if errors:
            # Salvage policy (Phase-4 live-run finding): when the surviving
            # errors are confined to individual regions and enough valid
            # regions remain, drop the broken ones instead of failing the
            # image. Global/structural errors still raise.
            selection, dropped = _drop_broken_regions(kb, selection)
            if dropped and not _check_selection(kb, selection):
                selection["scene_summary"] = (selection.get("scene_summary", "")
                                              + f" [dropped invalid regions: {', '.join(dropped)}]")
            else:
                raise ReasonerError(errors)

    return _selection_to_plan(kb, selection, image_id, user_prompt)


def _drop_broken_regions(kb: KnowledgeBase, selection: dict) -> tuple[dict, list[str]]:
    """Remove regions that individually fail validation, if >=2 valid remain.

    Returns (possibly-modified selection, names of dropped regions). Leaves
    the selection untouched when salvage isn't possible.
    """
    valid, dropped = [], []
    for region in selection.get("regions", []):
        probe = {"regions": [region], "global_modifiers": []}
        if _check_selection(kb, probe):
            dropped.append(str(region.get("object", "?")))
        else:
            valid.append(region)
    # global_modifiers errors are not region-salvageable
    global_errors = bool(_check_selection(kb, {"regions": valid[:1] or [],
                                               "global_modifiers": selection.get("global_modifiers", [])})) \
        if valid else True
    if len(valid) >= 2 and dropped and not global_errors:
        return {**selection, "regions": valid}, dropped
    return selection, []
