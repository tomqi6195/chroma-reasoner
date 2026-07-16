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
    if selection.get("global_modifiers"):
        plan["global"] = {
            "modifiers": [{"family": m["family"], "value": m["value"],
                           "effect": kb.modifier_entry(m["family"], m["value"]).get("note", m["why"])}
                          for m in selection["global_modifiers"]],
            "rationale": selection.get("scene_summary", ""),
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
            raise ReasonerError(errors)

    return _selection_to_plan(kb, selection, image_id, user_prompt)
