"""The KB-off arm of the ablation: same regions, colours chosen directly by
the VLM instead of resolved from the KB.

Given an existing (KB-resolved) plan, the model is shown the grayscale image,
the context prompt, and the region list, and asked for one specific hex
colour per region. Everything else — grounding phrases, masks, luminance —
is held fixed, so any scoring difference is attributable to the colour
source alone (roadmap §6 kill-criterion: "if ΔE-to-intent and human
preference don't move, the KB is redundant").
"""

from __future__ import annotations

import copy

from ..plan.colors import LabColor, project_chroma_into_gamut, srgb_to_lab
from ..plan.schema import assert_valid
from ..reasoner.backend import Backend, image_block

ABLATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["colours"],
    "properties": {
        "colours": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "hex"],
                "properties": {
                    "id": {"type": "string", "description": "the region id, exactly as given"},
                    "hex": {"type": "string", "description": "chosen colour, like #a3b2c1"},
                },
            },
        }
    },
}


def ablation_format_contract() -> str:
    return ('\nRespond with ONLY a JSON object, no commentary, exactly:\n'
            '{"colours": [{"id": "region_id", "hex": "#rrggbb"}]}\n'
            'One entry per region, ids exactly as given.')


def _hex_to_lab(hex_str: str) -> LabColor:
    h = hex_str.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"bad hex colour: {hex_str!r}")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return srgb_to_lab((r, g, b))


ABLATION_SYSTEM = """You are an expert photo colourist. You will see a GRAYSCALE
photograph, a context prompt, and a list of regions (with the phrase locating each).
Choose ONE specific, realistic colour per region, honouring the context prompt and
the region's visible lightness. Output hex colours only."""


def llm_color_plan(plan: dict, backend: Backend, image_path: str) -> dict:
    """Build the KB-off variant of a plan: colours from the model directly."""
    region_lines = []
    for region in plan["regions"]:
        L = region["resolved_colour"]["L"]
        mods = ", ".join(f"{m['family']}:{m['value']}" for m in region.get("modifiers", []))
        region_lines.append(f"- id={region['id']!r} object={region['object']} "
                            f"lightness L~{L:g} context=[{mods}] — {region['grounding_phrase']}")
    prompt_txt = plan.get("prompt") or "(none)"
    user_text = (f"Context prompt: {prompt_txt}\n\nRegions:\n" + "\n".join(region_lines)
                 + "\n\nChoose one colour per region.")

    selection = backend.complete(
        ABLATION_SYSTEM,
        [{"role": "user", "content": [image_block(image_path),
                                      {"type": "text", "text": user_text}]}],
        schema=ABLATION_SCHEMA,
        format_contract=ablation_format_contract(),
    )
    by_id = {c["id"]: c["hex"] for c in selection["colours"]}

    out = copy.deepcopy(plan)
    for region in out["regions"]:
        hex_str = by_id.get(region["id"])
        if hex_str is None:
            raise ValueError(f"model omitted colour for region {region['id']!r}")
        lab = _hex_to_lab(hex_str)
        # keep the region's luminance (colorization can't change L anyway);
        # take the model's chroma, made feasible at that L
        L = region["resolved_colour"]["L"]
        feasible = project_chroma_into_gamut(LabColor(L, lab.a, lab.b))
        region["resolved_colour"] = {"space": "Lab", "L": round(L, 1),
                                     "a": round(feasible.a, 1), "b": round(feasible.b, 1)}
        region["base_prior"] = None
        region["rationale"] += f" | ABLATION: colour {hex_str} chosen by model, KB bypassed"
    return assert_valid(out)
