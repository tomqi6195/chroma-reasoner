"""Prompt construction: the KB vocabulary is embedded in the system prompt so
the LLM can only select entries that exist. The vocabulary text is generated
deterministically from the KB (sorted), so the system prompt is stable across
requests — prompt-cache friendly."""

from __future__ import annotations

from ..kb.store import KnowledgeBase

CANONICAL_ORDER = ["era", "geography", "season", "weather", "time_of_day", "mood"]


def kb_vocabulary(kb: KnowledgeBase) -> str:
    lines = ["## Object classes (use these names or their aliases, exactly)"]
    for name in sorted(kb.objects):
        entry = kb.objects[name]
        aliases = ", ".join(entry.get("aliases", []))
        alias_txt = f" (aliases: {aliases})" if aliases else ""
        cats = ", ".join(entry.get("categories", []))
        modes = ", ".join(m["name"] for m in entry["modes"])
        lines.append(f"- {name}{alias_txt} [categories: {cats}] modes: {modes}")

    lines.append("")
    lines.append("## Modifier catalog (family:value -> what it does, and what it applies to)")
    for family in sorted(kb.modifiers, key=lambda f: (CANONICAL_ORDER.index(f)
                                                      if f in CANONICAL_ORDER else 99)):
        for value in sorted(kb.modifiers[family]):
            entry = kb.modifiers[family][value]
            applies = ", ".join(entry["applies_to"])
            lines.append(f"- {family}:{value} (applies to: {applies}) — {entry.get('note', '')}")
    return "\n".join(lines)


def system_prompt(kb: KnowledgeBase) -> str:
    return f"""You are the reasoning stage of a context-aware image colorization system.

You will be shown a GRAYSCALE photograph and a user's context prompt. Your job is
SELECTION, not colour choice: identify the major colourable regions, match each to an
object class from the knowledge-base vocabulary below, select which contextual modifiers
are operative for each object, and estimate each region's luminance. An explicit
knowledge base resolves the actual colours from your selections — you never output
colours.

Rules:
- Use ONLY object names/aliases and modifier family:value pairs from the vocabulary
  below. Anything else will be rejected.
- Pick 3-6 regions covering the visually dominant, colourable areas. Prefer large
  coherent regions (sky, ground, walls, dominant garments) plus any object with a
  strong colour identity.
- grounding_phrase must be specific enough for an open-vocabulary detector to find
  exactly that region (e.g. "the dark jumper of the boy sitting front left", not
  "clothing").
- estimated_L: read it off the grayscale image (0 = black, 50 = mid-grey, 100 = white).
  This matters: the achievable colour depends on the region's true lightness.
- Modifiers: include only factors that genuinely act on that object ("autumn does
  nothing to a car"). Order them: era, geography, season, weather, time_of_day, mood
  (facts first, stylistic intent last). If the user's prompt implies a factor that
  does not apply to an object, leave it off that object.
- global_modifiers: factors that act image-wide rather than through one object
  (era film rendering, overall mood). Use sparingly.
- confidence: high (>0.8) when content strongly constrains colour (sky, foliage,
  regulation objects), low (<0.6) when luminance under-determines it (garments).
  Vary it region by region; identical confidence everywhere is wrong.
- Never invent contextual factors: only select era/geography/season values stated
  in the user's prompt or unmistakable in the image. No prompt mention of a
  period means no era modifier.
- Each region exactly once — no duplicates.
- Match object classes to the scene: interior classes (wall_interior, radiator)
  only indoors; exterior masonry is stone_wall or brick.
- rationale: one or two sentences, concrete, referencing the evidence.

{kb_vocabulary(kb)}"""


def user_message_text(user_prompt: str) -> str:
    if user_prompt.strip():
        return (f"Context prompt from the user: {user_prompt!r}\n\n"
                "Analyze the grayscale image above and produce your selection.")
    return ("No context prompt was given; select objects and only the modifiers "
            "evident from the image itself.\n\nAnalyze the grayscale image above "
            "and produce your selection.")


def json_format_instructions() -> str:
    """Output-format contract for backends WITHOUT structured outputs (open
    VLMs). Describes the selection JSON concretely; the backend still
    extracts/validates, and the planner's repair round catches the rest."""
    return """
Respond with ONLY a JSON object — no markdown fences, no commentary — exactly this shape:

{
  "scene_summary": "one or two sentences about the scene",
  "global_modifiers": [
    {"family": "era", "value": "1940s", "why": "reason"}
  ],
  "regions": [
    {
      "object": "a KB object class or alias, exactly as listed",
      "grounding_phrase": "specific phrase locating this region",
      "estimated_L": 45,
      "modifiers": [
        {"family": "weather", "value": "overcast", "why": "reason"}
      ],
      "confidence": 0.8,
      "rationale": "one or two sentences"
    }
  ]
}

estimated_L is 0-100 (0 black, 50 mid-grey, 100 white). confidence is 0-1.
global_modifiers and modifiers may be empty arrays. 3-6 regions."""


def repair_message(errors: list[str]) -> str:
    listing = "\n".join(f"- {e}" for e in errors)
    return (f"Your selection had problems that must be fixed:\n{listing}\n\n"
            "Re-emit the FULL corrected selection. Use only object names/aliases and "
            "modifier family:value pairs from the vocabulary in the system prompt.")
