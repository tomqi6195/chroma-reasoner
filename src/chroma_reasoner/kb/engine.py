"""Composition engine: apply selected modifiers to an object's colour prior
and resolve a concrete, gamut-feasible colour.

Composition semantics (the research artifact of Phase 3 — documented in
docs/phase3.md):

- Modifiers apply **in the order given** (the plan schema's `modifiers` array
  is ordered for exactly this reason). Canonical authoring order: identity/
  cultural constraints first (era, geography), then physical state (season,
  weather, time_of_day), then stylistic intent (mood) last — mood modulates
  the final look rather than competing with facts.
- A modifier whose `applies_to` selector doesn't match the object is a
  recorded no-op, not an error: selection mistakes by the reasoner must be
  visible but not fatal.
- Ops act on the whole mode list; weights are renormalized once at the end.
- Resolution is luminance-conditioned (Phase-2 finding): modes are filtered
  by L_range against the region's measured L, and the winning mode's chroma
  is projected into the sRGB gamut at that L.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from ..plan.colors import LabColor, project_chroma_into_gamut
from .store import KBError, KnowledgeBase

MIN_TOLERANCE = 6.0
SIGMA_TO_TOLERANCE = 1.5


@dataclass
class Resolution:
    object: str
    modes: list[dict]                      # post-composition distribution
    resolved: LabColor                     # gamut-feasible at the given L
    chosen_mode: str
    tolerance_delta_e: float
    applied: list[str]                     # modifiers that acted
    skipped: list[str]                     # modifiers that didn't apply to this object
    rationale: str
    trace: list[str] = field(default_factory=list)

    def to_region(self, grounding_phrase: str, modifiers: list[dict],
                  region_id: str | None = None, confidence: float = 0.7) -> dict:
        """Emit a plan-schema-valid region dict."""
        region = {
            "object": self.object,
            "grounding_phrase": grounding_phrase,
            "base_prior": f"kb:{self.object}",
            "modifiers": modifiers,
            "resolved_colour": self.resolved.to_plan(),
            "tolerance_delta_e": round(self.tolerance_delta_e, 1),
            "confidence": confidence,
            "rationale": self.rationale,
        }
        if region_id:
            region["id"] = region_id
        return region


def _chroma(mode: dict) -> float:
    a, b = mode["ab"]
    return math.hypot(a, b)


def _apply_op(modes: list[dict], op: dict, trace: list[str]) -> list[dict]:
    kind = op["op"]
    if kind == "scale_chroma":
        f = op["factor"]
        for m in modes:
            m["ab"] = [m["ab"][0] * f, m["ab"][1] * f]
        trace.append(f"scale_chroma x{f}")
    elif kind == "shift_ab":
        da, db = op.get("da", 0), op.get("db", 0)
        for m in modes:
            m["ab"] = [m["ab"][0] + da, m["ab"][1] + db]
        trace.append(f"shift_ab ({da:+g},{db:+g})")
    elif kind == "clamp_chroma":
        cmax = op["max"]
        for m in modes:
            c = _chroma(m)
            if c > cmax:
                s = cmax / c
                m["ab"] = [m["ab"][0] * s, m["ab"][1] * s]
        trace.append(f"clamp_chroma <= {cmax}")
    elif kind == "scale_sigma":
        for m in modes:
            m["sigma"] *= op["factor"]
        trace.append(f"scale_sigma x{op['factor']}")
    elif kind == "reweight":
        hit = False
        for m in modes:
            if m["name"] == op["mode"]:
                m["weight"] *= op["factor"]
                hit = True
        trace.append(f"reweight {op['mode']} x{op['factor']}" + ("" if hit else " (mode absent)"))
    elif kind == "add_mode":
        modes.append(copy.deepcopy(op["mode"]))
        trace.append(f"add_mode {op['mode']['name']}")
    elif kind == "remove_mode":
        modes = [m for m in modes if m["name"] != op["mode"]]
        trace.append(f"remove_mode {op['mode']}")
    else:  # pragma: no cover - store validation rejects unknown ops
        raise KBError(f"unknown op {kind!r}")
    return modes


def compose(kb: KnowledgeBase, object_name: str, modifiers: list[dict]) -> tuple[list[dict], list[str], list[str], list[str]]:
    """Apply (family, value) modifiers in order to the object's prior.

    Returns (modes, applied, skipped, trace).
    """
    entry = kb.object_entry(object_name)
    selectors = kb.selectors_for(object_name)
    modes = copy.deepcopy(entry["modes"])
    applied, skipped, trace = [], [], []

    for mod in modifiers:
        family, value = mod["family"], mod["value"]
        m_entry = kb.modifier_entry(family, value)
        label = f"{family}:{value}"
        if not set(m_entry["applies_to"]) & selectors:
            skipped.append(label)
            trace.append(f"{label}: not operative for {object_name} (applies_to {m_entry['applies_to']})")
            continue
        applied.append(label)
        for op in m_entry["ops"]:
            modes = _apply_op(modes, op, trace)

    total = sum(m["weight"] for m in modes)
    if total <= 0:
        raise KBError(f"composition left {object_name!r} with no weighted modes")
    for m in modes:
        m["weight"] /= total
    return modes, applied, skipped, trace


def resolve(kb: KnowledgeBase, object_name: str, modifiers: list[dict],
            measured_L: float | None = None) -> Resolution:
    """Compose modifiers onto the prior and pick a concrete colour.

    measured_L: the region's actual median luminance (from the grayscale
    image). Modes whose L_range excludes it are dropped (unless that would
    drop everything); chroma is made feasible at that L.
    """
    modes, applied, skipped, trace = compose(kb, object_name, modifiers)

    candidates = modes
    if measured_L is not None:
        in_range = [m for m in modes
                    if m.get("L_range", [0, 100])[0] <= measured_L <= m.get("L_range", [0, 100])[1]]
        if in_range:
            candidates = in_range
        else:
            trace.append(f"no mode's L_range contains L={measured_L:g}; keeping all modes")

    best = max(candidates, key=lambda m: m["weight"])
    L = measured_L if measured_L is not None else sum(best.get("L_range", [0, 100])) / 2
    raw = LabColor(L, best["ab"][0], best["ab"][1])
    feasible = project_chroma_into_gamut(raw)
    if (feasible.a, feasible.b) != (raw.a, raw.b):
        trace.append(f"chroma projected into sRGB gamut at L={L:g}: "
                     f"({raw.a:.1f},{raw.b:.1f}) -> ({feasible.a:.1f},{feasible.b:.1f})")

    entry = kb.object_entry(object_name)
    parts = [f"KB prior '{entry['_canonical']}' mode '{best['name']}'"]
    if applied:
        parts.append("modifiers: " + ", ".join(applied))
    if skipped:
        parts.append("not operative: " + ", ".join(skipped))
    if best.get("note"):
        parts.append(best["note"])

    return Resolution(
        object=entry["_canonical"],
        modes=modes,
        resolved=LabColor(round(feasible.L, 1), round(feasible.a, 1), round(feasible.b, 1)),
        chosen_mode=best["name"],
        tolerance_delta_e=max(MIN_TOLERANCE, SIGMA_TO_TOLERANCE * best["sigma"]),
        applied=applied,
        skipped=skipped,
        rationale="; ".join(parts),
        trace=trace,
    )
