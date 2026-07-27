"""Counterfactual evaluation: does the system actually respond to context?

The Phase-5 scaled ablation showed the KB does not beat implicit model
knowledge on *realistic* prompts — where reality is available to both arms.
The roadmap (§6, §7) predicts the KB's accuracy case lives where reality is
NOT available: counterfactual era/mood prompts. This module tests that with
no human judging and no external reference.

Two measurements per contrast (e.g. "1910s" vs "1970s"):

1. **Separation** — how far apart are the two conditions' palettes for the
   same regions? A system that ignores the prompt scores ~0. This is the
   roadmap's kill-criterion made objective: "if humans can't distinguish
   1910s from 1970s outputs, the plan lacks discriminative period features."
2. **Direction correctness** — separation alone is not enough; the shift must
   be *right*. 1910s must be lower-chroma than 1970s; melancholic cooler and
   duller than cheerful. Expectations are declared below and checked with the
   same exact sign test used for the ablation.

Regions are held FIXED across conditions (the same plan is re-resolved under
each), so what is measured is the colour response to context, not selection
noise. For the KB arm this needs no model at all — the KB is deterministic,
so this whole protocol runs locally.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from ..kb.engine import resolve
from ..kb.store import KnowledgeBase
from ..plan.schema import assert_valid
from .paired import bootstrap_median_ci, sign_test

# --- the condition matrix ---------------------------------------------------

@dataclass(frozen=True)
class Condition:
    name: str
    prompt: str                      # what the LLM arm is told
    modifiers: tuple = ()            # (family, value) pairs the KB arm applies


CONDITIONS: dict[str, Condition] = {
    "neutral": Condition("neutral", "", ()),
    "era_1910s": Condition("era_1910s", "a photograph taken in the 1910s",
                           (("era", "1910s"),)),
    "era_1940s": Condition("era_1940s", "a photograph taken in the 1940s",
                           (("era", "1940s"),)),
    "era_1970s": Condition("era_1970s", "a photograph taken in the 1970s",
                           (("era", "1970s"),)),
    "mood_melancholic": Condition("mood_melancholic", "a melancholic, sombre scene",
                                  (("mood", "melancholic"),)),
    "mood_cheerful": Condition("mood_cheerful", "a cheerful, upbeat scene",
                               (("mood", "cheerful"),)),
    "season_autumn": Condition("season_autumn", "the same scene in autumn",
                               (("season", "autumn"),)),
    "season_summer": Condition("season_summer", "the same scene in high summer",
                               (("season", "summer"),)),
}


# --- declared expectations (pre-registered) --------------------------------

@dataclass(frozen=True)
class Contrast:
    a: str
    b: str
    # (metric, direction) — direction describes condition A relative to B.
    # metrics: chroma = hypot(a,b); warmth = b; redness = a
    expectations: tuple = ()
    note: str = ""


CONTRASTS: tuple[Contrast, ...] = (
    Contrast("era_1910s", "era_1970s", (("chroma", "lower"),),
             "1910s muted dyes vs 1970s avocado/harvest-gold saturation"),
    Contrast("era_1940s", "era_1970s", (("chroma", "lower"),),
             "wartime austerity vs 1970s saturation"),
    Contrast("mood_melancholic", "mood_cheerful",
             (("chroma", "lower"), ("warmth", "lower")),
             "melancholic desaturates and cools; cheerful saturates and warms"),
    Contrast("season_autumn", "season_summer", (("redness", "higher"),),
             "senescent foliage shifts toward red/orange"),
)

METRICS = ("chroma", "warmth", "redness")


def _metric(colour: dict, metric: str) -> float:
    a, b = colour["a"], colour["b"]
    if metric == "chroma":
        return math.hypot(a, b)
    if metric == "warmth":
        return b
    if metric == "redness":
        return a
    raise ValueError(f"unknown metric {metric!r}")


# --- KB arm: apply a condition locally (no model) ---------------------------

def apply_condition(kb: KnowledgeBase, plan: dict, condition: Condition) -> dict:
    """Re-resolve every region of `plan` under `condition`, regions unchanged.

    Modifiers of the condition's families replace any existing ones of the
    same family (so conditions don't stack); other families (weather, geo)
    are preserved. Luminance is preserved from the plan, so only chroma can
    move — matching what a colorizer can actually change.
    """
    families = {family for family, _ in condition.modifiers}
    out = copy.deepcopy(plan)
    out["prompt"] = condition.prompt
    for region in out["regions"]:
        kept = [m for m in region.get("modifiers", []) if m["family"] not in families]
        added = [{"family": f, "value": v,
                  "effect": kb.modifier_entry(f, v).get("note", "")}
                 for f, v in condition.modifiers]
        modifiers = kept + added
        res = resolve(kb, region["object"], modifiers,
                      measured_L=region["resolved_colour"]["L"])
        region["modifiers"] = modifiers
        region["resolved_colour"] = res.resolved.to_plan()
        region["tolerance_delta_e"] = round(res.tolerance_delta_e, 1)
        region["rationale"] = f"[{condition.name}] {res.rationale}"
    return assert_valid(out)


def condition_variants(kb: KnowledgeBase, plan: dict,
                       names: list[str] | None = None) -> dict[str, dict]:
    """One plan -> {condition_name: plan} for the KB arm."""
    chosen = names or list(CONDITIONS)
    return {name: apply_condition(kb, plan, CONDITIONS[name]) for name in chosen}


# --- measurement -----------------------------------------------------------

@dataclass
class ContrastResult:
    contrast: Contrast
    n_regions: int = 0
    n_active: int = 0                       # regions whose colour actually moved
    separations: list = field(default_factory=list)
    directions: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        lo, hi = bootstrap_median_ci(self.separations)
        return {
            "a": self.contrast.a, "b": self.contrast.b, "note": self.contrast.note,
            "n_regions": self.n_regions, "n_active": self.n_active,
            "active_share": round(self.n_active / self.n_regions, 3) if self.n_regions else None,
            "mean_separation": round(sum(self.separations) / len(self.separations), 2)
            if self.separations else None,
            "median_separation": round(_median(self.separations), 2),
            "median_separation_ci95": [round(lo, 2), round(hi, 2)],
            "directions": self.directions,
        }


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def evaluate_contrast(plans_a: dict[str, dict], plans_b: dict[str, dict],
                      contrast: Contrast, active_threshold: float = 1.0) -> dict:
    """Compare two conditions across images.

    plans_a / plans_b: {image_id: plan} for condition A / B. Regions are
    paired by id within each image.
    """
    result = ContrastResult(contrast)
    per_expectation = {m: {"wins": 0, "losses": 0, "ties": 0, "margins": []}
                       for m, _ in contrast.expectations}

    for image_id, plan_a in plans_a.items():
        plan_b = plans_b.get(image_id)
        if plan_b is None:
            continue
        by_id_b = {r.get("id") or r["object"]: r for r in plan_b["regions"]}
        for region_a in plan_a["regions"]:
            key = region_a.get("id") or region_a["object"]
            region_b = by_id_b.get(key)
            if region_b is None:
                continue
            ca, cb = region_a["resolved_colour"], region_b["resolved_colour"]
            sep = math.hypot(ca["a"] - cb["a"], ca["b"] - cb["b"])
            result.n_regions += 1
            result.separations.append(sep)
            if sep < active_threshold:
                continue                     # condition did not touch this object
            result.n_active += 1
            for metric, direction in contrast.expectations:
                va, vb = _metric(ca, metric), _metric(cb, metric)
                margin = (vb - va) if direction == "lower" else (va - vb)
                bucket = per_expectation[metric]
                bucket["margins"].append(margin)
                if margin > 0.5:
                    bucket["wins"] += 1
                elif margin < -0.5:
                    bucket["losses"] += 1
                else:
                    bucket["ties"] += 1

    for metric, direction in contrast.expectations:
        b = per_expectation[metric]
        result.directions[metric] = {
            "expected": f"A {direction}",
            "as_expected": b["wins"], "against": b["losses"], "ties": b["ties"],
            "median_margin": round(_median(b["margins"]), 2),
            "sign_test_p": round(sign_test(b["wins"], b["losses"]), 5),
        }
    return result.to_dict()


def evaluate_all(plans_by_condition: dict[str, dict[str, dict]],
                 contrasts: tuple = CONTRASTS) -> list[dict]:
    """Run every contrast whose two conditions are both present."""
    out = []
    for contrast in contrasts:
        if contrast.a in plans_by_condition and contrast.b in plans_by_condition:
            out.append(evaluate_contrast(plans_by_condition[contrast.a],
                                         plans_by_condition[contrast.b], contrast))
    return out


def format_contrast(res: dict) -> str:
    lines = [f"{res['a']} vs {res['b']}  ({res['note']})",
             f"  separation: median {res['median_separation']} "
             f"[{res['median_separation_ci95'][0]}, {res['median_separation_ci95'][1]}]  "
             f"mean {res['mean_separation']}  "
             f"active {res['n_active']}/{res['n_regions']} regions "
             f"({res['active_share']})"]
    for metric, d in res["directions"].items():
        decided = d["as_expected"] + d["against"]
        lines.append(f"  {metric:>8} {d['expected']:>10}: "
                     f"{d['as_expected']}/{decided} as expected "
                     f"({d['ties']} ties)  median margin {d['median_margin']:+g}  "
                     f"p={d['sign_test_p']}")
    return "\n".join(lines)
