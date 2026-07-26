"""Loading and validation for the explicit knowledge base (Phase 3).

The KB is two hand-authored YAML files:

- kb/objects.yaml    object class -> colour-distribution prior (weighted
                     modes in the ab plane, each with a validity L range)
- kb/modifiers.yaml  (family, value) -> documented distribution deformation
                     (an ordered list of ops), with applies_to selectors and
                     provenance

Everything is plain data on purpose: the KB must be inspectable, diffable,
and correctable without touching code (roadmap §4.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

KB_DIR = Path(__file__).resolve().parents[3] / "kb"

VALID_OPS = {"scale_chroma", "shift_ab", "clamp_chroma", "scale_sigma",
             "reweight", "add_mode", "remove_mode"}


class KBError(ValueError):
    pass


def _name_candidates(name: str) -> list[str]:
    """Progressively normalized forms to try for an object name.

    Live-run finding: reasoners emit phrasings, not identifiers — "woman's
    dress", "The Wall", "leafy greens". Normalizing here is far cheaper than
    enumerating every phrasing as an alias, and it is conservative: only
    case, separators, articles, and possessive prefixes are stripped. No
    last-word fallback (that would wrongly match "fire truck" to "truck"
    semantics we did not intend).
    """
    seen, out = set(), []

    def add(candidate: str) -> None:
        if candidate and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)

    add(name)
    base = name.strip().lower().replace("-", " ")
    add(base.replace(" ", "_"))
    for article in ("the ", "a ", "an "):
        if base.startswith(article):
            base = base[len(article):]
    add(base.replace(" ", "_"))
    if "'s " in base:                       # "woman's dress" -> "dress"
        add(base.split("'s ", 1)[1].strip().replace(" ", "_"))
    if "s' " in base:                       # "boys' jumpers" -> "jumpers"
        add(base.split("s' ", 1)[1].strip().replace(" ", "_"))
    for candidate in list(out):             # naive singular
        if candidate.endswith("s") and not candidate.endswith("ss"):
            add(candidate[:-1])
    return out


@dataclass
class KnowledgeBase:
    objects: dict
    modifiers: dict
    objects_path: Path
    modifiers_path: Path
    # name/alias/category -> canonical object name(s)
    _selector_index: dict = field(default_factory=dict, repr=False)

    def object_entry(self, name: str) -> dict:
        for candidate in _name_candidates(name):
            entry = self.objects.get(candidate)
            if entry is not None:
                return {**entry, "_canonical": candidate}
            for canonical, e in self.objects.items():
                if candidate in e.get("aliases", []):
                    return {**e, "_canonical": canonical}
        raise KBError(f"object class not in KB: {name!r}")

    def modifier_entry(self, family: str, value: str) -> dict:
        fam = self.modifiers.get(family)
        if fam is None:
            raise KBError(f"modifier family not in KB: {family!r}")
        entry = fam.get(value)
        if entry is None:
            raise KBError(f"modifier not in KB: {family}:{value}")
        return entry

    def selectors_for(self, object_name: str) -> set[str]:
        """Every selector token this object matches: name, aliases, categories, '*'."""
        entry = self.object_entry(object_name)
        return ({entry["_canonical"], "*"}
                | set(entry.get("aliases", []))
                | set(entry.get("categories", [])))


def _validate_mode(mode: dict, where: str) -> None:
    for key in ("name", "ab", "sigma", "weight"):
        if key not in mode:
            raise KBError(f"{where}: mode missing {key!r}: {mode}")
    a, b = mode["ab"]
    if not (-128 <= a <= 127 and -128 <= b <= 127):
        raise KBError(f"{where}: ab out of range: {mode['ab']}")
    if mode["sigma"] <= 0 or mode["weight"] <= 0:
        raise KBError(f"{where}: sigma and weight must be positive: {mode}")
    lo, hi = mode.get("L_range", [0, 100])
    if not (0 <= lo < hi <= 100):
        raise KBError(f"{where}: bad L_range: {mode.get('L_range')}")


def _validate_objects(objects: dict) -> None:
    for name, entry in objects.items():
        modes = entry.get("modes")
        if not modes:
            raise KBError(f"object {name!r}: no modes")
        for mode in modes:
            _validate_mode(mode, f"object {name!r}")
        if "provenance" not in entry:
            raise KBError(f"object {name!r}: provenance required")


def _validate_modifiers(modifiers: dict) -> None:
    for family, values in modifiers.items():
        for value, entry in values.items():
            where = f"modifier {family}:{value}"
            if not entry.get("applies_to"):
                raise KBError(f"{where}: applies_to required")
            if "provenance" not in entry:
                raise KBError(f"{where}: provenance required")
            ops = entry.get("ops")
            if not ops:
                raise KBError(f"{where}: ops required")
            for op in ops:
                if op.get("op") not in VALID_OPS:
                    raise KBError(f"{where}: unknown op {op.get('op')!r}")
                if op["op"] == "add_mode":
                    _validate_mode(op["mode"], where)


def load_kb(kb_dir: Path | None = None) -> KnowledgeBase:
    kb_dir = Path(kb_dir) if kb_dir else KB_DIR
    objects_path = kb_dir / "objects.yaml"
    modifiers_path = kb_dir / "modifiers.yaml"
    with open(objects_path, encoding="utf-8") as f:
        objects = yaml.safe_load(f)["objects"]
    with open(modifiers_path, encoding="utf-8") as f:
        modifiers = yaml.safe_load(f)["families"]
    _validate_objects(objects)
    _validate_modifiers(modifiers)
    return KnowledgeBase(objects=objects, modifiers=modifiers,
                         objects_path=objects_path, modifiers_path=modifiers_path)
