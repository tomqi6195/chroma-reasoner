"""Validation for the object-centric colour plan (Phase 1 lock).

The single source of truth is schemas/color_plan.schema.json. This module
wraps it in a Python API that reports *all* violations at once (an LLM
reasoner's output usually has several problems at a time, and iterating one
error per round-trip is slow).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import jsonschema

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "color_plan.schema.json"


class PlanValidationError(ValueError):
    """Raised when a plan violates the schema. `errors` lists every violation."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("invalid colour plan:\n" + "\n".join(f"  - {e}" for e in errors))


@lru_cache(maxsize=1)
def _schema() -> dict:
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        return json.load(f)


def validate_plan(plan: dict) -> list[str]:
    """Return a list of violation messages (empty = valid). Does not raise."""
    validator = jsonschema.Draft202012Validator(_schema())
    errors = []
    for err in sorted(validator.iter_errors(plan), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{where}: {err.message}")
    return errors


def assert_valid(plan: dict) -> dict:
    """Raise PlanValidationError on any violation; return the plan unchanged."""
    errors = validate_plan(plan)
    if errors:
        raise PlanValidationError(errors)
    return plan


def load_plan(path: str | Path) -> dict:
    """Load a plan JSON file and validate it. Raises PlanValidationError."""
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    return assert_valid(plan)
