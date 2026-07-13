import copy
import json
from pathlib import Path

import pytest

from chroma_reasoner.plan import PlanValidationError, load_plan, validate_plan

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "plans" / "melancholic_1910s_seaside.json"


@pytest.fixture()
def valid_plan() -> dict:
    with open(EXAMPLE, encoding="utf-8") as f:
        return json.load(f)


def test_example_plan_is_valid(valid_plan):
    assert validate_plan(valid_plan) == []


def test_load_plan_roundtrip():
    plan = load_plan(EXAMPLE)
    assert plan["plan_version"] == "1.0"
    assert len(plan["regions"]) == 4


def test_missing_required_field_rejected(valid_plan):
    bad = copy.deepcopy(valid_plan)
    del bad["regions"][0]["rationale"]
    errors = validate_plan(bad)
    assert any("rationale" in e for e in errors)


def test_unknown_field_rejected(valid_plan):
    """additionalProperties: false — catches LLM output drift and typos."""
    bad = copy.deepcopy(valid_plan)
    bad["regions"][0]["colour"] = "blue"
    errors = validate_plan(bad)
    assert any("colour" in e for e in errors)


def test_lab_out_of_range_rejected(valid_plan):
    bad = copy.deepcopy(valid_plan)
    bad["regions"][0]["resolved_colour"]["L"] = 140
    assert validate_plan(bad)


def test_confidence_out_of_range_rejected(valid_plan):
    bad = copy.deepcopy(valid_plan)
    bad["regions"][0]["confidence"] = 1.5
    assert validate_plan(bad)


def test_empty_regions_rejected(valid_plan):
    bad = copy.deepcopy(valid_plan)
    bad["regions"] = []
    assert validate_plan(bad)


def test_wrong_version_rejected(valid_plan):
    bad = copy.deepcopy(valid_plan)
    bad["plan_version"] = "2.0"
    assert validate_plan(bad)


def test_all_errors_reported_at_once(valid_plan):
    bad = copy.deepcopy(valid_plan)
    del bad["regions"][0]["rationale"]
    bad["regions"][1]["confidence"] = 2
    bad["plan_version"] = "9"
    with pytest.raises(PlanValidationError) as exc:
        from chroma_reasoner.plan.schema import assert_valid

        assert_valid(bad)
    assert len(exc.value.errors) >= 3
