"""Validate a colour-plan JSON file and print a human-readable summary.

Usage:
    python scripts/validate_plan.py examples/plans/melancholic_1910s_seaside.json
"""

import argparse
import sys
from pathlib import Path

from chroma_reasoner.plan import PlanValidationError, LabColor, lab_to_hex, load_plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", type=Path)
    args = ap.parse_args()

    try:
        plan = load_plan(args.plan)
    except PlanValidationError as e:
        print(f"INVALID: {args.plan}")
        for err in e.errors:
            print(f"  - {err}")
        sys.exit(1)

    print(f"VALID: {args.plan}  (plan_version {plan['plan_version']})")
    if plan.get("prompt"):
        print(f"prompt: {plan['prompt']!r}")
    for region in plan["regions"]:
        lab = LabColor.from_plan(region["resolved_colour"])
        mods = ", ".join(f"{m['family']}:{m['value']}" for m in region.get("modifiers", []))
        print(f"  [{region['object']:>10}] {lab_to_hex(lab)}  Lab({lab.L:g},{lab.a:g},{lab.b:g})"
              f"  conf={region['confidence']:.2f}  mods=[{mods}]")
        print(f"              {region['rationale']}")


if __name__ == "__main__":
    main()
