"""Counterfactual protocol: does the plan respond correctly to era/mood context?

Takes reasoned plans, re-resolves each under every condition (KB arm — local
and deterministic, no model needed), and scores the pre-registered contrasts:
prompt separation + direction correctness.

Usage:
    python scripts/counterfactual.py --plans plans/reasoned \
        --out results/phase6/counterfactual.json [--write-plans plans/counterfactual]
"""

import argparse
import json
from pathlib import Path

from chroma_reasoner.eval.counterfactual import (CONDITIONS, condition_variants,
                                                 evaluate_all, format_contrast)
from chroma_reasoner.kb import load_kb
from chroma_reasoner.plan import load_plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", type=Path, required=True, help="dir of reasoned plans")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--write-plans", type=Path, default=None,
                    help="also write per-condition plans here (for the LLM arm / rendering)")
    ap.add_argument("--conditions", type=str, default=None,
                    help="comma-separated subset of condition names")
    args = ap.parse_args()

    kb = load_kb()
    names = args.conditions.split(",") if args.conditions else list(CONDITIONS)

    plans_by_condition: dict[str, dict[str, dict]] = {name: {} for name in names}
    for plan_path in sorted(args.plans.glob("*.json")):
        plan = load_plan(plan_path)
        for name, variant in condition_variants(kb, plan, names).items():
            plans_by_condition[name][plan["image_id"]] = variant
            if args.write_plans:
                out_dir = args.write_plans / name
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / f"{plan['image_id']}.json", "w", encoding="utf-8") as f:
                    json.dump(variant, f, indent=2)

    n_images = len(next(iter(plans_by_condition.values()), {}))
    print(f"{n_images} images x {len(names)} conditions\n")

    results = evaluate_all(plans_by_condition)
    for res in results:
        print(format_contrast(res))
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"n_images": n_images, "conditions": names,
                       "contrasts": results}, f, indent=2)
        print(f"report: {args.out}")


if __name__ == "__main__":
    main()
