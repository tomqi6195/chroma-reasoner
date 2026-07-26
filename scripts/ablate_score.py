"""Score one or more plan directories against the original photographs.

Each arm (hand-authored / KB-reasoned / LLM-ablation) is a directory of plan
JSONs sharing image ids; masks may differ per arm. Monochrome originals
(no chroma reference) are excluded automatically.

Usage:
    python scripts/ablate_score.py --originals data/coco/val2017_subset \
        --arm human=examples/plans/phase2:masks \
        --arm kb=plans/reasoned:masks_reasoned \
        --arm llm=plans/ablation_llm:masks_reasoned
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from chroma_reasoner.eval import compare_arms, format_comparison, plan_vs_reference
from chroma_reasoner.plan import load_plan
from chroma_reasoner.plan.colors import srgb_array_to_lab
from chroma_reasoner.plan.masks import load_masks

MONO_CHROMA_THRESHOLD = 5.0  # mean |ab| below this = original is monochrome


def original_is_monochrome(rgb: np.ndarray) -> bool:
    lab = srgb_array_to_lab(rgb[::4, ::4].astype(np.float64) / 255.0)
    return float(np.abs(lab[..., 1:]).mean()) < MONO_CHROMA_THRESHOLD


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--originals", type=Path, required=True)
    ap.add_argument("--arm", action="append", required=True,
                    metavar="NAME=PLANS_DIR:MASKS_DIR")
    ap.add_argument("--out", type=Path, default=None, help="optional JSON report path")
    args = ap.parse_args()

    report: dict = {}
    for arm in args.arm:
        name, _, rest = arm.partition("=")
        plans_dir, _, masks_dir = rest.partition(":")
        rows = []
        for plan_path in sorted(Path(plans_dir).glob("*.json")):
            plan = load_plan(plan_path)
            iid = plan["image_id"]
            orig_path = args.originals / f"{iid}.jpg"
            if not orig_path.exists():
                continue
            original = cv2.cvtColor(cv2.imread(str(orig_path)), cv2.COLOR_BGR2RGB)
            if original_is_monochrome(original):
                print(f"[{name}] {iid}: original is monochrome - excluded")
                continue
            gray_shape = original.shape[:2]
            try:
                masks = load_masks(Path(masks_dir), iid, plan, shape=gray_shape,
                                   allow_missing=True)
            except ValueError as e:
                print(f"[{name}] {iid}: masks unavailable ({e})")
                continue
            if not masks:
                print(f"[{name}] {iid}: no masks at all")
                continue
            res = plan_vs_reference(plan, masks, original)
            rows.append({"image": iid, **res})
            print(f"[{name}] {iid}: mean dE {res['mean_delta_e']} | "
                  f"chroma deficit {res['mean_chroma_deficit']} | "
                  f"undercommitted {res['n_undercommitted']}/{res['n_regions']}")
            for r in res["regions"]:
                if "delta_e" in r:
                    print(f"    {r['region']:>16} planned ab{tuple(r['planned_ab'])} "
                          f"vs real ab{tuple(r['reference_ab'])}  dE={r['delta_e']}"
                          f"  deficit={r['chroma_deficit']}")
        des = [x["mean_delta_e"] for x in rows if x["mean_delta_e"] is not None]
        defs_ = [x["mean_chroma_deficit"] for x in rows if x["mean_chroma_deficit"] is not None]
        under = sum(x["n_undercommitted"] for x in rows)
        total = sum(x["n_regions"] for x in rows)
        summary = round(float(np.mean(des)), 2) if des else None
        deficit_summary = round(float(np.mean(defs_)), 2) if defs_ else None
        print(f"== arm '{name}': mean dE {summary} | mean chroma deficit {deficit_summary} "
              f"| undercommitted {under}/{total} regions over {len(rows)} images\n")
        report[name] = {"images": rows, "mean_delta_e": summary,
                        "mean_chroma_deficit": deficit_summary,
                        "n_undercommitted": under, "n_regions": total}

    # Paired analysis over shared regions — the defensible comparison when
    # arms are matched region-for-region (kb vs llm). Arms with disjoint
    # region ids (e.g. human) simply produce 0 shared regions.
    comparisons = []
    names = list(report)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            cmp = compare_arms(report[name_a], report[name_b], name_a, name_b)
            if cmp["n_paired_regions"] == 0:
                continue
            comparisons.append(cmp)
            print(format_comparison(cmp))
            print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"arms": report, "paired": comparisons}, f, indent=2)
        print(f"report: {args.out}")


if __name__ == "__main__":
    main()
