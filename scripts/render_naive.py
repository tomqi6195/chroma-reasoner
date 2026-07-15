"""Render a plan with the naive Lab-paste baseline and score adherence.

Proves the consumption path (plan + masks -> image) with zero model variance.

Usage:
    python scripts/render_naive.py --plan examples/plans/phase2/000000010092.json \
        --gray data/coco/gray/000000010092.png --masks data/masks \
        --out results/phase2/naive
"""

import argparse
import json
from pathlib import Path

import cv2

from chroma_reasoner.plan import load_plan
from chroma_reasoner.plan.adherence import evaluate_adherence
from chroma_reasoner.plan.hints import render_naive
from chroma_reasoner.plan.masks import load_masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--gray", type=Path, required=True, help="L-channel PNG (cv2 0-255 scaling)")
    ap.add_argument("--masks", type=Path, required=True, help="masks root (masks/{image_id}/{region}.png)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    plan = load_plan(args.plan)
    image_id = plan.get("image_id") or args.gray.stem
    gray = cv2.imread(str(args.gray), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"unreadable grayscale image: {args.gray}")
    masks = load_masks(args.masks, image_id, plan, shape=gray.shape)

    rgb = render_naive(gray, masks, plan)
    args.out.mkdir(parents=True, exist_ok=True)
    out_img = args.out / f"{image_id}.png"
    cv2.imwrite(str(out_img), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    report = evaluate_adherence(rgb, masks, plan)
    with open(args.out / f"{image_id}_adherence.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"rendered: {out_img}")
    for r in report["regions"]:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"  [{status}] {r['region']:>12}  dE={r.get('delta_e', '-'):>6}  tol={r.get('tolerance', '-')}")
    print(f"mean dE={report['mean_delta_e']}  pass {report['n_pass']}/{report['n_regions']}")


if __name__ == "__main__":
    main()
