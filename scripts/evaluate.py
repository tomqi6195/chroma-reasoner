"""Run the Phase 0 metric suite on a directory of colorized outputs.

Usage:
    python scripts/evaluate.py --pred results/ddcolor --gt data/coco/val2017_subset \
        --manifest data/coco/manifest.json --out results/phase0/ddcolor
"""

import argparse
import json
from pathlib import Path

from chroma_reasoner.metrics.evaluate import evaluate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-clip", action="store_true", help="skip CLIP-score (no captions needed)")
    args = ap.parse_args()

    report = evaluate(args.pred, args.gt, args.manifest, args.out, skip_clip=args.skip_clip)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
