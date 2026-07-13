"""Run the DDColor baseline over the smoke subset.

Usage:
    python scripts/run_ddcolor.py --input data/coco/val2017_subset --output results/ddcolor_tiny \
        [--model ddcolor_paper_tiny] [--input-size 512]

Note: the pipeline only reads the L channel, so pointing --input at the
colour images is correct and leak-free.
"""

import argparse
from pathlib import Path

from chroma_reasoner.baselines.ddcolor import MODEL_NAMES, colorize_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", type=str, default="ddcolor_paper_tiny",
                    help=f"one of {MODEL_NAMES} or a full HF repo id")
    ap.add_argument("--input-size", type=int, default=512)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    n = colorize_dir(args.input, args.output, model_name=args.model,
                     input_size=args.input_size, device=args.device)
    print(f"colorized {n} images -> {args.output}")


if __name__ == "__main__":
    main()
