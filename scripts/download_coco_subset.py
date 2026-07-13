"""Download the COCO val2017 smoke subset and synthesize grayscale inputs.

Usage:
    python scripts/download_coco_subset.py [--n 300] [--seed 0] [--root data/coco]
"""

import argparse
from pathlib import Path

from chroma_reasoner.data.coco_subset import SubsetPaths, download_subset
from chroma_reasoner.data.grayscale import make_grayscale_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", type=Path, default=Path("data/coco"))
    args = ap.parse_args()

    manifest = download_subset(args.root, n=args.n, seed=args.seed)
    paths = SubsetPaths(args.root)
    count = make_grayscale_dir(paths.images_dir, paths.gray_dir)
    print(f"manifest: {manifest}")
    print(f"colour images: {paths.images_dir} | grayscale L-channel: {paths.gray_dir} ({count})")


if __name__ == "__main__":
    main()
