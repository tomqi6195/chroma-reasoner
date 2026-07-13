"""Synthesize grayscale inputs as the Lab L channel of the colour originals.

This matches how the literature constructs colorization inputs (§2.7 of the
roadmap): "ground truth" is the original colour image and the input is its
luminance. We store L as an 8-bit single-channel PNG.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from tqdm import tqdm


def to_l_channel(image_bgr) -> "cv2.typing.MatLike":
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    return lab[:, :, 0]


def make_grayscale_dir(images_dir: Path, gray_dir: Path) -> int:
    """Convert every image in images_dir to its L channel PNG in gray_dir."""
    gray_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    for src in tqdm(files, desc="grayscale"):
        dest = gray_dir / (src.stem + ".png")
        if dest.exists():
            count += 1
            continue
        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"unreadable image: {src}")
        cv2.imwrite(str(dest), to_l_channel(img))
        count += 1
    return count
