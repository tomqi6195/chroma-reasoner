"""Build a deterministic smoke-scale subset of COCO val2017 with captions.

The subset is the evaluation substrate for Phase 0: ground-truth colour images,
synthesized grayscale inputs (Lab L channel), and one caption per image for
CLIP-score. Images are selected deterministically (sorted ids, fixed seed) so
the subset is reproducible across machines, including Colab.
"""

from __future__ import annotations

import json
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGE_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{file_name}"


@dataclass
class SubsetPaths:
    root: Path

    @property
    def annotations_zip(self) -> Path:
        return self.root / "annotations_trainval2017.zip"

    @property
    def captions_json(self) -> Path:
        return self.root / "annotations" / "captions_val2017.json"

    @property
    def images_dir(self) -> Path:
        return self.root / "val2017_subset"

    @property
    def gray_dir(self) -> Path:
        return self.root / "gray"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"


def _download(url: str, dest: Path, desc: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.rename(dest)


def ensure_captions(paths: SubsetPaths) -> Path:
    """Download and extract captions_val2017.json if not already present."""
    if paths.captions_json.exists():
        return paths.captions_json
    if not paths.annotations_zip.exists():
        _download(ANNOTATIONS_URL, paths.annotations_zip, "annotations zip")
    with zipfile.ZipFile(paths.annotations_zip) as zf:
        zf.extract("annotations/captions_val2017.json", paths.root)
    return paths.captions_json


def select_images(captions_json: Path, n: int, seed: int = 0) -> list[dict]:
    """Pick n images deterministically; attach all their captions."""
    with open(captions_json, encoding="utf-8") as f:
        data = json.load(f)
    captions_by_image: dict[int, list[str]] = {}
    for ann in data["annotations"]:
        captions_by_image.setdefault(ann["image_id"], []).append(ann["caption"].strip())
    images = sorted(data["images"], key=lambda im: im["id"])
    rng = random.Random(seed)
    chosen = rng.sample(images, n)
    chosen.sort(key=lambda im: im["id"])
    return [
        {
            "id": im["id"],
            "file_name": im["file_name"],
            "width": im["width"],
            "height": im["height"],
            "captions": captions_by_image.get(im["id"], []),
        }
        for im in chosen
    ]


def download_subset(root: Path, n: int = 300, seed: int = 0) -> Path:
    """Materialize the subset: images + manifest. Returns manifest path."""
    paths = SubsetPaths(root)
    ensure_captions(paths)
    records = select_images(paths.captions_json, n=n, seed=seed)
    paths.images_dir.mkdir(parents=True, exist_ok=True)
    for rec in tqdm(records, desc="images"):
        dest = paths.images_dir / rec["file_name"]
        if not dest.exists():
            _download(IMAGE_URL_TEMPLATE.format(file_name=rec["file_name"]), dest, rec["file_name"])
    manifest = {"dataset": "coco-val2017-subset", "n": n, "seed": seed, "images": records}
    with open(paths.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return paths.manifest
