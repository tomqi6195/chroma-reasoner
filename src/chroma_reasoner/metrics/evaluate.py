"""Phase 0 evaluation runner.

Given a directory of colorized outputs, the ground-truth colour images, and
the subset manifest (for captions), computes:

  - FID            (clean-fid, pred vs. ground truth)
  - HI-FID         (FID after per-image chroma scaling to match GT colorfulness)
  - CF / dCF       (Hasler-Süsstrunk colorfulness of pred, GT, and the delta)
  - CLIP-score     (pred vs. caption; GT vs. caption reported as the ceiling)

Outputs a JSON report. FID on a few hundred images carries high variance —
treat smoke-scale numbers as plumbing validation, not publishable results.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .colorfulness import colorfulness
from .hue_invariant import build_hue_corrected_dir

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def _image_files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _mean_cf(d: Path) -> float:
    vals = []
    for p in _image_files(d):
        img = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        vals.append(colorfulness(img))
    return float(np.mean(vals))


def captions_from_manifest(manifest_path: Path) -> dict[str, str]:
    """stem -> first caption."""
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    out = {}
    for rec in manifest["images"]:
        stem = Path(rec["file_name"]).stem
        if rec["captions"]:
            out[stem] = rec["captions"][0]
    return out


def evaluate(pred_dir: Path, gt_dir: Path, manifest_path: Path | None,
             out_dir: Path, skip_clip: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "n_pred": len(_image_files(pred_dir)),
        "n_gt": len(_image_files(gt_dir)),
    }

    # Colorfulness first: cheap, no model downloads.
    report["cf_pred"] = _mean_cf(pred_dir)
    report["cf_gt"] = _mean_cf(gt_dir)
    report["delta_cf"] = report["cf_pred"] - report["cf_gt"]

    # FID (clean-fid downloads Inception weights on first use).
    # num_workers=0: clean-fid's resizer closure can't be pickled by Windows
    # spawn-based DataLoader workers.
    from cleanfid import fid as cleanfid

    report["fid"] = float(cleanfid.compute_fid(str(pred_dir), str(gt_dir), num_workers=0))

    # Hue-invariant FID: chroma-match pred to GT colorfulness, then FID.
    hi_dir = out_dir / "hi_corrected"
    alphas = build_hue_corrected_dir(pred_dir, gt_dir, hi_dir)
    report["hi_fid"] = float(cleanfid.compute_fid(str(hi_dir), str(gt_dir), num_workers=0))
    report["hi_alpha_mean"] = float(np.mean(alphas))

    if manifest_path is not None and not skip_clip:
        from .clip_score import ClipScorer, mean_or_nan

        captions = captions_from_manifest(manifest_path)
        scorer = ClipScorer()
        report["clip_score_pred"] = mean_or_nan(scorer.score_dir(pred_dir, captions).values())
        report["clip_score_gt"] = mean_or_nan(scorer.score_dir(gt_dir, captions).values())

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
