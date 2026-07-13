"""CLIP-score: 100 * cosine similarity between image and caption embeddings.

The standard proxy for prompt faithfulness in the language-colorization line.
Uses openai/clip-vit-base-patch32 by default (the common choice in the
benchmark papers).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

DEFAULT_MODEL = "openai/clip-vit-base-patch32"


class ClipScorer:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # use_safetensors: torch<2.6 + transformers refuses .bin checkpoints
        # (CVE-2025-32434); the safetensors variant loads fine.
        self.model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @torch.no_grad()
    def score(self, image: Image.Image, text: str) -> float:
        inputs = self.processor(text=[text], images=[image], return_tensors="pt",
                                padding=True, truncation=True).to(self.device)
        # Full forward: outputs.image_embeds/text_embeds are the projected,
        # L2-normalized joint-space embeddings in every transformers version
        # (get_image_features changed return type across versions).
        outputs = self.model(**inputs)
        img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
        txt_emb = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
        return float((img_emb * txt_emb).sum() * 100.0)

    def score_dir(self, image_dir: Path, captions_by_stem: dict[str, str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        files = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
        for path in tqdm(files, desc=f"CLIP-score {image_dir.name}"):
            caption = captions_by_stem.get(path.stem)
            if caption is None:
                continue
            scores[path.stem] = self.score(Image.open(path).convert("RGB"), caption)
        return scores


def mean_or_nan(values) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float("nan")
