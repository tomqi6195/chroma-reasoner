"""DDColor baseline runner (automatic colorization, no language conditioning).

Wraps the vendored upstream repo (third_party/DDColor). The upstream pipeline
consumes only the Lab L channel of whatever it is given, so passing the
original colour images is the standard leak-free protocol: colour information
never reaches the model, and the output reuses the input's exact L channel.

Weights come from HuggingFace (piddnad/ddcolor_*). If HF is unreachable, set
HF_ENDPOINT=https://hf-mirror.com before running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
DDCOLOR_DIR = REPO_ROOT / "third_party" / "DDColor"

MODEL_NAMES = ("ddcolor_modelscope", "ddcolor_paper", "ddcolor_artistic", "ddcolor_paper_tiny")


def _import_ddcolor():
    if str(DDCOLOR_DIR) not in sys.path:
        sys.path.insert(0, str(DDCOLOR_DIR))
    from ddcolor import ColorizationPipeline, DDColor  # noqa: PLC0415

    return DDColor, ColorizationPipeline


def load_pipeline(model_name: str = "ddcolor_paper_tiny", input_size: int = 512,
                  device: str | None = None):
    DDColor, ColorizationPipeline = _import_ddcolor()
    from huggingface_hub import PyTorchModelHubMixin

    class DDColorHF(DDColor, PyTorchModelHubMixin):
        def __init__(self, config=None, **kwargs):
            if isinstance(config, dict):
                kwargs = {**config, **kwargs}
            super().__init__(**kwargs)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    repo_id = model_name if "/" in model_name else f"piddnad/{model_name}"
    model = DDColorHF.from_pretrained(repo_id).to(device).eval()
    return ColorizationPipeline(model, input_size=input_size, device=torch.device(device))


def colorize_dir(input_dir: Path, output_dir: Path, model_name: str = "ddcolor_paper_tiny",
                 input_size: int = 512, device: str | None = None) -> int:
    """Colorize every image in input_dir; outputs saved as PNG, same stems."""
    pipeline = load_pipeline(model_name, input_size=input_size, device=device)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    count = 0
    for path in tqdm(files, desc=f"DDColor({model_name})"):
        dest = output_dir / (path.stem + ".png")
        if dest.exists():
            count += 1
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"unreadable image: {path}")
        out = pipeline.process(img)
        cv2.imwrite(str(dest), out)
        count += 1
    return count
