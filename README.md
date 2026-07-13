# Chroma Reasoner

Context-reasoned colorization: an LLM infers which contextual factors (era, mood, season, geography, …) are operative for each object in a grayscale image, resolves colours against an explicit knowledge base, and emits an **object-centric, editable colour plan** that conditions a diffusion colorizer.

See [literature-review-and-roadmap.md](literature-review-and-roadmap.md) for the full framing and [docs/phase0.md](docs/phase0.md) for the current status.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\pip install -e .
git clone --depth 1 https://github.com/piddnad/DDColor.git third_party/DDColor
```

## Phase 0 workflow (baselines & evaluation protocol)

```powershell
# 1. Data: deterministic 300-image COCO val2017 smoke subset + grayscale L inputs
python scripts/download_coco_subset.py --n 300 --seed 0

# 2. Automatic baseline: DDColor (reads only the L channel; leak-free)
python scripts/run_ddcolor.py --input data/coco/val2017_subset --output results/ddcolor_tiny --model ddcolor_paper_tiny

# 3. Metrics: FID, hue-invariant FID, colorfulness, CLIP-score
python scripts/evaluate.py --pred results/ddcolor_tiny --gt data/coco/val2017_subset --manifest data/coco/manifest.json --out results/phase0/ddcolor_tiny
```

The language baseline (L-CAD) needs more than 4 GB VRAM — run [notebooks/lcad_baseline_colab.ipynb](notebooks/lcad_baseline_colab.ipynb) on Colab and evaluate the downloaded outputs locally with the same `scripts/evaluate.py`.

## Repo layout

- `src/chroma_reasoner/data/` — COCO subset + grayscale synthesis
- `src/chroma_reasoner/metrics/` — colorfulness (Hasler-Süsstrunk), FID (clean-fid), hue-invariant FID (arXiv:2503.14974 §5.2.1), CLIP-score
- `src/chroma_reasoner/baselines/` — DDColor runner (wraps `third_party/DDColor`)
- `scripts/` — CLI entrypoints
- `notebooks/` — Colab notebooks for models that don't fit local VRAM
- `tests/` — metric unit tests (`pytest`)
