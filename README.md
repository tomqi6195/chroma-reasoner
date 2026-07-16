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

## The colour plan (Phase 1, locked)

The system's intermediate representation is an object-centric JSON plan — see [docs/phase1.md](docs/phase1.md), [schemas/color_plan.schema.json](schemas/color_plan.schema.json), and the [canonical example](examples/plans/melancholic_1910s_seaside.json).

```powershell
python scripts/validate_plan.py examples/plans/melancholic_1910s_seaside.json
```

## Phase 2: manual consumption path

Hand-authored plans ([examples/plans/phase2/](examples/plans/phase2)) → Grounded-SAM masks → naive Lab-paste render / Control Color hints → per-region ΔE adherence. See [docs/phase2.md](docs/phase2.md); masks come from [notebooks/phase2_masks_colab.ipynb](notebooks/phase2_masks_colab.ipynb).

```powershell
python scripts/render_naive.py --plan examples/plans/phase2/000000010092.json --gray data/coco/gray/000000010092.png --masks masks --out results/phase2/naive
```

## The knowledge base (Phase 3)

Object → colour-distribution priors and modifier tables live in [kb/](kb) as auditable YAML; the composition engine is `chroma_reasoner.kb`. See [docs/phase3.md](docs/phase3.md).

```powershell
python scripts/kb_resolve.py dress --mod era=1910s --mod mood=melancholic --L 45 --trace
```

## The reasoner (Phase 4)

Text prompt + grayscale image → colour plan: Claude (vision + structured outputs) selects objects and operative modifiers; the KB resolves colours. See [docs/phase4.md](docs/phase4.md).

```powershell
python scripts/reason_plan.py --image data/coco/gray/000000002299.png --prompt "British school class photo, late 1940s, overcast day" --out plans/reasoned
```

## Repo layout

- `src/chroma_reasoner/data/` — COCO subset + grayscale synthesis
- `src/chroma_reasoner/metrics/` — colorfulness (Hasler-Süsstrunk), FID (clean-fid), hue-invariant FID (arXiv:2503.14974 §5.2.1), CLIP-score
- `src/chroma_reasoner/plan/` — colour-plan validation + Lab colour math (ΔE, Lab↔sRGB, gamut projection)
- `src/chroma_reasoner/kb/` — knowledge-base loading + modifier composition + luminance-conditioned resolution
- `src/chroma_reasoner/baselines/` — DDColor runner (wraps `third_party/DDColor`)
- `kb/` — the knowledge base itself (objects.yaml, modifiers.yaml)
- `schemas/` — the locked plan JSON Schema
- `examples/plans/` — hand-authored reference plans
- `scripts/` — CLI entrypoints
- `notebooks/` — Colab notebooks for models that don't fit local VRAM
- `tests/` — unit tests (`pytest`)
