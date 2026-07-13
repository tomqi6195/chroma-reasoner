# Phase 0 — Baselines & evaluation protocol

Status log for Phase 0 of the [roadmap](../literature-review-and-roadmap.md#5-implementation-plan-phased). Updated 2026-07-12.

## Compute reality

- Local: RTX 3050 laptop, **4 GB VRAM**, CUDA 12.1, torch 2.5.1+cu121. Runs DDColor (tiny confirmed; large untested), all metrics, and the data pipeline.
- Heavy models (L-CAD, Control Color, later the ControlNet colorizer): **Colab Pro** — see `notebooks/lcad_baseline_colab.ipynb`.

## Checkpoint availability (the roadmap's flagged risk) — ✅ cleared

| Model | Weights | Source |
|---|---|---|
| DDColor | ✅ four variants (`ddcolor_paper`, `ddcolor_modelscope`, `ddcolor_artistic`, `ddcolor_paper_tiny`) | HuggingFace `piddnad/*`, ModelScope |
| L-CAD | ✅ Google Drive folder (also Baidu) | linked from repo README |
| Control Color | ✅ colorization model + VAE released 2024-12-16 | Google Drive, linked from repo README |

## Dataset

Deterministic smoke subset: **300 images of COCO val2017**, `seed=0`, sorted-id sampling — reproducible on any machine (`scripts/download_coco_subset.py`). Each record carries all COCO captions; evaluation uses the first. Grayscale inputs are the Lab L channel (`data/coco/gray/`), matching field convention. The full protocol targets (later, on Colab): Extended COCO-Stuff (1,946 images / 3,520 prompts) and ImageNet val5k with BLIP captions, per arXiv:2503.14974.

**Caveat:** FID over 300 images has high variance. Smoke numbers validate plumbing and give rough ordering; they are not publishable and not comparable to paper numbers.

## Metrics (`src/chroma_reasoner/metrics/`)

- **FID** — clean-fid, pred vs. GT colour images.
- **Hue-invariant FID (HI-FID)** — per arXiv:2503.14974 §5.2.1: scale each prediction's chroma in YUV by α\* chosen so its Hasler-Süsstrunk colorfulness matches its ground truth's, then compute FID. Removes the saturation-pumping degree of freedom. Implementation: bisection on the monotone CF(α); `hi_alpha_mean` in the report shows how much correction was needed (α<1 ⇒ over-saturated outputs, α>1 ⇒ dull).
- **Colorfulness / ΔCF** — Hasler & Süsstrunk 2003.
- **CLIP-score** — `openai/clip-vit-base-patch32`, image vs. first COCO caption; GT score reported as the ceiling.

Unit tests: `tests/test_metrics.py`.

## Baselines

- **DDColor** (automatic): local, `scripts/run_ddcolor.py`. Uses the upstream pipeline, which consumes only the L channel of its input — feeding colour images is leak-free and preserves the exact original L in the output.
- **L-CAD** (language): Colab notebook scaffolded. Open item: L-CAD's sampling config for custom (image, caption) pairs must be adapted on first Colab run — its dataset plumbing assumes Extended COCO-Stuff layout.

## Results (smoke scale: 300 COCO val2017 images, seed 0)

| Baseline | FID ↓ | HI-FID ↓ | CF (GT 41.5) | ΔCF | CLIP ↑ (GT 30.7) |
|---|---|---|---|---|---|
| DDColor tiny (automatic) | **50.6** | **48.6** | 39.9 | −1.5 | 29.9 |
| L-CAD (COCO caption) | 61.8 | 60.0 | 40.3 | −1.2 | **30.6** |

Raw numbers in `results/phase0/*/report.json`. Reading them:

- **The ordering is exactly the expected pattern**, which validates the harness: the language-conditioned model wins CLIP-score (30.58, essentially at the GT ceiling of 30.67, vs. 29.92 for automatic), while the automatic model wins realism (FID/HI-FID). Caption conditioning measurably moves prompt faithfulness.
- **FID magnitudes are dominated by n=300 small-sample bias** (DDColor reports ≈3.9 on full ImageNet val). Smoke numbers are only comparable *between systems on this exact subset*.
- **L-CAD's FID carries a resolution confound**: it outputs 256×256 while DDColor preserves original resolution; clean-fid resizes internally but upscaling artifacts still cost L-CAD some FID. Don't over-read the FID gap.
- **Both models are slightly under-saturated** (ΔCF ≈ −1.2 to −1.5; HI-FID α* ≈ 1.11–1.20).
- **Reference targets for the plan-conditioned system** (roadmap §6 kill-criteria, on this subset): CLIP-score > 30.6 (beat L-CAD) and FID < 50.6 / HI-FID < 48.6 (beat DDColor).
- L-CAD produced 299/300 outputs (000000560371 missing — single sampling failure, not investigated).

### L-CAD-on-Colab port notes (all patches are in the notebook)

Getting L-CAD's 2023 research code running on a 2026 Colab required: skipping its conda-dump requirements.txt (torch 1.12 pin, local file:// paths); `torchviz` (debugging leftover import); patching the hardcoded placeholder checkpoint path and the author's local CLIP path (`/data/pretrained/...` → `openai/clip-vit-large-patch14`); `transformers<5` + non-strict weight load (the ckpt uses the 4.x CLIP key layout with `.text_model` nesting); and a Drive "Make a copy" workaround for the weights (public folder is perpetually quota-blocked). Weight choice: `coco_weight.ckpt` = caption-conditioned Extended-COCO-Stuff model; `multi_weight` = instance/SAM variant; `auto_weight` = no-caption.

## Open items / next

1. ~~Run L-CAD notebook on Colab Pro~~ — done 2026-07-13, results above.
2. Decide whether to also run `ddcolor_paper` (large) locally — may OOM on 4 GB; fall back to Colab if so.
3. Small user study deferred until there is a system output to compare (Phase 2+).
4. **Phase 1 next**: lock the object-centric plan JSON schema + validator (colour space: Lab).
