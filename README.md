# Chroma Reasoner

Context-reasoned colorization. A VLM reads a greyscale image and an abstract
context prompt ("a melancholic 1910s seaside", "British school photo, late
1940s") and **selects** which objects are present and which contextual factors
are operative for each. An explicit, auditable **knowledge base** — not the
model — resolves those selections into colours. The result is an
**object-centric, editable colour plan** that grounds to masks and conditions a
diffusion colorizer.

The novelty is the reasoning-to-plan layer and its evaluation, not the pixel
model. See [literature-review-and-roadmap.md](literature-review-and-roadmap.md)
for the full framing.

```
greyscale + "melancholic 1910s seaside"
   │
   ├─ VLM  ── selects objects, grounding phrases, operative modifiers, luminance
   │          (never colours — its output schema has no colour field)
   ├─ KB   ── resolves each selection to a gamut-feasible Lab colour
   │          (weighted mode distributions + documented modifier ops)
   ▼
colour plan (JSON, schema-locked, user-editable)
   │
   ├─ Grounded-SAM ── grounding phrases → masks
   ├─ colorizer    ── L channel + per-region hints → pixels
   ▼
evaluation ── palette adherence · ΔE-to-reality · chroma commitment · counterfactual separation
```

## Status

| Phase | What | State |
|---|---|---|
| 0 | Baselines (DDColor, L-CAD) + metric protocol | [done](docs/phase0.md) |
| 1 | Colour-plan schema, locked at `plan_version 1.0` | [done](docs/phase1.md) |
| 2 | Manual consumption path (masks → hints → pixels → adherence) | [done](docs/phase2.md) |
| 3 | The knowledge base (priors, modifiers, composition) | [done](docs/phase3.md) |
| 4 | The reasoner (open VLM, no API key) | [done](docs/phase4.md) |
| 5 | KB-on/off ablation with paired statistics | [done](docs/phase5.md) |
| 6 | Counterfactual protocol + narrowing | [KB arm done](docs/phase6.md); LLM arm pending one Colab run |

## Findings so far

**The KB does *not* beat implicit model knowledge on colour accuracy for
realistic prompts.** Over 61 region pairs with identical masks, KB-resolved and
model-chosen colours are statistically indistinguishable against the original
photographs (23/49 decided regions, sign test p = 0.78). This is the roadmap's
own redundancy risk, measured rather than assumed.

**The KB decisively wins colour commitment.** The same comparison: 32/39
decided regions, p = 7e-05. Left to itself the model hedges toward grey — the
classic desaturation failure mode — while the KB commits to real chroma. Our
first intent metric had to be redesigned when a degenerate model output
exploited it, re-deriving the field's PSNR bias; ΔE is now always paired with
a **chroma-deficit** term.

**Counterfactual context works, and mood is the axis with reach.** Every
pre-registered direction held with no counter-examples: melancholic vs cheerful
moves 97% of regions (warmth 73/74 correct, p ≈ 0); autumn, 1910s and 1970s
contrasts are correct but touch ~10% of regions, because most scenes contain
no era-bearing or vegetated content. This quantifies the roadmap's prose
prediction that mood is the cleaner novelty claim and era is one family among
several, not a pillar.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
.venv\Scripts\pip install -e .
git clone --depth 1 https://github.com/piddnad/DDColor.git third_party/DDColor
python -m pytest tests -q
```

Everything except the neural models runs on CPU. The VLM, Grounded-SAM, and the
diffusion colorizers need a GPU — those live in [notebooks/](notebooks) and run
on Colab. **No API keys are required anywhere**; the default reasoner backend is
open-weights Qwen2.5-VL.

## Local workflows

```powershell
# Data: deterministic COCO val2017 subset + greyscale L inputs
python scripts/download_coco_subset.py --n 300 --seed 0

# Baseline + metrics (FID, hue-invariant FID, colourfulness, CLIP-score)
python scripts/run_ddcolor.py --input data/coco/val2017_subset --output results/ddcolor_tiny
python scripts/evaluate.py --pred results/ddcolor_tiny --gt data/coco/val2017_subset --manifest data/coco/manifest.json --out results/phase0/ddcolor_tiny

# Knowledge base: resolve an object under context, with the full audit trail
python scripts/kb_resolve.py dress --mod era=1910s --mod mood=melancholic --L 45 --trace

# Plans: validate, render deterministically, score adherence
python scripts/validate_plan.py examples/plans/melancholic_1910s_seaside.json
python scripts/render_naive.py --plan examples/plans/phase2/000000010092.json --gray data/coco/gray/000000010092.png --masks masks --out results/phase2/naive

# Evaluation: three-arm ablation with paired per-region statistics
python scripts/ablate_score.py --originals data/coco/val2017_subset --arm kb=plans/reasoned:masks_reasoned --arm llm=plans/ablation_llm:masks_reasoned --out results/phase5/scores.json

# Counterfactuals: prompt separation + direction correctness (KB arm, no model)
python scripts/counterfactual.py --plans plans/reasoned --out results/phase6/counterfactual.json
```

## Colab notebooks

| Notebook | Purpose |
|---|---|
| [lcad_baseline_colab](notebooks/lcad_baseline_colab.ipynb) | L-CAD language baseline (Phase 0) |
| [phase2_masks_colab](notebooks/phase2_masks_colab.ipynb) | Grounded-SAM masks + Control Color diffusion render |
| [phase4_reasoner_colab](notebooks/phase4_reasoner_colab.ipynb) | Reason plans end to end, ground them, render, score |
| [phase5_scaled_ablation_colab](notebooks/phase5_scaled_ablation_colab.ipynb) | Ablation at ~30 images with captions as prompts |
| [phase6_counterfactual_colab](notebooks/phase6_counterfactual_colab.ipynb) | KB vs LLM under counterfactual context prompts |
| [showcase_render_colab](notebooks/showcase_render_colab.ipynb) | The visual artifact: original / greyscale / DDColor / plan-conditioned |

The Phase 4–6 notebooks write a run log into their output zip, so failures are
diagnosable locally without re-running the GPU work.

## Repo layout

- `src/chroma_reasoner/data/` — COCO subset + greyscale synthesis
- `src/chroma_reasoner/plan/` — plan validation, Lab colour math (ΔE, gamut projection), masks, hints, adherence
- `src/chroma_reasoner/kb/` — KB loading, modifier composition, luminance-conditioned resolution
- `src/chroma_reasoner/reasoner/` — VLM backends (Qwen2.5-VL, Claude), prompts, planner
- `src/chroma_reasoner/eval/` — ΔE-to-reality, KB-off ablation, paired statistics, counterfactual protocol
- `src/chroma_reasoner/metrics/` — FID, hue-invariant FID (arXiv:2503.14974 §5.2.1), colourfulness, CLIP-score
- `src/chroma_reasoner/baselines/` — DDColor runner
- `kb/` — the knowledge base itself: `objects.yaml`, `modifiers.yaml`
- `schemas/` — the locked plan JSON Schema
- `examples/plans/` — hand-authored reference plans (the human baseline)
- `docs/` — per-phase design records, decisions, and results
- `scripts/`, `notebooks/`, `tests/`

## Design decisions worth knowing

- **The model never outputs colours.** Its schema has no colour field. Selection
  errors are correctable and attributable; colour errors would be neither.
- **Colours are resolved at the region's measured luminance.** Chroma feasible
  at L=75 does not exist at L=20 — school-bus yellow in a dim reflection is a
  dark olive-gold, and the KB returns that.
- **Masks nest, so painting is largest-first** and each region is scored on its
  exclusive pixels; a "wall" mask that contains the floor must not be blamed
  for it.
- **Adherence is ab-only.** Colorization cannot change luminance, so scoring L
  would punish the pipeline for the plan author's guess.
- **Arms are compared with paired statistics.** They share regions by
  construction; means over images hide the effect behind heavy tails.
