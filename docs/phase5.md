# Phase 5 — Evaluation harness & the KB-on/off ablation (harness complete 2026-07-13)

The roadmap's kill-criterion for the whole project (§6): *"ablate KB-on vs
KB-off; if ΔE-to-intent and human preference don't move, the KB is redundant."*
This phase builds that experiment.

## What already existed (built early, on purpose)

- **Palette adherence** (objective half): `plan/adherence.py`, Phase 2
- **Realism/faithfulness metrics**: FID, HI-FID, colourfulness, CLIP-score, Phase 0

## New: ΔE-to-reality scoring (`eval/reference.py`)

When the context prompt describes the real scene, the original photograph's
colours are a fair proxy for intent. Per region: masked-median Lab of the
original vs the plan's colour, ab-plane CIE76. Any colour source — KB
resolution, LLM guess, human author — scores on the same masks with the same
metric. Monochrome originals (the 1940s school photo) are auto-excluded.

## New: the KB-off arm (`eval/ablation.py`)

`llm_color_plan(plan, backend, image)` takes an existing KB-resolved plan and
asks the VLM for one hex colour per region directly — same regions, same
grounding phrases, same masks, same luminance. The only difference between
arms is the colour source, so the comparison isolates exactly the KB's
contribution. Runs in the Phase-4 notebook while the backend is loaded;
plans land in `plans/ablation_llm/`.

## Three arms

| Arm | Plans | Masks | Colour source |
|---|---|---|---|
| `human` | `examples/plans/phase2` | `masks` | hand-authored (Phase 2) |
| `kb` | `plans/reasoned` | `masks_reasoned` | VLM selection → KB resolution |
| `llm` | `plans/ablation_llm` | `masks_reasoned` | VLM direct hex choice |

`kb` vs `llm` is the clean comparison (identical masks). `human` uses its own
masks, so cross-arm reads against it are mask-confounded — treat it as
context, not as a controlled arm.

```powershell
python scripts/ablate_score.py --originals data/coco/val2017_subset `
    --arm human=examples/plans/phase2:masks `
    --arm kb=plans/reasoned:masks_reasoned `
    --arm llm=plans/ablation_llm:masks_reasoned `
    --out results/phase5/reference_scores.json
```

## HEADLINE — scaled ablation, 61 shared regions (2026-07-14)

30 smoke-subset images, COCO captions as context prompts, 23 reasoned
successfully, 18 scorable (5 originals are monochrome), **61 region pairs**.
Both arms share masks by construction.

| kb vs llm | wins (decided) | median margin | 95% CI | sign test |
|---|---|---|---|---|
| ΔE-to-reality | 23/49 | −0.14 | [−1.23, +0.55] | **p = 0.78** |
| chroma deficit | **32/39** | +1.02 | [+0.00, +1.85] | **p = 0.00007** |

Arm means: ΔE 14.8 (kb) vs 17.8 (llm); chroma deficit 3.3 vs 5.5;
undercommitted regions 5/61 vs 10/61.

**This confirms the n=21 finding at 3× the sample, and it is the project's
first decision-grade result:**

1. **On colour accuracy the two arms are indistinguishable** — 23/49 is a
   coin flip, the median margin is essentially zero, and the CI straddles it.
   The mean difference (14.8 vs 17.8) is heavy-tail artefact, exactly what
   the paired test exists to strip out. *For scenes a model can describe,
   the explicit KB does not beat implicit model knowledge on accuracy.*
2. **On chroma commitment the KB wins decisively** — 32 of 39 decided
   regions, p < 1e-4. The LLM arm hedges toward gray; the KB commits to real
   colour. This is the roadmap's §2.8 desaturation failure mode reproduced
   as a measurable, significant difference between arms.

Per the roadmap's own kill-criterion, result (1) means the KB is **not**
earning its keep on realistic prompts, and the response is the one §7
prescribes: concentrate it where reality is unavailable to either arm —
under-determined regions and counterfactual (era/mood) prompts. Result (2)
is the standing evidence that it does something implicit knowledge does not.

Consistent per-region losses are actionable KB feedback, not noise, and
repeat across runs: `wood_floor` and `rug` (warm-varnished priors vs pale
near-neutral reality), `curtain`/`bedding` (over-warm), `building_facade`.
These priors' mode weights want revisiting before any further scaling.

Robustness fixes prompted by this run (7/30 reasoner failures, all
vocabulary): object-name **normalization** in the KB lookup (possessives,
articles, case, separators, naive plurals — "woman's dress" now resolves to
`dress`), plus a COCO-common vocabulary batch (food, vegetables, fruit,
hair, flowers, porcelain/bathroom, metal fixtures, towels, animal fur,
zebra, `ground`).

## Paired analysis changes the reading (2026-07-14)

Means over images are the wrong summary when the arms are **matched
region-for-region** (the ablation copies the KB arm's regions and masks). The
paired view (`eval/paired.py`, printed by `ablate_score.py`) — per-region
margins, win/loss counts, exact sign test, bootstrap CI on the median margin
— tells a different and more honest story on the same run-6 data:

| kb vs llm (21 shared regions) | result |
|---|---|
| ΔE-to-reality | kb wins **10/19** decided, median margin **+0.0**, p = 1.0 |
| chroma deficit | kb wins **15/16** decided, median margin **+3.2**, p = **0.0005** |

**Revised claim: on realistic prompts the two arms are statistically
indistinguishable in colour accuracy; the KB's decisive, significant
advantage is chroma commitment.** The earlier mean-based readings (both the
"LLM wins" of run 4 and the "KB matches human" of run 6) were being swung by
a handful of badly-grounded regions — exactly what a paired test exists to
neutralise.

This is the roadmap's §7 redundancy risk landing in the data: on scenes a
model can simply describe, an explicit KB does not beat implicit knowledge on
accuracy. It earns its keep by refusing to hedge into gray — and, per §6, the
place to look for a decisive accuracy win is **under-determined regions and
counterfactual prompts**, where reality is not available to either arm.

The per-region losses are also actionable KB feedback rather than noise: the
worst are `wood_floor` (prior is warm varnished brown; both test floors were
near-neutral pale) and `rug` — priors whose mode weights want revisiting.

## Current result (run 6, 2026-07-14 — 4 fresh images, both metrics)

| arm | mean ΔE ↓ | chroma deficit ↓ | undercommitted |
|---|---|---|---|
| human | 18.4 | 5.4 | 2/17 |
| **kb** | 18.0 | **4.4** | **2/21** |
| llm | 16.3 | 8.3 | 5/21 |

**The KB matches the human baseline on ΔE with the best chroma commitment of
any arm, humans included.** The LLM's ΔE edge remains fully gray-bought
(double the deficit; in 22755 its ΔE literally equals its deficit). On the
freshly-cleared 139 with shared masks, KB beats LLM on both metrics
(13.4/0.0 vs 15.8/0.8). Still n=4; scale next.

Robustness added after this run: region-level salvage — when the repair round
leaves only region-local errors (a 'tennis_racket') and ≥2 valid regions
remain, the broken regions are dropped (recorded in scene_summary) instead of
failing the image.

## The desaturation exploit (found 2026-07-14, fixed)

Run 4's LLM arm "beat" the KB on mean ΔE (14.6 vs 17.8) by collapsing to the
same near-gray hex for almost every region — and since real walls, asphalt,
and sky are near-neutral, gray scores well. **Our ΔE-to-reality metric had
re-derived the field's classic PSNR bias: rewarding conservative
desaturation** (roadmap §2.8). Fix: per-region **chroma deficit**
`max(0, C_ref − C_planned)` reported alongside ΔE, plus an
undercommitted-region count (deficit > 10 where reference chroma > 15). A
gray guess for a gray wall is fine; a gray guess for a colourful region now
shows up.

Two-metric view of run 4 (both must be read together):

| arm | mean ΔE ↓ | chroma deficit ↓ | undercommitted |
|---|---|---|---|
| human | 18.4 | **5.4** | 2/17 |
| kb | 17.8 | 5.7 | 3/15 |
| llm | **14.6** | 7.2 | **4/15** |

The LLM's ΔE edge is bought with gray: worst deficit, most undercommitted
regions, and (run 3, controlled masks) wildly wrong colours whenever it did
commit — yellow-green tennis court, purple 1940s dresses. The KB commits to
real chroma at human-baseline deficit. Small-n caveats stand; the metric
pair is now exploit-resistant in both directions.

## Earlier three-arm numbers (2026-07-13/14)

- `human`: mean ΔE-to-reality **18.4** (4 images, own masks — context only)
- `kb` (7B reasoner): **17.9** (2 images)
- `llm` (same regions, model-picked colours): **40.4** (1 scorable image)

**The controlled comparison (identical masks, image 1000): KB wins 5/6
regions and ties the 6th** — 24.6 vs 40.4 mean. Failure modes are
characteristic: the LLM picked yellow-green (b=+44) for a blue tennis court,
saturated orange for a green windscreen, and — on the unscorable 1940s photo —
**vivid purple (ab 36,−58) for wartime dresses** where the KB's era-composed
prior gave drab neutral. Direction strongly favours the KB; n is far too
small to close the question. Per-region rows remain the grounding auditor
(large ΔEs still trace to mask errors more often than colour errors).

## Phase-4 live-run feedback loop (what the log-driven runs fixed)

- 2/5 image failures were **KB vocabulary gaps**, not model errors: the
  reasoner wanted table/chair/rug/counter, mosquito_net/tile_floor/window,
  building/street/motorcycle, geography:america. KB expanded accordingly
  (aliases + 8 new classes + geography:usa); the vocabulary now covers all
  selections attempted across three live runs.
- 7B selections are **run-to-run unstable** (22755 passed one run, failed the
  next; 1000's classes changed between runs). Expected to shrink with the
  vocabulary fix; a larger open model is the next lever if not.

## Known limitations (v1)

- ΔE-to-reality only works where prompt ≈ reality; counterfactual prompts
  ("make it autumn") need the VLM-judge/human protocol (not yet built).
- The ablation arm's chroma is gamut-projected at the *estimated* L (it runs
  before masks exist); the KB arm gets mask-measured re-resolution. Minor
  asymmetry, ab-comparison mostly unaffected; fix if the margin is ever close.
- n=5 images. Scale via the smoke subset once the pipeline is trusted.

## Decision rule (pre-registered)

On shared masks, if `kb` mean ΔE-to-reality is not clearly below `llm` across
images (and region-level wins are not majority-kb), the KB as built is not
earning its keep on realistic prompts — per the roadmap, concentrate it on
under-determined regions (garments, ambiguous fabrics) or era/mood
counterfactuals where the LLM has no reference to lean on.
