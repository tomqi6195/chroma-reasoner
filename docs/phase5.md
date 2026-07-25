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
