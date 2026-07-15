# Phase 2 — Manual end-to-end consumption path (complete 2026-07-13)

Goal (roadmap §5): prove that a hand-authored colour plan flows all the way to
pixels — masks from grounding phrases, colours into the colorizer, adherence
measured — with **zero LLM/KB variance**. "The phase people skip and regret."

## Components (local, tested — 26 tests pass)

| Piece | Where | What |
|---|---|---|
| Mask conventions | `plan/masks.py` | `masks/{image_id}/{region_key}.png`, 0/255, region_key = region `id` or `object` |
| Naive renderer | `plan/hints.py::render_naive` | Lab-paste: keep input L, set region ab to plan colour. The deterministic control: proves the path, ΔE≈0 by construction, realism floor for the diffusion model |
| Hint generator | `plan/hints.py::make_hint_image` | Control Color's interface: grayscale + colour strokes painted inside an **eroded** mask (strokes stay off boundaries → less bleed) |
| Adherence evaluator | `plan/adherence.py` | Per-region median-Lab ΔE (CIE76) vs `tolerance_delta_e` (default 10). The objective half of the Phase-5 protocol, built early to debug hint-following |
| Gamut guard | `plan/colors.py::is_in_srgb_gamut` | Out-of-gamut plan colours clip at render and can never adhere; `validate_plan.py` warns |
| CLI | `scripts/render_naive.py` | plan + gray + masks → naive render + adherence report |

## The 5 hand-authored plans (`examples/plans/phase2/`)

Each chosen to stress a different part of the path:

- `000000000139` dining room — indoor domestic palette, several mid-size objects
- `000000001000` tennis kids — **multi-instance binding**: one red shirt among white ones ("the red t-shirt of the boy holding the trophy")
- `000000002299` vintage school photo — **a real B&W photograph**: no colour ground truth exists; the 1940s-Britain era/geo priors do all the work. This is the project's thesis in one image
- `000000010092` jungle lodge — strong saturated palette (orange walls, olive net)
- `000000022755` school bus mirror — the strongest object prior there is (US regulation school-bus yellow), plus a reflection

All 5 validate with zero gamut warnings.

## Colab workflow (`notebooks/phase2_masks_colab.ipynb`)

1. **Grounded-SAM** via HuggingFace `transformers` (`grounding-dino-base` + `sam-vit-huge`) — no CUDA-extension builds. Grounding runs on the **grayscale** image (the honest test-time input). Mask overlays are previewed inline; bad masks → edit the plan's `grounding_phrase`, re-run.
2. **Naive render + adherence** immediately in-notebook — the instant end-to-end signal.
3. **Control Color** — interface confirmed from its `test.py`: `process(input_image, hint_image, prompt, ...)` where `hint_image` = input with strokes painted (it diffs the two to find hints); ckpts `main_model.ckpt` + `content-guided_deformable_vae.ckpt` from its README's Google Drive (expect the quota dance again). Section is scaffolded; first run will need the usual porting iteration (`transformers<5` likely).
4. Export `masks/`, `hints/`, `results/` as a zip for local evaluation.

## What "done" looks like (exit criteria)

- Masks visually correct for ≥ 4/5 images (grounding phrases may need tuning)
- Naive render: all regions pass adherence (proves masks+colours+evaluator agree)
- Control Color output: adherence measured; where it fails, we learn whether
  the failure is *hint strength* (increase stroke area / strength param) or
  *bleeding* (roadmap's structural-consistency vs generative-freedom trade-off)
- Qualitative: the 1940s school photo colorized under its plan looks period-plausible

## Results (final run with fixed hints, 2026-07-13)

All 21 regions across 5 images grounded successfully by Grounded-SAM on
grayscale input. Adherence (ab-plane ΔE, median over exclusive mask):

| image | naive mean ΔE / pass | Control Color mean ΔE / pass |
|---|---|---|
| 000000000139 dining room | 0.08 · 4/4 | 1.82 · 4/4 |
| 000000001000 tennis kids | 0.12 · 4/4 | 8.53 · 3/4 |
| 000000002299 1940s school | 0.18 · 4/4 | 4.21 · 4/4 |
| 000000010092 jungle lodge | 0.10 · 5/5 | 5.07 · 4/5 |
| 000000022755 bus mirror | 0.35 · 4/4 | 4.88 · 3/4 |

**Naive: 21/21 — the consumption path is proven.** Control Color: 18/21,
following clean hints at ΔE 1–8. The three failures are the genuinely hard
cases, each mapping to a known tuning lever (the roadmap's structural-
consistency vs. generative-freedom trade-off), deferred to Phase 4:

- `court` ΔE 26 — strong model prior overrides sparse strokes on a
  fragmented region → densify strokes (erosion 0.15→0.05) and/or raise
  `strength`.
- `net` ΔE 18 — translucency: the model blends what is behind the net,
  which is arguably more physically correct than an opaque plan colour →
  either widen tolerance for translucent objects or model them as tints.
- `bus` ΔE 13 (tol 10) — tiny dark region, nearly passes; was 69 before the
  luminance-feasibility fix.

## Findings log (each one moved a design decision)

1. **Out-of-sRGB-gamut plan colours are unreachable by construction** — now
   guarded (`is_in_srgb_gamut`, CLI warns).
2. **Adherence must be ab-only.** Colorization cannot change L; scoring the L
   term punished the pipeline for authoring guesses (naive paste showed ΔE 88
   on an exact-colour paste). ΔL is now reported separately as authoring
   feedback.
3. **Region masks overlap, and order matters.** Real grounded masks nest:
   "walls" contains the floor, "mirror rim" contains the bus reflected in it,
   the translucent net contains the bed. Fix: paint large→small (specific
   overrides broad) and evaluate each region on its exclusive pixels. Plan
   JSON order is irrelevant; mask area decides.
4. **Chroma feasibility depends on the region's actual luminance, not the
   object's canonical colour.** "School-bus yellow" (b≈68) does not exist in
   sRGB at the L≈20 of a dim mirror reflection; the feasible resolved colour
   is a dark olive-gold (b≈30). Consequence for Phase 4: the reasoner must
   resolve chroma *conditioned on the region's measured L*, which the VLM/
   pipeline can supply per region.
5. **The thesis, photographed.** On the real 1940s B&W school photo, Control
   Color followed all four plan hints (ΔE 2–5.5) but colorized every
   *unplanned* region with vivid modern colours — neon purple/cyan cardigans
   on 1940s schoolchildren. Realistic, and completely anachronistic. This is
   precisely the gap the KB + reasoner fill; interim lever: translate the
   plan's `global` era modifiers into the diffusion prompt.

## Remaining / handoff to later phases

- Hint-strength tuning for the 3 hard cases (Phase 4, once the reasoner emits plans)
- Optional: `using_deformable_vae=True` pass for structure preservation
- Notebook footgun fixed: `transformers<5` is pinned in cell 1 (downgrading
  mid-session after section 2 imports 5.x corrupts the kernel; restart required)
- Phase 3 next: the knowledge base (object priors + modifier tables + composition rules)
