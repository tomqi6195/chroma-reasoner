# Phase 2 — Manual end-to-end consumption path (in progress)

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

## Findings log

- Out-of-sRGB-gamut plan colours are unreachable by construction — caught by
  a pipeline test, now guarded in authoring (`is_in_srgb_gamut`). Rule: author
  colours that survive the sRGB round-trip.
- (to fill after first Colab mask run)
