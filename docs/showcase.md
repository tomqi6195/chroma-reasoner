# Showcase render — the coverage finding (2026-07-27)

First end-to-end visual output: reasoned plan → Grounded-SAM masks → colour
hints → Control Color. Four panels per image (original / greyscale / DDColor /
plan-conditioned) in `results/showcase/results/figures/`.

**The renders are worse than the automatic baseline on sparsely-planned
images, and the cause is measurable.**

## Result

| image | mask coverage | CF plan-cond. | CF outside masks | CF DDColor | CF original |
|---|---|---|---|---|---|
| 000000022755 bus mirror | **70.0%** | 45.9 | **27.5** | 42.4 | 33.1 |
| 000000010092 jungle lodge | 34.7% | 49.5 | 42.2 | 33.8 | 52.1 |
| 000000001000 tennis kids | 28.6% | 77.4 | 71.7 | 44.5 | 67.4 |
| **000000002299 1940s school** | **19.8%** | **79.4** | **77.4** | **1.8** | **0.0** |
| 000000000139 dining room | 10.3% | 54.0 | 53.0 | 25.7 | 61.1 |

Colourfulness overshoot scales inversely with mask coverage, and it is
concentrated **outside** the planned regions in every case. The 1940s school
photo is the worst case in the set: a monochrome original (CF 0.0) where
DDColor sensibly stays near-neutral (1.8) while the plan-conditioned render
reaches 79.4 — neon purple and orange cardigans on 1940s schoolchildren, with
77.4 of that colourfulness coming from the 80% of pixels the plan never
masked. At 70% coverage (the bus image) the output is well-controlled.

**Per-region hints constrain only masked pixels. Everything else is filled by
the colorizer's own prior, which reaches harder for vivid modern colour the
less it is given to anchor on.** This is the Phase-2 finding #5 observation,
now quantified.

## The architectural gap it exposed

The plan schema has carried a `global` block since Phase 1 — for exactly the
effects that "do not route through a single object": film-stock rendering, era
fade, saturation ceilings. The 2299 plan *has* `era:1940s` there, noting
"muted saturation". **The renderer never used it.** Per-region hints
structurally cannot express whole-image rendering, and nothing else was
carrying it to the colorizer.

Fix: `plan.hints.global_prompt_terms(plan)` translates global modifiers into
positive/negative diffusion prompt fragments (era:1940s → "1940s colour
photography, muted utility palette" / "vivid, neon, saturated modern
colours"), now applied in the showcase notebook's render cell. Unmapped
modifiers fall back to their `effect` text, so the KB can grow without code
changes.

## Still open after that fix

- **Coverage is the real lever.** The reasoner selects 3–6 regions regardless
  of scene complexity; a 40-child class photo needs far more, or a background
  region, to be constrained. Consider making the reasoner target *coverage*
  rather than region count, and adding a catch-all background region.
- Control Color's `strength` / `scale` are untuned — worth a sweep on 2299.
- Re-run the showcase after the prompt fix and compare CF against this table;
  2299's CF is the headline number to move.
