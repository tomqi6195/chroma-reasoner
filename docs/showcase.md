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

## Result of the fix (re-run, same seed)

Only 2 of 23 reasoned plans carry a `global` block at all, so only those
images could change — a clean natural control:

| image | global block | CF before | CF after | CF outside masks |
|---|---|---|---|---|
| **000000002299** | `era:1940s` | **79.4** | **51.6** (−35%) | 77.4 → **49.0** |
| 000000022755 | none | 45.9 | 47.4 | 27.5 → 26.8 |
| 000000000139 / 001000 / 010092 | none | unchanged | unchanged | unchanged |

**The one image with an era global block dropped 35% in colourfulness; every
image without one is bit-identical.** Visually the change is larger than the
number suggests: the neon purple and magenta cardigans are gone, replaced by
amber and maroon knitwear that reads as period-plausible — 1940s home-dyed
wool is exactly what the KB's era table describes. The remaining artefacts
are a teal cast on the unmasked back wall and an overall amber heaviness.

So the global block was a real, load-bearing part of the design that had
simply never been connected. It is worth ~a third of the anachronism on the
image it applies to — and does nothing at all elsewhere, which is the correct
behaviour, not a limitation.

## Still open after that fix

- **51.6 is still far from DDColor's 1.8** on a monochrome original. The
  prompt fix addresses the symptom; coverage is the cause.
## Global-block rate: the prompt fix, and its side effect (2026-07-27)

Under the original wording (*"use sparingly"*) only **2 of 23** plans carried
a global block — advice that was backwards once the block became the only
lever over unmasked pixels. Rewriting it to require one whenever the context
prompt implies a period, mood, or lighting condition took the regenerated
showcase set to **4 of 5**, and the fifth (`000000000139`) has no context
prompt at all, so no global block is correct. Every prompted image gained one:

| image | context prompt | global modifiers |
|---|---|---|
| 000000001000 | "summer tennis camp group photo" | season:summer, mood:cheerful |
| 000000002299 | "British school class photo, late 1940s, overcast day" | era:1940s |
| 000000010092 | "tropical jungle lodge interior" | geography:tropics, mood:nostalgic |
| 000000022755 | "overcast day in an American town" | era:1940s, geography:usa, **season:autumn**, weather:overcast |

**The side effect is visible in the last row.** Only `geography:usa` and
`weather:overcast` are in that prompt; `era:1940s` and `season:autumn` are
invented — the plan's own rationale gives it away ("a *vintage car* driving
down a street"), inferring a decade from one object glimpsed in a mirror.
Pushing for globals traded under-application for over-application.

The instruction now carries its own bound: at most 2 global modifiers, each
stated in the prompt or beyond doubt from the whole image, with the failure
named explicitly ("one old-looking car does not make a 1940s photograph") and
an empty prompt required to yield no globals. Worth re-checking on the next
run — the general lesson is that this model needs *both* directions of a
constraint stated, since strengthening one side reliably breaks the other.

- **Coverage is the real lever.** The reasoner selects 3–6 regions regardless
  of scene complexity; a 40-child class photo needs far more, or a background
  region, to be constrained. Consider making the reasoner target *coverage*
  rather than region count, and adding a catch-all background region.
- Control Color's `strength` / `scale` are untuned — worth a sweep on 2299.
- Re-run the showcase after the prompt fix and compare CF against this table;
  2299's CF is the headline number to move.
