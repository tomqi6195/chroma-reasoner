# Phase 6 — Counterfactual protocol & narrowing the contribution

Phase 5 established that on **realistic** prompts the KB does not beat implicit
model knowledge on colour accuracy (p = 0.78 over 61 region pairs) — the
roadmap's §7 redundancy risk, in data. Its prescribed response is to look where
reality is unavailable to either arm: **counterfactual context**. This phase
builds that test and uses it to pick the sharpest claim (roadmap §5, Phase 6).

## Protocol (`eval/counterfactual.py`, `scripts/counterfactual.py`)

The same plan is re-resolved under each of 8 conditions (neutral, era 1910s /
1940s / 1970s, mood melancholic / cheerful, season autumn / summer).
**Regions are held fixed** across conditions, and luminance is preserved, so
what is measured is the *colour* response to context — not selection noise,
and not something a colorizer couldn't apply. For the KB arm this needs no
model at all: the KB is deterministic, so the protocol runs locally.

Two pre-registered measurements per contrast:

1. **Separation** — palette distance between the two conditions on the same
   regions, plus the *active share*: the fraction of regions whose colour
   moved at all. A system that ignores context scores ≈ 0. This is the
   roadmap's kill-criterion made objective ("if humans can't distinguish
   1910s from 1970s outputs, the plan lacks discriminative period features").
2. **Direction correctness** — the shift must be *right*, not merely present.
   Expectations are declared in code (`CONTRASTS`) before running and checked
   with the same exact sign test as the Phase-5 ablation.

```powershell
python scripts/counterfactual.py --plans plans/reasoned --out results/phase6/counterfactual.json
```

## Results (KB arm, 23 images / 76 regions, 2026-07-14)

| contrast | active share | median margin | direction | sign test |
|---|---|---|---|---|
| melancholic vs cheerful — chroma | **97%** | +10.0 | 54/74 as expected | **p = 0.0001** |
| melancholic vs cheerful — warmth | 97% | +12.1 | **73/74** as expected | **p ≈ 0** |
| autumn vs summer — redness | 11% | +43.2 | 8/8 as expected | p = 0.008 |
| 1910s vs 1970s — chroma | 11% | +15.1 | 8/8 as expected | p = 0.008 |
| 1940s vs 1970s — chroma | 8% | +22.9 | 6/6 as expected | p = 0.03 |

**Every declared direction held, with zero counter-examples outside the mood
contrast.** The system responds to counterfactual context correctly — which
is exactly what the realistic-prompt ablation could not show.

The decisive split is **coverage, not correctness**:

- **Mood moves 97% of regions.** `mood:*` applies to `'*'`, so every object
  responds, and the effect is large and correctly signed (warmth 73/74).
- **Era and season move ~10%.** They are correct where they bite but bite
  rarely: era applies to clothing/vehicles, season to vegetation, and most
  COCO val2017 scenes (kitchens, bathrooms, food, animals) contain neither.
  Broadening era's `applies_to` to vehicles and 1970s dress — historically
  documented (Ford's black-only policy; the decade's earth-tone fashion and
  automotive catalogues) — **tripled the effect size** (median margin 1.3 →
  15.1) but barely moved coverage (7 → 8 regions). Coverage is bounded by
  scene content, not KB design.

## Narrowing decision (roadmap §5 Phase 6)

The evidence selects **mood as the primary axis**, and it does so by
quantifying what the roadmap predicted in prose (§1: *"Mood is the cleaner
novelty claim… treat era as one modifier family among several, not a
pillar"*):

1. **Mood** — broad coverage, large correctly-signed effect, no ground truth
   to be redundant with, and the KB's chroma-commitment advantage (Phase 5,
   p = 7e-05) matters most exactly where a model would hedge to gray.
2. **Era** — keep as a *disambiguation prior on era-bearing content*, not a
   standalone axis. To evaluate it fairly it needs era-appropriate imagery
   (portraits, street scenes — i.e. MHMD, which is precisely that), or the
   roadmap's own decomposition: era as global film/dye rendering via the
   plan's `global` block rather than per-object modifiers.

## HEADLINE — counterfactual arms at n=23 (2026-07-27)

23 images × 4 conditions, 72 shared region pairs, zero failures. Both arms
get identical regions and identical condition prompts.

### Era: the KB knows the period, the model does not

| era_1910s vs era_1970s | KB | LLM |
|---|---|---|
| chroma direction (1910s must be duller) | **6/6 as expected** | **6/12 — chance** |
| sign test | **p = 0.031** | **p = 1.0** |
| active regions | 6/72 | 13/72 |

**This is the project's first clean accuracy win, and it is exactly the one
the roadmap predicted.** Asked to colour the same regions for "1910s" versus
"1970s", the open VLM is a coin flip on which decade should be more muted —
it moves colours, but not in any systematic period direction. The KB is right
every time, because `era:1910s` (scale chroma ×0.6, clamp 32) and `era:1970s`
(add avocado / harvest-gold / burnt-orange modes) are documented operations,
not recollection. At n=5 the LLM scored 8/8 here; at n=23 it regresses to
chance, which is what a small sample of a coin flip looks like.

### Mood: both respond, but only the KB keeps a palette

| mood_melancholic vs mood_cheerful | KB | LLM |
|---|---|---|
| warmth direction | **69/69** (p ≈ 0) | 50/68 (p = 0.0001) |
| chroma direction | 50/69 (p = 0.0002) | 66/67 (p ≈ 0) |
| median separation | 13.1 | 29.9 |
| **palette spread / distinct colours under "melancholic"** | **14.4 / 3.0** | **9.2 / 1.5** |

The n=5 collapse finding holds at scale: under "melancholic" the LLM assigns
**1.5 distinct colours per image**, flattening the scene to a single tone,
then swings to 45.4 spread under "cheerful". Its larger separation and better
chroma score are both products of that global tint. The KB moves every object
relative to its own prior — 3.0 distinct colours in both conditions — and
gets the warmth direction **perfectly, 69 out of 69**, where the model manages
73%.

(The KB's 50/69 on chroma is expected, not a defect: melancholic scales chroma
×0.7 *and* applies a cool shift, so on already-neutral objects the shift adds
a little chroma. Muting a grey wall by 30% leaves it grey; tinting it cool
makes it slightly blue.)

### What this establishes

> On realistic prompts the KB matches implicit model knowledge (Phase 5,
> p = 0.78). On **counterfactual** prompts it separates from it: the model
> has no systematic period mapping and collapses the palette under mood,
> while the KB applies documented, per-object, correctly-signed deformations.

That is the roadmap's §7 prescription carried out and confirmed — concentrate
the KB where reality is unavailable to either arm — and it is the evidence
behind narrowing to mood with era as a disambiguation prior.

## The LLM arm at n=5 — prediction wrong, mechanism right (2026-07-14)

Both arms, same 26 regions, same condition prompts. The **pre-registered
prediction was that the LLM would show weaker separation. It is wrong**: the
LLM separates roughly twice as far as the KB.

| mood_melancholic vs mood_cheerful | KB | LLM |
|---|---|---|
| median separation | 13.1 | **25.7** |
| active share | **96%** | 77% |
| direction: chroma / warmth | 21/25, **24/25** | 17/20, 16/20 |
| **palette spread within image** | **11.4 → 20.8** | **0.4 → 26.7** |
| **distinct colours per image** | **4.8 / 5.0** | **1.2 / 4.2** |

The last two rows explain the first. Under "melancholic" the LLM assigns
**1.2 distinct colours per image** — it paints the whole scene a single flat
tone, then blasts everything saturated under "cheerful". Its large separation
is a **global tint swing, not object-aware reasoning**. The KB keeps ~5
distinct colours in both conditions: each object is muted or warmed relative
to *its own* prior, so a sky still reads as sky and wood as wood.

So the mechanism cited in the prediction (degenerate hedging) is what is
happening — it simply shows up as *too much* uniform movement rather than too
little. The defensible claim is not "the LLM ignores context" (it does not;
directions are correct, p < 0.05) but:

> **Both arms respond to counterfactual context; only the KB applies it
> per-object while preserving the scene's palette structure.**

That is the object-centric thesis of the project, measured.

This is the third time a magnitude metric alone proved gameable — ΔE rewarded
grey (Phase 5), separation rewards global tinting (here). Each needed a
companion: **chroma deficit** next to ΔE, **palette spread** next to
separation. Treat any single-number colour metric as exploitable until paired.

Caveat: 5 images / 26 regions (the notebook read the repo's older plan set).
Re-run with the 23-image set for publishable n.

## Not yet done

- **Re-run the LLM arm at n=23** (commit `plans/reasoned/` first, or use the
  notebook's upload cell). The mechanism is unambiguous but the sample is small.
- **A paraphrase control.** (A same-condition rerun would be uninformative:
  the backend decodes greedily, so identical input gives identical output and
  separation is 0 by construction.) The informative control is to re-run a
  condition under a *paraphrase* — "a sombre, downcast scene" against "a
  melancholic, sombre scene". If paraphrase separation approaches
  opposite-mood separation, the arm is reacting to surface wording rather
  than meaning. This is the last alternative explanation still standing for
  the LLM arm's response.
- Human/VLM-judge protocol for plan quality (roadmap §6's subjective half).
- Era evaluated on era-appropriate imagery (MHMD).
