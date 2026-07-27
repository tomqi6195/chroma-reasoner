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

## Not yet done

- **The LLM counterfactual arm** — the decisive comparison. Same fixed
  regions, but colours chosen by the model under each condition prompt
  (`eval/ablation.py` already supports this; the condition plans are written
  by `--write-plans`). Prediction to test: the LLM shows *weaker separation*
  than the KB, because it has no systematic period/mood mapping and its
  gray-hedging habit gives it nowhere to move. Needs one Colab run.
- Human/VLM-judge protocol for plan quality (roadmap §6's subjective half).
- Era evaluated on era-appropriate imagery (MHMD).
