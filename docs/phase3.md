# Phase 3 — The knowledge base (v1, 2026-07-13)

The ownable core (roadmap §4.1): an explicit, auditable store mapping
**object → colour-distribution prior**, with **modifier tables** that deform
distributions and **composition rules** for stacking them. The LLM's job
(Phase 4) shrinks to *selection* — identify objects, pick operative
modifiers — while colours come from here, inspectable and correctable.

## Layout

- [`kb/objects.yaml`](../kb/objects.yaml) — 22 object classes, each a set of
  weighted modes `{name, ab, sigma, weight, L_range, note}` + provenance
- [`kb/modifiers.yaml`](../kb/modifiers.yaml) — 6 families (era, season,
  weather, geography, time_of_day, mood), each value an ordered op list +
  `applies_to` selectors + provenance
- `chroma_reasoner.kb` — loading/validation (`store.py`), composition +
  resolution (`engine.py`)
- CLI: `python scripts/kb_resolve.py dress --mod era=1910s --mod mood=melancholic --L 45 [--trace]`
- Tests: `tests/test_kb.py` (13 tests; suite total 40)

## Representation decisions

**Distributions, not colours.** A prior is a weighted mode list in the ab
plane. "car" is broad (4 modes, achromatic-heavy per automotive colour
statistics); "school_bus" is one tight regulation mode. Mode `sigma` drives
the suggested adherence tolerance (`max(6, 1.5σ)`) — weakly-constrained
objects automatically get wider tolerances.

**Chroma is conditioned on luminance** (Phase-2 finding institutionalized).
Modes carry an `L_range`; resolution takes the region's *measured* L, filters
modes by it, and projects the winning chroma into the sRGB gamut at that L
(`project_chroma_into_gamut`). `kb_resolve school_bus --L 20` returns a
feasible dark gold, with the projection recorded in the trace.

**Modifiers are op lists, not colour overrides.** Seven ops:
`scale_chroma, shift_ab, clamp_chroma, scale_sigma, reweight, add_mode,
remove_mode`. Reading an entry tells you exactly what it does — e.g.
era:1970s *adds* avocado/harvest-gold/burnt-orange modes to interior objects;
mood:melancholic scales chroma ×0.7 and shifts (−2,−5). Every entry carries
provenance.

**Selection routes through objects.** `applies_to` matches object names,
aliases, or categories; a non-matching modifier is a *recorded no-op*
(`skipped` in the resolution), never an error — the reasoner's selection
mistakes must be visible but not fatal. Test:
`season:autumn` does nothing to a car and everything to foliage (roadmap §1
verbatim).

## Composition rules (the research artifact)

1. **Order = application order**, matching the plan schema's ordered
   `modifiers` array. Ops compose naturally: chroma scales multiply, shifts
   add, reweights multiply; weights renormalize once at the end.
2. **Canonical authoring order**: era → geography → season → weather →
   time_of_day → mood. Rationale: cultural/identity constraints (what colours
   *existed*) first, physical state (what the scene *does* to them) second,
   stylistic intent (how it should *feel*) last — mood modulates the outcome
   rather than competing with facts.
3. Order sensitivity is intended and tested (`test_composition_order_matters`):
   golden_hour-then-melancholic ≠ melancholic-then-golden_hour.

## Validation highlights

- The KB *derives* the Phase-1 canonical example: `dress + era:1910s +
  mood:melancholic` at L=45 resolves to `#646c70` muted slate — the
  hand-authored plan had `#576d7e`. Convergent evidence the modifier
  semantics are right.
- Every base mode is sRGB-feasible at the midpoint of its L_range (tested).
- `Resolution.to_region()` emits plan-schema-valid regions with
  `base_prior: "kb:<object>"` — colours become traceable to auditable priors.

## Known limits / next

- Isotropic sigma (no ab covariance); fine at this scale.
- Era coverage is 4 decades × narrow object sets; expand by mining MHMD
  (period labels) once the general pipeline demands it.
- `resolve` picks the max-weight mode deterministically; sampling and
  multi-candidate output are trivial extensions when the reasoner wants
  alternatives.
- Phase 4: VLM captions/tags the grayscale + measures per-region L → LLM
  selects objects + modifiers → `resolve()` → plan JSON → the proven Phase-2
  pipeline. The reasoner gets `applied`/`skipped`/`trace` back, so its
  selections are checkable.
