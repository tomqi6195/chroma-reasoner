# Phase 1 — The object-centric colour plan (locked 2026-07-13)

The plan is the system's intermediate representation: the LLM reasoner emits
it, the user may edit it, Grounded-SAM grounds it, the colorizer consumes it,
and the evaluator scores it. It is **locked** as of `plan_version: "1.0"` —
schema changes require a version bump and a migration note in this file.

- Schema (single source of truth):
  [`schemas/color_plan.schema.json`](../schemas/color_plan.schema.json)
- Python API: `chroma_reasoner.plan` — `load_plan` / `validate_plan` (reports
  **all** violations at once), `LabColor`, `lab_to_hex`, `delta_e76`
- Canonical example: [`melancholic_1910s_seaside.json`][canonical-plan]
- CLI: `python scripts/validate_plan.py <plan.json>` — validates and prints per-region hex swatches
- Tests: `tests/test_plan_schema.py`, `tests/test_colors.py`

## Design decisions and why

**Object-centric, not context-centric.** Each region carries
`(object, modifiers, resolved_colour, rationale)`. Factors deform an object's
colour prior; they do not act on pixels directly.

**`additionalProperties: false` everywhere.** LLM outputs drift through
invented fields and renamed keys. A closed schema turns drift into immediate,
legible validation errors instead of silently ignored data.

**Lab (CIE, D65) as the only colour space.** `L` is 0–100 and `a`/`b` are
signed — *true* CIE values, not cv2's 0–255 encoding. `plan.colors` provides
exact pure-numpy conversions; cv2's scaling stays inside image pipelines. ΔE
and colorization math live in Lab; hex/RGB are display-only derivatives.

**`grounding_phrase` separate from `object`.** `object` is the canonical KB key
(`dress`); `grounding_phrase` is what Grounding DINO needs ("the woman's long
dress"). Conflating them would force the KB vocabulary into detection
phrasing.

**`modifiers` is ordered.** Composition is order-sensitive, so array order is
application order. The precedence rules are part of the research artifact.

**Open modifier vocabulary.** `family` is a free string, not an enum. The
factor list is unbounded by design; an enum would need a schema bump for every
new family. Established families are documented in the schema description.

**`tolerance_delta_e` per region.** Palette adherence (Phase 5) measures ΔE
from realized colour to `resolved_colour`. Objects differ in how constrained
they are: a car could be many blues, while grass cannot be purple. Encoding
tolerance in the plan makes adherence object-aware. The default is 10.

**`base_prior` is nullable.** Plans authored before the Phase 3 KB set it to
`null`. KB-generated plans store the prior's ID, making every colour traceable
to an auditable source.

**`confidence` + `rationale` required.** Every colour decision must be
justified and hedged. Requiring both fields makes the reasoner and human
authors preserve that evidence.

**Optional `global` block.** Effects such as film-stock rendering, fade, and
saturation ceilings do not route through objects. They apply image-wide after
per-region colours.

**ΔE = CIE76 for now.** Euclidean Lab distance, with JND ≈ 2.3, is adequate at
this scale. CIEDE2000 is the upgrade path if thresholds need more perceptual
precision; that would change the metric, not the schema.

## What Phase 2 consumes

Phase 2 (manual end-to-end) takes a plan file plus a grayscale image and:

1. `grounding_phrase` → Grounding DINO → SAM → mask per region
2. `(mask, resolved_colour)` → colorizer hint format (scribbles/points inside the mask)
3. L channel + hints → colorized output
4. ΔE(realized colour in mask, `resolved_colour`) vs `tolerance_delta_e` → adherence check

[canonical-plan]: ../examples/plans/melancholic_1910s_seaside.json
