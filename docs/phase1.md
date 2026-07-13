# Phase 1 — The object-centric colour plan (locked 2026-07-13)

The plan is the system's intermediate representation: the LLM reasoner emits it, the user may edit it, Grounded-SAM grounds it, the colorizer consumes it, and the evaluator scores against it. It is **locked** as of `plan_version: "1.0"` — schema changes from here require a version bump and a migration note in this file.

- Schema (single source of truth): [`schemas/color_plan.schema.json`](../schemas/color_plan.schema.json)
- Python API: `chroma_reasoner.plan` — `load_plan` / `validate_plan` (reports **all** violations at once), `LabColor`, `lab_to_hex`, `delta_e76`
- Canonical example: [`examples/plans/melancholic_1910s_seaside.json`](../examples/plans/melancholic_1910s_seaside.json)
- CLI: `python scripts/validate_plan.py <plan.json>` — validates and prints per-region hex swatches
- Tests: `tests/test_plan_schema.py`, `tests/test_colors.py`

## Design decisions and why

**Object-centric, not context-centric.** Each region carries `(object, modifiers, resolved_colour, rationale)` — the roadmap's core reframe: factors deform an object's colour prior; they don't act on pixels directly.

**`additionalProperties: false` everywhere.** LLM outputs drift (invented fields, renamed keys). A closed schema turns drift into immediate, legible validation errors instead of silently ignored data.

**Lab (CIE, D65) as the only colour space.** `L` 0–100, `a`/`b` signed — *true* CIE values, not cv2's 0–255 encoding (`plan.colors` has exact pure-numpy conversions; cv2's scaling stays contained inside image pipelines). ΔE and colorization math live in Lab; hex/RGB are display-only derivatives.

**`grounding_phrase` separate from `object`.** `object` is the canonical KB key ('dress'); `grounding_phrase` is what Grounding DINO needs ("the woman's long dress"). Conflating them would force the KB vocabulary into detection phrasing.

**`modifiers` is ordered.** Composition is order-sensitive (roadmap §4.1: precedence rules are a research artifact). The array order is the application order.

**Open modifier vocabulary.** `family` is a free string, not an enum — the factor list is unbounded by design; enums would need a schema bump per new family. Established families are documented in the schema description.

**`tolerance_delta_e` per region.** Palette adherence (Phase 5) measures ΔE from realized colour to `resolved_colour`. Objects differ in how constrained they are — a car could be many blues (wide tolerance), grass cannot be purple (tight). Encoding tolerance in the plan makes adherence evaluation object-aware. Default 10 when absent.

**`base_prior` is nullable.** The KB doesn't exist until Phase 3; plans authored before then (Phase 2's hand-authored plans) set it to `null`. Once the KB lands, it holds the prior's id, making every colour traceable to an auditable source.

**`confidence` + `rationale` required.** The interpretability claim of the whole project lives here: every colour decision must be justified and hedged. Making them required forces the reasoner (and human authors) to produce them.

**Optional `global` block.** Some effects don't route through objects — film-stock rendering, fade, global saturation ceilings (the "era largely decomposes into palette prior + film emulation" note). They apply image-wide after per-region colours.

**ΔE = CIE76 for now.** Euclidean in Lab; JND ≈ 2.3. Adequate for adherence scoring at this scale; CIEDE2000 is the upgrade path if thresholds ever need perceptual precision (would be a metric change, not a schema change).

## What Phase 2 consumes

Phase 2 (manual end-to-end) takes a plan file plus a grayscale image and:
1. `grounding_phrase` → Grounding DINO → SAM → mask per region
2. `(mask, resolved_colour)` → colorizer hint format (scribbles/points inside the mask)
3. L channel + hints → colorized output
4. ΔE(realized colour in mask, `resolved_colour`) vs `tolerance_delta_e` → adherence check
