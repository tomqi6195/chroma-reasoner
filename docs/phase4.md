# Phase 4 — The reasoner (implementation complete 2026-07-13)

Text prompt + grayscale image → validated colour plan, with the VLM doing
**selection only** (roadmap §4.2): it names objects, grounding phrases,
operative modifiers, and per-region luminance estimates; the KB resolves every
colour. This keeps colour knowledge auditable and the model swappable.

Two interchangeable backends behind one `Backend` protocol:

| Backend | Model | Where it runs | Needs |
|---|---|---|---|
| `QwenVLBackend` (**default**) | Qwen2.5-VL-7B-Instruct (3B for T4) | Colab GPU | nothing — no key, no billing |
| `AnthropicBackend` | claude-opus-4-8, vision + structured outputs | anywhere | ANTHROPIC_API_KEY |

The open backend has no structured-outputs guarantee, so it adds: an explicit
JSON contract appended to the prompt, balanced-brace JSON extraction from the
generation, and one format-retry — all before the planner's content-level
repair round. Live runs: `notebooks/phase4_reasoner_colab.ipynb` (the full
loop: reason → ground → render → adherence, in one session).

## Pipeline

```
grayscale image + "melancholic 1910s seaside"
        │
        ▼  AnthropicBackend (claude-opus-4-8, vision, adaptive thinking,
        │   structured outputs → selection JSON is schema-guaranteed)
        ▼
selection: {scene_summary, global_modifiers, regions: [{object,
            grounding_phrase, estimated_L, modifiers, confidence, rationale}]}
        │
        ▼  content validation against the KB (unknown object/modifier,
        │   L/confidence ranges) → ONE repair round with the error list
        ▼
kb.resolve(object, modifiers, measured_L=estimated_L) per region
        │
        ▼
plan JSON (schema-validated) → the proven Phase-2 pipeline
```

## Design decisions

- **The LLM never outputs colours.** Its output schema (`selection_schema.py`)
  has no colour field at all. Selection mistakes are correctable (repair
  round); colour mistakes wouldn't be attributable.
- **Structured outputs** (`output_config.format`) make the selection JSON
  parse-proof; content errors (a made-up object class) are caught by
  `_check_selection` and fed back once. Still-broken selections **raise**
  with the full error list — silent degradation would hide reasoner failures
  from evaluation.
- **`estimated_L` from vision.** The Phase-2/3 luminance-conditioning finding
  reaches the reasoner: the model reads each region's lightness off the
  grayscale, and `resolve()` projects chroma into gamut at that L. The Colab
  pipeline can later re-resolve with mask-measured L (deterministic, cheap).
- **KB vocabulary lives in the system prompt**, generated deterministically
  (sorted) so the prompt is byte-stable → prompt-cache friendly
  (`cache_control` on the system block). Growing the KB automatically grows
  the reasoner's vocabulary.
- **One backend interface** (`Backend` protocol): `AnthropicBackend` today;
  an open VLM (Qwen-VL/LLaVA) implements `complete()` later without touching
  the pipeline.
- **Rationales are two-part**: the LLM's evidence ("weathered institutional
  stone") + the KB's provenance chain ("KB prior 'stone_wall' mode 'grey';
  modifiers: geography:britain") — the full audit trail per region.

## Usage

```powershell
# credentials: setx ANTHROPIC_API_KEY <key>  (or `ant auth login`)
python scripts/reason_plan.py --image data/coco/gray/000000002299.png `
    --prompt "British school class photo, late 1940s, overcast day" --out plans/reasoned
```

Output plans feed directly into the Phase-2 notebook (Grounded-SAM → hints →
Control Color) and `scripts/render_naive.py`.

## Tests

`tests/test_reasoner.py` — mock backend, no API: happy path (alias resolution,
luminance-conditioned colours, global block), repair round, hard failure with
error list, unique region ids, prompt determinism. Suite total: 46.

## First live run (Qwen2.5-VL-7B, 2026-07-13)

3/5 images produced valid plans end-to-end (text → plan → masks → pixels,
adherence 16/16 at ΔE ≈ 0 — the machinery is sound); 2/5 raised
`ReasonerError` (unrepairable selections). Findings, each now addressed:

1. **Open-VLM luminance estimates are unreliable** (ΔL errors up to 60: a
   bright wall estimated L=20). Fix: `re_resolve_with_masks()` — after
   grounding, colours re-resolve at the mask-measured median L
   (deterministic; in the notebook between grounding and rendering). The
   VLM's `estimated_L` is now only a bootstrap.
2. **Measured L doubles as a grounding diagnostic**: multiple regions with
   near-identical measured L means their phrases grabbed the same wrong box
   (2299's dress/suit/wall all measured L=79.6; 22755's "sky" measured 15.7).
   Root cause: the 7B writes vague plural phrases ("the dresses worn by the
   girls") where the human baseline wrote singular specific ones.
3. **Verbatim duplicate regions** (2299 emitted wall+brick twice) — planner
   now dedupes on (object, grounding_phrase).
4. **Hallucinated factors**: era:1940s applied to a "vintage car" in an image
   whose prompt said nothing about a period; system prompt now forbids
   inventing era/geography beyond prompt or unmistakable evidence.
5. **Misc 7B weaknesses**: flat 0.8 confidence, interior classes used
   outdoors (prompt rules added), and the school bus — the strongest prior in
   the KB — was missed entirely.

Quality levers, in expected order of impact: a larger open model
(`Qwen2.5-VL-32B/72B` quantized, same backend, `--model` flag), the Claude
backend as a quality ceiling for comparison, richer grounding-phrase
instructions, and KB vocabulary growth.

## Next

1. Rerun the Phase-4 notebook after pulling (dedup + re-resolve + prompt
   rules); diagnose the 2 unrepairable images from their printed error lists.
2. Reasoned plans + masks → Control Color (Phase-2 notebook section 4) for
   the diffusion render.
3. Phase 5/6: KB-on/off ablation (roadmap kill-criteria) — same selection,
   colours from the LLM instead of the KB, measure ΔE-to-intent and human
   preference.
