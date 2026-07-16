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

## Next

1. Live run on the 5 Phase-2 images (needs ANTHROPIC_API_KEY) — compare
   reasoned plans against the hand-authored ones (same images, same pipeline:
   the hand plans are now the human baseline).
2. Colab: reasoned plans → masks → measure true median L per mask →
   re-resolve → hints → Control Color → adherence + the Phase-0 metrics.
3. Phase 5/6: KB-on/off ablation (roadmap kill-criteria) — same selection,
   colours from the LLM instead of the KB, measure ΔE-to-intent and human
   preference.
