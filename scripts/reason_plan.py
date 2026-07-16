"""Text prompt + grayscale image -> colour plan, via the VLM reasoner + KB.

Usage (open VLM, needs a GPU - run on Colab; no API key):
    python scripts/reason_plan.py --backend qwen --image gray/000000002299.png \
        --prompt "British school class photo, late 1940s, overcast day" --out plans/reasoned

Usage (Claude API; needs ANTHROPIC_API_KEY or an `ant auth login` profile):
    python scripts/reason_plan.py --backend anthropic --image ... --prompt ... --out ...
"""

import argparse
import json
from pathlib import Path

from chroma_reasoner.kb import load_kb
from chroma_reasoner.plan import LabColor, lab_to_hex
from chroma_reasoner.reasoner import reason_plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True, help="grayscale image (png/jpg)")
    ap.add_argument("--prompt", type=str, default="", help="abstract context prompt")
    ap.add_argument("--out", type=Path, required=True, help="output directory for the plan JSON")
    ap.add_argument("--backend", choices=["qwen", "anthropic"], default="qwen")
    ap.add_argument("--model", type=str, default=None,
                    help="override model id for the chosen backend")
    args = ap.parse_args()

    kb = load_kb()
    if args.backend == "qwen":
        from chroma_reasoner.reasoner.backend_open import DEFAULT_OPEN_MODEL, QwenVLBackend

        backend = QwenVLBackend(model_id=args.model or DEFAULT_OPEN_MODEL)
    else:
        from chroma_reasoner.reasoner.backend import DEFAULT_MODEL, AnthropicBackend

        backend = AnthropicBackend(model=args.model or DEFAULT_MODEL)
    plan = reason_plan(kb, backend, args.image, user_prompt=args.prompt)

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{plan['image_id']}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print(f"plan: {out_path}")
    if plan.get("prompt"):
        print(f"prompt: {plan['prompt']!r}")
    for region in plan["regions"]:
        lab = LabColor.from_plan(region["resolved_colour"])
        mods = ", ".join(f"{m['family']}:{m['value']}" for m in region.get("modifiers", []))
        print(f"  [{region['object']:>16}] {lab_to_hex(lab)}  Lab({lab.L:g},{lab.a:g},{lab.b:g})"
              f"  conf={region['confidence']:.2f}  tol={region.get('tolerance_delta_e')}  [{mods}]")
        print(f"       {region['grounding_phrase']}")


if __name__ == "__main__":
    main()
