"""Resolve an object's colour against the KB with selected modifiers.

Usage:
    python scripts/kb_resolve.py dress --mod era=1910s --mod mood=melancholic --L 45
    python scripts/kb_resolve.py foliage --mod season=autumn
"""

import argparse

from chroma_reasoner.kb import load_kb, resolve
from chroma_reasoner.plan.colors import LabColor, lab_to_hex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("object")
    ap.add_argument("--mod", action="append", default=[], metavar="FAMILY=VALUE")
    ap.add_argument("--L", type=float, default=None, help="region's measured luminance (0-100)")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    modifiers = []
    for m in args.mod:
        family, _, value = m.partition("=")
        modifiers.append({"family": family, "value": value, "effect": ""})

    kb = load_kb()
    res = resolve(kb, args.object, modifiers, measured_L=args.L)

    print(f"object: {res.object}   chosen mode: {res.chosen_mode}")
    print(f"resolved: {lab_to_hex(res.resolved)}  Lab({res.resolved.L:g},{res.resolved.a:g},{res.resolved.b:g})"
          f"   tolerance dE: {res.tolerance_delta_e:g}")
    print(f"rationale: {res.rationale}")
    print("distribution:")
    for mode in sorted(res.modes, key=lambda m: -m["weight"]):
        lab = LabColor(sum(mode.get("L_range", [0, 100])) / 2, *mode["ab"])
        print(f"  {mode['weight']:.2f}  {mode['name']:>16}  ab({mode['ab'][0]:.0f},{mode['ab'][1]:.0f})"
              f"  ~{lab_to_hex(lab)}")
    if args.trace:
        print("trace:")
        for line in res.trace:
            print(f"  - {line}")


if __name__ == "__main__":
    main()
