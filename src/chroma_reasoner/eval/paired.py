"""Paired per-region statistics for the KB-on/off ablation.

The arms are matched by construction: the ablation arm copies the KB arm's
regions and masks, changing only the colour. So the correct analysis is
paired — per-region margins, win/loss counts, and a test on the margins —
not a comparison of two independent means. At n=30 images the mean ΔE is a
poor summary (one badly-grounded region swings it); "KB wins 71 of 94
regions, median margin 6.2, p=0.002" is the defensible claim.

Test: exact two-sided sign test on the win/loss counts. No distributional
assumption, no scipy dependency, and robust to the heavy tails that mask
errors produce. Ties are excluded (standard). A bootstrap CI accompanies the
median margin.
"""

from __future__ import annotations

import math
import random

METRICS = ("delta_e", "chroma_deficit")


def _region_index(arm_report: dict) -> dict[tuple[str, str], dict]:
    """(image_id, region_key) -> region row, for one arm's report."""
    index = {}
    for image in arm_report["images"]:
        for row in image["regions"]:
            if "delta_e" in row:
                index[(image["image"], row["region"])] = row
    return index


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided sign test p-value. Ties excluded before calling."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def bootstrap_median_ci(values: list[float], iterations: int = 2000,
                        seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the median. Deterministic given seed."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    medians = []
    for _ in range(iterations):
        sample = sorted(values[rng.randrange(n)] for _ in range(n))
        mid = n // 2
        medians.append(sample[mid] if n % 2 else (sample[mid - 1] + sample[mid]) / 2)
    medians.sort()
    return (medians[int(0.025 * iterations)], medians[int(0.975 * iterations)])


def _median(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2


def compare_arms(report_a: dict, report_b: dict, name_a: str, name_b: str,
                 tie_threshold: float = 0.5) -> dict:
    """Paired comparison of two arms over their shared regions.

    Lower is better for both metrics, so a positive margin (b - a) means
    arm A won that region. tie_threshold guards against calling
    quantization-scale differences a win.
    """
    index_a, index_b = _region_index(report_a), _region_index(report_b)
    shared = sorted(set(index_a) & set(index_b))

    out: dict = {"arm_a": name_a, "arm_b": name_b, "n_paired_regions": len(shared),
                 "metrics": {}}
    for metric in METRICS:
        margins, wins, losses, ties = [], 0, 0, 0
        worst = []
        for key in shared:
            a, b = index_a[key].get(metric), index_b[key].get(metric)
            if a is None or b is None:
                continue
            margin = b - a          # >0 : A is closer/committed => A wins
            margins.append(margin)
            if margin > tie_threshold:
                wins += 1
            elif margin < -tie_threshold:
                losses += 1
                worst.append((margin, key, a, b))
            else:
                ties += 1
        lo, hi = bootstrap_median_ci(margins)
        worst.sort()
        out["metrics"][metric] = {
            "wins_a": wins, "wins_b": losses, "ties": ties,
            "median_margin": round(_median(margins), 2),
            "median_margin_ci95": [round(lo, 2), round(hi, 2)],
            "mean_margin": round(sum(margins) / len(margins), 2) if margins else None,
            "sign_test_p": round(sign_test(wins, losses), 5),
            "worst_losses": [{"image": k[0], "region": k[1],
                              f"{name_a}": round(av, 2), f"{name_b}": round(bv, 2)}
                             for _, k, av, bv in worst[:5]],
        }
    return out


def format_comparison(cmp: dict) -> str:
    a, b = cmp["arm_a"], cmp["arm_b"]
    lines = [f"paired comparison: {a} vs {b}  ({cmp['n_paired_regions']} shared regions)"]
    for metric, s in cmp["metrics"].items():
        decided = s["wins_a"] + s["wins_b"]
        share = f"{s['wins_a']}/{decided}" if decided else "0/0"
        lines.append(
            f"  {metric:>15}: {a} wins {share} decided ({s['ties']} ties)  "
            f"median margin {s['median_margin']:+g} "
            f"[{s['median_margin_ci95'][0]:+g}, {s['median_margin_ci95'][1]:+g}]  "
            f"p={s['sign_test_p']}")
        for loss in s["worst_losses"][:3]:
            lines.append(f"      worst loss: {loss['image']}/{loss['region']} "
                         f"({a}={loss[a]} vs {b}={loss[b]})")
    return "\n".join(lines)
