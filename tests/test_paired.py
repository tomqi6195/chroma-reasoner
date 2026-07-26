"""Paired-statistics tests: the analysis that turns per-region margins into
a defensible claim."""

import pytest

from chroma_reasoner.eval.paired import (bootstrap_median_ci, compare_arms,
                                         format_comparison, sign_test)


def _report(values: dict[str, dict[str, tuple[float, float]]]) -> dict:
    """values: {image: {region: (delta_e, chroma_deficit)}} -> arm report."""
    images = []
    for image, regions in values.items():
        rows = [{"region": r, "delta_e": de, "chroma_deficit": cd}
                for r, (de, cd) in regions.items()]
        images.append({"image": image, "regions": rows})
    return {"images": images}


def test_sign_test_symmetric_and_exact():
    assert sign_test(0, 0) == 1.0
    assert sign_test(5, 5) == 1.0
    assert sign_test(10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert sign_test(0, 10) == sign_test(10, 0)     # symmetric
    assert sign_test(9, 1) < 0.05                   # decisive
    assert sign_test(6, 4) > 0.5                    # not decisive


def test_bootstrap_ci_brackets_median_and_is_deterministic():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    lo, hi = bootstrap_median_ci(values)
    assert lo <= 4.0 <= hi
    assert bootstrap_median_ci(values) == (lo, hi)   # same seed -> same CI


def test_compare_arms_counts_wins_for_lower_values():
    a = _report({"img1": {"sky": (5.0, 1.0), "grass": (8.0, 2.0)}})
    b = _report({"img1": {"sky": (15.0, 9.0), "grass": (3.0, 12.0)}})
    cmp = compare_arms(a, b, "kb", "llm")

    assert cmp["n_paired_regions"] == 2
    de = cmp["metrics"]["delta_e"]
    assert (de["wins_a"], de["wins_b"]) == (1, 1)     # sky to kb, grass to llm
    cd = cmp["metrics"]["chroma_deficit"]
    assert (cd["wins_a"], cd["wins_b"]) == (2, 0)     # kb commits more on both


def test_compare_arms_only_pairs_shared_regions():
    a = _report({"img1": {"sky": (5.0, 1.0)}, "img2": {"road": (4.0, 1.0)}})
    b = _report({"img1": {"sky": (9.0, 2.0), "extra": (1.0, 1.0)}})
    cmp = compare_arms(a, b, "kb", "llm")
    assert cmp["n_paired_regions"] == 1               # only img1/sky is shared


def test_disjoint_arms_pair_nothing():
    a = _report({"img1": {"walls": (5.0, 1.0)}})
    b = _report({"img1": {"wall_interior": (5.0, 1.0)}})
    assert compare_arms(a, b, "human", "kb")["n_paired_regions"] == 0


def test_near_equal_values_count_as_ties():
    a = _report({"img1": {"sky": (5.0, 1.0), "road": (5.2, 1.0)}})
    b = _report({"img1": {"sky": (5.1, 1.0), "road": (5.0, 1.0)}})
    de = compare_arms(a, b, "kb", "llm")["metrics"]["delta_e"]
    assert de["ties"] == 2 and de["wins_a"] == 0 and de["wins_b"] == 0


def test_format_comparison_is_readable():
    a = _report({"img1": {"sky": (5.0, 1.0)}})
    b = _report({"img1": {"sky": (25.0, 9.0)}})
    text = format_comparison(compare_arms(a, b, "kb", "llm"))
    assert "kb vs llm" in text
    assert "delta_e" in text and "p=" in text
