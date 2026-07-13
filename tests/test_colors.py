import pytest

from chroma_reasoner.plan import LabColor, delta_e76, lab_to_hex, lab_to_srgb, srgb_to_lab


def test_white_roundtrip():
    lab = srgb_to_lab((1.0, 1.0, 1.0))
    assert lab.L == pytest.approx(100, abs=0.01)
    assert lab.a == pytest.approx(0, abs=0.01)
    assert lab.b == pytest.approx(0, abs=0.01)


def test_black_roundtrip():
    lab = srgb_to_lab((0.0, 0.0, 0.0))
    assert lab.L == pytest.approx(0, abs=0.01)


def test_srgb_lab_srgb_roundtrip():
    for rgb in [(0.2, 0.5, 0.8), (0.9, 0.1, 0.3), (0.5, 0.5, 0.5), (0.0, 1.0, 0.0)]:
        back = lab_to_srgb(srgb_to_lab(rgb))
        assert back == pytest.approx(rgb, abs=1e-4)


def test_known_red():
    # Pure sRGB red is approximately Lab(53.2, 80.1, 67.2)
    lab = srgb_to_lab((1.0, 0.0, 0.0))
    assert lab.L == pytest.approx(53.2, abs=0.5)
    assert lab.a == pytest.approx(80.1, abs=0.5)
    assert lab.b == pytest.approx(67.2, abs=0.5)


def test_hex_output():
    assert lab_to_hex(LabColor(100, 0, 0)) == "#ffffff"
    assert lab_to_hex(LabColor(0, 0, 0)) == "#000000"


def test_delta_e_identity_and_symmetry():
    c1, c2 = LabColor(50, 10, -10), LabColor(55, 12, -8)
    assert delta_e76(c1, c1) == 0
    assert delta_e76(c1, c2) == delta_e76(c2, c1)
    assert delta_e76(c1, c2) == pytest.approx((25 + 4 + 4) ** 0.5)


def test_out_of_gamut_clips_not_crashes():
    # Extreme Lab values fall outside sRGB; conversion must clip gracefully.
    rgb = lab_to_srgb(LabColor(50, 120, -120))
    assert all(0 <= c <= 1 for c in rgb)
