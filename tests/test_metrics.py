import numpy as np

from chroma_reasoner.metrics.colorfulness import colorfulness
from chroma_reasoner.metrics.hue_invariant import match_colorfulness, scale_chroma_yuv


def test_colorfulness_zero_for_grayscale():
    gray = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert colorfulness(gray) == 0.0


def test_colorfulness_positive_for_color():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:16, :, 0] = 255  # red half
    img[16:, :, 2] = 255  # blue half
    assert colorfulness(img) > 50


def test_scale_chroma_alpha_zero_desaturates():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    desat = scale_chroma_yuv(img, 0.0)
    # alpha=0 collapses chroma; YUV<->RGB roundtrip keeps a small residue
    assert colorfulness(desat) < 5.0


def test_scale_chroma_monotone_in_alpha():
    rng = np.random.default_rng(1)
    img = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
    cfs = [colorfulness(scale_chroma_yuv(img, a)) for a in (0.25, 0.5, 1.0, 1.5)]
    assert cfs == sorted(cfs)


def test_match_colorfulness_hits_target():
    rng = np.random.default_rng(2)
    img = rng.integers(60, 200, (64, 64, 3), dtype=np.uint8)
    target = colorfulness(img) * 0.5  # ask for a much duller version
    corrected, alpha = match_colorfulness(img, target)
    assert abs(colorfulness(corrected) - target) < 1.0
    assert 0 < alpha < 1
