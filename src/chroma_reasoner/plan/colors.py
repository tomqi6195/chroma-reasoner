"""CIE L*a*b* (D65) colour math for the plan pipeline.

Pure-numpy reference implementations (not cv2's scaled 8-bit Lab): plans store
true CIE values (L 0-100, a/b signed), and evaluation needs exact ΔE. cv2's
Lab is only used inside image pipelines where its 0-255 scaling is contained.

ΔE here is CIE76 (Euclidean in Lab) — adequate for palette adherence at
Phase-1/2 scale; CIEDE2000 is the documented upgrade path if adherence
thresholds ever need perceptual precision near the JND.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# D65 reference white
_XN, _YN, _ZN = 0.95047, 1.0, 1.08883
_DELTA = 6 / 29


@dataclass(frozen=True)
class LabColor:
    L: float
    a: float
    b: float

    @classmethod
    def from_plan(cls, resolved_colour: dict) -> "LabColor":
        assert resolved_colour.get("space") == "Lab"
        return cls(resolved_colour["L"], resolved_colour["a"], resolved_colour["b"])

    def to_plan(self) -> dict:
        return {"space": "Lab", "L": self.L, "a": self.a, "b": self.b}


def _f_inv(t: float) -> float:
    return t ** 3 if t > _DELTA else 3 * _DELTA ** 2 * (t - 4 / 29)


def _f(t: float) -> float:
    return t ** (1 / 3) if t > _DELTA ** 3 else t / (3 * _DELTA ** 2) + 4 / 29


def lab_to_srgb(lab: LabColor) -> tuple[float, float, float]:
    """Lab (D65) -> sRGB in [0,1], gamut-clipped."""
    fy = (lab.L + 16) / 116
    x = _XN * _f_inv(fy + lab.a / 500)
    y = _YN * _f_inv(fy)
    z = _ZN * _f_inv(fy - lab.b / 200)

    r_lin = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g_lin = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b_lin = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    def gamma(c: float) -> float:
        c = min(max(c, 0.0), 1.0)
        return 12.92 * c if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055

    return gamma(r_lin), gamma(g_lin), gamma(b_lin)


def srgb_to_lab(rgb: tuple[float, float, float]) -> LabColor:
    """sRGB in [0,1] -> Lab (D65)."""

    def linear(c: float) -> float:
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(c) for c in rgb)
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / _XN
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / _YN
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / _ZN

    fx, fy, fz = _f(x), _f(y), _f(z)
    return LabColor(116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))


def lab_to_hex(lab: LabColor) -> str:
    r, g, b = lab_to_srgb(lab)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def delta_e76(c1: LabColor, c2: LabColor) -> float:
    """CIE76 ΔE: Euclidean distance in Lab. ~2.3 is a just-noticeable difference."""
    return math.sqrt((c1.L - c2.L) ** 2 + (c1.a - c2.a) ** 2 + (c1.b - c2.b) ** 2)


def is_in_srgb_gamut(lab: LabColor, tolerance: float = 1.0) -> bool:
    """True if this Lab colour survives a round-trip through sRGB.

    Out-of-gamut plan colours get clipped at render time, so the realized
    colour will miss the plan's target no matter how well the colorizer
    follows hints. Plans should only specify in-gamut colours; the
    validate_plan CLI warns about violations.
    """
    return delta_e76(lab, srgb_to_lab(lab_to_srgb(lab))) <= tolerance


def project_chroma_into_gamut(lab: LabColor, iters: int = 22) -> LabColor:
    """Scale ab toward neutral until the colour is sRGB-representable at its L.

    The Phase-2 school-bus finding as an operation: the maximum feasible
    chroma depends on luminance, so a KB-resolved chroma must be projected
    into gamut at the region's measured L before it goes into a plan.
    """
    if is_in_srgb_gamut(lab):
        return lab
    lo, hi = 0.0, 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if is_in_srgb_gamut(LabColor(lab.L, lab.a * mid, lab.b * mid)):
            lo = mid
        else:
            hi = mid
    return LabColor(lab.L, lab.a * lo, lab.b * lo)


def srgb_array_to_lab(rgb: "np.ndarray") -> "np.ndarray":
    """Vectorized sRGB->Lab. rgb: (..., 3) float in [0,1]. Returns (..., 3) Lab.

    Same math as srgb_to_lab; used where per-pixel conversion matters
    (adherence evaluation over masked regions).
    """
    import numpy as np

    rgb = np.asarray(rgb, dtype=np.float64)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / _XN
    y = (0.2126729 * r + 0.7151522 * g + 0.0721750 * b) / _YN
    z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / _ZN
    xyz = np.stack([x, y, z], axis=-1)
    f = np.where(xyz > _DELTA ** 3, np.cbrt(xyz), xyz / (3 * _DELTA ** 2) + 4 / 29)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)
