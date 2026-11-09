"""Finite-thickness slab impedance, exp(+jwt), including internal inductance.

`faces=1` assumes current enters from one face (field zero at the other);
`faces=2` assumes equal current on both faces. These are slab approximations,
not a solution for arbitrary through-thickness proximity currents or barrels.
"""
import numpy as np

from ._numerics.base import MU0


def surface_impedance(f, thickness, sigma=5.8e7, faces=2):
    """Ohms per square; DC limit 1/(sigma*t), high-frequency skin impedance.
    f is Hz, thickness m. Returns complex values, including internal L.
    """
    if faces not in (1, 2) or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('faces must be 1 or 2 and sigma finite and positive')
    f, thickness = np.asarray(f, float), np.asarray(thickness, float)
    if not np.isfinite(f).all() or (f < 0).any() or not np.isfinite(thickness).all() or (thickness <= 0).any():
        raise ValueError('frequency must be nonnegative and thickness finite and positive')
    u = np.sqrt(2j * np.pi * f * MU0 * sigma) * thickness / faces
    small = np.abs(u) < 1e-3
    safe = np.where(small, 1.0, u)
    ratio = np.where(small, 1 + u*u/3 - u**4/45, safe / np.tanh(safe))
    return ratio / (sigma * thickness)


def face_impedance(f, thickness, sigma=5.8e7, vacuum=True):
    """(self, mutual) ohms per square between a slab's two face sheets (currents per width K1, K2 on the faces of a
    slab `thickness` m thick): the exact 1D two-port Zs [coth kt, csch kt], k = sqrt(j w mu0 sigma), Zs = k / sigma,
    less the vacuum field between two sheets t apart (j w mu0 t / 4 [[1, -1], [-1, 1]]) that a partial-inductance
    model of the faces already counts. DC: both 1 / (sigma t) (the total current's resistance, however it divides);
    low frequency: the slab's internal inductance mu0 t / 12 for a uniform current."""
    f, t = float(f), float(thickness)
    w = 2 * np.pi * f
    u = np.sqrt(1j * w * MU0 * sigma) * t                  # kt
    if abs(u) < 1e-3:
        zs_coth = 1 / (sigma * t) + 1j * w * MU0 * t / 3 if u else 1 / (sigma * t)
        zs_csch = 1 / (sigma * t) - 1j * w * MU0 * t / 6 if u else 1 / (sigma * t)
    else:
        zs = u / (sigma * t)
        zs_coth, zs_csch = zs / np.tanh(u), zs / np.sinh(u)
    vac = 1j * w * MU0 * t / 4 if vacuum else 0.0     # vacuum=False: the raw two-port (MQS removes its own share)
    return zs_coth - vac, zs_csch + vac
