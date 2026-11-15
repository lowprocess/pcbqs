"""
The droop of a load step in time from the rail's impedance at the load (R-E14): a current step of `step` A rising
in `rise` s, the voltage it takes off the rail there.

For a causal impedance Z(w) the step response (V per A) is h(t) = R0 + (2 / pi) int_0^inf g(w) cos(w t) dw with
g = Im Z / w (its inductance; R0 = Re Z at the lowest frequency, the DC path), and the ramp's response
v(t) = step / rise (H(t) - H(t - rise)), H = int_0^t h. With g linear in w between the samples, each segment's
integral is exact through the sine integral, a(Si(w2 t) - Si(w1 t)) + b / t (cos(w1 t) - cos(w2 t)), so the
samples' log spacing and a long t cost no accuracy; below the lowest frequency g is held at its value there, above
the highest at its value there (the mounting's inductance). It is the impedance as solved, between the
frequencies it was solved at: a resonance between two samples is as sharp as they show it.

    t, v = droop(f, Z, step=1.0, rise=1e-9)                     # v: the rail's drop, V (positive: down)
"""
import numpy as np
from scipy.special import sici


def _H(f, Z, t):
    """H(t) = int_0^t h (V s per A) of the impedance samples Z at f (Hz), at times t (s, an array)."""
    w = 2 * np.pi * np.asarray(f, float)
    g = np.imag(Z) / w
    R0 = float(np.real(Z[0]))
    t = np.asarray(t, float)
    out = R0 * t
    tt = np.maximum(t, 1e-300)[:, None]
    si, ci = sici(w[None, :] * tt)
    cos = np.cos(w[None, :] * tt)
    b = np.diff(g) / np.diff(w)                                    # g = a + b w on each segment
    a = g[:-1] - b * w[:-1]
    seg = a * (si[:, 1:] - si[:, :-1]) + b / tt * (cos[:, :-1] - cos[:, 1:])
    tail = g[0] * si[:, 0] + g[-1] * (np.pi / 2 - si[:, -1])        # below and above the samples, held
    out = out + (2 / np.pi) * (seg.sum(1) + tail)                   # (int_0^t cos(w s) ds = sin(w t) / w)
    return np.where(t > 0, out, 0.0)


def droop(f, Z, step=1.0, rise=1e-9, t=None, points=600):
    """(t s, v V) of a current step of `step` A rising in `rise` s at a load whose impedance is Z (complex, at f
    Hz, rising): v is the rail's drop there. t: the times (default: log-spaced from a tenth of the rise to two
    periods of the lowest frequency)."""
    f, Z = np.asarray(f, float), np.asarray(Z, complex)
    if t is None:
        t = np.r_[0.0, np.geomspace(max(rise, 1e-12) / 10, 2 / f[0], points)]
    t = np.asarray(t, float)
    if rise > 0:
        v = step / rise * (_H(f, Z, t) - _H(f, Z, t - rise))
    else:                                                          # an ideal step: h itself, by differencing
        dt = 1e-3 / f[-1]
        v = step * (_H(f, Z, t) - _H(f, Z, t - dt)) / dt
    return t, v


def summary(f, Z, step, rise):
    """dict(peak_V, at_s, settled_V, t_s, v_V) of droop's response (the times decimated for the JSON)."""
    t, v = droop(f, Z, step, rise)
    k = int(np.argmax(np.abs(v)))
    keep = np.unique(np.r_[np.linspace(0, len(t) - 1, 200).astype(int), k])
    return dict(peak_V=float(v[k]), at_s=float(t[k]), settled_V=float(v[-1]), step_A=float(step), rise_s=float(rise),
                t_s=t[keep].tolist(), v_V=v[keep].tolist())
