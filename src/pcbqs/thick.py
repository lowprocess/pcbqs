"""Finite-thickness kernels for close sheet pairs (experimental): a sheet standing for a slab of copper t thick
interacts, with its near neighbours, through the 1/R kernel averaged over both slabs' thickness instead of between
two zero-thickness sheets.

Why: in the inductive regime a zero-thickness sheet's current is singular at its edges, so its loss grows without
bound as the edges are refined (an isolated strip at 10 MHz: +26 / +76 / +82% against FastHenry at 4 / 8 / 16
cells across). Copper caps the crowding at about its thickness; averaging the kernel over it does the same, as
FastHenry's rectangular filaments do.

    thicken(nodes, tris, z, t, M0, Mrx, Mry, Mrr)      # in place, on static_moments' dense moments

For a pair at heights zp, zq with thicknesses tp, tq the offset between a point of each slab is spread over a
trapezoid (the convolution of two uniform distributions) centred on zp - zq; the moments are even in the offset and
exact for any offset (_numerics.sing.offplane_pairs), so the average is a Gauss sum over the offset, split where the
trapezoid bends and at zero (the |offset| kink).
"""
import numpy as np
from scipy.spatial import cKDTree

from ._numerics.sing import offplane_pairs

_GX, _GW = np.polynomial.legendre.leggauss(4)


def _trapezoid_rule(delta, tp, tq):
    """Nodes and weights (summing to 1) for the offset between uniform points of two slabs, centred delta apart."""
    a, b = abs(tp - tq) / 2, (tp + tq) / 2
    breaks = sorted({delta - b, delta - a, delta + a, delta + b} | ({0.0} if delta - b < 0 < delta + b else set()))
    xs, ws = [], []
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        if hi - lo <= 0:
            continue
        x = 0.5 * (hi + lo) + 0.5 * (hi - lo) * _GX
        u = np.abs(x - delta)
        dens = np.where(u <= a, 1 / max(tp, tq), (b - u) / (tp * tq))       # the trapezoid's density
        xs.append(x)
        ws.append(0.5 * (hi - lo) * _GW * dens)
    x, w = np.concatenate(xs), np.concatenate(ws)
    return x, w / w.sum()


def close_pairs(nodes, tris, z, t, reach=3.0):
    """Pairs (p <= q) whose thickness matters: on overlapping or touching slabs (|zp - zq| < 2 (tp + tq)) and
    closer, centroid to centroid, than their summed radii plus `reach` thicknesses."""
    V = nodes[tris]
    c = V.mean(1)
    rad = np.linalg.norm(V - c[:, None], axis=2).max(1)
    tree = cKDTree(c)
    cand = tree.query_pairs(2 * rad.max() + reach * t.max(), output_type='ndarray')
    cand = np.vstack([cand, np.c_[np.arange(len(tris)), np.arange(len(tris))]])
    p, q = cand[:, 0], cand[:, 1]
    d = np.linalg.norm(c[p] - c[q], axis=1)
    keep = (np.abs(z[p] - z[q]) < 2 * (t[p] + t[q])) & (d < rad[p] + rad[q] + reach * np.maximum(t[p], t[q]))
    return p[keep], q[keep]


def thicken(nodes, tris, z, t, M0, Mrx, Mry, Mrr, outer_levels=1, reach=3.0, log=None):
    """Replace the close pairs' moments by their thickness averages, in place (both orders). Returns the count."""
    p, q = close_pairs(nodes, tris, z, t, reach)
    # group the pairs by (offset, thicknesses): each group shares one quadrature
    key = np.round(np.c_[z[p] - z[q], t[p], t[q]] * 1e9).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    m0 = np.zeros(len(p)); mro = np.zeros((len(p), 2)); mrs = np.zeros((len(p), 2)); mrr = np.zeros(len(p))
    for g in range(inv.max() + 1):
        sel = np.flatnonzero(inv == g)
        k = sel[0]
        xs, ws = _trapezoid_rule(z[p[k]] - z[q[k]], t[p[k]], t[q[k]])
        for x, w in zip(xs, ws):
            a, b, c_, d_ = offplane_pairs(nodes, tris, p[sel], q[sel], x, outer_levels=outer_levels)
            m0[sel] += w * a; mro[sel] += w * b; mrs[sel] += w * c_; mrr[sel] += w * d_
    off = p != q
    M0[p, q] = m0; M0[q, p] = m0
    Mrr[p, q] = mrr; Mrr[q, p] = mrr
    Mrx[p[off], q[off]], Mry[p[off], q[off]] = mro[off, 0], mro[off, 1]
    Mrx[q[off], p[off]], Mry[q[off], p[off]] = mrs[off, 0], mrs[off, 1]
    dg = ~off
    Mrx[p[dg], p[dg]] = 0.5 * (mro[dg, 0] + mrs[dg, 0])
    Mry[p[dg], p[dg]] = 0.5 * (mro[dg, 1] + mrs[dg, 1])
    if log:
        log(f'mqs: thickness-averaged kernels on {len(p):,} close pairs ({inv.max() + 1} offset groups)')
    return len(p)
