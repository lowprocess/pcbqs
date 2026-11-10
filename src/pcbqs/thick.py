"""Finite-thickness kernels for close sheet pairs (experimental): a sheet standing for a slab of copper t thick
interacts, with its near neighbours, through the 1/R kernel averaged over both slabs' thickness instead of between
two zero-thickness sheets.

Why: in the inductive regime a zero-thickness sheet's current is singular at its edges, so its loss grows without
bound as the edges are refined (an isolated strip at 10 MHz: +26 / +76 / +82% against FastHenry at 4 / 8 / 16
cells across). Copper caps the crowding at about its thickness; averaging the kernel over it does the same, as
FastHenry's rectangular filaments do.

    D = delta_inductance(rwg, z, t)    # sparse, over RWG branches: added to either backend's point-kernel L

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


def delta_inductance(rwg, z, t, outer_levels=1, reach=3.0, log=None):
    """The thickness correction as a sparse matrix over RWG branches (edge currents i = l I, as MQS's L): for every
    close triangle pair, (thickness-averaged - point) moments, combined over the RWG halves each triangle supports.
    Added to the point-kernel L of either backend (dense or compressed), it gives thick_kernel's L."""
    from scipy.sparse import coo_matrix
    MU0 = 4e-7 * np.pi
    nodes, tris = rwg.nodes, rwg.tris
    p, q = close_pairs(nodes, tris, z, t, reach)
    key = np.round(np.c_[z[p] - z[q], t[p], t[q]] * 1e9).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    d0 = np.zeros(len(p)); dro = np.zeros((len(p), 2)); drs = np.zeros((len(p), 2)); drr = np.zeros(len(p))
    for g in range(inv.max() + 1):
        sel = np.flatnonzero(inv == g)
        k = sel[0]
        pt = offplane_pairs(nodes, tris, p[sel], q[sel], z[p[k]] - z[q[k]], outer_levels=outer_levels)
        xs, ws = _trapezoid_rule(z[p[k]] - z[q[k]], t[p[k]], t[q[k]])
        acc = [np.zeros_like(a) for a in pt]
        for x, w in zip(xs, ws):
            for a, b in zip(acc, offplane_pairs(nodes, tris, p[sel], q[sel], x, outer_levels=outer_levels)):
                a += w * b
        d0[sel] = acc[0] - pt[0]; dro[sel] = acc[1] - pt[1]; drs[sel] = acc[2] - pt[2]; drr[sel] = acc[3] - pt[3]
    # each triangle's RWG halves: (edge, alpha, free vertex)
    halves = [[] for _ in range(len(tris))]
    for e in range(rwg.Ne):
        halves[rwg.tp[e]].append((e, rwg.l[e] / (2 * rwg.area[rwg.tp[e]]), rwg.vp[e]))
        halves[rwg.tm[e]].append((e, -rwg.l[e] / (2 * rwg.area[rwg.tm[e]]), rwg.vm[e]))
    rows, cols, vals = [], [], []
    for k in range(len(p)):
        a, b = p[k], q[k]
        for ea, aa, va in halves[a]:
            for eb, ab, vb in halves[b]:
                v = aa * ab * (drr[k] - vb @ dro[k] - va @ drs[k] + (va @ vb) * d0[k])
                rows.append(ea); cols.append(eb); vals.append(v)
                if a != b:                                   # the pair's other order: observation <-> source
                    rows.append(eb); cols.append(ea); vals.append(v)
    s = 1.0 / rwg.l
    vals = MU0 / (4 * np.pi) * np.array(vals) * s[np.array(rows)] * s[np.array(cols)]
    D = coo_matrix((vals, (rows, cols)), shape=(rwg.Ne, rwg.Ne)).tocsr()
    D = 0.5 * (D + D.T)
    if log:
        log(f'mqs: thickness correction on {len(p):,} close triangle pairs, {D.nnz:,} entries')
    return D
