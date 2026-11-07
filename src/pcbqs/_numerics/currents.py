"""Private numerical snapshot; independently maintained. See PROVENANCE.json."""
import numpy as np
from .base import _BAR6, _W6, tri_area2

def rwg_currents(rwg, I):
    """(a, b): J = a r + b on each triangle (a (Nt,), b (Nt, 2)) from RWG coefficients I (Ne,); I (Ne, k): a (Nt, k),
    b (Nt, 2, k)."""
    I = np.asarray(I)
    multi = I.ndim > 1
    I2 = I if multi else I[:, None]
    k = I2.shape[1]
    a = np.zeros((len(rwg.tris), k), complex)
    b = np.zeros((len(rwg.tris), 2, k), complex)
    for t, alpha, v in ((rwg.tp, rwg.l / (2 * rwg.area[rwg.tp]), rwg.vp), (rwg.tm, -rwg.l / (2 * rwg.area[rwg.tm]), rwg.vm)):
        ia = I2 * alpha[:, None]
        np.add.at(a, t, ia)
        np.add.at(b, t, -ia[:, None, :] * v[:, :, None])
    return (a, b) if multi else (a[:, 0], b[:, :, 0])


def _split4(V):
    """Each triangle (M, 3, 2) into its 4 midpoint triangles (4M, 3, 2)."""
    m01, m12, m20 = (V[:, 0] + V[:, 1]) / 2, (V[:, 1] + V[:, 2]) / 2, (V[:, 2] + V[:, 0]) / 2
    return np.concatenate([np.stack(t, 1) for t in ((V[:, 0], m01, m20), (m01, V[:, 1], m12),
                                                      (m20, m12, V[:, 2]), (m01, m12, m20))])


def current_deposit(sim, I, x, y, k, per_cell=2.0):
    """The surface current of interface k as a density on the uniform grid x, y (m), for the near field's FFT:
    each triangle's current integrated by a 6-point rule (triangles larger than `per_cell` cells split into four
    until they are not), each point's current moment spread onto its four grid points (cloud in cell). The total
    current and its centre are kept on any grid; sampling the current at the grid points instead was fragile (it
    crowds at a trace's edges: a row of points on one edge and none on the other put rf_gcpw's 1 mm near field
    0.15 mm off the trace's centre, and at another step halved it). Returns Jx, Jy ((nx, ny) complex, A/m)."""
    r = sim.rwg
    a, b = rwg_currents(r, np.asarray(I)[:r.Ne])
    idx = sim.tri_idx[k]
    V, aa, bb = r.nodes[r.tris[idx]][:, :, :2], a[idx], b[idx]
    dx, dy = x[1] - x[0], y[1] - y[0]
    for _ in range(6):
        size = np.sqrt(np.abs(tri_area2(V)) / 2)
        big = size > per_cell * min(dx, dy)
        if not big.any():
            break
        V = np.concatenate([V[~big], _split4(V[big])])
        aa = np.concatenate([aa[~big], np.tile(aa[big], 4)])
        bb = np.concatenate([bb[~big], np.tile(bb[big], (4, 1))])
    area = 0.5 * np.abs(tri_area2(V))
    Q = np.einsum('qk,tkc->tqc', _BAR6, V)                              # (M, 6, 2)
    mom = (aa[:, None, None] * Q + bb[:, None, :]) * (area[:, None] * _W6[None])[..., None]   # A m per point
    fx, fy = (Q[..., 0].ravel() - x[0]) / dx, (Q[..., 1].ravel() - y[0]) / dy
    i0, j0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - i0, fy - j0
    nx, ny = len(x), len(y)
    out = [np.zeros((nx, ny), complex), np.zeros((nx, ny), complex)]
    for di, dj, wgt in ((0, 0, (1 - tx) * (1 - ty)), (1, 0, tx * (1 - ty)), (0, 1, (1 - tx) * ty), (1, 1, tx * ty)):
        ii, jj = i0 + di, j0 + dj
        ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        for c in range(2):
            np.add.at(out[c], (ii[ok], jj[ok]), (mom[..., c].ravel() * wgt)[ok])
    return out[0] / (dx * dy), out[1] / (dx * dy)

