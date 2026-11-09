"""1/R moments between triangles of any orientation (side-wall panels as well as the horizontal sheets).

    m0, mro, mrs, mrr = pair_moments(Vo, Vs, outer_levels=1)

For each pair of an observation triangle Vo[k] and a source triangle Vs[k] ((m, 3, 3), metres):

    m0  = int_o int_s 1/R            mro = int_o r int_s 1/R      (3-vectors)
    mrr = int_o int_s r.r'/R         mrs = int_o int_s r'/R

The source integral is analytic: in the source triangle's own frame (two in-plane axes and its normal) an
observation point is (u, v) over the plane at height w, which is the horizontal kernel's (x, y) at offset d
(_numerics.sing.offplane_pairs), so the same edge formulas hold; the in-plane moment comes back to 3D through the
frame. The observation integral is the 6-point rule, on 4**outer_levels sub-triangles. For two horizontal triangles
this is offplane_pairs exactly (tests/test_panels3d.py).

From the moments an RWG pair's  int int f_m . f_n / R  (f = alpha (r - v) on each triangle) is
alpha alpha' (mrr - v'.mro - v.mrs + v.v' m0)  (rwg_pair).
"""
import numpy as np

from ._numerics.base import _BAR6, _W6
from ._numerics.currents import _split4


def _rule(outer_levels):
    bar, w = _BAR6, _W6
    if outer_levels:
        ref = np.eye(3)[None]
        for _ in range(outer_levels):
            ref = _split4(ref)
        bar = np.einsum('qk,tkc->tqc', _BAR6, ref).reshape(-1, 3)
        w = np.tile(_W6, len(ref)) / len(ref)
    return bar, w


def frames(V):
    """Per triangle (V (m, 3, 3)): origin (vertex 0), in-plane axes e1, e2, unit normal n, area. e1 runs along
    edge 0 and n = e1 x e2 follows the vertex order, so the triangle is counter-clockwise in (e1, e2)."""
    a, b = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]
    c = np.cross(a, b)
    area2 = np.linalg.norm(c, axis=1)
    if (area2 <= 0).any():
        raise ValueError('degenerate triangle')
    n = c / area2[:, None]
    e1 = a / np.linalg.norm(a, axis=1)[:, None]
    e2 = np.cross(n, e1)
    return V[:, 0], e1, e2, n, 0.5 * area2


def pair_moments(Vo, Vs, outer_levels=1, scalar_only=False):
    """(m0, mro, mrs, mrr) of each pair (Vo[k], Vs[k]): m0, mrr (m,), mro, mrs (m, 3). See the module."""
    Vo, Vs = np.asarray(Vo, float), np.asarray(Vs, float)
    bar, w = _rule(outer_levels)
    o, e1, e2, n, _ = frames(Vs)
    _, _, _, _, Ao = frames(Vo)
    P = np.einsum('qk,mkc->mqc', bar, Vo)                     # observation points (m, q, 3)
    # the source in its frame: vertices (m, 3, 2), its edges' vectors, lengths, tangents, outward normals
    rel = Vs - o[:, None]
    S = np.stack([(rel * e1[:, None]).sum(-1), (rel * e2[:, None]).sum(-1)], -1)
    e = np.roll(S, -1, axis=1) - S
    L = np.linalg.norm(e, axis=2)
    tau = e / L[..., None]
    un = np.stack([tau[..., 1], -tau[..., 0]], -1)
    pr = P - o[:, None]
    r2 = np.stack([(pr * e1[:, None]).sum(-1), (pr * e2[:, None]).sum(-1)], -1)   # (m, q, 2)
    dz = (pr * n[:, None]).sum(-1)[..., None]                                    # (m, q, 1)
    ad = np.abs(dz)
    dv = S[:, None] - r2[:, :, None]                                             # (m, q, 3, 2)
    sm = (dv * tau[:, None]).sum(-1)
    sp = sm + L[:, None]
    t0 = (dv * un[:, None]).sum(-1)
    R02 = t0 ** 2 + dz * dz
    R0 = np.sqrt(R02)
    Rm, Rp = np.sqrt(R02 + sm ** 2), np.sqrt(R02 + sp ** 2)
    ok = R0 > 1e-14 * L[:, None]
    R0s = np.where(ok, R0, 1.0)
    lg = np.where(ok, np.arcsinh(sp / R0s) - np.arcsinh(sm / R0s), 0.0)
    beta = np.arctan2(t0 * sp, R02 + ad * Rp) - np.arctan2(t0 * sm, R02 + ad * Rm)
    I1 = (t0 * lg - ad * beta).sum(-1)                                           # int_s 1/R (m, q)
    wq = w[None] * Ao[:, None]
    m0 = (wq * I1).sum(1)
    if scalar_only:
        return m0
    eR = 0.5 * (R02 * lg + sp * Rp - sm * Rm)
    IR = (eR[..., None] * un[:, None]).sum(-2)                                   # int (rho' - rho)/R, in-plane
    W2 = IR + r2 * I1[..., None]                                                 # int (u', v')/R
    W = o[:, None] * I1[..., None] + W2[..., :1] * e1[:, None] + W2[..., 1:] * e2[:, None]   # int r'/R (m, q, 3)
    mro = (wq[..., None] * P * I1[..., None]).sum(1)
    mrs = (wq[..., None] * W).sum(1)
    mrr = (wq * (P * W).sum(-1)).sum(1)
    return m0, mro, mrs, mrr


def rwg_pair(moments, ao, vo, as_, vs):
    """int int f_o . f_s / R for basis halves f_o = ao (r - vo) on the observation triangle and f_s = as_ (r' - vs) on
    the source triangle, from pair_moments' (m0, mro, mrs, mrr); ao, as_ (m,), vo, vs (m, 3)."""
    m0, mro, mrs, mrr = moments
    return ao * as_ * (mrr - (vs * mro).sum(-1) - (vo * mrs).sum(-1) + (vo * vs).sum(-1) * m0)
