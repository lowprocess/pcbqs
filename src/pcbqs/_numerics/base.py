"""Private numerical snapshot; independently maintained. See PROVENANCE.json."""
import numpy as np
from scipy import sparse

MU0 = 4e-7 * np.pi
C0 = 299792458.0
EPS0 = 1.0 / (MU0 * C0 ** 2)


class RWG:
    """RWG basis functions on interior edges. f = alpha*(r - v) on each adjacent triangle."""
    def __init__(self, nodes, tris):
        self.nodes, self.tris = nodes, tris
        V = nodes[tris]
        e1, e2 = V[:, 1] - V[:, 0], V[:, 2] - V[:, 0]
        self.area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        assert (self.area > 0).all(), "triangles must be counter-clockwise"
        self.cent = V.mean(1)
        d = {}
        for t, (a, b, c) in enumerate(tris):
            for i, (u, v) in enumerate(((a, b), (b, c), (c, a))):
                d.setdefault((min(u, v), max(u, v)), []).append((t, i))
        E = [(k, v) for k, v in d.items() if len(v) == 2]
        self.Ne = Ne = len(E)
        u, v = np.array([k for k, _ in E], int).reshape(-1, 2).T                       # the edge's end nodes
        (t1, i1), (t2, i2) = np.array([s for _, s in E], int).reshape(-1, 2, 2).transpose(1, 2, 0)   # T+, T-
        self.tp, self.tm = t1, t2
        self.vp, self.vm = nodes[tris[t1, (i1 + 2) % 3]], nodes[tris[t2, (i2 + 2) % 3]]   # the free vertices
        pu, pv = nodes[u], nodes[v]
        self.l = np.linalg.norm(pv - pu, axis=1)
        self.mid, self.ends = 0.5 * (pu + pv), np.stack([pu, pv], 1)
        tau = (pv - pu) / self.l[:, None]
        n = np.c_[tau[:, 1], -tau[:, 0]]
        n[((self.cent[t2] - self.cent[t1]) * n).sum(1) < 0] *= -1
        self.nrm = n                         # direction of positive current (T+ -> T-)
        alpha = np.concatenate([self.l / (2 * self.area[t1]), -self.l / (2 * self.area[t2])])
        vv = np.vstack([self.vp, self.vm])
        sp = lambda vals: sparse.csr_matrix((vals, (np.r_[t1, t2], np.r_[np.arange(Ne), np.arange(Ne)])),
                                            shape=(len(tris), Ne))
        self.Aal, self.Bx, self.By = sp(alpha), sp(alpha * vv[:, 0]), sp(alpha * vv[:, 1])
        self.D = sp(2 * alpha)               # divergence: +l/A on T+, -l/A on T-


def tri_area2(V):
    """Twice the signed area of planar triangles V (..., 3, 2+): the z of (V1-V0) x (V2-V0). np.cross no
    longer takes 2-vectors (NumPy 2.5), so it is written out."""
    e1, e2 = V[..., 1, :] - V[..., 0, :], V[..., 2, :] - V[..., 0, :]
    return e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0]


def qform(S1, M, S2):
    """S1^T M S2 for sparse S1,S2 (Nt x Ne) and dense M (Nt x Nt)."""
    return (S2.T @ (S1.T @ M).T).T


def combine(rwg, Mrr, Mrx, Mry, M0):
    """Assemble the Ne x Ne matrix of  int int f_m . f_n K  from triangle-pair moments."""
    A, Bx, By = rwg.Aal, rwg.Bx, rwg.By
    return (qform(A, Mrr, A) - qform(A, Mrx, Bx) - qform(A, Mry, By) - qform(Bx, Mrx.T, A) - qform(By, Mry.T, A)
            + qform(Bx, M0, Bx) + qform(By, M0, By))


# ---- Integrals --------------------------------------------------------------
# 6-point degree-4 rule (weights sum to 1) for the analytic-inner / numerical-outer integrals
_a1, _a2 = 0.445948490915965, 0.091576213509771
_BAR6 = np.array([[1 - 2 * _a1, _a1, _a1], [_a1, 1 - 2 * _a1, _a1], [_a1, _a1, 1 - 2 * _a1],
                  [1 - 2 * _a2, _a2, _a2], [_a2, 1 - 2 * _a2, _a2], [_a2, _a2, 1 - 2 * _a2]])
_W6 = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)
# 3-point degree-2 rule for the smooth remainder
_BAR3 = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])
_W3 = np.ones(3) / 3


def graded(u, grade):
    """u in 0..1 graded toward both ends (cosine) by `grade` (0: uniform): the cells across a strip."""
    return (1 - grade) * u + grade * (1 - np.cos(np.pi * u)) / 2


def quad_pts(rwg, bar):
    return np.einsum('qk,tkc->tqc', bar, rwg.nodes[rwg.tris])

def edge_frames(V):
    """Per triangle (vertices V, (Nt, 3, 2), counter-clockwise) and edge: vector, length, unit tangent, outward
    normal."""
    e = np.roll(V, -1, axis=1) - V
    L = np.linalg.norm(e, axis=2)
    tau = e / L[..., None]
    return e, L, tau, np.stack([tau[..., 1], -tau[..., 0]], -1)
