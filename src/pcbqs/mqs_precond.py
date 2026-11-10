"""Preconditioners of the compressed MQS loop system C_f^T (R + jw L) C_f (mqs_iterative.py).

graph: the loop system inverted exactly for a block-diagonal branch impedance (each branch's own, or a face pair's
2x2 block), one sparse graph-Laplacian solve per frequency. Cheap to set up; the default.

FaceLoops: for face sheets (thickness_sheets=2), where the graph preconditioner's iterations grow with frequency
(wpthb, 24 ports: 966 at DC, 3,540 at 900 kHz). The two copies of the copper carry a common current, which sees
the board's full inductance, and a difference current circulating between the faces 35 um apart, whose field is
local. So the loops split into
  - the common-mode loops, as many as one sheet has: solved exactly, a dense matrix from compressed products;
  - the difference loops, one per edge pair: solved with the inductance among edges that share a triangle only,
    a sparse matrix,
applied one after the other (the difference solve on the residual the common solve leaves). 217 to 618 iterations
on that case. The setup is one compressed product per common-mode loop, so it pays for many right-hand sides
(the circuit tool's ports x frequencies), not for a single solve.
"""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import coo_matrix, csr_matrix, diags, identity
from scipy.sparse.linalg import LinearOperator, splu


class Unusable(Exception):
    """This preconditioner does not apply to the model (the caller falls back to graph)."""


def graph(p, d, zpair):
    """(C_f^T D C_f)^-1 applied exactly for a block-diagonal D: each branch's own impedance d, and for face-sheet
    pairs the 2x2 block with their coupling zpair. The loop currents under a force y on the loops' cotree branches
    are i = D^-1 (f - A^T v) with A i = 0, i.e. (A D^-1 A^T) v = A D^-1 f on the passive branches (one node grounded
    per connected piece of copper); both copies of an edge join the same nodes, so A D^-1 A^T stays sparse."""
    A, passive, keep, cotree, pairs = p['A'], p['passive'], p['keep'], p['cotree'], p['pairs']
    nb = len(d)
    inv = np.zeros(nb, complex)
    inv[passive] = 1 / d[passive]
    rows, cols, vals = [np.arange(nb)], [np.arange(nb)], [inv]
    if pairs.shape[1]:
        e, g = pairs
        det = d[e] * d[g] - zpair ** 2
        inv[e], inv[g] = d[g] / det, d[e] / det
        rows += [e, g]
        cols += [g, e]
        vals += [-zpair / det, -zpair / det]
    Dinv = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nb, nb)).tocsr()
    Dinv = diags(np.where(np.isin(np.arange(nb), passive), 1.0, 0.0)) @ Dinv
    lap = (A @ Dinv @ A.T).tocsc()[keep][:, keep]
    lu = splu(lap.tocsc())

    def apply(y):                                        # y: (loops,) or (loops, k) columns
        force = np.zeros((nb,) + np.shape(y)[1:], complex)
        force[cotree] = y
        trial = Dinv @ force
        v = np.zeros((A.shape[0],) + np.shape(y)[1:], complex)
        v[keep] = lu.solve(np.ascontiguousarray(np.asarray(A @ trial)[keep]))
        return (trial - Dinv @ np.asarray(A.T @ v))[cotree]
    size = sum(s.data.nbytes + s.indices.nbytes + s.indptr.nbytes for s in (lu.L, lu.U))
    M = LinearOperator((len(cotree), len(cotree)), matvec=apply, matmat=apply, dtype=complex)
    M.block = apply
    return M, size


def _near_inductance(model, Le, partner, nb):
    """The exact partial inductances among the edges of each triangle and their face partners, sparse (nb, nb)."""
    r, n = model.rwg, model.n_rwg
    order = np.argsort(np.r_[r.tp, r.tm], kind='stable')
    tri = np.r_[r.tp, r.tm][order]
    edge = np.r_[np.arange(n), np.arange(n)][order]
    rows, cols, vals, seen = [], [], [], set()
    for ids in np.split(edge, np.flatnonzero(np.diff(tri)) + 1):
        group = np.unique(np.r_[ids, partner[ids][partner[ids] >= 0]])
        key = group.tobytes()
        if key in seen:                                  # the other face's triangle: the same edges
            continue
        seen.add(key)
        block = np.asarray(Le.entry(group, group))
        rows.append(np.repeat(group, len(group)))
        cols.append(np.tile(group, len(group)))
        vals.append(block.ravel())
    rows, cols, vals = map(np.concatenate, (rows, cols, vals))
    _, first = np.unique(rows * nb + cols, return_index=True)        # an edge pair in two groups: once, not summed
    near = coo_matrix((vals[first], (rows[first], cols[first])), shape=(nb, nb)).tocsr()
    if getattr(model, 'thick_kernel', False):
        D = model._thick_correction().tocoo()
        near = near + coo_matrix((D.data, (D.row, D.col)), shape=(nb, nb)).tocsr()
    return near


class FaceLoops:
    """The common-mode / difference-loop preconditioner for face sheets (the module docstring). Built once per
    model: products(left, right) gives left^T L right densely from compressed products. at(R, w) is the
    preconditioner at one frequency, R the branch resistance there (sparse)."""

    def __init__(self, model, p, products, max_coarse):
        m = model._mats
        Le = m['L'][0]
        e, g = p['pairs']
        if not len(e):
            raise Unusable('no face sheets')
        if getattr(Le, 'entry', None) is None:
            raise Unusable('the compressed inductance keeps no exact entries (a reloaded operator)')
        nb, Cf, cotree = m['nb'], p['Cf'], p['cotree']
        partner = np.full(nb, -1)
        partner[e], partner[g] = g, e
        swap = np.arange(nb)
        swap[e], swap[g] = g, e
        mean = 0.5 * (identity(nb, format='csr') + csr_matrix((np.ones(nb), (np.arange(nb), swap)), shape=(nb, nb)))
        B = (mean @ Cf).tocsc()                          # each free loop with its current shared by the two faces
        lower = np.ones(nb, bool)
        lower[g] = False                                 # one of each pair's two loops: they symmetrise to one
        norm = np.sqrt(np.asarray(B.multiply(B).sum(0)).ravel())
        B = B[:, np.flatnonzero(lower[cotree] & (norm > 1e-9))]
        W = B[cotree].tocsr()                            # the same currents as combinations of the free loops
        if B.shape[1] > max_coarse:
            raise Unusable(f'{B.shape[1]:,} common-mode loops (over {max_coarse:,})')
        if B.shape[1] == 0 or abs(B - Cf @ W).max() > 1e-9:
            raise Unusable('the common-mode currents are not combinations of the free loops')
        k = np.arange(len(e))
        V = coo_matrix((np.r_[np.ones(len(e)), -np.ones(len(e))], (np.r_[g, e], np.r_[k, k])),
                       shape=(nb, len(e))).tocsc()        # a pair's difference loop: up one face, back the other
        self.Cf, self.B, self.W, self.V, self.Wd = Cf, B, W, V, V[cotree].tocsr()
        self.CfL = products(Cf, B)                       # (free loops, common loops), real
        Lc = np.asarray(W.T @ self.CfL)
        self.Lc = 0.5 * (Lc + Lc.T)
        self.Dl = (V.T @ _near_inductance(model, Le, partner, nb) @ V).tocsc()
        self.size = B.shape[1]

    def at(self, R, w):
        RB = R @ self.B
        lu = lu_factor((self.B.T @ RB).toarray() + 1j * w * self.Lc, check_finite=False)
        pivots = abs(np.diagonal(lu[0]))
        if not np.isfinite(pivots).all() or pivots.min() < 1e-13 * pivots.max():
            raise Unusable('the common-mode system is singular')
        CfRB = (self.Cf.T @ RB).tocsr()
        dlu = splu((self.V.T @ R @ self.V + 1j * w * self.Dl).astype(complex).tocsc())
        W, Wd, CfL = self.W, self.Wd, self.CfL

        def apply(r):
            y = lu_solve(lu, np.asarray(W.T @ r), check_finite=False)
            left = r - CfRB @ y                          # what the common-mode solve leaves
            if w:                                        # real matrix, complex y: two real products, no copy
                left -= 1j * w * (CfL @ y.real) - w * (CfL @ y.imag)
            return W @ y + Wd @ dlu.solve(np.ascontiguousarray(np.asarray(Wd.T @ left)))
        M = LinearOperator((W.shape[0],) * 2, matvec=apply, matmat=apply, dtype=complex)
        M.block = apply
        return M
