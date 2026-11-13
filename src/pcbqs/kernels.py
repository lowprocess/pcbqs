"""
Kernels with a compiled twin: the dense systems of the frequency loops, filled in one pass, and points located in
a triangle mesh. Each is the compiled core's (pcbqs._core, core/dense_kernels.hpp) where it is built, and numpy's
here otherwise (no core, PCBQS_NO_CORE set, or a core that predates the kernel); `numpy=True` asks for the twin,
which is how the tests hold the two to each other.

    loop_system(R, L, jw, rows, cols, S)        R[rows, cols] + jw L[rows, cols] + S, complex (rows, cols)
    peec_system(R, L, jw, pscale, nl, electro, S)   the charge solve's scaled system and its scale
    tri_locate(nodes, tris, points)             each point's triangle (-1: none) and its weights on the corners
    lu_factor_rows(a), lu_solve_rows(lu, b)     LAPACK's LU of a row-major matrix in place (no copy of it)
    solve_symmetric(P, B)                       P^-1 B for a symmetric P: by Cholesky where it is real and definite

The numpy twins make a temporary of the whole matrix per operation, which is what the compiled ones are for: on
18,000 unknowns that was four matrices of 5 GB and several seconds a frequency.
"""
import os
from importlib import import_module

import numpy as np


def core(name):
    """The compiled core's function of that name, or None (no core built, PCBQS_NO_CORE set, or an older core)."""
    if os.environ.get('PCBQS_NO_CORE'):
        return None
    try:
        return getattr(import_module('pcbqs._core'), name, None)
    except ImportError:
        return None


def _workers():
    from .static_backends import resolve_workers
    return resolve_workers()


def _csr(S, shape):
    """A sparse term's CSR arrays as the core takes them (None, or no entries: three empty ones)."""
    if S is None or S.nnz == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, complex)
    S = S.tocsr()
    if S.shape != tuple(shape):
        raise ValueError(f"sparse term of shape {S.shape}, not {tuple(shape)}")
    S.sum_duplicates()
    return S.indptr.astype(np.int64), S.indices.astype(np.int64), S.data.astype(complex)


def loop_system(R, L, jw, rows, cols, S=None, numpy=False):
    """R[rows, cols] + jw L[rows, cols] + S: the loop impedance on the unknowns kept, complex (len(rows), len(cols)),
    row-major. R and L are the real loop matrices; S (sparse, of the result's shape, or None) is what changes with
    frequency: the parts' own impedances and the conductors' resistance."""
    rows, cols = np.asarray(rows, np.int64), np.asarray(cols, np.int64)
    fn = None if numpy or np.iscomplexobj(R) or np.iscomplexobj(L) else core('loop_system')
    if fn is not None:
        return fn(R, L, complex(jw), rows, cols, *_csr(S, (len(rows), len(cols))), _workers())
    ix = np.ix_(rows, cols)
    out = R[ix] + complex(jw) * L[ix]
    if S is not None and S.nnz:
        out += S.toarray()
    return out


def peec_system(R, L, jw, pscale, nl, electro, S=None, numpy=False):
    """The charge solve's system at one frequency and its scale: (system, zscale). Z = R + jw L + S (S sparse:
    the parts and the resistance's change), zscale = max |Z|; the first nl columns (the loops' currents) are
    Z / zscale, the others (the tree's, as charge) jw Z / pscale, with `electro` added to the tree's own block."""
    fn = None if numpy or np.iscomplexobj(R) or np.iscomplexobj(L) else core('peec_system')
    n = R.shape[0]
    if fn is not None:
        return fn(R, L, complex(jw), float(pscale), int(nl), electro, *_csr(S, (n, n)), _workers())
    Z = R + complex(jw) * L
    if S is not None and S.nnz:
        Z += S.toarray()
    zscale = max(np.max(np.abs(Z)), np.finfo(float).tiny) if Z.size else np.finfo(float).tiny
    system = Z / zscale
    system[:, nl:] = complex(jw) * Z[:, nl:] / pscale
    system[nl:, nl:] += electro
    return system, float(zscale)


def tri_locate(nodes, tris, points, tol=1e-9, numpy=False):
    """Which triangle of the mesh (nodes (N, 2), tris (M, 3)) holds each of `points` (K, 2): (index (K,), -1
    where none does; weights (K, 3) on the triangle's corners, NaN where none). A value at the nodes is then
    (values[tris[index]] * weights).sum(1) at the points: the mesh's own linear interpolation."""
    nodes, tris = np.ascontiguousarray(nodes, float), np.ascontiguousarray(tris, np.int64)
    points = np.ascontiguousarray(points, float).reshape(-1, 2)
    fn = None if numpy else core('tri_locate')
    if fn is not None:
        return fn(nodes, tris, points, float(tol), _workers())
    from matplotlib.tri import Triangulation
    index = np.full(len(points), -1, np.int64)
    weight = np.full((len(points), 3), np.nan)
    if len(tris) and len(points):
        find = Triangulation(nodes[:, 0], nodes[:, 1], tris).get_trifinder()
        index = np.asarray(find(points[:, 0], points[:, 1]), np.int64)
        ok = index >= 0
        a, b, c = (nodes[tris[index[ok], k]] for k in range(3))
        e1, e2, d = b - a, c - a, points[ok] - a
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        w1 = (d[:, 0] * e2[:, 1] - d[:, 1] * e2[:, 0]) / det
        w2 = (e1[:, 0] * d[:, 1] - e1[:, 1] * d[:, 0]) / det
        weight[ok] = np.c_[1 - w1 - w2, w1, w2]
    return index, weight


def lu_factor_rows(a):
    """LAPACK's LU of the square row-major matrix `a`, in a's own memory (a is destroyed): what lu_solve_rows
    takes. scipy's lu_factor copies a row-major matrix to column-major first, a second matrix of the same size;
    this factors a's transpose, which is a's memory read by columns, and lu_solve_rows solves with the transpose
    of that, so the answers are a's."""
    from scipy.linalg import lu_factor
    a = np.asarray(a)
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    return lu_factor(a.T, overwrite_a=True, check_finite=False)


def lu_solve_rows(lu, b):
    """The solution of a x = b for lu = lu_factor_rows(a)."""
    from scipy.linalg import lu_solve
    return lu_solve(lu, b, trans=1, check_finite=False)


def solve_symmetric(P, B):
    """P^-1 B for the symmetric matrix P (coefficients of potential): by Cholesky where P is real, or complex
    with no imaginary part (a layered stack without loss: its images make the matrix complex), which such a
    matrix is positive definite for; by LAPACK's symmetric indefinite solve where it is complex (a lossy stack) or
    Cholesky finds it is not definite. That solve is what this always was: 1.7 s on the 6,359 unknowns of two
    traces on FR4, for Cholesky's 0.6 s."""
    from scipy.linalg import solve, LinAlgError
    real = P if not np.iscomplexobj(P) else (P.real if not P.imag.any() else None)
    if real is not None:
        try:
            return solve(real, B, assume_a='pos')
        except LinAlgError:                             # not definite to rounding (a mesh with a sliver)
            pass
    return solve(P, B, assume_a='sym')
