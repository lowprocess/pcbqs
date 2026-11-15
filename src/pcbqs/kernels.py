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


def blas_threads():
    """BLAS on the fast physical cores only (emsdev's core_threads), as mom25d's LU: on a hybrid CPU OpenBLAS's
    24 threads made an LU of 2,532 unknowns take 0.11-0.88 s, from one call to the next, for 0.041 s on the 8
    P-cores. A context manager; MOM25D_BLAS_THREADS / OPENBLAS_NUM_THREADS set by hand are left as they are."""
    import contextlib
    if os.environ.get('OPENBLAS_NUM_THREADS') or os.environ.get('PCBQS_BLAS_THREADS') == '0':
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
        from emsdev.cpu import core_threads
        n = int(os.environ.get('PCBQS_BLAS_THREADS') or core_threads())
        return threadpool_limits(max(1, n), user_api='blas')
    except Exception:                                           # noqa: BLE001  (no threadpoolctl or emsdev: as is)
        return contextlib.nullcontext()


MIXED_MIN = 4000                # the smallest system factored in single precision (2,532 unknowns: no gain)


class MixedLU:
    """The LU of a complex row-major matrix in single precision, refined to double's accuracy (R-D13).

    LAPACK's cgetrf is about 2.3 times as fast as zgetrf (6,233 unknowns: 0.57 s for 1.30). Each solve takes the
    single-precision solution and corrects it by its residual, worked out with the double matrix (kept, so the
    factor is 1.5 times a double factor's memory), until the residual is a double solve's (1e-12 of b; three or
    four steps). A system it cannot bring there (single precision's error times the condition number near 1) is
    factored in double, once, and solved as before."""

    TOL, STEPS = 1e-12, 12

    def __init__(self, a):
        from scipy.linalg import lu_factor
        self.a = a                                              # (row-major, double: the residuals)
        with blas_threads():
            self.lu = lu_factor(np.array(a.T, dtype=np.complex64), overwrite_a=True, check_finite=False)
        self.double = None

    @property
    def nbytes(self):
        if self.double is not None:
            return self.double[0].nbytes + self.double[1].nbytes
        return self.a.nbytes + self.lu[0].nbytes + self.lu[1].nbytes

    def _double(self):
        if self.double is None:                                 # the single factor first freed, then the matrix
            from scipy.linalg import lu_factor                  # factored in its own memory (no third copy)
            self.lu = None
            self.double = lu_factor(self.a.T, overwrite_a=True, check_finite=False)
            self.a = None
        return self.double

    def solve(self, b):
        with blas_threads():
            return self._solve(b)

    def _solve(self, b):
        from scipy.linalg import lu_solve
        b = np.asarray(b)
        if self.double is not None:
            return lu_solve(self.double, b, trans=1, check_finite=False)
        one = lambda r: lu_solve(self.lu, r.astype(np.complex64), trans=1, check_finite=False).astype(complex)
        x = one(b)
        nb = max(float(np.linalg.norm(b)), 1e-300)
        last = np.inf
        for _ in range(self.STEPS):
            r = b - self.a @ x
            res = float(np.linalg.norm(r)) / nb
            if res <= self.TOL:
                return x
            if res > 0.5 * last:                                # no longer converging: double from here on
                break
            last = res
            x = x + one(r)
        if last <= 1e-10:                                       # (stalled at a double solve's own level)
            return x
        return lu_solve(self._double(), b, trans=1, check_finite=False)


def _mixed_fits(a):
    """Whether the single-precision copy fits beside the double matrix in the memory free now, with room to
    spare (twice the copy: the solver's other arrays come and go meanwhile)."""
    try:
        from emsdev.memory import available
        free = available()
    except Exception:                                           # noqa: BLE001  (no probe: assume it does)
        return True
    return free is None or free > 2 * (a.size * 8) + 2e9


def _rmul(G, v):
    """G @ v for a real G and a complex v, without making G complex (numpy would copy it as complex)."""
    v = np.asarray(v)
    if not np.iscomplexobj(v):
        return G @ v
    w = G @ np.concatenate([v.real, v.imag], axis=1)
    k = v.shape[1]
    return w[:, :k] + 1j * w[:, k:]


class Pencil:
    """(R + s L + P^T diag(z) P)^-1 for any s from one generalized eigendecomposition (R-D13.2): R and L real
    symmetric, R positive definite (the loops' DC resistance), P sparse (k, n) (the loops through k parts of
    impedance z(s)). With R = C C^T and C^-1 L C^-T = W diag(lam) W^T, G = C^-T W puts both on the diagonal:
    (R + s L)^-1 = G diag(1 / (1 + s lam)) G^T, and the parts come in by Woodbury, a k x k solve. The
    eigendecomposition costs about PENCIL_LUS(n) complex LUs, after which a frequency costs three products with
    G: a sweep of more frequencies than that is quicker (pencil_pays). Raises numpy.linalg.LinAlgError where R
    is not positive definite."""

    def __init__(self, R, L, P=None):
        from scipy.linalg import cholesky, eigh, solve_triangular
        with blas_threads():
            c = cholesky(R, lower=True, check_finite=False)
            X = solve_triangular(c, L, lower=True, check_finite=False)
            X = solve_triangular(c, np.ascontiguousarray(X.T), lower=True, check_finite=False, overwrite_b=True)
            X = (X + X.T) / 2
            self.lam, W = eigh(X, overwrite_a=True, check_finite=False, driver='evd')
            del X
            self.G = solve_triangular(c, W, lower=True, trans='T', check_finite=False, overwrite_b=True)
        self.P = None if P is None or P.shape[0] == 0 else P.tocsr()
        self.PG = None if self.P is None else np.asarray(self.P @ self.G)

    def solve(self, s, z, b):
        """x of (R + s L + P^T diag(z) P) x = b (b: (n,) or (n, m))."""
        b = np.asarray(b, complex)
        one = b.ndim == 1
        b = b.reshape(len(b), -1)
        d = 1 / (1 + complex(s) * self.lam)
        with blas_threads():
            u = d[:, None] * _rmul(self.G.T, b)
            x = _rmul(self.G, u)
            if self.P is not None and np.any(z):
                z = np.asarray(z, complex)
                S = (self.PG * d) @ self.PG.T                       # P Z0^-1 P^T
                t = np.linalg.solve(np.eye(len(z)) + z[:, None] * S, z[:, None] * (self.PG @ u))
                x -= _rmul(self.G, d[:, None] * (self.PG.T @ t))
        return x[:, 0] if one else x


def pencil_lus(n):
    """About how many complex LUs of n unknowns the Pencil's eigendecomposition costs (measured on a Ryzen/
    Core desktop: 7 at 2,000, 10.5 at 4,000, 13.6 at 6,000), and a frequency's three products."""
    return 4.0 + 1.6 * n / 1000


def pencil_pays(n, frequencies):
    """Whether a sweep of `frequencies` on n loops is quicker by one Pencil than by an LU each, and fits in the
    memory free (four real matrices of n x n at once). PCBQS_PENCIL=always or never decides it."""
    choice = os.environ.get('PCBQS_PENCIL', '').lower()
    if choice in ('always', 'never'):
        return choice == 'always'
    if n < 200 or frequencies < 1.25 * pencil_lus(n) + 1:
        return False
    try:
        from emsdev.memory import available
        free = available()
    except Exception:                                           # noqa: BLE001  (no probe: assume it does)
        return True
    return free is None or free > 4 * n * n * 8 + 2e9


def lu_factor_rows(a, mixed='auto'):
    """LAPACK's LU of the square row-major matrix `a`: what lu_solve_rows takes. A complex one of MIXED_MIN
    unknowns or more is factored in single precision and refined (MixedLU: a is kept, not destroyed) where the
    copy fits in memory; otherwise in double, in a's own memory (a is destroyed): scipy's lu_factor copies a
    row-major matrix to column-major first, a second matrix of the same size, so this factors a's transpose,
    which is a's memory read by columns, and lu_solve_rows solves with the transpose of that, so the answers are
    a's. PCBQS_LU=double: always double."""
    from scipy.linalg import lu_factor
    a = np.asarray(a)
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    if mixed == 'auto':
        mixed = (os.environ.get('PCBQS_LU', '').lower() != 'double' and np.iscomplexobj(a)
                 and a.shape[0] >= MIXED_MIN and _mixed_fits(a))
    if mixed:
        return MixedLU(a)
    with blas_threads():
        return lu_factor(a.T, overwrite_a=True, check_finite=False)


def lu_solve_rows(lu, b):
    """The solution of a x = b for lu = lu_factor_rows(a)."""
    if isinstance(lu, MixedLU):
        return lu.solve(b)
    from scipy.linalg import lu_solve
    with blas_threads():
        return lu_solve(lu, b, trans=1, check_finite=False)


def factor_bytes(lu):
    """The memory a factor of lu_factor_rows holds."""
    return lu.nbytes if isinstance(lu, MixedLU) else lu[0].nbytes + lu[1].nbytes


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
