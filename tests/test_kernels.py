"""The kernels with a compiled twin (pcbqs/kernels.py): the compiled core's against numpy's, the LU of a row-major
matrix in its own memory, and a mesh's own interpolation where the thermal maps use it."""
import numpy as np
import pytest
import scipy.sparse as sp

from pcbqs import kernels as K

core = pytest.mark.skipif(K.core('loop_system') is None, reason='no compiled core with the dense kernels')


def _matrices(n, seed=0):
    rng = np.random.default_rng(seed)
    R = rng.normal(size=(n, n))
    L = rng.normal(size=(n, n))
    S = (sp.random(n, n, 0.02, random_state=seed + 1) * (1 + 2j)).tocsr()
    return rng, R + R.T + 4 * n * np.eye(n), L + L.T, S


@core
@pytest.mark.parametrize('n', [1, 7, 400])
def test_loop_system_is_numpy_s(n):
    rng, R, L, S = _matrices(n)
    rows = np.delete(np.arange(n), [0] if n > 1 else [])
    cols = rng.permutation(n)[:max(1, n // 2)]
    jw = 2j * np.pi * 3e5
    for term in (None, S[rows][:, cols], sp.csr_matrix((len(rows), len(cols)), dtype=complex)):
        a = K.loop_system(R, L, jw, rows, cols, term)
        b = K.loop_system(R, L, jw, rows, cols, term, numpy=True)
        assert a.shape == b.shape == (len(rows), len(cols)) and a.flags.c_contiguous
        assert np.allclose(a, b, rtol=1e-14, atol=0)
    with pytest.raises(ValueError):
        K.loop_system(R, L, jw, [n], cols)                         # a row that is not the matrix's
    with pytest.raises(ValueError):
        K.loop_system(R, L, jw, rows, cols, sp.csr_matrix(np.ones((len(rows) + 1, len(cols)))))   # another shape


@core
@pytest.mark.parametrize('lossy', [False, True])
def test_peec_system_is_numpy_s(lossy):
    n, nl = 300, 110
    rng, R, L, S = _matrices(n, 3)
    electro = rng.normal(size=(n - nl, n - nl)) * ((1 + 0.2j) if lossy else 1)
    for jw in (2j * np.pi * 1e3, 2j * np.pi * 1e9):
        for term in (S, None):
            a, za = K.peec_system(R, L, jw, 7.5, nl, electro, term)
            b, zb = K.peec_system(R, L, jw, 7.5, nl, electro, term, numpy=True)
            assert za == pytest.approx(zb, rel=1e-15) and np.allclose(a, b, rtol=1e-13, atol=1e-300)
    a, _ = K.peec_system(R, L, 1j, 1.0, n, np.zeros((0, 0)))        # no tree at all, and no loops at all
    assert a.shape == (n, n)
    a, _ = K.peec_system(R, L, 1j, 1.0, 0, rng.normal(size=(n, n)))
    assert a.shape == (n, n)
    with pytest.raises(ValueError):
        K.peec_system(R, L, 1j, 1.0, nl, electro[:-1, :-1])


def test_complex_matrices_keep_numpy_s_path():
    """R or L complex (no solver makes them so): the compiled kernel would drop the imaginary part."""
    _, R, L, S = _matrices(20)
    a = K.loop_system(R * (1 + 1j), L, 2j, np.arange(20), np.arange(20), S)
    assert np.allclose(a, R * (1 + 1j) + 2j * L + S.toarray())


def test_lu_of_a_row_major_matrix_in_its_own_memory():
    rng = np.random.default_rng(5)
    a = rng.normal(size=(60, 60)) + 1j * rng.normal(size=(60, 60))
    b = rng.normal(size=(60, 3)) + 0j
    keep = a.copy()
    lu = K.lu_factor_rows(a)
    assert np.shares_memory(lu[0], a)                               # no second matrix
    assert np.allclose(keep @ K.lu_solve_rows(lu, b), b)
    assert np.allclose(keep @ K.lu_solve_rows(lu, b[:, 0]), b[:, 0])


def _mesh():
    """A 6 x 4 grid of squares, each cut in two; a hole where one square is left out."""
    x, y = np.meshgrid(np.arange(7.0), np.arange(5.0))
    nodes = np.c_[x.ravel(), y.ravel()]
    tris = []
    for j in range(4):
        for i in range(6):
            if (i, j) == (2, 1):
                continue
            a = j * 7 + i
            tris += [(a, a + 1, a + 8), (a, a + 8, a + 7)]
    return nodes, np.array(tris)


@pytest.mark.parametrize('numpy', [False, True])
def test_points_in_a_mesh(numpy):
    if not numpy and K.core('tri_locate') is None:
        pytest.skip('no compiled core with the dense kernels')
    nodes, tris = _mesh()
    v = 3 * nodes[:, 0] - 2 * nodes[:, 1] + 1                       # linear: the mesh interpolates it exactly
    rng = np.random.default_rng(2)
    pts = rng.uniform(-1, 7, (4000, 2))
    index, w = K.tri_locate(nodes, tris, pts, numpy=numpy)
    inside = (pts[:, 0] > 0) & (pts[:, 0] < 6) & (pts[:, 1] > 0) & (pts[:, 1] < 4) & \
        ~((pts[:, 0] > 2) & (pts[:, 0] < 3) & (pts[:, 1] > 1) & (pts[:, 1] < 2))
    assert ((index >= 0) == inside).all()                           # none outside, none in the hole
    assert np.isnan(w[~inside]).all() and np.allclose(w[inside].sum(1), 1) and (w[inside] > -1e-9).all()
    got = (v[tris[index[inside]]] * w[inside]).sum(1)
    assert np.allclose(got, 3 * pts[inside, 0] - 2 * pts[inside, 1] + 1)
    empty, we = K.tri_locate(nodes, np.zeros((0, 3), int), pts[:5], numpy=numpy)
    assert (empty == -1).all() and np.isnan(we).all()


@core
def test_points_on_nodes_and_edges_have_a_triangle():
    nodes, tris = _mesh()
    v = np.arange(len(nodes), dtype=float) ** 1.5
    index, w = K.tri_locate(nodes, tris, nodes)                     # every node: its own value
    assert (index >= 0).all() and np.allclose((v[tris[index]] * w).sum(1), v)
    mid = 0.5 * (nodes[tris[:, 0]] + nodes[tris[:, 1]])             # every first edge's midpoint
    index, w = K.tri_locate(nodes, tris, mid)
    assert (index >= 0).all()
    assert np.allclose((v[tris[index]] * w).sum(1), 0.5 * (v[tris[:, 0]] + v[tris[:, 1]]))
    a, wa = K.tri_locate(nodes, tris, mid)
    b, wb = K.tri_locate(nodes, tris, mid, numpy=True)
    ok = b >= 0                                                     # (matplotlib's finder may drop a boundary point)
    assert np.allclose((v[tris[a[ok]]] * wa[ok]).sum(1), (v[tris[b[ok]]] * wb[ok]).sum(1))


def test_a_thermal_mesh_is_sampled_on_its_own_triangles():
    from types import SimpleNamespace
    from pcbqs.thermal.problem import Result
    nodes, tris = _mesh()
    T = np.vstack([20 + nodes[:, 0] ** 2, 30 + nodes[:, 1]])
    r = Result('mesh', SimpleNamespace(index=lambda name: 1), T, {}, len(nodes), {}, 0.0, 1.0, nodes=nodes, tris=tris)
    assert np.allclose(r.sample(0, nodes[:, 0], nodes[:, 1]), T[0])                   # at the nodes: the solution
    x, y = np.meshgrid(np.linspace(0.1, 5.9, 30), np.linspace(0.1, 3.9, 20))
    got = r.sample('B.Cu', x, y)
    hole = (x > 2) & (x < 3) & (y > 1) & (y < 2)
    assert got.shape == x.shape and np.isnan(got[hole]).all() and np.allclose(got[~hole], 30 + y[~hole])
    assert np.isnan(r.sample(0, [-1.0, 9.0], [1.0, 1.0])).all()                       # off the board
    a = r.sample(0, x, y)                                           # x^2 on each triangle: between its corners' values
    assert np.nanmax(np.abs(a - (20 + x ** 2))) < 0.26


def test_symmetric_solve_by_cholesky_where_it_can():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(40, 40))
    P = a @ a.T + 40 * np.eye(40)                                   # definite: Cholesky
    B = rng.normal(size=(40, 3))
    assert np.allclose(P @ K.solve_symmetric(P, B), B)
    x = K.solve_symmetric(P.astype(complex), B)                     # complex with nothing imaginary: the same
    assert not np.iscomplexobj(x) and np.allclose(P @ x, B)
    Q = P.copy()
    Q[0, 0] = -3.0                                                  # symmetric, not definite
    assert np.allclose(Q @ K.solve_symmetric(Q, B), B)
    Z = P * (1 + 0.3j)                                              # complex symmetric (a lossy stack)
    assert np.allclose(Z @ K.solve_symmetric(Z, B), B)
