"""panels3d.pair_moments: triangles of any orientation (side walls) against the horizontal kernel, rotations and
brute-force quadrature."""
import numpy as np
import pytest

from pcbqs._numerics.base import tri_area2
from pcbqs._numerics.sing import offplane_pairs
from pcbqs.panels3d import frames, pair_moments, rwg_pair


def _rot(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _horizontal_pairs(rng, m=40):
    """Random horizontal triangle pairs, near and far, incl. coplanar touching ones (mm-scale, in metres)."""
    nodes = rng.uniform(-1e-3, 1e-3, (3 * 2 * m, 2))
    tris = np.arange(len(nodes)).reshape(-1, 3)
    tris[tri_area2(nodes[tris]) < 0] = tris[tri_area2(nodes[tris]) < 0][:, [0, 2, 1]]
    z = rng.choice([0.0, 35e-6, 0.2e-3], len(tris))
    p, q = np.arange(m), np.arange(m, 2 * m)
    return nodes, tris, z, p, q


@pytest.mark.parametrize('levels', [0, 1])
def test_horizontal_pairs_match_the_sheet_kernel(levels):
    rng = np.random.default_rng(1)
    nodes, tris, z, p, q = _horizontal_pairs(rng)
    m0, mro, mrs, mrr = offplane_pairs(nodes, tris, p, q, z[p] - z[q], outer_levels=levels)
    V = np.concatenate([nodes[tris], z[:, None, None] * np.ones((1, 3, 1))], -1)
    a0, aro, ars, arr = pair_moments(V[p], V[q], outer_levels=levels)
    np.testing.assert_allclose(a0, m0, rtol=1e-12, atol=0)
    np.testing.assert_allclose(aro[:, :2], mro, rtol=1e-11, atol=1e-30)
    np.testing.assert_allclose(ars[:, :2], mrs, rtol=1e-11, atol=1e-30)
    np.testing.assert_allclose(aro[:, 2], z[p] * m0, rtol=1e-12, atol=1e-40)
    np.testing.assert_allclose(ars[:, 2], z[q] * m0, rtol=1e-12, atol=1e-40)
    np.testing.assert_allclose(arr, mrr + z[p] * z[q] * m0, rtol=1e-11, atol=1e-40)


@pytest.mark.parametrize('axis,angle', [((1, 0, 0), np.pi / 2), ((0, 1, 0), np.pi / 2), ((1, 2, 3), 0.7)])
def test_moments_rotate_with_the_pair(axis, angle):
    """A rigid rotation (about the origin) of both triangles: m0 and mrr unchanged, mro and mrs rotated. A quarter
    turn about x or y stands horizontal triangles on end, as side walls."""
    rng = np.random.default_rng(2)
    nodes, tris, z, p, q = _horizontal_pairs(rng)
    V = np.concatenate([nodes[tris], z[:, None, None] * np.ones((1, 3, 1))], -1)
    R = _rot(axis, angle)
    W = V @ R.T
    a = pair_moments(V[p], V[q])
    b = pair_moments(W[p], W[q])
    np.testing.assert_allclose(b[0], a[0], rtol=1e-11)
    np.testing.assert_allclose(b[1], a[1] @ R.T, rtol=1e-9, atol=1e-12 * np.abs(a[1]).max())
    np.testing.assert_allclose(b[2], a[2] @ R.T, rtol=1e-9, atol=1e-12 * np.abs(a[2]).max())
    np.testing.assert_allclose(b[3], a[3], rtol=1e-9, atol=1e-12 * np.abs(a[3]).max())


def _brute(Vo, Vs, levels=4):
    """Both integrals by the 6-point rule on 4**levels sub-triangles each (pairs well apart)."""
    from pcbqs.panels3d import _rule
    bar, w = _rule(levels)
    _, _, _, _, Ao = frames(Vo)
    _, _, _, _, As = frames(Vs)
    P = np.einsum('qk,mkc->mqc', bar, Vo)
    Q = np.einsum('qk,mkc->mqc', bar, Vs)
    g = 1 / np.linalg.norm(P[:, :, None] - Q[:, None], axis=-1)
    W = (w[None] * Ao[:, None])[:, :, None] * (w[None] * As[:, None])[:, None]
    gw = g * W
    return (gw.sum((1, 2)), np.einsum('mab,mac->mc', gw, P), np.einsum('mab,mbc->mc', gw, Q),
            np.einsum('mab,mac,mbc->m', gw, P, Q))


def test_separated_pairs_at_any_angle_match_quadrature():
    rng = np.random.default_rng(3)
    m = 30
    base = np.array([[0, 0, 0], [0.3e-3, 0, 0], [0.1e-3, 35e-6, 0]])          # a wall-like sliver, 0.3 mm x 35 um
    Vo = np.stack([base @ _rot(rng.normal(size=3), rng.uniform(0, np.pi)).T + rng.uniform(-1e-3, 1e-3, 3) for _ in range(m)])
    Vs = np.stack([base @ _rot(rng.normal(size=3), rng.uniform(0, np.pi)).T + rng.uniform(-1e-3, 1e-3, 3) + [2e-3, 0, 0]
                   for _ in range(m)])
    a, b = pair_moments(Vo, Vs, outer_levels=2), _brute(Vo, Vs)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=2e-6)


def test_wall_meeting_a_sheet_converges_and_is_reciprocal():
    """A side wall (35 um tall) standing on a horizontal sheet triangle's edge: the singular, perpendicular case a
    face-to-wall RWG function spans. Outer refinement converges, and swapping roles (observation <-> source) gives
    the same m0 and mrr."""
    sheet = np.array([[[0, 0, 0], [0.4e-3, 0, 0], [0.2e-3, 0.3e-3, 0]]])
    wall = np.array([[[0.4e-3, 0, 0], [0, 0, 0], [0, 0, -35e-6]]])         # down from the sheet's edge 0
    ref = [pair_moments(wall, sheet, outer_levels=5), pair_moments(sheet, wall, outer_levels=5)]
    np.testing.assert_allclose(ref[0][0], ref[1][0], rtol=1e-4)
    np.testing.assert_allclose(ref[0][3], ref[1][3], rtol=1e-4)
    np.testing.assert_allclose(ref[0][1], ref[1][2], rtol=1e-4, atol=1e-6 * np.abs(ref[0][1]).max())
    for levels, tol in ((1, 2e-2), (3, 2e-3)):
        np.testing.assert_allclose(pair_moments(wall, sheet, outer_levels=levels)[0], ref[0][0], rtol=tol)


def test_rwg_pair_is_the_galerkin_product():
    """rwg_pair from the moments equals the brute-force int int (r - vo).(r' - vs)/R, scaled."""
    Vo = np.array([[[0, 0, 0], [0.3e-3, 0, 0], [0, 0, 35e-6]]], float)
    Vs = Vo + [[1.5e-3, 0.4e-3, 0.2e-3]]
    vo, vs = Vo[:, 2], Vs[:, 0]
    from pcbqs.panels3d import _rule
    bar, w = _rule(4)
    P = np.einsum('qk,mkc->mqc', bar, Vo)[0]
    Q = np.einsum('qk,mkc->mqc', bar, Vs)[0]
    Ao, As = frames(Vo)[4][0], frames(Vs)[4][0]
    f = ((P - vo)[:, None] * (Q - vs)[None]).sum(-1) / np.linalg.norm(P[:, None] - Q[None], axis=-1)
    brute = (w[:, None] * w[None] * f).sum() * Ao * As * 2.0 * 3.0
    got = rwg_pair(pair_moments(Vo, Vs, outer_levels=3), np.array([2.0]), vo, np.array([3.0]), vs)
    np.testing.assert_allclose(got, brute, rtol=1e-6)
