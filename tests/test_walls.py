"""walls: boundary loops, wall panels and their rooftop currents' inductance (side walls of thick copper)."""
import numpy as np
import pytest
from shapely.geometry import box

from pcbqs._numerics.base import RWG, tri_area2
from pcbqs._numerics.geo import mesh_polygons
from pcbqs.walls import boundary_loops, boundary_walls, chain_walls, wall_inductance, wall_rwg_inductance

MU0 = 4e-7 * np.pi
T = 35e-6


def grover_tape(l, w):
    """Grover's partial self inductance of a thin straight tape, length l and width w (uniform current)."""
    return MU0 * l / (2 * np.pi) * (np.log(2 * l / w) + 0.5 + 0.2235 * w / l)


def grover_rect_loop(a, b, w):
    """Grover's rectangular loop a x b of thin tape of width w: the round-wire formula with the strip's geometric
    mean distance 0.2235 w as radius, less the round wire's internal term 0.25 (a + b) (the GMD already holds the
    strip's own field; validation/mqs_palace.py subtracts it alike)."""
    r = 0.2235 * w
    d = np.hypot(a, b)
    return MU0 / np.pi * (a * np.log(2 * a * b / (r * (a + d))) + b * np.log(2 * a * b / (r * (b + d)))
                          - 2 * (a + b) + 2 * d)


def _mesh(geom, dmax=0.4e-3):
    nodes, tris, _ = mesh_polygons(geom, dmax, w_ref=1e-3)
    tris = np.asarray(tris)
    flip = tri_area2(nodes[tris]) < 0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return nodes, tris


def test_boundary_loops_outer_and_hole():
    nodes, tris = _mesh(box(0, 0, 4e-3, 3e-3).difference(box(1e-3, 1e-3, 2e-3, 2e-3)))
    loops = boundary_loops(tris)
    assert len(loops) == 2
    area = []
    for loop in loops:
        assert all(b == loop[(k + 1) % len(loop)][0] for k, (a, b, t) in enumerate(loop))   # closed, chained
        p = nodes[[a for a, b, t in loop]]
        area.append(0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    assert sorted(np.sign(area)) == [-1, 1]                    # outer counter-clockwise, hole clockwise
    assert max(area) == pytest.approx(12e-6, rel=1e-6) and min(area) == pytest.approx(-1e-6, rel=1e-6)
    w = boundary_walls(nodes, tris, 1.0e-3, T)
    assert len(w['roof']) == len(w['ends']) == sum(len(l) for l in loops)
    assert w['z1'] - w['z0'] == pytest.approx(T)


def test_straight_wall_matches_a_thin_tape():
    """Every rooftop of an open straight wall at 1 A: uniform current but for the end panels' ramps, whose half
    weight shortens the tape by one panel. Grover's tape (10 mm x 35 um) within 1%."""
    n, l = 100, 10e-3
    w = chain_walls(np.c_[np.linspace(0, l, n + 1), np.zeros(n + 1)], 0.0, T)
    L = wall_inductance(w)
    c = np.ones(len(w['roof']))
    assert c @ L @ c == pytest.approx(grover_tape(l - l / n, T), rel=1e-2)


def test_closed_wall_loop_matches_grovers_loop_and_is_positive_definite():
    """A 4 x 2 mm rectangle of wall (35 um tall) carrying 1 A all round: Grover's rectangular loop of tape within
    1% (0.1% at this mesh); the rooftop matrix symmetric positive definite."""
    a, b, k = 4e-3, 2e-3, 20
    pts = np.concatenate([np.c_[np.linspace(0, a, 2 * k, endpoint=False), np.zeros(2 * k)],
                          np.c_[np.full(k, a), np.linspace(0, b, k, endpoint=False)],
                          np.c_[np.linspace(a, 0, 2 * k, endpoint=False), np.full(2 * k, b)],
                          np.c_[np.zeros(k), np.linspace(b, 0, k, endpoint=False)]])
    w = chain_walls(pts, 0.0, T, closed=True)
    L = wall_inductance(w)
    np.testing.assert_allclose(L, L.T)
    assert np.linalg.eigvalsh(L).min() > 0
    c = np.ones(len(w['roof']))
    assert c @ L @ c == pytest.approx(grover_rect_loop(a, b, T), rel=1e-2)


def test_wall_on_a_sheet_couples_the_same_either_way():
    """Walls standing on a sheet's boundary (the singular, perpendicular case): the rooftop x RWG mutual inductance
    with the wall observing equals that with the sheet observing, to the observation rule's accuracy; finer
    observation rules converge."""
    nodes, tris = _mesh(box(0, 0, 2e-3, 1e-3), dmax=0.5e-3)
    rwg = RWG(nodes, tris)
    z = np.zeros(len(tris))
    w = boundary_walls(nodes, tris, 0.0, T)
    w = dict(w, z0=0.0, z1=T)                                   # standing on the sheet, up from its plane
    from pcbqs.walls import _galerkin, _wall_pieces
    walls = _wall_pieces(w)
    V = np.concatenate([nodes[tris], np.zeros((len(tris), 3, 1))], -1)
    sheet = [(e, V[t][None], a * np.eye(3), np.r_[v, 0.0])
             for e in range(rwg.Ne)
             for t, a, v in ((rwg.tp[e], rwg.l[e] / (2 * rwg.area[rwg.tp[e]]), rwg.vp[e]),
                             (rwg.tm[e], -rwg.l[e] / (2 * rwg.area[rwg.tm[e]]), rwg.vm[e]))]
    one, two = np.zeros((len(w['roof']), rwg.Ne)), np.zeros((len(w['roof']), rwg.Ne))
    r, c, v = _galerkin(walls, sheet, 3)
    np.add.at(one, (r, c), v)
    r, c, v = _galerkin(sheet, walls, 3)
    np.add.at(two, (c, r), v)
    scale = np.abs(one).max()
    assert np.abs(one - two).max() < 2e-2 * scale
    M2, M3 = wall_rwg_inductance(w, rwg, z, outer_levels=2), wall_rwg_inductance(w, rwg, z, outer_levels=3)
    assert np.abs(M2 - M3).max() < 2e-2 * scale
