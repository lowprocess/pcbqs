"""
The board's temperature on pcbqs's triangle mesh (linear elements), the mesher the DC drop uses.

One mesh of the board's outline carries every layer: the outlines of each layer's copper, of the heated pads and a
ring at each via are made mesh edges, so a triangle is all copper or none on each layer and the layers share their
vertices (one unknown per vertex per layer).

    r = solve(problem, dmax=1.0)            # problem.Result: r.nodes, r.tris, r.T (layers, N) C, r.parts
"""
import time

import numpy as np
import shapely
from shapely.affinity import scale
from scipy.sparse import coo_matrix, diags

from .._numerics.geo import mesh_polygons
from ..pdn.mesh import stiffness, mass, overlapping_vias
from .problem import Result, part_temperatures, solve_spd


def _segments(geom):
    out = []
    for q in getattr(geom, 'geoms', [geom]):
        if q.geom_type != 'Polygon':
            continue
        for ring in [q.exterior, *q.interiors]:
            c = np.asarray(ring.coords)[:, :2] * 1e-3
            out += [(tuple(a), tuple(b)) for a, b in zip(c[:-1], c[1:])]
    return out


def solve(problem, dmax=1.0, simplify=None, coarsen=None, quality=25, log=print):
    """simplify: the copper's outlines are straightened by up to this (mm; default dmax / 20), so that the short
    segments of arcs and zone fills do not each make their own triangles."""
    p = problem
    t0 = time.perf_counter()
    tol = dmax / 20 if simplify is None else simplify
    copper = [None if l.copper is None else l.copper.simplify(tol).buffer(0) for l in p.layers]
    lines = []
    for g in copper:
        if g is not None:
            lines += _segments(g)
    for s in p.sources:
        lines += _segments(s.geom)
    pts = [(v['x'], v['y'], v['r']) for v in p.vias]
    gone = overlapping_vias(pts)                        # drilled into one another: one ring
    keep = [i for i in range(len(pts)) if i not in gone]
    vias = [tuple(c * 1e-3 for c in pts[i]) for i in keep]
    geom = scale(p.outline, 1e-3, 1e-3, origin=(0, 0))
    nodes, tris, _ = mesh_polygons(geom, dmax * 1e-3, dmax * 1e-3, ncross=1, lines=lines, vias=vias,
                                   coarsen=coarsen, quality=quality)
    nodes = nodes * 1e3
    N, L = len(nodes), len(p.layers)
    cen = nodes[tris].mean(axis=1)
    M = mass(nodes, tris).diagonal() * 1e-6             # m2 at each vertex
    P = nodes[tris]
    tri_area = 0.5 * np.abs((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1])
                            - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))
    blocks = []
    for l, lay in enumerate(p.layers):
        coef = np.full(len(tris), lay.laminate)
        if copper[l] is not None:
            coef = coef + p.k_cu * lay.t * 1e-3 * shapely.contains_xy(copper[l], cen[:, 0], cen[:, 1])
        K = stiffness(nodes, tris, coef).tocoo()
        blocks.append((K.row + l * N, K.col + l * N, K.data))
    rows, cols, vals = [], [], []

    def link(a, b, g):
        rows.append(np.r_[a, b, a, b]); cols.append(np.r_[a, b, b, a]); vals.append(np.r_[g, g, -g, -g])
    every = np.arange(N)
    for l, gap in enumerate(p.gaps):
        link(every + l * N, every + (l + 1) * N, gap * M)
    host = {}
    for i, v in enumerate(p.vias):
        host.setdefault(gone.get(i, i), []).append(v)
    for i, group in host.items():
        x, y, r = pts[i]
        d = np.hypot(nodes[:, 0] - x, nodes[:, 1] - y)
        ring = np.flatnonzero(d <= r * 1.02 + 1e-9)
        if not len(ring):
            ring = np.array([int(np.argmin(d))])
        for v in group:
            for l in range(v['lo'], v['hi']):
                link(ring + l * N, ring + (l + 1) * N, np.full(len(ring), p.barrel(v, l) / len(ring)))
    film = np.zeros(L * N)
    film[:N] += p.film[0] * M
    film[(L - 1) * N:] += p.film[1] * M
    q = np.zeros(L * N)
    weights = []
    for s in p.sources:
        inside = shapely.contains_xy(s.geom.buffer(1e-6), cen[:, 0], cen[:, 1])
        w = np.zeros(N)
        if inside.any():
            np.add.at(w, tris[inside].ravel(), np.repeat(tri_area[inside] / 3, 3))
        else:                                       # a pad smaller than a triangle: the vertex nearest its middle
            c = s.geom.representative_point()
            w[int(np.argmin(np.hypot(nodes[:, 0] - c.x, nodes[:, 1] - c.y)))] = 1.0
        w /= w.sum()
        q[s.layer * N:(s.layer + 1) * N] += s.watts * w
        weights.append(w)
    r_ = np.concatenate([b[0] for b in blocks] + rows)
    c_ = np.concatenate([b[1] for b in blocks] + cols)
    d_ = np.concatenate([b[2] for b in blocks] + vals)
    A = (coo_matrix((d_, (r_, c_)), shape=(L * N, L * N)) + diags(film)).tocsc()
    t1 = time.perf_counter()
    rise = solve_spd(A, q)
    t2 = time.perf_counter()
    T = p.ambient + rise.reshape(L, N)
    by = {id(s): w for s, w in zip(p.sources, weights)}
    parts = part_temperatures(p, lambda s: float(T[s.layer] @ by[id(s)]))
    res = Result('mesh', p, T, parts, L * N, dict(build=t1 - t0, solve=t2 - t1), float(film @ rise), dmax,
                 nodes=nodes, tris=tris)
    log(f"thermal mesh: {N:,} vertices, {len(tris):,} triangles of up to {dmax:g} mm on {L} layer(s), "
        f"{L * N:,} unknowns, {t2 - t0:.2f} s; hottest {res.hottest:.1f} C")
    return res
