"""
The board's temperature on a regular grid (finite volumes): square cells of `pitch` mm over the outline's bounding
box, one unknown per cell on the board per copper layer.

The copper of each layer is drawn onto a raster finer than the cells (1/16 mm unless asked otherwise). The conductance between two
neighbouring cells comes from that raster: between their centres, each fine row is a chain in series, the rows in
parallel. A slot narrower than a cell still blocks, a trace narrower than a cell still carries, and nothing has to
follow the outline of the copper, so the cost does not depend on how intricate the copper is.

    r = solve(problem, pitch=0.5)           # problem.Result: r.T (layers, ny, nx) C, r.parts, r.hottest
"""
import math
import time

import numpy as np
from scipy.sparse import coo_matrix
from shapely.geometry.polygon import orient

from .problem import Result, part_temperatures, solve_spd

DETAIL = 0.0625     # mm: the fine raster's cell unless asked otherwise
RASTER = 4e7        # the most fine cells a layer's raster may have
OVER = 4            # bands across each fine cell


def _polys(geom):
    return [q for q in getattr(geom, 'geoms', [geom]) if q.geom_type == 'Polygon' and not q.is_empty]


def raster(geom, x0, y0, h, nx, ny):
    """The fraction of each h x h cell (ny, nx) from (x0, y0) covered by geom (shapely, mm), by signed areas: each
    cell is cut into OVER bands, and each part of an outline within a band adds the area to its right (the cell
    it crosses gets the part of its width beyond the crossing). The total is the geometry's area exactly, so thin
    traces and narrow gaps keep their widths wherever they fall on the raster."""
    acc = np.zeros(ny * (nx + 2))
    if geom is None or geom.is_empty:
        return np.zeros((ny, nx))
    U0, V0, U1, V1 = [], [], [], []
    for q in _polys(geom):
        q = orient(q, 1.0)                              # outside anticlockwise, holes clockwise
        for ring in [q.exterior, *q.interiors]:
            c = np.asarray(ring.coords)[:, :2]
            U0.append((c[:-1, 0] - x0) / h); V0.append((c[:-1, 1] - y0) / h)
            U1.append((c[1:, 0] - x0) / h); V1.append((c[1:, 1] - y0) / h)
    u0, v0, u1, v1 = (np.concatenate(a) for a in (U0, V0, U1, V1))
    keep = v0 != v1
    u0, v0, u1, v1 = u0[keep], v0[keep], u1[keep], v1[keep]
    sign = np.where(v1 < v0, 1.0, -1.0)                 # copper lies to the right of an edge that runs down
    up = v1 > v0
    ua, va, ub, vb = np.where(up, u0, u1), np.where(up, v0, v1), np.where(up, u1, u0), np.where(up, v1, v0)
    k0 = np.clip(np.floor(va * OVER), 0, ny * OVER).astype(np.int64)        # band k: [k, k + 1) / OVER
    k1 = np.clip(np.ceil(vb * OVER), 0, ny * OVER).astype(np.int64)
    n = np.maximum(k1 - k0, 0)
    e = np.repeat(np.arange(len(n)), n)
    k = k0[e] + (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))
    lo, hi = np.maximum(va[e], k / OVER), np.minimum(vb[e], (k + 1) / OVER)    # the edge's part in the band
    u = ua[e] + ((lo + hi) / 2 - va[e]) * (ub[e] - ua[e]) / (vb[e] - va[e])
    u = np.clip(u, 0.0, nx)
    c = np.floor(u).astype(np.int64)
    f = u - c
    at = (k // OVER) * (nx + 2) + c
    w = sign[e] * (hi - lo)
    acc += np.bincount(at, w * (1 - f), minlength=len(acc)) + np.bincount(at + 1, w * f, minlength=len(acc))
    return np.clip(np.cumsum(acc.reshape(ny, nx + 2), axis=1)[:, :nx], 0.0, 1.0)


def _between(g, sub, axis):
    """The conductance between neighbouring cells along `axis` (1: x, 0: y) from the fine sheet conductance g
    (W/K per square, 0 where there is no board): between the two centres each fine row is a chain in series, the
    rows in parallel. Where the board ends inside a cell, short of its centre, a row's chain starts at the board's
    edge; a slot (board again beyond the centre) still breaks the chain."""
    if axis == 0:
        return _between(g.T, sub, 1).T
    if sub % 2:                                 # centre to centre needs half cells: split each fine cell in two
        g, sub = np.repeat(np.repeat(g, 2, 0), 2, 1), 2 * sub
    half = sub // 2
    rows, nx = g.shape[0], g.shape[1] // sub
    c = g.reshape(rows, nx, 2, half)            # each fine row of each cell: its left half, its right half
    with np.errstate(divide='ignore'):
        r = 1.0 / c

    def out(r, on, other):                      # from the centre outwards: r, on (.., half), nearest the centre first
        lead = np.cumsum(on, axis=-1) == 0      # off the board, before the first of the board
        skip = lead & ~other.any(axis=-1)[..., None] & on.any(axis=-1)[..., None]
        return np.where(skip, 0.0, r).sum(axis=-1)
    on = c > 0
    right = out(r[:, :, 1], on[:, :, 1], on[:, :, 0])
    left = out(r[:, :, 0, ::-1], on[:, :, 0, ::-1], on[:, :, 1])
    with np.errstate(divide='ignore'):
        k = 1.0 / (right[:, :-1] + left[:, 1:])                 # (fine rows, nx - 1)
    return k.reshape(rows // sub, sub, nx - 1).sum(axis=1)


def cover(geom, x0, y0, pitch, sub, nx, ny):
    """(rows, columns, fractions of geom's area) of the cells geom lies on."""
    bx0, by0, bx1, by1 = geom.bounds
    c0, r0 = max(int(math.floor((bx0 - x0) / pitch)), 0), max(int(math.floor((by0 - y0) / pitch)), 0)
    c1, r1 = min(int(math.floor((bx1 - x0) / pitch)) + 1, nx), min(int(math.floor((by1 - y0) / pitch)) + 1, ny)
    if c1 <= c0 or r1 <= r0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    w, h = c1 - c0, r1 - r0
    a = raster(geom, x0 + c0 * pitch, y0 + r0 * pitch, pitch / sub, w * sub, h * sub)
    a = a.reshape(h, sub, w, sub).mean(axis=(1, 3))
    if a.sum() <= 0:                            # smaller than the raster sees: the cell of its middle
        p = geom.representative_point()
        c = min(max(int((p.x - x0) / pitch), 0), nx - 1)
        r = min(max(int((p.y - y0) / pitch), 0), ny - 1)
        return np.array([r]), np.array([c]), np.array([1.0])
    r, c = np.nonzero(a)
    return r + r0, c + c0, a[r, c] / a.sum()


def edge_lengths(problem, x0, y0, pitch, live, piece):
    """m of the board's edges in each cell (ny, nx): the outline cut into pieces of at most `piece` mm, each
    given to the cell its middle is in (moved a hair into the board, so a piece on a cell's side goes to the cell
    of the board), or, if that cell has no board, to the nearest one that does."""
    ny, nx = live.shape
    out = np.zeros((ny, nx))
    a, b = problem.edges()
    if not len(a) or not live.any():
        return out
    d = b - a
    seg = np.hypot(d[:, 0], d[:, 1])
    keep = seg > 0
    a, d, seg = a[keep], d[keep], seg[keep]
    n = np.maximum(np.ceil(seg / piece).astype(np.int64), 1)
    e = np.repeat(np.arange(len(n)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 0.5) / n[e]
    left = np.c_[-d[:, 1], d[:, 0]] / seg[:, None]                 # towards the board
    m = a[e] + t[:, None] * d[e] + left[e] * (1e-3 * piece)
    c = np.clip(np.floor((m[:, 0] - x0) / pitch).astype(np.int64), 0, nx - 1)
    r = np.clip(np.floor((m[:, 1] - y0) / pitch).astype(np.int64), 0, ny - 1)
    if not live[r, c].all():
        from scipy.ndimage import distance_transform_edt
        _, (rr, cc) = distance_transform_edt(~live, return_indices=True)
        r, c = rr[r, c], cc[r, c]
    np.add.at(out, (r, c), (seg / n)[e] * 1e-3)
    return out


def solve(problem, pitch=0.5, sub=None, detail=DETAIL, log=print):
    """pitch: the cells, mm. sub: fine raster cells across one cell (default: enough for `detail` mm, the narrowest
    gap or trace the conductances still see; fewer on a board too large for that)."""
    p = problem
    t0 = time.perf_counter()
    if pitch <= 0 or (sub is not None and sub < 1) or detail <= 0:
        raise ValueError('pitch and detail must be positive and sub at least 1')
    x0, y0, x1, y1 = p.outline.bounds
    nx, ny = max(int(math.ceil((x1 - x0) / pitch - 1e-9)), 1), max(int(math.ceil((y1 - y0) / pitch - 1e-9)), 1)
    if sub is None:
        sub = max(2, 2 * int(math.ceil(pitch / detail / 2 - 1e-9)))
        while sub > 2 and nx * ny * sub * sub > RASTER:
            sub -= 2
    L = len(p.layers)
    h = pitch / sub
    board = raster(p.outline, x0, y0, h, nx * sub, ny * sub)
    area = board.reshape(ny, sub, nx, sub).mean(axis=(1, 3)) * pitch ** 2 * 1e-6        # m2 of board per cell
    live = area > 0
    idx = np.full((L, ny, nx), -1)
    n1 = int(live.sum())
    for l in range(L):
        idx[l][live] = np.arange(n1) + l * n1
    n = L * n1
    rows, cols, vals = [], [], []

    def link(a, b, g):
        rows.append(np.r_[a, b, a, b]); cols.append(np.r_[a, b, b, a]); vals.append(np.r_[g, g, -g, -g])
    diag = np.zeros(n)
    for l, lay in enumerate(p.layers):
        g = lay.laminate * board
        if lay.copper is not None:
            g = g + p.k_cu * lay.t * 1e-3 * raster(lay.copper, x0, y0, h, nx * sub, ny * sub) * board
        gx, gy = _between(g, sub, 1), _between(g, sub, 0)
        a, b = idx[l][:, :-1], idx[l][:, 1:]
        m = (a >= 0) & (b >= 0) & (gx > 0)
        link(a[m], b[m], gx[m])
        a, b = idx[l][:-1], idx[l][1:]
        m = (a >= 0) & (b >= 0) & (gy > 0)
        link(a[m], b[m], gy[m])
    for l, gap in enumerate(p.gaps):
        link(idx[l][live], idx[l + 1][live], gap * area[live])
    for v in p.vias:
        c, r = int((v['x'] - x0) / pitch), int((v['y'] - y0) / pitch)
        if not (0 <= c < nx and 0 <= r < ny and live[r, c]):
            continue
        for l in range(v['lo'], v['hi']):
            link(np.array([idx[l, r, c]]), np.array([idx[l + 1, r, c]]), np.array([p.barrel(v, l)]))
    film = np.zeros(n)
    film[idx[0][live]] += p.film[0] * area[live]
    film[idx[-1][live]] += p.film[1] * area[live]
    edge = np.zeros(n)
    if p.edge_film > 0:
        run = edge_lengths(p, x0, y0, pitch, live, h)[live]
        for l in range(L):
            edge[idx[l][live]] += p.edge_conductance(l) * run
    film += edge
    diag += film
    q = np.zeros(n)
    covers = []
    for s in p.sources:
        r, c, f = cover(s.geom, x0, y0, pitch, sub, nx, ny)
        ok = live[r, c]
        if not ok.any():
            covers.append(None)
            continue
        r, c, f = r[ok], c[ok], f[ok] / f[ok].sum()
        np.add.at(q, idx[s.layer, r, c], s.watts * f)
        covers.append((idx[s.layer, r, c], f))
    A = coo_matrix((np.concatenate(vals + [diag]), (np.concatenate(rows + [np.arange(n)]),
                                                    np.concatenate(cols + [np.arange(n)]))), shape=(n, n)).tocsc()
    t1 = time.perf_counter()
    rise = solve_spd(A, q)
    t2 = time.perf_counter()
    T = np.full((L, ny, nx), np.nan)
    for l in range(L):
        T[l][live] = p.ambient + rise[idx[l][live]]
    by = {id(s): cv for s, cv in zip(p.sources, covers)}
    parts = part_temperatures(p, lambda s: p.ambient + (float(rise[by[id(s)][0]] @ by[id(s)][1])
                                                        if by[id(s)] is not None else 0.0))
    res = Result('grid', p, T, parts, n, dict(build=t1 - t0, solve=t2 - t1), float(film @ rise), pitch, x0=x0, y0=y0,
                 edge_W=float(edge @ rise))
    log(f"thermal grid: {nx} x {ny} cells of {pitch:g} mm on {L} layer(s), {n:,} unknowns, "
        f"{t2 - t0:.2f} s; hottest {res.hottest:.1f} C")
    return res
