"""Side walls of thick copper (experimental, for MQS's two face sheets): the copper's vertical edges as panels, and
the current along them.

A layer's copper is two face sheets (thickness_sheets=2) whose triangle copies are one circuit node, so current
already passes between the faces at every triangle. What the faces lack is the current on the copper's sides,
where it crowds at high frequency. Here each boundary edge of a layer's mesh stands up as a wall panel of the
copper's thickness, owned by the boundary triangle beneath it (its node), and the wall carries current along itself:
a rooftop across each vertex shared by two consecutive panels, rising over one panel and falling over the next,
uniform up the wall's height and normalised to 1 A through the vertex's vertical line.

    w = boundary_walls(nodes, tris, z, t)        # nodes (n, 2) m, tris (m, 3) counter-clockwise, z and t m
    L = wall_inductance(w)                       # rooftop x rooftop partial inductance (H), dense
    M = wall_rwg_inductance(w, rwg, z_tri)       # rooftop x RWG edge, with a sheet's RWG functions (nodes 2D)

Open: the walls' impedance. At DC the faces already carry all the copper's cross-section, so a wall must add none
then; see ROADMAP (C14, side walls).
"""
import numpy as np

from .panels3d import affine_pair, pair_moments

MU0 = 4e-7 * np.pi


def boundary_loops(tris):
    """The mesh's boundary as closed loops of directed edges [(a, b, triangle)], following each triangle's own
    (counter-clockwise) order: outer boundaries run counter-clockwise, holes clockwise."""
    count = {}
    for t, tri in enumerate(tris):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(a, b), max(a, b))
            count.setdefault(key, []).append((int(a), int(b), t))
    nxt = {}
    for edges in count.values():
        if len(edges) == 1:
            a, b, t = edges[0]
            if a in nxt:
                raise ValueError('boundary vertex with two outgoing edges (a pinched mesh)')
            nxt[a] = (b, t)
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, a = [], start
        while a not in seen:
            seen.add(a)
            b, t = nxt[a]
            loop.append((a, b, t))
            a = b
        if a != start:
            raise ValueError('open boundary chain')
        loops.append(loop)
    return loops


def boundary_walls(nodes, tris, z, t):
    """Wall panels and rooftops of one layer's mesh (nodes (n, 2) m; z, t m: the copper's centre and thickness).
    Returns dict(ends (np, 2, 2): each panel's ends along the boundary; z0, z1; owner (np,): the boundary
    triangle beneath;
    roof (nr, 2): the two consecutive panels of each rooftop)."""
    nodes, tris = np.asarray(nodes, float), np.asarray(tris)
    ends, owner, roof = [], [], []
    for loop in boundary_loops(tris):
        first = len(ends)
        for a, b, tri in loop:
            ends.append((nodes[a], nodes[b]))
            owner.append(tri)
        n = len(loop)
        roof += [(first + k, first + (k + 1) % n) for k in range(n)]
    ends = np.array(ends)
    return dict(ends=ends, z0=z - t / 2, z1=z + t / 2, owner=np.array(owner), roof=np.array(roof, int))


def chain_walls(points, z, t, closed=False):
    """Walls along a polyline (points (n, 2) m), for tests and hand-built cases: as boundary_walls, no owners."""
    p = np.asarray(points, float)
    ends = np.stack([p[:-1], p[1:]], 1)
    if closed:
        ends = np.concatenate([ends, [[p[-1], p[0]]]])
    n = len(ends)
    roof = [(k, k + 1) for k in range(n - 1)] + ([(n - 1, 0)] if closed else [])
    return dict(ends=ends, z0=z - t / 2, z1=z + t / 2, owner=np.full(n, -1), roof=np.array(roof, int))


def _panel_triangles(w):
    """Each panel's two triangles (2 np, 3, 3) and the panel of each."""
    e, z0, z1 = w['ends'], w['z0'], w['z1']
    a0 = np.c_[e[:, 0], np.full(len(e), z0)]
    b0 = np.c_[e[:, 1], np.full(len(e), z0)]
    a1, b1 = a0 + [0, 0, z1 - z0], b0 + [0, 0, z1 - z0]
    T = np.concatenate([np.stack([a0, b0, b1], 1), np.stack([a0, b1, a1], 1)])
    return T, np.r_[np.arange(len(e)), np.arange(len(e))]


def _halves(w):
    """Each rooftop's two halves as affine currents f = M (r - v): (rooftop, panel, M (3, 3), v (3,))."""
    e, h = w['ends'], w['z1'] - w['z0']
    out = []
    for n, (k, j) in enumerate(w['roof']):
        for panel, rising in ((k, True), (j, False)):
            a, b = e[panel]
            d = np.r_[b - a, 0.0]
            L = np.linalg.norm(d)
            tau = d / L
            M = np.outer(tau, tau) / (L * h)
            if rising:                       # 0 at the panel's start, 1/h at its end (the shared vertex)
                out.append((n, panel, M, np.r_[a, w['z0']]))
            else:                            # 1/h at its start, 0 at its end
                out.append((n, panel, -M, np.r_[b, w['z0']]))
    return out


def _galerkin(obs, src, outer_levels):
    """sum over observation and source pieces: obs, src lists of (basis index, triangles (k, 3, 3), M, v)."""
    rows, cols, Vo, Vs, Mo, Ms, vo, vs = [], [], [], [], [], [], [], []
    for i, To, Ma, va in obs:
        for j, Ts, Mb, vb in src:
            for to in To:
                for ts in Ts:
                    rows.append(i); cols.append(j); Vo.append(to); Vs.append(ts)
                    Mo.append(Ma); Ms.append(Mb); vo.append(va); vs.append(vb)
    m = pair_moments(np.array(Vo), np.array(Vs), outer_levels=outer_levels, tensor=True)
    val = affine_pair(m, np.array(Mo), np.array(vo), np.array(Ms), np.array(vs))
    return np.array(rows), np.array(cols), MU0 / (4 * np.pi) * val


def _wall_pieces(w):
    T, panel = _panel_triangles(w)
    tri_of = {p: T[panel == p] for p in range(len(w['ends']))}
    return [(n, tri_of[p], M, v) for n, p, M, v in _halves(w)]


def wall_inductance(w, outer_levels=2):
    """The rooftops' partial inductance matrix (H), dense; symmetrised (observation and source swapped)."""
    pieces = _wall_pieces(w)
    rows, cols, val = _galerkin(pieces, pieces, outer_levels)
    n = len(w['roof'])
    L = np.zeros((n, n))
    np.add.at(L, (rows, cols), val)
    return 0.5 * (L + L.T)


def wall_rwg_inductance(w, rwg, z_tri, outer_levels=2):
    """Partial inductance (H) between the rooftops (rows) and a sheet's RWG functions (columns): rwg built on 2D
    nodes, z_tri each triangle's height (m). Averaged over both orders (rooftop observing, then RWG observing)."""
    walls = _wall_pieces(w)
    V = rwg.nodes[rwg.tris]
    tri3 = np.concatenate([V, np.asarray(z_tri, float)[:, None, None] * np.ones((1, 3, 1))], -1)
    sheet = []
    for e in range(rwg.Ne):
        for t, alpha, v in ((rwg.tp[e], rwg.l[e] / (2 * rwg.area[rwg.tp[e]]), rwg.vp[e]),
                            (rwg.tm[e], -rwg.l[e] / (2 * rwg.area[rwg.tm[e]]), rwg.vm[e])):
            sheet.append((e, tri3[t][None], alpha * np.eye(3), np.r_[v, z_tri[t]]))
    r1, c1, v1 = _galerkin(walls, sheet, outer_levels)
    r2, c2, v2 = _galerkin(sheet, walls, outer_levels)
    M = np.zeros((len(w['roof']), rwg.Ne))
    np.add.at(M, (r1, c1), 0.5 * v1)
    np.add.at(M, (c2, r2), 0.5 * v2)
    return M
