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


# ---- vectorized Galerkin for the solver -----------------------------------------------------------------------------
_B3 = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


def wall_pieces(walls):
    """Affine pieces of every rooftop of several wall sets (each a boundary_walls dict, with its own z0, z1):
    (basis index (P,), triangles (P, 3, 3), M (P, 3, 3), v (P, 3)), rooftops numbered across the sets in order."""
    idx, T, M, V = [], [], [], []
    off = 0
    for w in walls:
        tri, panel = _panel_triangles(w)
        for n, p, m, v in _halves(w):
            for t in tri[panel == p]:
                idx.append(off + n); T.append(t); M.append(m); V.append(v)
        off += len(w['roof'])
    return np.array(idx, int), np.array(T), np.array(M), np.array(V)


def rwg_pieces(rwg, z_tri):
    """Affine pieces of RWG functions on horizontal triangles (rwg on 2D nodes, z_tri m): f = alpha (r - v), alpha
    = +-l / 2A (unit normal current across the edge)."""
    V = rwg.nodes[rwg.tris]
    tri3 = np.concatenate([V, np.asarray(z_tri, float)[:, None, None] * np.ones((1, 3, 1))], -1)
    e = np.arange(rwg.Ne)
    t = np.r_[rwg.tp, rwg.tm]
    alpha = np.r_[rwg.l / (2 * rwg.area[rwg.tp]), -rwg.l / (2 * rwg.area[rwg.tm])]
    v = np.r_[rwg.vp, rwg.vm]
    v3 = np.c_[v, np.asarray(z_tri)[t]]
    return np.r_[e, e], tri3[t], alpha[:, None, None] * np.eye(3)[None], v3


def galerkin(obs, src, n_obs, n_src, near=3.0, outer_levels=2, chunk=(200, 3000)):
    """sum int int f_o . f_s / R over the pieces of each pair of bases (obs, src as from wall_pieces/rwg_pieces):
    (n_obs, n_src) dense. Pairs within `near` x their summed radii analytically (panels3d), the rest by a 3 x 3
    point product rule."""
    from scipy.sparse import csr_matrix
    io, To, Mo, vo = obs
    is_, Ts, Ms, vs = src
    co, cs = To.mean(1), Ts.mean(1)
    ro = np.linalg.norm(To - co[:, None], axis=2).max(1)
    rs = np.linalg.norm(Ts - cs[:, None], axis=2).max(1)
    ao = 0.5 * np.linalg.norm(np.cross(To[:, 1] - To[:, 0], To[:, 2] - To[:, 0]), axis=1)
    as_ = 0.5 * np.linalg.norm(np.cross(Ts[:, 1] - Ts[:, 0], Ts[:, 2] - Ts[:, 0]), axis=1)
    Po = np.einsum('qk,pkc->pqc', _B3, To)                       # (Po, 3, 3) points
    Ps = np.einsum('qk,pkc->pqc', _B3, Ts)
    Fo = np.einsum('pij,pqj->pqi', Mo, Po - vo[:, None]) * (ao / 3)[:, None, None]   # weighted f at the points
    Fs = np.einsum('pij,pqj->pqi', Ms, Ps - vs[:, None]) * (as_ / 3)[:, None, None]
    Bo = csr_matrix((np.ones(len(io)), (io, np.arange(len(io)))), shape=(n_obs, len(io)))
    Bs = csr_matrix((np.ones(len(is_)), (np.arange(len(is_)), is_)), shape=(len(is_), n_src))
    out = np.zeros((n_obs, n_src))
    for a in range(0, len(io), chunk[0]):
        sa = slice(a, min(len(io), a + chunk[0]))
        for b in range(0, len(is_), chunk[1]):
            sb = slice(b, min(len(is_), b + chunk[1]))
            d = Po[sa][:, None, :, None, :] - Ps[sb][None, :, None, :, :]        # (A, B, 3, 3, 3)
            R = np.sqrt((d * d).sum(-1))
            dot = np.einsum('aqi,bri->abqr', Fo[sa], Fs[sb])
            with np.errstate(divide='ignore', invalid='ignore'):  # coincident points: near pairs, replaced below
                val = (dot / R).sum((2, 3))
            dist = np.linalg.norm(co[sa][:, None] - cs[sb][None], axis=2)
            ii, jj = np.nonzero(dist < near * (ro[sa][:, None] + rs[sb][None]))
            if len(ii):
                gi, gj = ii + a, jj + b
                m = pair_moments(To[gi], Ts[gj], outer_levels=outer_levels, tensor=True)
                val[ii, jj] = affine_pair(m, Mo[gi], vo[gi], Ms[gj], vs[gj])
            out += Bo[:, sa] @ (Bs[sb].T @ val.T).T
    return MU0 / (4 * np.pi) * out


def rooftop_gram(walls):
    """int f_m . f_n dS of the rooftops (sparse, square): a rooftop's two halves L/(3h) each, two rooftops on one
    panel L/(6h)."""
    from scipy.sparse import coo_matrix
    rows, cols, vals, off = [], [], [], 0
    for w in walls:
        h = w['z1'] - w['z0']
        L = np.linalg.norm(w['ends'][:, 1] - w['ends'][:, 0], axis=1)
        roof = w['roof']
        for n, (k, j) in enumerate(roof):
            rows.append(off + n); cols.append(off + n); vals.append((L[k] + L[j]) / (3 * h))
        first = {k: n for n, (k, j) in enumerate(roof)}               # the rooftop rising on panel k ...
        for n, (k, j) in enumerate(roof):                              # ... and the one falling on it (n): share j
            m = first.get(j)
            if m is not None:
                rows += [off + n, off + m]; cols += [off + m, off + n]; vals += [L[j] / (6 * h)] * 2
        off += len(roof)
    return coo_matrix((vals, (rows, cols)), shape=(off, off)).tocsr()
