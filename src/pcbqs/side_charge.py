"""Charge on the copper's sides (capacitance with face sheets; experimental, dense, homogeneous dielectric).

Two face sheets (thickness_sheets=2) give thick copper's charge its two faces, but not its sides: between two traces
0.1 mm apart the facing sides are a small parallel-plate capacitor, and without them the mutual capacitance was 30%
low against FasterCap's copper boxes. Here every boundary edge of a layer's face mesh stands up as a vertical panel
between the two faces, carrying a constant charge density, at its boundary triangle's conductor potential (no
current: these panels add charge unknowns, not circuit branches).

    panels = side_panels(model)                    # per panel: two triangles (m), area, owning triangle
    Ptt, Ptp, Ppp = side_moments(model, panels)    # 1/R moments: triangle x panel, panel x panel (m^3)
"""
import numpy as np

from .panels3d import pair_moments

_B3 = np.array([[2 / 3, 1 / 6, 1 / 6], [1 / 6, 2 / 3, 1 / 6], [1 / 6, 1 / 6, 2 / 3]])


def boundary_loops(tris):
    """The mesh's boundary as closed loops of directed edges [(a, b, triangle)] in each triangle's own order."""
    count = {}
    for t, tri in enumerate(tris):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            count.setdefault((min(a, b), max(a, b)), []).append((int(a), int(b), t))
    nxt = {e[0][0]: (e[0][1], e[0][2]) for e in count.values() if len(e) == 1}
    loops, seen = [], set()
    for start in nxt:
        if start in seen:
            continue
        loop, a = [], start
        while a not in seen and a in nxt:
            seen.add(a)
            b, t = nxt[a]
            loop.append((a, b, t))
            a = b
        loops.append(loop)
    return loops


def side_panels(model):
    """The side panels of every face-sheet layer: dict(tris (2P, 3, 3) m, panel (2P,) index, area (P,) m^2, owner
    (P,) the triangle each stands on). A layer's bottom-face mesh gives the boundary; the panel spans its two faces."""
    T, idx, area, owner = [], [], [], []
    for lay in dict.fromkeys(model.sheet_layer.values()):
        top = model.layer_tris.get(f'{lay}#1')
        if top is None:
            continue
        bottom = model.layer_tris[lay]
        z0, z1 = float(model.z_of[bottom[0]]), float(model.z_of[top[0]])
        tris = model.tris[bottom]
        for loop in boundary_loops(tris):
            for a, b, t in loop:
                pa, pb = model.nodes[a], model.nodes[b]
                a0, b0 = np.r_[pa, z0], np.r_[pb, z0]
                a1, b1 = np.r_[pa, z1], np.r_[pb, z1]
                k = len(area)
                T += [np.stack([a0, b0, b1]), np.stack([a0, b1, a1])]
                idx += [k, k]
                area.append(np.linalg.norm(pb - pa) * (z1 - z0))
                owner.append(int(bottom[t]))
    return dict(tris=np.array(T), panel=np.array(idx, int), area=np.array(area), owner=np.array(owner, int))


def _scalar_moments(To, Ts, near=3.0, chunk=(400, 4000)):
    """int int 1/R over every pair (To[i], Ts[j]) (m^4/m = m^3): near pairs analytic (panels3d), far ones 3 x 3
    points. Returns (len(To), len(Ts))."""
    co, cs = To.mean(1), Ts.mean(1)
    ro = np.linalg.norm(To - co[:, None], axis=2).max(1)
    rs = np.linalg.norm(Ts - cs[:, None], axis=2).max(1)
    ao = 0.5 * np.linalg.norm(np.cross(To[:, 1] - To[:, 0], To[:, 2] - To[:, 0]), axis=1)
    as_ = 0.5 * np.linalg.norm(np.cross(Ts[:, 1] - Ts[:, 0], Ts[:, 2] - Ts[:, 0]), axis=1)
    Po = np.einsum('qk,pkc->pqc', _B3, To)
    Ps = np.einsum('qk,pkc->pqc', _B3, Ts)
    out = np.zeros((len(To), len(Ts)))
    for a in range(0, len(To), chunk[0]):
        sa = slice(a, min(len(To), a + chunk[0]))
        for b in range(0, len(Ts), chunk[1]):
            sb = slice(b, min(len(Ts), b + chunk[1]))
            d = Po[sa][:, None, :, None, :] - Ps[sb][None, :, None, :, :]
            with np.errstate(divide='ignore'):
                val = (1 / np.sqrt((d * d).sum(-1))).sum((2, 3)) * (ao[sa] / 3)[:, None] * (as_[sb] / 3)[None]
            ii, jj = np.nonzero(np.linalg.norm(co[sa][:, None] - cs[sb][None], axis=2)
                                < near * (ro[sa][:, None] + rs[sb][None]))
            if len(ii):
                val[ii, jj] = pair_moments(To[sa][ii], Ts[sb][jj], outer_levels=2, scalar_only=True)
            out[sa, sb] = val
    return out


def side_moments(model, panels):
    """1/R moments, triangle x panel-triangle (Nt, 2P) and panel-triangle x panel-triangle (2P, 2P)."""
    V = model.nodes[model.tris]
    tri3 = np.concatenate([V, model.z_of[:, None, None] * np.ones((1, 3, 1))], -1)
    Mtp = _scalar_moments(tri3, panels['tris'])
    Mpp = _scalar_moments(panels['tris'], panels['tris'])
    return Mtp, 0.5 * (Mpp + Mpp.T)
