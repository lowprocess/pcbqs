"""
The steady temperature of a board as one problem that both solvers (grid.py, mesh.py) take.

The board is a stack of sheets, one per copper layer. A sheet conducts along the board: its copper (k x thickness
where there is copper) and its share of the laminate (half of the dielectric on each side of it). Sheets next to
each other exchange heat through the dielectric between them (k_z / its thickness, per area) and through the vias
(the plated barrel). The two outer sheets lose heat to the air through a film coefficient (convection and
radiation together, W/m2/K), and every sheet through the board's edges (the outline and its cut-outs) over the
height of the board it stands for, with the edge's own coefficient. Heat comes in over the pads of the parts that dissipate: each part's loss, shared
over its pads by their area.

    p = Problem.of(board, {'Q1': 0.35, 'D1': 0.5})      # board.json's dict (mm), watts per footprint
    p.theta = {'Q1': 2.0}                               # K/W from each part's pads to the part itself (optional)

A part's temperature is the mean temperature of the copper under its pads plus its loss times its theta.

What this leaves out: the heat the bodies of the parts give to the air themselves, air warmed by one part reaching
another, and any change of the film with temperature. The first two make it read high, never low.
"""
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import shape, box
from shapely.ops import unary_union

K_CU = 390.0            # W/m/K, electro-deposited copper
K_XY = 0.8              # W/m/K, FR-4 along the weave
K_Z = 0.3               # W/m/K, FR-4 through it
FILM = 10.0             # W/m2/K, still air and radiation off a board near 50 C
AMBIENT = 25.0          # C
PLATING = 0.025         # mm, via barrel
OFF_BOARD = ('J', 'P', 'X', 'TP', 'CN')     # reference letters of connectors: what is bound to one is off the board


@dataclass
class Layer:
    name: str
    z: float                # mm
    t: float                # mm of copper
    copper: object          # shapely, mm (every net), or None
    laminate: float         # W/K per square: this sheet's share of the laminate's conduction along the board
    edge: float = 0.0       # mm: the height of the board's edge this sheet stands for (its share of the thickness)


@dataclass
class Source:
    name: str               # the part
    layer: int              # index into Problem.layers
    geom: object            # shapely, mm
    watts: float


@dataclass
class Problem:
    outline: object                         # shapely, mm
    layers: list                            # [Layer], bottom first
    gaps: list                              # W/m2/K between layers[i] and layers[i + 1]
    vias: list                              # [dict(x, y, r, lo, hi)]: layer indices it joins
    sources: list = field(default_factory=list)
    powers: dict = field(default_factory=dict)      # {part: W}
    theta: dict = field(default_factory=dict)       # {part: K/W, its pads to the part}
    film: tuple = (FILM, FILM)              # W/m2/K: bottom face, top face
    edge_film: float = FILM                 # W/m2/K: the board's edges (0: insulated)
    ambient: float = AMBIENT
    k_cu: float = K_CU
    plating: float = PLATING
    notes: list = field(default_factory=list)

    @classmethod
    def of(cls, board, losses=None, film=FILM, ambient=AMBIENT, k_cu=K_CU, k_xy=K_XY, k_z=K_Z, plating=PLATING,
           theta=None, edge_film=None):
        """The problem of a board (board.json's dict) with `losses` {footprint reference: W}. film: one value for
        both faces or (bottom, top). edge_film: the edges' coefficient (default: the mean of the faces'; 0: none)."""
        cu = sorted(board['stack']['copper'], key=lambda c: c['z'])
        if not cu:
            raise ValueError('the board has no copper layer')
        zs = [c['z'] for c in cu]
        thick = float(board['stack'].get('thickness') or (zs[-1] - zs[0]) or 1.6)
        lam = np.zeros(len(cu))
        for i in range(len(cu) - 1):
            d = zs[i + 1] - zs[i]
            lam[i] += 0.5 * d
            lam[i + 1] += 0.5 * d
        spare = max(thick - (zs[-1] - zs[0]), 0.0)          # dielectric outside the copper (one layer: all of it)
        lam[0] += 0.5 * spare
        lam[-1] += 0.5 * spare
        layers = []
        for c, l in zip(cu, lam):
            gs = [shape(q['polygon']) for q in board['copper'].get(c['name'], [])]
            g = unary_union(gs).buffer(0) if gs else None
            layers.append(Layer(c['name'], c['z'], c['t'], None if g is None or g.is_empty else g,
                                k_xy * l * 1e-3, float(l)))
        out = board.get('outline')
        outline = shape(out).buffer(0) if out else None
        if outline is None or outline.is_empty:
            b = unary_union([l.copper for l in layers if l.copper is not None]).bounds
            outline = box(*b)
        gaps = [k_z / ((zs[i + 1] - zs[i]) * 1e-3) for i in range(len(cu) - 1)]
        vias = []
        for v in board.get('vias', []):
            on = [i for i, z in enumerate(zs) if v['z0'] - 1e-6 <= z <= v['z1'] + 1e-6]
            if len(on) > 1 and outline.contains(shape({'type': 'Point', 'coordinates': (v['x'], v['y'])})):
                vias.append(dict(x=v['x'], y=v['y'], r=v['r'], lo=on[0], hi=on[-1]))
        film = (film, film) if np.isscalar(film) else tuple(film)
        edge_film = 0.5 * (film[0] + film[1]) if edge_film is None else float(edge_film)
        if edge_film < 0 or min(film) < 0:
            raise ValueError('a film coefficient cannot be negative')
        p = cls(outline, layers, gaps, vias, film=film, edge_film=edge_film, ambient=ambient, k_cu=k_cu,
                plating=plating, theta=dict(theta or {}))
        if losses:
            p.dissipate(board, losses)
        return p

    def edges(self):
        """The board's edges as segments: (a, b) arrays (n, 2) mm, the board on the left going from a to b (the
        outline anticlockwise, its cut-outs clockwise)."""
        from shapely.geometry.polygon import orient
        A, B = [], []
        for q in getattr(self.outline, 'geoms', [self.outline]):
            if q.geom_type != 'Polygon' or q.is_empty:
                continue
            q = orient(q, 1.0)
            for ring in [q.exterior, *q.interiors]:
                c = np.asarray(ring.coords)[:, :2]
                A.append(c[:-1]); B.append(c[1:])
        if not A:
            return np.zeros((0, 2)), np.zeros((0, 2))
        return np.concatenate(A), np.concatenate(B)

    def edge_conductance(self, layer):
        """W/K per metre of edge from sheet `layer` to the air."""
        return self.edge_film * self.layers[layer].edge * 1e-3

    def index(self, name):
        return [l.name for l in self.layers].index(name)

    def barrel(self, via, gap):
        """A via's conductance across gap `gap` (between layers gap and gap + 1), W/K."""
        dz = (self.layers[gap + 1].z - self.layers[gap].z) * 1e-3
        ri = max(via['r'] - self.plating, 0.0)
        return self.k_cu * np.pi * (via['r'] ** 2 - ri ** 2) * 1e-6 / dz

    def heat(self, name, pads, watts):
        """`watts` into the board over `pads` [(layer name, shapely polygon mm)], shared by area."""
        pads = [(self.index(lay), g) for lay, g in pads if lay in [l.name for l in self.layers] and g.area > 0]
        total = sum(g.area for _, g in pads)
        if not pads or total <= 0:
            self.notes.append(f'{name}: no pad on a copper layer; its {watts:g} W left out')
            return
        for i, g in pads:
            self.sources.append(Source(name, i, g, watts * g.area / total))
        self.powers[name] = self.powers.get(name, 0.0) + watts

    def dissipate(self, board, losses):
        """Each footprint's loss {reference: W} over its pads."""
        from ..pdn.rail import parts_of
        parts = {p.ref: p for p in parts_of(board)}
        for ref, w in losses.items():
            if ref not in parts:
                self.notes.append(f'{ref}: not a footprint of the board; its {w:g} W left out')
                continue
            if w <= 0:
                continue
            pads = [(lay, pad.geom.intersection(self.outline)) for pad in parts[ref].pads for lay in pad.layers]
            self.heat(ref, pads, float(w))

    @property
    def watts(self):
        return float(sum(s.watts for s in self.sources))


def solve_spd(A, q):
    """A x = q for the symmetric positive definite A (sparse): SuperLU ordered for a symmetric matrix, which on these
    sheets is two to three times quicker than its default ordering."""
    from scipy.sparse.linalg import splu
    return splu(A.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options=dict(SymmetricMode=True)).solve(q)


def losses_of(case, model):
    """A circuit run's losses as heat per footprint: ({footprint reference: W}, [what was left out and why]).
    `case` is one case of circuit.json, `model` the circuit model it ran. A source and a load are not on the board;
    nor is a part bound to a connector (a transformer on flying leads). Parts bound to one footprint (the windings
    of a transformer) are summed."""
    bound = (model.get('board') or {}).get('parts', {})
    kinds = {p['ref']: p.get('type') for p in model.get('parts', [])}
    out, left = {}, []
    for ref, st in case.get('parts', {}).items():
        w = float(st.get('power_W', 0.0))
        fp = bound.get(ref, ref)
        if kinds.get(ref) in ('source', 'load'):
            left.append(f'{ref}: a {kinds[ref]}, not on the board')
        elif ref in bound and fp.rstrip('0123456789').upper() in OFF_BOARD:
            left.append(f'{ref}: at connector {fp}, not on the board')
        else:
            out[fp] = out.get(fp, 0.0) + w
    for fp in [f for f, w in out.items() if w <= 0]:
        left.append(f'{fp}: no loss')
        del out[fp]
    return out, left


@dataclass
class Result:
    """Temperatures, C. grid: T (layers, ny, nx), NaN off the board, cell centres at x0 + (i + 0.5) pitch.
    mesh: nodes (N, 2) mm, tris, T (layers, N)."""
    method: str
    problem: Problem
    T: np.ndarray
    parts: dict                     # {part: dict(power_W, board_C, part_C)}
    unknowns: int
    seconds: dict                   # build, solve
    out_W: float                    # the heat leaving through the films (faces and edges)
    size: float                     # mm: the grid's pitch or the mesh's longest edge
    x0: float = 0.0
    y0: float = 0.0
    nodes: np.ndarray = None
    tris: np.ndarray = None
    notes: list = field(default_factory=list)
    edge_W: float = 0.0             # of out_W, the heat leaving through the edges

    @property
    def hottest(self):
        return float(np.nanmax(self.T))

    def sample(self, layer, x, y):
        """The temperature of layer `layer` (index or name) at points x, y (mm); NaN off the board."""
        i = layer if isinstance(layer, int) else self.problem.index(layer)
        x, y = np.asarray(x, float), np.asarray(y, float)
        if self.method == 'grid':
            T = self.T[i]
            ny, nx = T.shape
            u, v = (x - self.x0) / self.size - 0.5, (y - self.y0) / self.size - 0.5     # in cell centres
            c, r = np.floor(u).astype(int), np.floor(v).astype(int)
            fu, fv = u - c, v - r
            pad = np.full((ny + 2, nx + 2), np.nan)
            pad[1:-1, 1:-1] = T
            ok = (c >= -1) & (c < nx) & (r >= -1) & (r < ny)
            c, r = np.clip(c, -1, nx - 1) + 1, np.clip(r, -1, ny - 1) + 1
            q = np.stack([pad[r, c], pad[r, c + 1], pad[r + 1, c], pad[r + 1, c + 1]])
            w = np.stack([(1 - fu) * (1 - fv), fu * (1 - fv), (1 - fu) * fv, fu * fv])
            w = np.where(np.isnan(q), 0.0, w)                   # at the board's edge: the cells there are
            tot = w.sum(axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                out = np.where(tot > 0, (w * np.nan_to_num(q)).sum(axis=0) / tot, np.nan)
            return np.where(ok, out, np.nan)
        from scipy.interpolate import LinearNDInterpolator
        import shapely
        f = LinearNDInterpolator(self.nodes, self.T[i])
        out = f(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)
        out[~shapely.contains_xy(self.problem.outline, x, y)] = np.nan
        return out

    def summary(self):
        p = self.problem
        return dict(method=self.method, size_mm=self.size, unknowns=self.unknowns,
                    seconds={k: round(v, 4) for k, v in self.seconds.items()},
                    ambient_C=p.ambient, film_W_m2K=list(p.film), edge_film_W_m2K=p.edge_film,
                    heat_in_W=p.watts, heat_out_W=self.out_W, heat_out_edges_W=self.edge_W,
                    hottest_C=self.hottest,
                    layers={l.name: dict(hottest_C=float(np.nanmax(t)), mean_C=float(np.nanmean(t)))
                            for l, t in zip(p.layers, self.T)},
                    parts=self.parts, notes=p.notes + self.notes)


def part_temperatures(problem, board_of):
    """{part: dict(power_W, board_C, part_C)} from board_of(source) -> the mean temperature under that source."""
    acc = {}
    for s in problem.sources:
        a = acc.setdefault(s.name, [0.0, 0.0])
        a[0] += s.watts * board_of(s)
        a[1] += s.watts
    out = {}
    for name, (tw, w) in acc.items():
        tb = tw / w
        out[name] = dict(power_W=problem.powers[name], board_C=float(tb),
                         part_C=float(tb + problem.powers[name] * problem.theta.get(name, 0.0)))
    return out
