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

Beyond the films (R-E13): a part may have its own path to the air (p.air {part: K/W}: its body's, or a heatsink's
on it), which makes it a node of its own, joined to its pads through its theta and to the air through that, so its
heat splits between the two; a footprint may be held at a temperature (p.hold(board, {ref: C}): bolted to a cold
plate or a chassis), its pads tied to it. forced_film gives the faces' coefficient in moving air. grid.py also
gives the heating from cold (its transient), from the board's own heat capacity.

What this leaves out: air warmed by one part reaching another, any change of the film with temperature, and the
heat the parts' bodies hold (a transient's first seconds). The first makes it read high, never low.
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
RADIATION = 6.0         # W/m2/K of FILM: radiation off a board near 50 C (emissivity 0.9); the rest is convection
HOLD = 1e4              # W/K: a held footprint's tie to its temperature (a board's square of copper: about 0.01)
RHO_C_CU = 3.45e6       # J/m3/K: copper's heat capacity per volume
RHO_C_FR4 = 2.0e6       # J/m3/K: FR-4's (1850 kg/m3, 1100 J/kg/K)


def forced_film(speed, length_mm, still=FILM):
    """W/m2/K off a face in air moving at `speed` m/s along `length_mm` of board: radiation (RADIATION) and the
    convection of a laminar flat plate, 3.84 sqrt(v / L) (Nu = 0.664 Re^1/2 Pr^1/3, air near 40 C), joined to the
    still air's (`still` less its radiation) as assisting flows join, (h_n^3 + h_f^3)^(1/3)."""
    if speed < 0 or length_mm <= 0:
        raise ValueError('the air speed cannot be negative and the length must be positive')
    natural = max(still - RADIATION, 0.0)
    forced = 3.84 * np.sqrt(speed / (length_mm * 1e-3))
    return float(RADIATION + (natural ** 3 + forced ** 3) ** (1 / 3))


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
class Held:
    name: str               # the footprint
    layer: int
    geom: object            # shapely, mm
    temperature: float      # C


@dataclass
class Problem:
    outline: object                         # shapely, mm
    layers: list                            # [Layer], bottom first
    gaps: list                              # W/m2/K between layers[i] and layers[i + 1]
    vias: list                              # [dict(x, y, r, lo, hi)]: layer indices it joins
    sources: list = field(default_factory=list)
    powers: dict = field(default_factory=dict)      # {part: W}
    theta: dict = field(default_factory=dict)       # {part: K/W, its pads to the part}
    tmax: dict = field(default_factory=dict)        # {part: C, the most it may reach (its junction's limit)}
    air: dict = field(default_factory=dict)         # {part: K/W, from the part itself to the air: its body, a sink}
    held: list = field(default_factory=list)        # [Held]: pads tied to a temperature (Problem.hold)
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

    def hold(self, board, temperatures):
        """Each footprint's pads {reference: C} held at that temperature (a cold plate, a chassis screw)."""
        from ..pdn.rail import parts_of
        parts = {p.ref: p for p in parts_of(board)}
        names = [l.name for l in self.layers]
        for ref, c in temperatures.items():
            if ref not in parts:
                self.notes.append(f'{ref}: not a footprint of the board; not held at {c:g} C')
                continue
            pads = [(lay, pad.geom.intersection(self.outline)) for pad in parts[ref].pads for lay in pad.layers]
            pads = [(names.index(lay), g) for lay, g in pads if lay in names and g.area > 0]
            if not pads:
                self.notes.append(f'{ref}: no pad on a copper layer to hold at {c:g} C')
            self.held += [Held(ref, i, g, float(c)) for i, g in pads]

    def extend(self, A, q, covers, spot):
        """The surroundings beyond the films added to a solver's system A x = q (x: the rise over ambient of its n
        unknowns): each part of self.air a node of its own (n, n + 1, ...), its heat put there rather than on its
        pads, joined to its pads through its theta (by their shares of its heat) and to the air through its own
        resistance; each held footprint's pads tied to their temperature by HOLD. covers {id(source): (unknowns,
        weights summing to 1) or None}: where the solver put each source's heat; spot(layer, geom) -> the same for
        a held pad. Returns (A, q, read): read(x) -> dict(parts={part: rise of its node}, air_W={part: W},
        held_W={footprint: W out through it})."""
        from scipy.sparse import coo_matrix
        n = A.shape[0]
        own = [name for name in self.air if name in self.powers]
        for name in self.air:
            if name not in self.powers:
                self.notes.append(f'{name}: a path to the air but no loss on the board; left out')
        rows, cols, vals = [], [], []
        q = np.r_[q, np.zeros(len(own))]
        diag = np.zeros(n + len(own))

        def link(a, b, g):
            rows.append(np.r_[a, b, a, b]); cols.append(np.r_[a, b, b, a]); vals.append(np.r_[g, g, -g, -g])
        for j, name in enumerate(own):
            k = n + j
            total = self.powers[name]
            theta = max(float(self.theta.get(name) or 0.0), 1e-3)  # (none given: the part is at its pads')
            for s in self.sources:
                if s.name != name or covers.get(id(s)) is None:
                    continue
                idx, w = covers[id(s)]
                np.subtract.at(q, idx, s.watts * w)
                link(np.full(len(idx), k), idx, (s.watts / total) * w / theta)
            q[k] += total
            r = float(self.air[name])
            if r <= 0:
                raise ValueError(f'{name}: its path to the air must be a positive K/W')
            diag[k] += 1.0 / r
        ties = []
        for h in self.held:
            idx, w = spot(h.layer, h.geom)
            if len(idx):
                np.add.at(diag, idx, HOLD * w)
                np.add.at(q, idx, HOLD * w * (h.temperature - self.ambient))
                ties.append((h, idx, w))
        m = n + len(own)
        A = A.tocoo()
        B = coo_matrix((np.concatenate([A.data, *vals, diag]),
                        (np.concatenate([A.row, *rows, np.arange(m)]), np.concatenate([A.col, *cols, np.arange(m)]))),
                       shape=(m, m)).tocsc()

        def read(x):
            held = {}
            for h, idx, w in ties:
                held[h.name] = held.get(h.name, 0.0) + float(HOLD * w @ (x[idx] - (h.temperature - self.ambient)))
            return dict(parts={name: float(x[n + j]) for j, name in enumerate(own)},
                        air_W={name: float(x[n + j]) / float(self.air[name]) for j, name in enumerate(own)},
                        held_W=held)
        return B, q, read

    @property
    def watts(self):
        return float(sum(s.watts for s in self.sources))


def factor_spd(A):
    """The factor of the symmetric positive definite A (sparse): SuperLU ordered for a symmetric matrix, which on
    these sheets is two to three times quicker than its default ordering."""
    from scipy.sparse.linalg import splu
    return splu(A.tocsc(), permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))


def solve_spd(A, q):
    """A x = q for the symmetric positive definite A (sparse), by factor_spd."""
    return factor_spd(A).solve(q)


def losses_of(case, model):
    """A circuit run's losses as heat per footprint: ({footprint reference: W}, [what was left out and why]).
    `case` is one case of circuit.json, `model` the circuit model it ran. A source and a load are not on the board;
    nor is a part bound to a connector (a transformer on flying leads), one board.parts puts off it (null), or one
    whose pins are on two footprints (a winding between two test points: its loss is its own, not either's). Parts
    bound to one footprint (the windings of a transformer; a part with one pin off the board) are summed.

    A coupled winding's V·I is the power it passes to the others as well as its loss: its own loss is its
    resistances' (loss_W, which the run measures). A switch's gate drive is dissipated in its gate resistance:
    counted on the switch. An inductor's losses worked out after the run (added_loss_W: circuit/windings.py) are
    its too."""
    bound = (model.get('board') or {}).get('parts', {})
    kinds = {p['ref']: p.get('type') for p in model.get('parts', [])}
    parts = case.get('parts', {})

    def footprints(ref):
        given = bound.get(ref, ref)
        return sorted({f for f in (given if isinstance(given, list) else [given]) if f})
    out, left = {}, []
    for ref, st in parts.items():
        w = float(st.get('power_W', 0.0)) + float(st.get('gate_power_W', 0.0))
        names = footprints(ref)
        if st.get('coupled_with'):
            if 'loss_W' in st:
                w = float(st['loss_W'])
            elif any(footprints(r) != names for r in st['coupled_with']):     # (a run before loss_W: only a group
                left.append(f"{ref}: a coupled winding whose own loss the run did not measure (its V·I is the "  # on
                            "power it passes on too): run the circuit again")       # one footprint sums to its loss)
                continue
        w += float(st.get('added_loss_W', 0.0))                    # an inductor's AC and core losses (R-E10)
        if kinds.get(ref) in ('source', 'load'):
            left.append(f'{ref}: a {kinds[ref]}, not on the board')
        elif not names:
            left.append(f'{ref}: not on the board (board.parts)')
        elif len(names) > 1:
            left.append(f"{ref}: its pins on {' and '.join(names)}, so its loss is not one footprint's")
        elif ref in bound and names[0].rstrip('0123456789').upper() in OFF_BOARD:
            left.append(f'{ref}: at connector {names[0]}, not on the board')
        else:
            out[names[0]] = out.get(names[0], 0.0) + w
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
    extra: dict = None              # Problem.extend's read: the parts' own paths to the air, the held footprints
    transient: dict = None          # grid.py's heating from cold, when asked

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
        # on the mesh's own triangles (the solution is linear on each): a point's triangle and its weights on the
        # corners. (It was a Delaunay triangulation of the nodes, made again at every call and masked by the
        # outline: 1.7 s a layer on 60,000 nodes, and not the solution's triangles where the mesh is not Delaunay.)
        from ..kernels import tri_locate
        index, weight = tri_locate(self.nodes, self.tris, np.c_[x.ravel(), y.ravel()])
        out = np.full(len(index), np.nan)
        ok = index >= 0
        out[ok] = (self.T[i][self.tris[index[ok]]] * weight[ok]).sum(axis=1)
        return out.reshape(x.shape)

    def summary(self):
        p = self.problem
        return dict(method=self.method, size_mm=self.size, unknowns=self.unknowns,
                    seconds={k: round(v, 4) for k, v in self.seconds.items()},
                    ambient_C=p.ambient, film_W_m2K=list(p.film), edge_film_W_m2K=p.edge_film,
                    heat_in_W=p.watts, heat_out_W=self.out_W, heat_out_edges_W=self.edge_W,
                    **({} if not self.extra else dict(
                        heat_out_parts_air_W=float(sum(self.extra['air_W'].values())),
                        held={h: dict(C=next(x.temperature for x in p.held if x.name == h), out_W=w)
                              for h, w in self.extra['held_W'].items()})),
                    hottest_C=self.hottest,
                    layers={l.name: dict(hottest_C=float(np.nanmax(t)), mean_C=float(np.nanmean(t)))
                            for l, t in zip(p.layers, self.T)},
                    parts=self.parts, notes=p.notes + self.notes)


def part_temperatures(problem, board_of, extra=None):
    """{part: dict(power_W, board_C, part_C)} from board_of(source) -> the mean temperature under that source;
    extra (Problem.extend's read): a part with its own path to the air at its node's temperature, with the heat
    that path takes (to_air_W)."""
    acc = {}
    for s in problem.sources:
        a = acc.setdefault(s.name, [0.0, 0.0])
        a[0] += s.watts * board_of(s)
        a[1] += s.watts
    out = {}
    for name, (tw, w) in acc.items():
        tb = tw / w
        part = float(tb + problem.powers[name] * problem.theta.get(name, 0.0))
        out[name] = dict(power_W=problem.powers[name], board_C=float(tb), part_C=part,
                         theta_K_W=problem.theta.get(name))     # None: part_C is its pads' (no theta given)
        if extra and name in extra['parts']:
            part = problem.ambient + extra['parts'][name]
            out[name].update(part_C=float(part), to_air_W=extra['air_W'][name], air_K_W=float(problem.air[name]))
        if name in problem.tmax:
            out[name].update(tmax_C=float(problem.tmax[name]), margin_K=float(problem.tmax[name] - part))
    return out
