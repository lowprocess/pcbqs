"""
DC voltage drop (IR drop) of a power rail and its return: the steady conduction in the copper, each layer's copper a
resistive sheet (sigma x thickness, linear triangles on pcbqs's mesh), the vias plated barrels between the layers
they join, the pads equipotential terminals: the source's held at its voltage, each load's drawing its current.

    rail = Rail(board, '+3V3', loads={'U21': 2.0, 'U14': 1.0})       # rail.py: the tree, the source, the parts
    res = solve_rail(rail, volts=3.3, dmax=1.0)
    res['loads']['U21']          # dict(V: what it sees, its supply pads' minus its ground pads', drop, power_drop,
                                 #      ground_rise)
    res['power'].sheets          # {layer: Sheet with V (node volts), J (A per mm of width per triangle)}
    res['power'].vias            # [dict(x, y, r, I, segments)]; res['series']: each series part's current and drop

The network (Network) is the copper of a set of nets (a rail tree, or the ground); a terminal is a set of pads
([(layer, polygon)]: a through-hole pad on every layer it is plated through), meshed with its outline as mesh edges
and held to one voltage (its vertices tied to one junction); a series part (a sense resistor, a bead, a jumper) is
a resistance between its two pads' terminals. A via joins the copper of the nets on every layer between its ends that
it lies on, through a barrel of plating thickness `plating` (mm): rho dz / area.

solve(board, net, source, loads) is the one-net form: source dict(x, y, r, layer, V), loads [dict(name, x, y, r,
layer, I)], each a disc pad; ground=True solves a return (the source at 0 V, the loads' currents put back).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from shapely.geometry import Point
from shapely.ops import unary_union

from .mesh import net_sheets, stiffness, gradient, stack_z, SIGMA_CU
# S, from a terminal's or a via end's junction to EACH of its vertices. Per vertex, not shared: current enters a
# pad through the few vertices on its edge, and a total of 1e6 S split over thousands of vertices left a pad
# ~20 uV from equipotential per amp (+1% on a 40 mm strip, growing with mesh refinement; validation/dc_elmer.py).
TIE_G = 1e6


def _barrel_g(r_mm, dz_mm, plating_mm, sigma):
    """A via barrel's conductance, S: sigma x its plating's cross-section / its length."""
    ri = max(r_mm - plating_mm, 0.0)
    return sigma * np.pi * (r_mm ** 2 - ri ** 2) * 1e-6 / (dz_mm * 1e-3)


class Network:
    """The conduction network of the copper of `nets` with its terminals {name: [(layer, polygon)]}. fine: the
    names of terminals meshed finer around (within `fine_r` mm, at dmax_fine), the current crowding into them."""

    def __init__(self, board, nets, terminals, dmax=1.0, sigma=SIGMA_CU, plating=0.025, fine=(), fine_r=2.0,
                 dmax_fine=None, log=print, quality=25):
        self.board, self.nets = board, {nets} if isinstance(nets, str) else set(nets)
        self.sigma, self.terms = sigma, dict(terminals)
        layers = list(stack_z(board))
        self.vias = [v for v in board['vias'] if v.get('net') in self.nets]
        regions, self.tkey = {}, {}                 # tkey: terminal -> [(layer, index in that layer's regions)]
        for name, pads in self.terms.items():
            for lay, poly in pads:
                if lay not in layers:
                    continue
                regions.setdefault(lay, []).append(poly)
                self.tkey.setdefault(name, []).append((lay, len(regions[lay]) - 1))
        fz = [poly.buffer(fine_r) for n in fine for _, poly in self.terms.get(n, [])]
        fzone = unary_union(fz) if fz else None
        pts = [(v['x'], v['y'], v['r']) for v in self.vias]
        # quality: a minimum angle, so the cells grow out from each via's ring instead of reaching from it to far
        # vertices in slivers (which underestimated every via junction's spreading: validation/dc_elmer.py)
        self.mesh_info = {}
        self.sheets = net_sheets(board, self.nets, pts, dmax=dmax, regions=regions, fine=fzone,
                                 dmax_fine=dmax_fine or (dmax / 4 if fzone is not None else None), quality=quality,
                                 info=self.mesh_info)
        if not self.sheets:
            raise ValueError(f"no copper of {', '.join(sorted(self.nets))}")
        off = 0
        for sh in self.sheets.values():
            sh.off = off
            off += len(sh.nodes)
        self.n_mesh = off
        rows, cols, vals = [], [], []

        def link(a, b, g):
            """Conductance g between dof a and each of b (one dof, or an array of them)."""
            a, b = np.broadcast_arrays(np.asarray(a, int).ravel(), np.asarray(b, int).ravel())
            g = np.full(len(a), float(g))
            rows.append(np.r_[a, b, a, b]); cols.append(np.r_[a, b, b, a]); vals.append(np.r_[g, g, -g, -g])
        self._link = link
        blocks = []
        for sh in self.sheets.values():
            K = stiffness(sh.nodes, sh.tris, sigma * sh.t * 1e-3).tocoo()
            blocks.append((K.row + sh.off, K.col + sh.off, K.data))
        self.blocks = blocks
        j = off
        self.via_segs = []                      # per via: [(layer_a, layer_b, junction_a, junction_b, g)]
        for k, v in enumerate(self.vias):
            on = sorted([(sh.z, lay) for lay, sh in self.sheets.items() if sh.at(k) and
                         v['z0'] - 1e-6 <= sh.z <= v['z1'] + 1e-6])
            js = []
            for z, lay in on:
                sh = self.sheets[lay]
                ring = sh.at(k)[1:] or sh.at(k)
                link(j, sh.off + np.asarray(ring, int), TIE_G)
                js.append((j, lay, z))
                j += 1
            segs = []
            for (ja, la, za), (jb, lb, zb) in zip(js, js[1:]):
                g = _barrel_g(v['r'], zb - za, plating, sigma)
                link(ja, jb, g)
                segs.append((la, lb, ja, jb, g))
            self.via_segs.append(segs)
        self.tj, self.notes = {}, []            # terminal -> its junction
        for name, keys in self.tkey.items():
            ids = [self.sheets[lay].off + q for lay, i in keys if lay in self.sheets for q in self.sheets[lay].region(i)]
            if not ids:
                self.notes.append(f"terminal {name}: on none of the copper of {', '.join(sorted(self.nets))}")
                continue
            link(j, ids, TIE_G)
            self.tj[name] = j
            j += 1
        self.links = []                         # series parts: (name, terminal a, terminal b, R)
        self.n = j
        self._rows, self._cols, self._vals = rows, cols, vals
        if self.mesh_info.get('merged_vias'):
            self.notes.append(f"overlapping vias (drilled into one another) meshed as one ring: "
                              f"{self.mesh_info['merged_vias']} ring(s) merged over the layers")
        log(f"dc {', '.join(sorted(self.nets))}: {self.n_mesh:,} vertices on {len(self.sheets)} layer(s), "
            f"{sum(1 for s in self.via_segs if s)} vias, {len(self.tj)} terminals")

    def series(self, name, a, b, R):
        """A series part (resistance R, ohms) between terminals a and b."""
        if a in self.tj and b in self.tj:
            self._link(self.tj[a], self.tj[b], 1.0 / max(R, 1e-9))
            self.links.append((name, a, b, R))
        else:
            self.notes.append(f"{name}: a pad not on the copper: left out")

    def matrix(self):
        r_ = np.concatenate([b[0] for b in self.blocks] + self._rows).astype(int)
        c_ = np.concatenate([b[1] for b in self.blocks] + self._cols).astype(int)
        d_ = np.concatenate([b[2] for b in self.blocks] + self._vals).astype(float)
        return coo_matrix((d_, (r_, c_)), shape=(self.n, self.n)).tocsr()

    def solve(self, fixed, inject):
        """Terminal voltages fixed {name: V} and currents injected {name: A, + into the copper}: the voltages,
        currents and loss. Copper not connected to a fixed terminal is left out (NaN) and said."""
        G = self.matrix()
        fx = np.array([self.tj[t] for t in fixed])
        V = np.full(self.n, np.nan)
        V[fx] = [fixed[t] for t in fixed]
        inj = np.zeros(self.n)
        for t, i in inject.items():
            if t in self.tj:
                inj[self.tj[t]] += i
        _, lab = connected_components(G, directed=False)
        live = np.isin(lab, lab[fx])
        notes = list(self.notes)
        if not live[:self.n_mesh].all():
            notes.append(f"{(~live[:self.n_mesh]).sum()} mesh vertices on copper not connected to the source (left out)")
        lost = [t for t in inject if t in self.tj and not live[self.tj[t]]]
        if lost:
            raise ValueError(f"terminal(s) {', '.join(lost)} on copper not connected to the source")
        free = np.flatnonzero(live & ~np.isin(np.arange(self.n), fx))
        if not inj[free].any() and len(set(fixed.values())) == 1:
            V[free] = V[fx[0]]                  # no current anywhere (a return no load reaches): one voltage
        else:
            V[free] = spsolve(G[free][:, free].tocsc(), inj[free] - G[free][:, fx] @ V[fx])
        Vz = np.nan_to_num(V)
        # the loss as the power the terminals put in (Tellegen): V.G.V cancels across the stiff ties
        loss = float(Vz[fx] @ (G[fx] @ Vz) + Vz[free] @ inj[free])
        out = DCResult(self.nets, self.sheets, [], {}, loss, self.n_mesh, notes)
        for lay, sh in self.sheets.items():
            sh.V = V[sh.off:sh.off + len(sh.nodes)]
            gv = gradient(sh.nodes, sh.tris, np.nan_to_num(sh.V))      # V/mm
            sh.J = self.sigma * sh.t * 1e-3 * np.hypot(gv[:, 0], gv[:, 1])   # A per mm of width
        for v, segs in zip(self.vias, self.via_segs):
            ss = [(la, lb, float(g * (Vz[ja] - Vz[jb]))) for la, lb, ja, jb, g in segs]
            out.vias.append(dict(x=v['x'], y=v['y'], r=v['r'], segments=ss, I=max([abs(s[2]) for s in ss], default=0.0)))
        out.terminals = {t: float(V[j]) for t, j in self.tj.items()}
        out.series = {name: dict(I=abs(float((Vz[self.tj[a]] - Vz[self.tj[b]]) / R)),     # along its current
                                 drop=abs(float(Vz[self.tj[a]] - Vz[self.tj[b]])))
                      for name, a, b, R in self.links}
        return out


@dataclass
class DCResult:
    net: object
    sheets: dict                    # {layer: Sheet with .V (node volts), .J (A/mm per triangle), .off (dof offset)}
    vias: list                      # [dict(x, y, r, I, segments=[(layer_a, layer_b, I)])]
    loads: dict                     # {name: dict(V, I, layer)}
    loss: float                     # W in the copper
    nodes: int
    notes: list = field(default_factory=list)
    source: dict = None
    terminals: dict = None
    series: dict = None


def _disc(p):
    return [(p['layer'], Point(p['x'], p['y']).buffer(p['r'], quad_segs=4))]


def solve(board, net, source, loads, dmax=1.0, sigma=SIGMA_CU, plating=0.025, ground=False, log=print, coarsen=None,
          fine=True):
    """The one-net form (see the module docstring); pads are discs (x, y, r) on a layer."""
    pads = {'_source': source, **{l.get('name', f'L{j + 1}'): l for j, l in enumerate(loads)}}
    terms = {name: _disc(p) for name, p in pads.items()}
    nw = Network(board, net, terms, dmax=dmax, sigma=sigma, plating=plating, log=log,
                 fine=list(terms) if fine else ())
    for name, p in pads.items():
        if name not in nw.tj:
            raise ValueError(f"pad {name} at ({p['x']}, {p['y']}) is not on {net} copper")
    s = 1 if ground else -1
    named = list(pads.items())[1:]
    r = nw.solve({'_source': 0.0 if ground else float(source['V'])}, {name: s * l['I'] for name, l in named})
    r.loads = {name: dict(V=r.terminals[name], I=l['I'], layer=l['layer']) for name, l in named}
    r.source = dict(source)
    log(f"dc {net}: {r.loss * 1e3:.2f} mW in the copper")
    return r


def solve_rail(rail, volts, dmax=1.0, sigma=SIGMA_CU, plating=0.025, log=print, fine=True):
    """A rail (rail.Rail) and its return at DC: the source's pads held at `volts`, each load drawing rail.loads[ref]
    amps over its pads on the tree and returning it over its ground pads. {'loads': {ref: dict(V, drop, power_drop,
    ground_rise, I)}, 'series': {ref: dict(I, drop)}, 'power': DCResult, 'ground': DCResult, 'notes'}."""
    from .rail import terminal
    layers = list(stack_z(rail.board))
    g = set(rail.ground)
    pt = {'source': terminal(rail.source_pads, layers)}
    for s in rail.series:
        pt[f'{s.part.ref}:a'] = terminal(s.pads('a'), layers)
        pt[f'{s.part.ref}:b'] = terminal(s.pads('b'), layers)
    for ref in rail.loads:
        pt[ref] = terminal(rail.parts[ref].on(rail.nets), layers)
    pw = Network(rail.board, rail.nets, pt, dmax=dmax, sigma=sigma, plating=plating, log=log,
                 fine=['source', *rail.loads] if fine else ())
    for s in rail.series:
        pw.series(s.part.ref, f'{s.part.ref}:a', f'{s.part.ref}:b', s.R)
    rp = pw.solve({'source': volts}, {ref: -i for ref, i in rail.loads.items()})
    gt = {'source': terminal(rail.ground_source_pads, layers)}
    gt.update({ref: terminal(rail.pads_of(rail.parts[ref])[1], layers) for ref in rail.loads})   # (a --load @RETURN)
    no_gnd = [ref for ref in rail.loads if not gt[ref]]
    for ref in no_gnd:
        gt.pop(ref)
    gn = Network(rail.board, g, gt, dmax=dmax, sigma=sigma, plating=plating, log=log,
                 fine=['source', *rail.loads] if fine else ())
    rg = gn.solve({'source': 0.0}, {ref: i for ref, i in rail.loads.items() if ref in gt})
    loads = {}
    for ref, i in rail.loads.items():
        vp, vg = rp.terminals.get(ref, np.nan), rg.terminals.get(ref, 0.0)
        loads[ref] = dict(V=vp - vg, drop=volts - (vp - vg), power_drop=volts - vp, ground_rise=vg, I=i)
    notes = rp.notes + rg.notes + [f"{r}: no ground pad (its return not modelled; --load {r}=AMPS@RETURN names the "
                                   f"part it returns through)" for r in no_gnd]
    if loads:
        log(f"dc rail {rail.net}: {(rp.loss + rg.loss) * 1e3:.1f} mW in the copper; worst drop "
            f"{max(l['drop'] for l in loads.values()) * 1e3:.2f} mV")
    return dict(loads=loads, series=rp.series, power=rp, ground=rg, notes=notes, no_return=no_gnd)
