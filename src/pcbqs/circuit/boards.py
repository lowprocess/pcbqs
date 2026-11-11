"""
The example circuits' boards: a KiCad layout for each, written from a short description.

    python -m pcbqs.circuit.boards            # write every example's .kicad_pcb beside its model, and check each
    python -m pcbqs.circuit.boards --picture boards.png

A layout here is what a power board is: an island of copper for each net on the top layer, a ground strip
stitched to a ground plane on the bottom, and each part across the gap between the islands of its two nets. So a
description is the islands (rectangles, mm) and the package of each part; where a part goes follows from the
islands its nets have (`place`), a transformer or anything else unusual is put down by hand (`part`).

    b = Board('06_boost', 44, 24)
    b.row(['in', 'sw', 'out'], [15, 9, 16])          # three islands in a row over the ground strip
    b.place(model, packages={'V1': 'J_PADS', 'L1': 'L_7x7', ...}, names={'V1': 'J1', 'R1': 'J2'})
    b.check(); b.write(folder)

check() is the generator's own: every pad on copper of its net, every net's copper in one piece (through its vias
and the plane), no two nets closer than the clearance. tests/test_circuit_boards.py checks the written files
again, as kicad2ems reads them.
"""
import json
import uuid
from pathlib import Path

from shapely.geometry import box, Point
from shapely.ops import unary_union

GAP = 0.4               # mm between two islands, and the least clearance anywhere
STRIP = 4.0             # mm, the ground strip's height
VIA = (0.8, 0.4)        # a via's pad and drill, mm
GROUND = ('0', 'GND', 'gnd')

# A package's pads, the part lying along x: {pad: (x, y, w, h)}, and how much of an island's edge it takes (mm).
# Two-pad parts: pad 1 on the left (a diode's is its cathode, as KiCad numbers them). A switch: its drain right, its
# source and gate left, placed so the gap between two islands runs between them.
PACKAGES = {
    'R_0805': (dict(p1=(-0.95, 0, 1.0, 1.45), p2=(0.95, 0, 1.0, 1.45)), 2.4),
    'R_2512': (dict(p1=(-2.9, 0, 1.6, 3.3), p2=(2.9, 0, 1.6, 3.3)), 4.4),
    'C_0805': (dict(p1=(-0.95, 0, 1.0, 1.45), p2=(0.95, 0, 1.0, 1.45)), 2.4),
    'C_1210': (dict(p1=(-1.45, 0, 1.15, 2.7), p2=(1.45, 0, 1.15, 2.7)), 3.8),
    'CP_10x10': (dict(p1=(-4.2, 0, 4.0, 2.0), p2=(4.2, 0, 4.0, 2.0)), 10.8),
    'L_7x7': (dict(p1=(-2.6, 0, 2.2, 6.0), p2=(2.6, 0, 2.2, 6.0)), 7.6),
    'D_SMA': (dict(p1=(-2.0, 0, 2.2, 1.8), p2=(2.0, 0, 2.2, 1.8)), 3.4),
    'D_SMC': (dict(p1=(-3.4, 0, 2.6, 3.2), p2=(3.4, 0, 2.6, 3.2)), 6.4),
    'Q_SO8FL': (dict(D=(2.85, 0, 4.2, 4.4), S=(-1.25, -0.65, 1.0, 2.7), G=(-1.25, 1.85, 1.0, 0.7)), 5.6),
    'J_PADS': (dict(p1=(-2.5, 0, 3.0, 3.0), p2=(2.5, 0, 3.0, 3.0)), 4.2),
}
LIBS = dict(R_0805='Resistor_SMD:R_0805_2012Metric', R_2512='Resistor_SMD:R_2512_6332Metric',
            C_0805='Capacitor_SMD:C_0805_2012Metric', C_1210='Capacitor_SMD:C_1210_3225Metric',
            CP_10x10='Capacitor_SMD:CP_Elec_10x10', L_7x7='Inductor_SMD:L_7.3x7.3_H4.5',
            D_SMA='Diode_SMD:D_SMA', D_SMC='Diode_SMD:D_SMC', Q_SO8FL='Package_SO:SO-8FL_5x6mm',
            J_PADS='Connector_Wire:SolderWirePad_1x02_SMD_3x3mm', T_4PAD='Transformer_SMD:Transformer_4Pad')
PREFIX = dict(resistor='R', capacitor='C', inductor='L', diode='D', switch='Q', source='J', load='J')


_count = [0]


def uid():
    """A KiCad item's uuid: the same every time a board is written, so a board file only changes when its layout does."""
    _count[0] += 1
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"pcbqs.circuit.boards/{_count[0]}"))


def net_of(node):
    return 'GND' if node in GROUND else node


def package_of(part):
    """The package a part gets when none is named: by its type and its size."""
    kind = part['type']
    value = part.get('value')
    if kind == 'capacitor':
        return 'CP_10x10' if value >= 200e-6 else 'C_1210' if value >= 1e-6 else 'C_0805'
    return dict(resistor='R_0805', inductor='L_7x7', diode='D_SMA', switch='Q_SO8FL', source='J_PADS', load='J_PADS')[kind]


class Board:
    def __init__(self, name, width, height, thickness=1.5):
        self.name, self.W, self.H, self.thickness = name, width, height, thickness
        self.islands = []                                           # (net, polygon) on F.Cu
        self.boxes = []                                             # (net, (x0, y0, x1, y1)): the islands as drawn
        self.footprints = []                                        # dict(ref, value, lib, x, y, pads: [(name, fn, net, box)])
        self.vias = []                                              # (net, x, y)
        self.plane = box(0.5, 0.5, width - 0.5, height - 0.5)       # B.Cu: ground

    # ---- copper ----
    def island(self, net, x0, y0, x1, y1):
        self.islands.append((net, box(x0, y0, x1, y1)))
        self.boxes.append((net, (x0, y0, x1, y1)))                  # as drawn, before any notch
        return self.islands[-1][1]

    def ground(self, x0, y0, x1, y1, pitch=3.0):
        """A ground island on top, stitched to the plane underneath every `pitch` mm along its middle."""
        g = self.island('GND', x0, y0, x1, y1)
        long = x1 - x0 >= y1 - y0
        n = max(int(((x1 - x0) if long else (y1 - y0)) // pitch), 1)
        for k in range(n):
            t = (k + 0.5) / n
            self.vias.append(('GND', x0 + t * (x1 - x0), (y0 + y1) / 2) if long else ('GND', (x0 + x1) / 2, y0 + t * (y1 - y0)))
        return g

    def row(self, nets, widths, top=2.0, height=10.0, left=2.0, strip=STRIP):
        """Islands side by side from `left`, GAP apart, over a ground strip that runs under them all."""
        x = left
        for net, w in zip(nets, widths):
            self.island(net, x, top, x + w, top + height)
            x += w + GAP
        self.ground(left, top + height + GAP, x - GAP, top + height + GAP + strip)

    # ---- parts ----
    def part(self, ref, package, x, y, nets, value='', turn=0):
        """A footprint at (x, y), turned by quarter turns (0, 90, 180, 270: its pad 1 left, above, right, below);
        nets: {pad: net}. A pad on a net of its own (a gate) gets a notch in the island it sits on."""
        pads, _ = PACKAGES[package]
        functions = dict(D='D', S='S', G='G', p1='K', p2='A') if package.startswith(('Q_', 'D_')) else {}
        out = []
        for number, (name, (px, py, w, h)) in enumerate(pads.items(), 1):
            for _ in range(turn % 360 // 90):
                px, py, w, h = -py, px, h, w
            g = box(x + px - w / 2, y + py - h / 2, x + px + w / 2, y + py + h / 2)
            out.append((str(number), functions.get(name, ''), nets.get(name, ''), g))
            if nets.get(name, '').startswith('Net-('):
                self.notch(g)
        self.footprints.append(dict(ref=ref, value=value, lib=LIBS[package], x=x, y=y, pads=out))

    def notch(self, pad):
        """Clear a pad of another net out of the islands it sits on: a notch from the island's nearest edge."""
        x0, y0, x1, y1 = pad.buffer(GAP, join_style=2).bounds
        for k, (net, island) in enumerate(self.islands):
            if island.distance(pad) >= GAP:
                continue
            ix0, iy0, ix1, iy1 = island.bounds
            side = min((x0 - ix0, 'left'), (ix1 - x1, 'right'), (y0 - iy0, 'top'), (iy1 - y1, 'bottom'))[1]
            cut = dict(left=box(ix0 - 1, y0, x1, y1), right=box(x0, y0, ix1 + 1, y1), top=box(x0, iy0 - 1, x1, y1),
                       bottom=box(x0, y0, x1, iy1 + 1))[side]
            left = island.difference(cut)
            if left.geom_type != 'Polygon' or left.interiors:
                raise ValueError(f"{self.name}: a notch for a pad at ({pad.centroid.x:.1f}, {pad.centroid.y:.1f}) cuts {net} in two")
            self.islands[k] = (net, left)

    def transformer(self, ref, x, y, nets, value='', pitch=(9.0, 6.0), pad=(2.4, 2.6)):
        """Four pads: 1 and 2 the primary (left, top and bottom), 3 and 4 the secondary (right)."""
        out = []
        for number, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)), 1):
            cx, cy = x + sx * pitch[0] / 2, y + sy * pitch[1] / 2
            out.append((str(number), ('P1', 'P2', 'S1', 'S2')[number - 1], nets[number - 1],
                        box(cx - pad[0] / 2, cy - pad[1] / 2, cx + pad[0] / 2, cy + pad[1] / 2)))
        self.footprints.append(dict(ref=ref, value=value, lib=LIBS['T_4PAD'], x=x, y=y, pads=out))

    def edges(self, a, b):
        """Where an island of net a faces one of net b across a gap: [(axis, at, lo, hi, sign)], axis 'x' a
        vertical gap at x = at with a on its left when sign is 1."""
        out = []
        for na, (ax0, ay0, ax1, ay1) in self.boxes:
            for nb, (bx0, by0, bx1, by1) in self.boxes:
                if (na, nb) != (a, b):
                    continue
                for sign, (p, q) in ((1, (ax1, bx0)), (-1, (bx1, ax0))):
                    if 0 < q - p <= 3 * GAP + 1e-9 and min(ay1, by1) - max(ay0, by0) > 1:
                        out.append(('x', (p + q) / 2, max(ay0, by0), min(ay1, by1), sign))
                for sign, (p, q) in ((1, (ay1, by0)), (-1, (by1, ay0))):
                    if 0 < q - p <= 3 * GAP + 1e-9 and min(ax1, bx1) - max(ax0, bx0) > 1:
                        out.append(('y', (p + q) / 2, max(ax0, bx0), min(ax1, bx1), sign))
        return out

    def place(self, model, packages=None, names=None, skip=(), values=None):
        """Every part of a (resolved) model across the gap between its two nets' islands; parts that share a gap
        are spread along it, in the model's order. names: a part's reference on the board when it is not the
        model's (a source is a connector: {'V1': 'J1'}); skip: parts put down by hand."""
        packages, names = packages or {}, names or {}
        queue = {}
        for p in model['parts']:
            if p['ref'] in skip:
                continue
            a, b = (net_of(n) for n in p['nodes'])
            found = self.edges(a, b)
            if not found:
                raise ValueError(f"{self.name}: no gap between an island of {a} and one of {b} for {p['ref']}")
            edge = max(found, key=lambda e: e[3] - e[2])            # the longest facing edge
            queue.setdefault(edge, []).append(p)
        for (axis, at, lo, hi, sign), parts in queue.items():
            need = [PACKAGES[packages.get(p['ref'], package_of(p))][1] for p in parts]
            if sum(need) > hi - lo + 1e-9:
                raise ValueError(f"{self.name}: {[p['ref'] for p in parts]} need {sum(need):g} mm of a {hi - lo:g} mm edge")
            spare = (hi - lo - sum(need)) / len(parts)
            along = lo
            for p, w in zip(parts, need):
                centre = along + (w + spare) / 2
                along += w + spare
                ref = names.get(p['ref'], p['ref'])
                package = packages.get(p['ref'], package_of(p))
                a, b = (net_of(n) for n in p['nodes'])
                if package.startswith('Q_'):                        # drain, source: the source and gate pads left
                    nets, left = dict(D=a, S=b, G=f"Net-({ref}-G)"), b
                elif package.startswith('D_'):                      # anode, cathode: pad 1 is the cathode
                    nets, left = dict(p1=b, p2=a), b
                else:
                    nets, left = dict(p1=a, p2=b), a
                side = sign if left == a else -sign                 # 1: the left pads' island is the first (left, above)
                turn = {('x', 1): 0, ('x', -1): 180, ('y', 1): 90, ('y', -1): 270}[(axis, side)]
                x, y = (at, centre) if axis == 'x' else (centre, at)
                self.part(ref, package, x, y, nets, (values or {}).get(p['ref'], label(p)), turn)

    # ---- checks ----
    def copper(self):
        """{net: {layer: geometry}} of everything drawn."""
        out = {}
        for net, g in self.islands:
            out.setdefault(net, {}).setdefault('F.Cu', []).append(g)
        for fp in self.footprints:
            for _, _, net, g in fp['pads']:
                if net:
                    out.setdefault(net, {}).setdefault('F.Cu', []).append(g)
        out.setdefault('GND', {}).setdefault('B.Cu', []).append(self.plane)
        for net, x, y in self.vias:
            for layer in ('F.Cu', 'B.Cu'):
                out.setdefault(net, {}).setdefault(layer, []).append(Point(x, y).buffer(VIA[0] / 2))
        return {net: {l: unary_union(gs) for l, gs in layers.items()} for net, layers in out.items()}

    def check(self):
        foreign = [(net, g) for fp in self.footprints for _, _, net, g in fp['pads'] if net]
        self.vias = [(net, x, y) for net, x, y in self.vias          # no stitching via beside another net's pad
                     if all(n == net or g.distance(Point(x, y)) >= VIA[0] / 2 + GAP for n, g in foreign)]
        cu = self.copper()
        inside = box(0, 0, self.W, self.H)
        for net, layers in cu.items():
            for layer, g in layers.items():
                if not inside.contains(g):
                    raise ValueError(f"{self.name}: {net} on {layer} leaves the board")
                for other, more in cu.items():
                    if other > net and layer in more and g.distance(more[layer]) < GAP - 1e-6:
                        raise ValueError(f"{self.name}: {net} and {other} are {g.distance(more[layer]):.2f} mm apart on {layer}")
            if net.startswith('Net-('):
                continue                                            # a gate: its pad alone
            pieces = [p for l in layers.values() for p in (l.geoms if hasattr(l, 'geoms') else [l])]
            group = list(range(len(pieces)))                        # a via joins what it touches on each layer
            for vnet, x, y in self.vias:
                if vnet == net:
                    touched = [k for k, p in enumerate(pieces) if p.distance(Point(x, y)) < VIA[0] / 2]
                    for k in range(len(pieces)):
                        if group[k] in {group[t] for t in touched}:
                            group[k] = group[touched[0]]
            if len(set(group)) != 1:
                raise ValueError(f"{self.name}: the copper of {net} is in {len(set(group))} pieces")
        hulls = []
        for fp in self.footprints:
            for number, _, net, g in fp['pads']:
                if net and not net.startswith('Net-(') and not any(n == net and r.intersection(g).area > 0.5 * g.area for n, r in self.islands):
                    raise ValueError(f"{self.name}: {fp['ref']} pad {number} is not on an island of {net}")
            hull = unary_union([g for *_, g in fp['pads']]).envelope
            for other, h in hulls:
                if hull.intersects(h) and hull.intersection(h).area > 1e-6:
                    raise ValueError(f"{self.name}: {fp['ref']} and {other} overlap")
            hulls.append((fp['ref'], hull))
        return self

    # ---- the file ----
    def write(self, folder):
        _count[0] = sum(map(ord, self.name)) * 1000                 # this board's own run of uuids
        nets = [''] + sorted({n for n, _ in self.islands} | {n for fp in self.footprints for _, _, n, _ in fp['pads'] if n} | {'GND'})
        num = {n: i for i, n in enumerate(nets)}
        L = ['(kicad_pcb (version 20240108) (generator "pcbqs") (generator_version "8.0")',
             f'  (general (thickness {self.thickness + 0.07:g}))\n  (paper "A4")',
             '  (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (37 "F.SilkS" user) (39 "F.Mask" user) (44 "Edge.Cuts" user) (49 "F.Fab" user))',
             '  (setup (stackup', '    (layer "F.Cu" (type "copper") (thickness 0.035))',
             f'    (layer "dielectric 1" (type "core") (thickness {self.thickness:g}) (material "FR4") (epsilon_r 4.5) (loss_tangent 0.02))',
             '    (layer "B.Cu" (type "copper") (thickness 0.035))',
             '    (copper_finish "None") (dielectric_constraints no))\n    (pad_to_mask_clearance 0))',
             '  ' + ' '.join(f'(net {i} "{n}")' for i, n in enumerate(nets))]
        def rect(g, layer, net):                                   # an island: a rectangle, or its outline once notched
            xy = ' '.join(f'(xy {x:.4f} {y:.4f})' for x, y in list(g.exterior.coords)[:-1])
            return (f'  (gr_poly (pts {xy}) (stroke (width 0) (type solid)) (fill yes) (layer "{layer}") (net {num[net]}) '
                    f'(uuid "{uid()}"))')
        for net, g in self.islands:
            L.append(rect(g, 'F.Cu', net))
        L.append(rect(self.plane, 'B.Cu', 'GND'))
        for net, x, y in self.vias:
            L.append(f'  (via (at {x:.4f} {y:.4f}) (size {VIA[0]}) (drill {VIA[1]}) (layers "F.Cu" "B.Cu") (net {num[net]}) (uuid "{uid()}"))')
        for fp in self.footprints:
            L += [f'  (footprint "{fp["lib"]}" (layer "F.Cu") (uuid "{uid()}") (at {fp["x"]:.4f} {fp["y"]:.4f})',
                  f'    (property "Reference" "{fp["ref"]}" (at 0 -1.5 0) (layer "F.SilkS") (uuid "{uid()}"))',
                  f'    (property "Value" "{fp["value"]}" (at 0 1.5 0) (layer "F.Fab") (uuid "{uid()}"))', '    (attr smd)']
            for number, function, net, g in fp['pads']:
                x0, y0, x1, y1 = g.bounds
                L.append(f'    (pad "{number}" smd rect (at {(x0 + x1) / 2 - fp["x"]:.4f} {(y0 + y1) / 2 - fp["y"]:.4f}) (size {x1 - x0:.4f} {y1 - y0:.4f}) '
                         f'(layers "F.Cu" "F.Paste" "F.Mask")' + (f' (net {num[net]} "{net}")' if net else '') +
                         (f' (pinfunction "{function}")' if function else '') + f' (uuid "{uid()}"))')
            L.append('  )')
        L.append(f'  (gr_rect (start 0 0) (end {self.W:.4f} {self.H:.4f}) (stroke (width 0.1) (type default)) (fill none) '
                 f'(layer "Edge.Cuts") (uuid "{uid()}"))\n)')
        path = Path(folder) / f"{self.name}.kicad_pcb"
        path.write_text('\n'.join(L) + '\n', encoding='utf-8')
        return path


def label(part):
    """A part's value as its footprint's Value: '22u', '5R', the way a schematic writes it."""
    def si(v, unit=''):
        for e, p in ((6, 'M'), (3, 'k'), (0, ''), (-3, 'm'), (-6, 'u'), (-9, 'n'), (-12, 'p')):
            if abs(v) >= 10 ** e:
                return f"{v / 10 ** e:.4g}{p}{unit}"
        return f"{v:g}{unit}"
    kind = part['type']
    if kind in ('resistor', 'capacitor', 'inductor'):
        return si(part['value'], 'H' if kind == 'inductor' else '')
    if kind == 'load' and 'resistance_ohm' in part:
        return 'load ' + si(part['resistance_ohm'])
    return dict(diode='D', switch='MOSFET', source='supply', load='load')[kind]


# ---- the examples' layouts ----
def layouts():
    """{example: (Board, {model ref: board ref})} for every example circuit, placed and checked."""
    from . import model as M
    out = {}

    def start(name, width, height):
        stem = next(f.name[:-5] for f in sorted(Path(__file__).parent.joinpath('examples').glob('*.json'))
                    if f.name[:-5].split('_', 1)[1] == name)
        model = M.template(name)
        model.pop('board', None)
        resolved = M.resolve(model, M.cases(model)[0])
        return Board(stem, width, height), resolved

    def done(name, b, names):
        out[name] = (b.check(), names)

    b, m = start('rc_filter', 17.4, 16.4)
    b.row(['in', 'out'], [7, 6], height=6)
    names = dict(V1='J1')
    b.place(m, names=names)
    done('rc_filter', b, names)

    b, m = start('half_wave_rectifier', 33.4, 21.4)
    b.row(['ac', 'out'], [10, 19], height=8, strip=7)                # room for the electrolytic's ground pad
    names = dict(V1='J1', R1='J2')
    b.place(m, names=names)
    done('half_wave_rectifier', b, names)

    b, m = start('bridge_rectifier', 37, 34.2)                      # the AC pair left, one over the other; DC right
    b.ground(2, 2, 16, 6)
    b.island('l', 2, 6.4, 16, 15.4)
    b.island('n', 2, 15.8, 16, 24.8)
    b.island('p', 16.4, 6.4, 35, 24.8)
    b.ground(2, 25.2, 35, 32.2)
    names = dict(V1='J1', P1='J2')
    b.place(m, names=names)
    done('bridge_rectifier', b, names)

    for name, widths in (('buck', [10, 12, 10]), ('buck_synchronous', [10, 12, 10]), ('boost', [10, 13, 10]),
                         ('buck_boost', [10, 13, 10])):
        b, m = start(name, sum(widths) + 2 * GAP + 4, 20.4)
        b.row(['in', 'sw', 'out'], widths, height=10)
        names = {'V1': 'J1', next(p['ref'] for p in m['parts'] if p['type'] == 'load'): 'J2'}
        b.place(m, names=names)
        done(name, b, names)

    b, m = start('flyback', 58, 31.2)                               # primary left, the transformer across the gap
    b.ground(2, 2, 24, 6)
    b.island('in', 2, 6.4, 24, 14.4)
    b.island('clamp', 2, 14.8, 11.6, 24.8)
    b.island('sw', 12, 14.8, 24, 24.8)
    b.ground(2, 25.2, 24, 29.2)
    b.island('sec', 31.5, 6.4, 40, 14.4)
    b.ground(31.5, 14.8, 40, 24.8)
    b.island('out', 40.4, 6.4, 56, 14.4)
    b.ground(40.4, 14.8, 56, 18.8)
    names = dict(V1='J1', R1='J2', LP='T1', LS='T1')
    b.transformer('T1', 27.75, 14.6, ['in', 'sw', 'sec', 'GND'], value='200uH 2:1')
    b.place(m, names=names, skip=('LP', 'LS'), packages=dict(R2='R_2512'))
    done('flyback', b, names)

    b, m = start('half_bridge_resonant', 51.6, 20.4)
    b.row(['in', 'sw', 'a', 'b'], [10, 12, 8, 16], height=10)
    names = dict(V1='J1', R1='J2')
    b.place(m, names=names, packages=dict(CR='C_1210'))
    done('half_bridge_resonant', b, names)

    b, m = start('full_bridge_phase_shift', 62, 35.2)               # the bridge left (the bus over its two legs, the
    b.ground(2, 2, 38, 6)                                           # tank between them), the receiver right
    b.island('in', 2, 6.4, 38, 16.4)
    b.island('a', 2, 16.8, 14, 28.8)
    b.island('t', 14.4, 16.8, 23.6, 28.8)
    b.island('b', 24, 16.8, 38, 28.8)
    b.ground(2, 29.2, 38, 33.2)
    b.island('s', 40, 6.4, 50, 16.4)
    b.island('r', 50.4, 6.4, 60, 16.4)
    b.ground(40, 16.8, 60, 20.8)
    names = dict(V1='J1', L1='J2', L2='J3', R1='J4')
    b.place(m, names=names, packages=dict(L1='J_PADS', L2='J_PADS', C3='C_1210', C2='C_1210'),
            values=dict(L1='coil 24uH', L2='coil 24uH'))
    done('full_bridge_phase_shift', b, names)
    return out


def picture(path, columns=2):
    """Every example's layout in one picture: the islands by net, the ground plane under them, pads and vias."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    made = layouts()
    palette = ['#d95f02', '#1b9e77', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d']
    rows = -(-len(made) // columns)
    fig, axs = plt.subplots(rows, columns, figsize=(7.5 * columns, 4.2 * rows))
    for ax, (name, (b, _)) in zip(axs.ravel(), made.items()):
        colour = {}
        ax.add_patch(Polygon(list(b.plane.exterior.coords), fc='#dfe8f5', ec='none'))
        for net, g in b.islands:
            if net != 'GND':
                colour.setdefault(net, palette[len(colour) % len(palette)])
            ax.add_patch(Polygon(list(g.exterior.coords), fc='#9aa7b8' if net == 'GND' else colour[net], ec='k', lw=0.4, alpha=0.55))
            ax.text(g.bounds[0] + 0.3, g.bounds[1] + 1.1, net, fontsize=7)
        for _, x, y in b.vias:
            ax.add_patch(Circle((x, y), VIA[0] / 2, fc='w', ec='k', lw=0.5))
        for fp in b.footprints:
            for _, _, net, g in fp['pads']:
                ax.add_patch(Polygon(list(g.exterior.coords), fc='#b00020' if net.startswith('Net-(') else '#222222', ec='none', alpha=0.85))
            ax.text(fp['x'], fp['y'], fp['ref'], fontsize=7, color='w', ha='center', va='center', bbox=dict(fc='#333333', ec='none', pad=0.6))
        ax.set(xlim=(0, b.W), ylim=(b.H, 0), aspect='equal')
        ax.set_title(f"{name}   {b.W:g} x {b.H:g} mm", fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axs.ravel()[len(made):]:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['--picture']:
        return picture(argv[1])
    folder = Path(__file__).parent / 'examples'
    for name, (board, names) in layouts().items():
        path = board.write(folder)
        model_path = folder / (board.name + '.json')
        model = json.loads(model_path.read_text())
        model['board'] = dict(file=path.name, parts=names)
        model_path.write_text(json.dumps(model, indent=1))
        print(f"{path.name}: {board.W:g} x {board.H:g} mm, {len(board.footprints)} footprints, {len(board.islands)} islands, "
              f"{len(board.vias)} vias")


if __name__ == '__main__':
    main()
