"""
A power rail as the board has it: the nets it spans (the rail tree), the parts on it and what each one is.

    r = Rail(board, '+3V3', ground=('GND',), loads={'U21': 2.0, 'U14': 1.0})
    r.nets        # {'+3V3', '/DCDC/3V3_OUT'}: joined through the series parts in r.series
    r.series      # [Series(part, a, b, kind, R, z100)]: sense / 0 ohm resistors, beads, jumpers
    r.source      # the terminal the rail is fed from (a power inductor's output pad, or the part named source=)
    r.ground_source, r.decaps, r.loads

The tree is grown from `net` through two-pad parts between two non-ground nets that are, by their reference and value,
  resistors up to max_ohms (default 1 ohm: current-sense and 0 ohm links; a feedback divider's 10 k is not),
  ferrite beads (FB*, or an L* whose value reads like one: 600R@100MHz, BLM..., bead), jumpers and net ties (JP*, NT*),
and any part named in through=; never one in stop=. A power inductor (an L* in henries) is where it ends: the
regulator's output, and the default source (its pad on the tree, the regulator's ground pads the return's source).
An inductor named in through= is instead a series inductor of the tree (an input filter): its inductance and its
DC resistance (dcr=, else a DCR field, else an assumed DEFAULT_INDUCTOR_DCR) both stay in the model. dcr= {ref: ohms}
also overrides any series part's resistance.

The board is board.json's dict with its footprints (prims.footprints: ref, value, props, lib, side, dnp, pads with
name, net, layers, geom). A terminal is a part's pads on some nets: [(layer, shapely polygon)] per pad, a
through-hole pad on every copper layer of the stack.
"""
import re
from dataclasses import dataclass, field

from shapely.geometry import shape

_SI = {'f': 1e-15, 'p': 1e-12, 'n': 1e-9, 'u': 1e-6, 'µ': 1e-6, 'μ': 1e-6, 'm': 1e-3, 'k': 1e3, 'K': 1e3, 'M': 1e6,
       'G': 1e9, 'R': 1.0, 'r': 1.0, 'E': 1.0}
_NUM = re.compile(r'^(\d*)(?:[.,](\d+))?([fpnuµμmkKMGRrE]?)(\d*)\s*(F|H|Ω|ohms?|OHMS?|R)?$')
BEAD = re.compile(r'(\d+\s*(R|Ω|ohm)\s*@|bead|ferrite|^BLM|^MPZ|^FB)', re.I)
DEFAULT_BEAD_DCR, DEFAULT_BEAD_Z100 = 0.05, 600.0
DEFAULT_INDUCTOR_DCR = 0.01                 # a followed series inductor without a DCR: assumed, and flagged


def parse_value(text):
    """'0R001' 1e-3, '4k7' 4700, '1u8' 1.8e-6, '2.2uH' 2.2e-6, '100n' 1e-7, '10m' 0.01; None if not a value."""
    m = _NUM.match(str(text).strip())
    if not m or not (m.group(1) or m.group(2) or m.group(4)):
        return None
    whole, frac, pre, frac2, _ = m.groups()
    digits = f"{whole or '0'}.{frac or frac2 or '0'}"
    if frac and frac2:
        return None
    return float(digits) * _SI.get(pre, 1.0)


def part_value(fp):
    """The first value among a footprint's Val / Value fields and their '_'-separated pieces ('R_0R001_0805'), or
    None. fp: a footprint dict or a Part."""
    props, value = _pv(fp)
    for v in (props.get('Val'), props.get('Value'), value):
        if not v:
            continue
        for tok in [v] + re.split(r'[_\s]+', str(v)):
            x = parse_value(tok)
            if x is not None:
                return x
    return None


def _pv(fp):
    return (fp['props'], fp.get('value')) if isinstance(fp, dict) else (fp.props, fp.value)


def field_value(fp, *keys):
    """The value of the first of a part's fields `keys` (any case, or sim.<key>) that has one, or None."""
    props, _ = _pv(fp)
    for k in keys:
        for kk in (k, k.upper(), k.lower(), 'sim.' + k.lower()):
            if props.get(kk) not in (None, ''):
                x = parse_value(re.sub(r'(ohms?|Ω|H|F)$', '', str(props[kk]).strip()))
                if x is not None:
                    return x
    return None


LOAD_FIELD = 'pdn.load'                 # a part's field with the current it draws (any case)


def field_loads(board, nets):
    """The loads the parts' own fields give for a rail of `nets`: ({ref: amps}, {ref: its return's ref}, notes).

    A part's `pdn.load` field (KiCad: a field of the symbol, a property of the footprint) is the current it draws:
        0.5        500m       2 A          amps, from this rail when the part has a pad on it
        +3V3=0.4, +1V8=100m                a part on several rails: by net
        1.5@Q2                             returning through another part's ground pads (--load REF=A@RETURN)
    A part on none of the nets, or whose field names other nets, is not a load of this rail; a field that cannot
    be read is named in the notes and gives none."""
    loads, returns, notes = {}, {}, []
    for p in parts_of(board):
        text = next((str(v).strip() for k, v in p.props.items() if str(k).strip().lower() == LOAD_FIELD), '')
        if not text or not p.on(nets):
            continue
        amps, back, bad = 0.0, None, False
        for item in text.split(','):
            net, eq, value = item.rpartition('=')
            value, at, ret = value.partition('@')
            a = parse_value(re.sub(r'\s*A(mps?)?$', '', value.strip(), flags=re.I))
            if a is None or a < 0 or (at and not ret.strip()):
                bad = True
                break
            if eq and net.strip() not in nets:
                continue                                    # another rail's share
            amps += a
            back = ret.strip() or back
        if bad:
            notes.append(f"{p.ref}: its {LOAD_FIELD} field {text!r} is not AMPS, NET=AMPS or AMPS@RETURN (ignored)")
        elif amps > 0:
            loads[p.ref] = amps
            if back:
                returns[p.ref] = back
    return loads, returns, notes


@dataclass
class Pad:
    part: str
    name: str
    net: str
    layers: list
    geom: object                # shapely, mm
    xy: tuple


@dataclass
class Part:
    ref: str
    value: str
    props: dict
    lib: str
    side: str
    pads: list
    raw: dict = field(repr=False, default=None)

    def nets(self):
        return {p.net for p in self.pads if p.net}

    def on(self, nets):
        return [p for p in self.pads if p.net in nets]


RETURN_NET = 8          # a net with more parts than this is a rail, not the node a load returns through


def return_part(pad_nets, ref, ground, rail=()):
    """The one part a load with no pad on ground returns through, or None (none such, or several): of the load's
    kind (its reference's letters: a half bridge's low-side switch for its high-side one), on one of the load's
    nets that is no rail (not of the rail's tree `rail`, at most RETURN_NET parts) and on ground. pad_nets:
    {reference: the nets of its pads}."""
    kind = re.match(r'[A-Za-z]*', ref).group(0)
    ground, found = set(ground), set()
    for net in pad_nets[ref] - {'', None} - set(rail) - ground:
        on = [r for r, ns in pad_nets.items() if net in ns and r != ref]
        if len(on) <= RETURN_NET:
            found |= {r for r in on if pad_nets[r] & ground and re.match(r'[A-Za-z]*', r).group(0) == kind}
    return found.pop() if len(found) == 1 else None


@dataclass
class Series:
    part: Part
    a: str                      # the net nearer the rail's net
    b: str
    kind: str                   # 'R', 'bead', 'jumper', 'through', 'inductor'
    R: float                    # DC resistance, ohms
    z100: float = None          # a bead's impedance at 100 MHz
    defaulted: bool = False
    L: float = 0.0              # a series inductor's inductance, H

    def pads(self, side):
        """Its pads on net a or b (side 'a' or 'b')."""
        net = getattr(self, side)
        return [q for q in self.part.pads if q.net == net]


def parts_of(board):
    out = []
    for fp in board['prims']['footprints']:
        if fp.get('dnp'):
            continue
        pads = [Pad(fp['ref'], p.get('name'), p.get('net'), p.get('layers', []), shape(p['geom']), tuple(p['xy']))
                for p in fp.get('pads', []) if p.get('geom')]
        out.append(Part(fp['ref'], fp.get('value', ''), fp.get('props', {}), fp.get('lib', ''), fp.get('side', 'F'),
                        pads, fp))
    return out


def classify(p, max_ohms=1.0):
    """(kind, R ohms, z100, defaulted) of a two-net part that can join a rail tree, ('inductor', henries, ...) for a
    power inductor, or None."""
    ref, val = p.ref.upper(), ' '.join(str(p.props.get(k, '')) for k in ('Val', 'Value')) + ' ' + str(p.value)
    if ref.startswith(('JP', 'NT')):
        return ('jumper', 1e-5, None, False)
    if ref.startswith('FB') or (ref.startswith('L') and BEAD.search(val.strip())):
        dcr = field_value(p, 'DCR', 'dcr')
        z = re.search(r'(\d+)\s*(R|Ω|ohm)\s*@', val, re.I)
        return ('bead', dcr or DEFAULT_BEAD_DCR, float(z.group(1)) if z else DEFAULT_BEAD_Z100, dcr is None)
    if re.match(r'L\d', ref):
        h = part_value(p)
        return ('inductor', h, None, False) if h else None
    if re.match(r'R\d', ref):
        r = part_value(p)
        if r is not None and r <= max_ohms:
            return ('R', max(r, 1e-5), None, False)
    return None


@dataclass
class Rail:
    board: dict
    net: str
    ground: tuple = ('GND',)
    loads: dict = field(default_factory=dict)       # {ref: amps}
    source: str = None                              # a part's ref (default: the power inductor feeding the tree)
    ground_source: str = None
    max_ohms: float = 1.0
    through: tuple = ()
    stop: tuple = ()
    dcr: dict = field(default_factory=dict)         # {ref: ohms}: series parts' DC resistance
    returns: dict = field(default_factory=dict)     # {load ref: ref}: a load returning through another part's ground
                                                    # pads (a bridge's high side through its low side), C9

    def __post_init__(self):
        self.parts = {p.ref: p for p in parts_of(self.board)}
        g = set(self.ground)
        self.nets, self.series, self.inductors, self.notes = {self.net}, [], [], []
        todo = [self.net]
        two = [p for p in self.parts.values() if len(p.nets()) == 2 and not (p.nets() & g)]
        while todo:
            n = todo.pop()
            for p in two:
                if n not in p.nets() or p.ref in self.stop or any(s.part is p for s in self.series):
                    continue
                other = next(x for x in p.nets() if x != n)
                L = 0.0
                if p.ref in self.through:
                    found = classify(p, float('inf'))
                    dcr = field_value(p, 'DCR', 'dcr')
                    if found is not None and found[0] == 'inductor':      # a series filter inductor
                        L = found[1]
                        c = ('inductor', dcr or DEFAULT_INDUCTOR_DCR, None, dcr is None)
                    else:
                        c = ('through', dcr or 1e-5, None, False)
                else:
                    c = classify(p, self.max_ohms)
                if c is None:
                    continue
                if c[0] == 'inductor' and p.ref not in self.through:
                    if not any(q is p for q, _ in self.inductors):
                        self.inductors.append((p, n))
                    continue
                if p.ref in self.dcr:
                    c = (c[0], float(self.dcr[p.ref]), c[2], False)
                self.series.append(Series(p, n, other, c[0], c[1], c[2], c[3], L))
                if other not in self.nets:
                    self.nets.add(other)
                    todo.append(other)
        # the source: the named part's pads on the tree, else the power inductors' tree-side pads
        if self.source:
            src = [self.parts[self.source]]
        else:
            src = [p for p, _ in self.inductors]
            if not src:
                raise ValueError(f"{self.net}: no power inductor feeds the rail tree ({', '.join(sorted(self.nets))}); "
                                 f"name its source (source = <ref>)")
            if len(src) > 1:
                self.notes.append(f"{len(src)} power inductors feed the rail: {', '.join(p.ref for p in src)} (all sources)")
        self.source_parts = src
        self.source_pads = [q for p in src for q in p.on(self.nets)]
        if not self.source_pads:
            raise ValueError(f"source {', '.join(p.ref for p in src)} has no pad on the rail "
                             f"({', '.join(sorted(self.nets))})")
        # the return's source: the named part's ground pads, else those of the parts on each inductor's switch node
        if self.ground_source:
            gp = self.parts[self.ground_source].on(g)
        else:
            gp = []
            for p in src:
                sw = [x for x in p.nets() if x not in self.nets]
                for q in self.parts.values():
                    if q is not p and sw and sw[0] in q.nets():
                        gp += q.on(g)
        if not gp:
            raise ValueError(f"{self.net}: no ground pads for the return's source: name ground_source = <ref>")
        self.ground_source_pads = gp
        # decaps: two-pad capacitors from the tree to ground (C1, and prefixed series such as CA1 / CE1 / CP1;
        # one whose value is not a capacitance is reported and left out by the PDN model)
        self.decaps = []
        for p in self.parts.values():
            ns = p.nets()
            if re.match(r'C[A-Z]{0,2}\d', p.ref.upper()) and len(p.pads) == 2 and ns & self.nets and ns & g and len(ns) == 2:
                self.decaps.append(p)
        missing = [r for r in self.loads if r not in self.parts]
        if missing:
            raise ValueError(f"loads not on the board: {', '.join(missing)}")
        for r in self.loads:
            if not self.parts[r].on(self.nets):
                raise ValueError(f"load {r} has no pad on the rail ({', '.join(sorted(self.nets))})")
        for r, back in self.returns.items():
            if r not in self.loads:
                raise ValueError(f"return given for {r}, which is not a load")
            if back not in self.parts:
                raise ValueError(f"load {r}'s return {back} is not on the board")
            if not self.parts[back].on(g):
                raise ValueError(f"load {r}'s return {back} has no pad on the ground ({', '.join(sorted(g))})")
        # a load with no pad on ground (a bridge's high-side switch): its current returns through another part.
        # Not named (REF=AMPS@RETURN), the board says which where it has just one such (return_part); none or
        # several: its return is left out, and the result says so (dc.solve_rail)
        pad_nets = {r: p.nets() for r, p in self.parts.items()}
        for r in self.loads:
            if r in self.returns or self.parts[r].on(g):
                continue
            back = return_part(pad_nets, r, g, self.nets)
            if back:
                self.returns[r] = back
                self.notes.append(f"{r} has no pad on ground: its return is taken through {back}, on its other net "
                                  f"and on ground ({r}=AMPS@PART names another)")

    def pads_of(self, part, role=None):
        """(rail pads, ground pads) of a part: the source's are source_pads and ground_source_pads."""
        if role == 'source':
            return self.source_pads, self.ground_source_pads
        back = self.returns.get(part.ref)
        return part.on(self.nets), (self.parts[back] if back else part).on(set(self.ground))

    def describe(self):
        lines = [f"rail {self.net}: nets {', '.join(sorted(self.nets))}"]
        for s in self.series:
            lh = f" {s.L * 1e6:.3g} uH" if s.L else ''
            lines.append(f"  {s.part.ref} {s.kind}{lh} {s.R:.3g} ohm{' (default DCR)' if s.defaulted else ''}: {s.a} -> {s.b}")
        lines.append(f"  source {', '.join(p.ref for p in self.source_parts)} ({len(self.source_pads)} pads), return "
                     f"source {len(self.ground_source_pads)} ground pads; {len(self.decaps)} decaps, "
                     f"{len(self.loads)} loads")
        return lines + [f"  note: {n}" for n in self.notes]


def terminal(pads, layers):
    """[(layer, polygon)] of pads: an SMD pad on its layer, a through-hole one (*.Cu, or several layers) on each of
    `layers` (the stack's copper layers, names)."""
    out = []
    for p in pads:
        ls = layers if any(l in ('*.Cu', '*.Cu*') for l in p.layers) else [l for l in p.layers if l in layers]
        out += [(l, p.geom) for l in ls]
    return out
