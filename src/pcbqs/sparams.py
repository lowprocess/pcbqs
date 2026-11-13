"""
python -m pcbqs.sparams: low-frequency S-parameters of a board's ports, from the quasi-static currents of its copper.

    python -m pcbqs.sparams board.kicad_pcb --nets SIG,GND --fmin 1e3 --fmax 1e8 --out lf
    python -m pcbqs.sparams run/board_board.json --out lf          # a kicad2ems board file: its ports as they are
    python -m pcbqs.sparams --describe --json

From DC up to where the board is a tenth of a wavelength, the full-wave solvers are at their worst (a time-domain
run needs millions of steps, a moment method's matrix loses its conditioning), and what decides the answer is
where the current goes: the resistance of the copper it takes and the area of the loop it makes with its return.
A ground via missing or in the wrong place, a slot under a trace, a return that has to change planes: each shows
as loop inductance and as impedance two ports share. That is the magneto-quasistatic solve (pcbqs.mqs: the sheet
currents of every layer, the vias, the mutual inductance of all of it), with a port as it is in a measurement:

    each port: a 1 A source and its reference impedance, both between the trace's end and the copper under it;
    one solve per driven port; S from the currents in the terminations:
        S[j, k] = 2 I_j sqrt(z0_j / z0_k) - (j == k)        I_j: the current in port j's termination

so no impedance matrix is needed (the ports of a board without capacitance have none: nothing joins a trace to
its plane but the other ports).

Resistance and inductance alone hold while the copper's capacitance carries next to no current: a trace's own
capacitance moves S by 0.01 where pi f C z0 = 0.01 (a matched line's S11 first: its capacitance is what cancels
its inductance). Above that frequency the copper's charge is solved with its currents (pcbqs.peec: one model
for R, L and C, in the board's dielectrics, with their loss), which holds to about a tenth of a wavelength; there
is no radiation in either. `--capacitance auto` (the default) adds the charge when the band reaches above the
first limit; both limits are estimated and written with the result (`valid_to_Hz`).

It writes <out>/sparams.json (the ports, the frequencies, S, the path between each two ports as a series R and L,
the limits, the notes), <out>/<name>_pcbqs.s<N>p (Touchstone, with the port names as kicad2ems writes them) and,
where kicad2ems's viewer reads a run, the current of each layer with one port driven, at a few frequencies of the
band (<out>/sparams/exc_<port>/dump_J_<layer>_fd.npz, per ampere into the port): where the return goes, and how it
moves under the trace as the frequency rises.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

C0, EPS0 = 299792458.0, 8.8541878128e-12
G_BAND, G_PORTS, G_MESH, G_OUT = 'Frequencies', 'Ports', 'Mesh', 'Output'
OPTIONS = [            # (key, kind, default, help, group): _cli's table, which --describe gives a front end
    ('fmin', 'float', 1e3, 'the lowest frequency, Hz', G_BAND),
    ('fmax', 'float', 1e8, 'the highest frequency, Hz', G_BAND),
    ('points', 'int', 61, 'frequencies, spaced evenly on a log scale', G_BAND),
    ('nets', 'str', '', "the nets to model, comma separated (a KiCad board; default: all its copper); a kicad2ems "
     "board file holds its own", G_PORTS),
    ('z0', 'float', None, "every port's reference impedance, ohm (default: each port's own, 50)", G_PORTS),
    ('dmax', 'float', None, 'the largest mesh cell, mm (default: a twenty-fifth of the board, 0.5 to 3)', G_MESH),
    ('fine', 'float', None, "the cell at and around the ports' nets, mm (default: the narrowest port's width; the "
     "path's inductance is within 2 percent of a mesh three times finer)", G_MESH),
    ('backend', 'str', 'auto', 'dense, hierarchical (compressed, for large meshes) or auto', G_MESH),
    ('capacitance', 'str', 'auto', "the copper's charge with its currents (R, L and C): on, off, or auto (when the "
     "band reaches above where resistance and inductance alone hold)", G_MESH),
    ('thickness', 'bool', False, "the copper's thickness: its two faces as sheets, with the skin effect between "
     "them, instead of one sheet with its DC resistance (twice the unknowns; a narrow trace's inductance and "
     "capacitance to a few percent, with a cell no wider than the trace)", G_MESH),
    ('maps', 'str', '', 'the frequencies of the current maps, Hz, comma separated (default: three across the band; '
     'none: no maps)', G_OUT),
    ('map_port', 'str', '', 'the port driven in the current maps (default: the first)', G_OUT),
]
EDGE_RINGS = 6          # with the charge solved: cells across the copper's edges, graded (the edge charge: a matched
                        # line's |S11| 8.8 % above the field solver's line with none, 0.1 % with these; two traces
                        # 0.1 mm apart -23 % in mutual capacitance, then -3.6 %), over half the narrowest port's width
DENSE_MAX = 9000        # triangles: above it the mesh is made coarser (auto), or the matrices compressed
CHARGE_MAX = 5000       # the same with the charge solved too (a larger, dense system)
COARSEN = 1.4           # a mesh over the limit: both cell sizes by this, up to four times
CELLS = 500             # the maps' grid, along the board's longer side


def terminal_box(p):
    """The patch a port takes of its trace and of the copper under it: the trace's width across, half of it (at
    least 0.2 mm) along the trace from the port into the circuit."""
    from shapely.geometry import box
    a = int(p['axis'])
    reach = max(0.5 * p['w'], 0.2)
    lo, hi = [0.0, 0.0], [0.0, 0.0]
    lo[a], hi[a] = sorted((p['c'][a], p['c'][a] + p['sgn'] * reach))
    lo[1 - a], hi[1 - a] = p['c'][1 - a] - p['w'] / 2, p['c'][1 - a] + p['w'] / 2
    return box(lo[0], lo[1], hi[0], hi[1])


def part_patches(p):
    """The two patches of copper a board's part joins (a kicad2ems board's `parts`: the gap between its pads, from
    s to e, along its axis): the copper at each end of the gap, the gap's width across and half of that (at least
    0.2 mm) along."""
    from shapely.geometry import box
    a = int(p['axis'])
    g0, g1 = sorted((p['s'][a], p['e'][a]))
    w0, w1 = sorted((p['s'][1 - a], p['e'][1 - a]))
    reach = max(0.5 * (w1 - w0), 0.2)
    out = []
    for lo, hi in ((g0 - reach, g0), (g1, g1 + reach)):
        b = [0.0] * 4
        b[a], b[a + 2], b[1 - a], b[3 - a] = lo, hi, w0, w1
        out.append(box(*b))
    return out


def part_elements(board):
    """(the parts the solve takes: [dict(ref, layer, patches, nets, R, L, C)], those it leaves out: [(ref, why)]).
    A part is a series R, L and C between its two pads, as kicad2ems's other solvers have it (a value of one of
    them: that element alone); one with no value, or with a pad that is on no copper of its layer, is left out."""
    from shapely.geometry import shape
    took, left = [], []
    for p in board.get('parts') or []:
        vals = {k: p.get(k) for k in 'RLC'}
        if all(v is None for v in vals.values()):
            left.append((p['ref'], 'no value'))
            continue
        patches, nets = part_patches(p), []
        for patch in patches:
            on = [r.get('net') or '' for r in board['copper'].get(p['layer'], []) if shape(r['polygon']).intersects(patch)]
            nets.append(on[0] if on else None)
        if None in nets:
            left.append((p['ref'], f"a pad on no copper of {p['layer']}"))
            continue
        took.append(dict(ref=p['ref'], layer=p['layer'], patches=patches, nets=nets, R=float(vals['R'] or 0.0),
                         L=float(vals['L'] or 0.0), C=None if vals['C'] is None else float(vals['C'])))
    return took, left


def _nets(board):
    return {r['net'] for records in board['copper'].values() for r in records if r.get('net')}


def signal_nets(board, ports):
    """The nets the ports' signals are on: the ports' own, and those a part joins them to (a filter's inner
    nodes), but not the ground nets nor the nets of the copper under the ports."""
    from shapely.geometry import shape
    ground = set(board.get('ground_nets') or [])
    for p in ports:                                         # (a board file with no ground named: what is under a port)
        patch = terminal_box(p)
        ground |= {r.get('net') for r in board['copper'].get(p.get('ref_layer'), []) if r.get('net')
                   and shape(r['polygon']).intersects(patch)}
    nets = {p['net'] for p in ports if p.get('net')}
    links = [set(e['nets']) - ground for e in part_elements(board)[0]]
    grew = True
    while grew:
        grew = False
        for link in links:
            if link & nets and not link <= nets:
                nets |= link
                grew = True
    return nets


def _signal(board, ports):
    """The copper of the ports' signals (signal_nets), every layer's, as one geometry."""
    from shapely.geometry import shape
    from shapely.ops import unary_union
    nets = signal_nets(board, ports)
    return unary_union([shape(r['polygon']) for records in board['copper'].values() for r in records
                        if r.get('net') in nets])


def limits(board, ports):
    """(f_wave, f_cap, C): the frequency where the ports' copper is a tenth of a wavelength long; the one where its
    capacitance to the copper nearest above and below moves S by 0.01, pi f C z0 = 0.01; that capacitance, F (an
    estimate: each piece as a line of its mean width over each neighbour, parallel plates and Wheeler's fringe,
    three quarters of it where only one side has copper and part of the field is in the air)."""
    from shapely.geometry import shape
    zs = {c['name']: c['z'] for c in board['stack']['copper']}
    er = max((l.get('er', 1.0) for l in board['stack'].get('layers', [])), default=1.0)
    nets = signal_nets(board, ports)
    sig = _signal(board, ports)
    x0, y0, x1, y1 = sig.bounds if not sig.is_empty else (0, 0, 1, 1)
    f_wave = C0 / (10 * np.sqrt(er) * max(np.hypot(x1 - x0, y1 - y0), 1e-3) * 1e-3)
    cap = 0.0
    for lay, records in board['copper'].items():
        others = [zs[l] - zs[lay] for l, rs in board['copper'].items()
                  if l != lay and any(r.get('net') not in nets for r in rs)]
        sides = [min(side) for side in ([d for d in others if d > 0], [-d for d in others if d < 0]) if side]
        for r in records:
            if r.get('net') not in nets:
                continue
            g = shape(r['polygon'])
            w = 2 * g.area / max(g.length, 1e-9)                # a long piece's width
            for h in sides:
                fringe = (1.393 + 0.667 * np.log(w / h + 1.444)) * h / w
                cap += EPS0 * er * g.area * 1e-6 / (h * 1e-3) * max(1.0, (0.75 if len(sides) == 1 else 0.5) * (1 + fringe))
    z0 = max(p.get('z0', 50.0) for p in ports)
    return f_wave, (0.01 / (np.pi * z0 * cap) if cap else np.inf), cap


def solve(board, freqs, ports=None, z0=None, dmax=None, fine=None, backend='auto', capacitance='auto',
          thickness=False, log=print, edge_rings=None):
    """The S-parameters of a board's ports at freqs (Hz, all positive): dict(f, S (nf, n, n), names, zref, q, drive,
    info). board: a kicad2ems board dict (stack, copper, vias, ports, parts); ports: its own unless given, each
    dict(name, layer, ref_layer, c, axis, sgn, w, z0, net), single-ended, between a trace and the copper under it.
    Its parts are series R, L and C between their pads (part_elements).
    capacitance: 'on', 'off' or 'auto' (the charge solved too when freqs reach above where R and L alone hold).
    thickness: the copper as its two faces (MQS's conductor_loss='slab') instead of one sheet of its DC resistance.
    edge_rings: with the charge, the graded cells across the copper's edges (default EDGE_RINGS; 1: none).
    drive[k]: the solve with port k driven (pcbqs.mqs.Result) and the current that entered the port, for the
    maps. A mesh left to the defaults is made coarser until the dense solve takes it."""
    from shapely.geometry import shape
    from .mqs import MQS
    f = np.atleast_1d(np.asarray(freqs, float))
    if f.ndim != 1 or not len(f) or not np.isfinite(f).all() or (f <= 0).any():
        raise ValueError('frequencies must be finite and positive')
    ports = [dict(p) for p in (board.get('ports') if ports is None else ports)]
    if not ports:
        raise ValueError('the board has no ports: S-parameters need at least one')
    bad = [p['name'] for p in ports if p.get('kind', 'z') != 'z' or not p.get('ref_layer')]
    if bad:
        raise ValueError(f"ports {', '.join(bad)}: only ports between a trace and the copper under it are modelled "
                         "(no gap ports)")
    if backend not in ('auto', 'dense', 'hierarchical'):
        raise ValueError(f"backend {backend!r}: auto, dense or hierarchical")
    capacitance = {True: 'on', False: 'off'}.get(capacitance, capacitance)
    if capacitance not in ('auto', 'on', 'off'):
        raise ValueError(f"capacitance {capacitance!r}: auto, on or off")
    for p in ports:
        p['z0'] = float(z0 if z0 else p.get('z0') or 50.0)
    names = [p['name'] for p in ports]
    terms = {}
    for p in ports:
        patch = terminal_box(p)
        for sign, lay, what in (('+', p['layer'], 'its trace'), ('-', p['ref_layer'], 'the copper under it')):
            if not any(shape(r['polygon']).intersects(patch) for r in board['copper'].get(lay, [])):
                raise ValueError(f"port {p['name']}: no copper on {lay} at ({p['c'][0]:.2f}, {p['c'][1]:.2f}) for {what}")
            terms[p['name'] + sign] = [(lay, patch)]
    parts, left_out = part_elements(board)                  # the board's lumped parts: between their pads
    for e in parts:
        for i, patch in enumerate(e['patches']):
            terms[f"{e['ref']}:{i + 1}"] = [(e['layer'], patch)]
    f_wave, f_cap, cap = limits(board, ports)
    charge = capacitance == 'on' or (capacitance == 'auto' and f.max() > f_cap)
    sig = _signal(board, ports)
    zs = {c['name']: c['z'] for c in board['stack']['copper']}
    height = min(abs(zs[p['layer']] - zs[p['ref_layer']]) for p in ports)
    x0, y0, x1, y1 = shape(board['outline']).bounds if board.get('outline') else sig.buffer(10).bounds
    auto = not dmax and not fine                            # a mesh nobody chose may be made coarser
    dmax = float(dmax) if dmax else float(np.clip(max(x1 - x0, y1 - y0) / 25, 0.5, 3.0))
    fine = float(fine) if fine else max(min(p['w'] for p in ports), 0.2)
    limit = (CHARGE_MAX if charge else DENSE_MAX) * (2 if thickness else 1)     # a triangle on each face
    t0 = time.perf_counter()
    kw = dict(capacitance=True, dielectric='stack') if charge else {}
    if charge and edge_rings != 1:                          # the charge crowds at the copper's edges: graded rings
        kw.update(edge_rings=int(edge_rings or EDGE_RINGS), edge_width=max(0.5 * min(p['w'] for p in ports), 0.05))
    if thickness:
        kw['conductor_loss'] = 'slab'
    for attempt in range(5):
        # the return current of a trace lies within a few heights of it: the fine cells reach that far
        zone = sig.buffer(float(np.clip(3 * height, 2 * fine, 3.0)))
        q = MQS(board, _nets(board), terminals=terms, dmax=dmax, fine=zone, dmax_fine=min(fine, dmax), log=log, **kw)
        if not auto or backend == 'hierarchical' or len(q.tris) <= limit or attempt == 4:
            break
        log(f"sparams: {len(q.tris):,} triangles is over {limit:,}: a coarser mesh ({dmax * COARSEN:.2g} mm, "
            f"{fine * COARSEN:.2g} mm at the ports' nets)")
        dmax, fine = dmax * COARSEN, fine * COARSEN
    q.matrix_backend = ('hierarchical' if len(q.tris) > limit else 'dense') if backend == 'auto' else backend
    for p in ports:
        q.element(p['name'] + '.Z', p['name'] + '+', p['name'] + '-', R=p['z0'], part_h=0)
        q.source(p['name'] + '.I', p['name'] + '-', p['name'] + '+', part_h=0)     # into the trace
    for e in parts:                                         # (its own inductance is its value's: no loop over it)
        q.element(e['ref'], e['ref'] + ':1', e['ref'] + ':2', R=e['R'], L=e['L'], C=e['C'], part_h=0)
    if parts or left_out:
        log(f"sparams: {len(parts)} part(s) between their pads"
            + (f"; left out: {', '.join(f'{ref} ({why})' for ref, why in left_out)}" if left_out else ''))
    n = len(ports)
    S = np.zeros((len(f), n, n), complex)
    drive = []
    log(f"sparams: {n} port(s) driven in turn, {len(f)} frequencies, "
        + ('currents and charge (R, L, C)' if charge else 'currents (R, L)'))
    results = q.solve(f, amps=[{p['name'] + '.I': float(j == k) for j, p in enumerate(ports)} for k in range(n)])
    for k, (pk, r) in enumerate(zip(ports, results)):
        part = {x['name']: r.i.shape[1] - len(q.parts) + i for i, x in enumerate(q.parts)}     # the parts: last
        for j, pj in enumerate(ports):
            S[:, j, k] = 2 * r.i[:, part[pj['name'] + '.Z']] * np.sqrt(pj['z0'] / pk['z0']) - (j == k)
        drive.append((r, 1.0 - r.i[:, part[pk['name'] + '.Z']]))                    # what went into the board
    info = dict(triangles=int(len(q.tris)), backend=q.matrix_backend, dmax_mm=dmax, fine_mm=min(fine, dmax),
                seconds=round(time.perf_counter() - t0, 2), capacitance=bool(charge), thickness=bool(thickness),
                edge_rings=int(kw.get('edge_rings', 1)),
                valid_to_Hz=float(f_wave if charge else min(f_wave, f_cap)),
                tenth_wavelength_Hz=float(f_wave), capacitance_limit_Hz=float(f_cap), capacitance_F=float(cap),
                parts=[e['ref'] for e in parts], parts_left_out=[list(x) for x in left_out],
                model=("copper resistance, inductance and capacitance (quasi-static currents and charge); no "
                       "radiation" if charge else
                       'copper resistance and inductance (magneto-quasistatic); no capacitance, no radiation')
                + ("; the copper's two faces, the skin effect between them" if thickness else
                   '; the copper as one sheet of its DC resistance'))
    return dict(f=f, S=S, names=names, zref=[p['z0'] for p in ports], ports=ports, q=q, drive=drive, info=info)


OPEN = 1e-9             # |S| between two ports below this: no current gets from one to the other


def paths(f, S, names, zref):
    """Each two ports' path as a series R and L: [dict(a, b, joined, R_ohm, L_H)], per frequency, from the ports'
    admittance matrix: Z = -1 / Y[b, a], the element between a and b of the network of two-terminal elements that
    has that Y (what capacitance there is goes to the ports' own shunt elements, not into the path). Exact for
    ports joined by one path each; a figure to compare layouts by where paths share copper. joined False (R and L
    None): no current gets from one port to the other, the copper of the two, or of their returns, is not
    connected. C_F: the capacitance that closes such a path instead, where the charge is solved (the path is a
    capacitor at the lowest frequency: two planes that no via joins); else None."""
    n = len(names)
    g = 1 / np.sqrt(np.asarray(zref, float))
    eye = np.eye(n)
    Y = np.array([g[:, None] * np.linalg.solve(eye + s, eye - s) * g[None, :] for s in S])
    out = []
    for a in range(n):
        for b in range(a + 1, n):
            if np.abs(S[:, b, a]).max() < OPEN:
                out.append(dict(a=names[a], b=names[b], joined=False, R_ohm=None, L_H=None, C_F=None))
                continue
            Z = -1 / Y[:, b, a]
            if Z[0].imag < -10 * abs(Z[0].real) - 1e3:          # a capacitor: nothing joins them at DC
                out.append(dict(a=names[a], b=names[b], joined=False, R_ohm=None, L_H=None,
                                C_F=float(-1 / (2 * np.pi * f[0] * Z[0].imag))))
                continue
            out.append(dict(a=names[a], b=names[b], joined=True, R_ohm=Z.real.tolist(),
                            L_H=(Z.imag / (2 * np.pi * f)).tolist(), C_F=None))
    return out


def apart_note(p):
    """What to say of two ports that nothing joins at DC (paths()'s joined False)."""
    text = (f"{p['a']} and {p['b']} are not joined at DC: their copper, or the copper their returns take, is not "
            "connected (a ground via missing between two planes?).")
    if p.get('C_F'):
        return text + (f" {p['C_F'] * 1e12:.3g} pF closes the path as the frequency rises: an open circuit at the "
                       "low end.")
    return text + (" With resistance and inductance alone no current passes at any frequency; with the charge "
                   "solved too (--capacitance on), the copper's capacitance closes the path higher up.")


def write_touchstone(path, f, S, zref, names, comment=''):
    """A Touchstone v1 file (Hz, S, real and imaginary, one reference), its ports named as kicad2ems names them."""
    n = S.shape[1]
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(f"! pcbqs {comment}\n! ports: " + ', '.join(f"{i + 1}={nm}" for i, nm in enumerate(names)) + '\n')
        fh.write(f"# Hz S RI R {zref[0]:.6g}\n")
        for fi, s in zip(f, S):
            rows = [[s[0, 0], s[1, 0], s[0, 1], s[1, 1]]] if n == 2 else [list(r) for r in s]
            first = True
            for row in rows:
                for k in range(0, len(row), 4):
                    fh.write((f"{fi:.9e}" if first else ' ' * 15) + ' ' +
                             ' '.join(f"{v.real:.9e} {v.imag:.9e}" for v in row[k:k + 4]) + '\n')
                    first = False


def write_maps(out, board, r, port, freqs, name='sparams', log=print):
    """The sheet current of each layer with `port` driven, at the frequencies of the solve nearest to freqs, per
    ampere into the port, where kicad2ems's viewer reads a run: <out>/<name>_board.json and
    <out>/<name>/exc_<port>/dump_J_<layer>_fd.npz (A/mm; zero off the copper) with their dump_info.json. Returns
    dict(board, dumps=[{layer, file}], frequencies_Hz)."""
    from types import SimpleNamespace
    from matplotlib.tri import Triangulation
    from shapely.geometry import shape, box
    from ._artifacts import Dump, write_info, write_npz
    from .mqs_layout import crop_layout, geometry
    from .pdn.viewer import _copper, grid_of
    q, f = r['q'], r['f']
    k = r['names'].index(port)
    res, entered = r['drive'][k]
    idx = sorted({int(np.argmin(np.abs(np.log(f) - np.log(v)))) for v in freqs})
    outline = shape(board['outline']) if board.get('outline') else \
        box(*np.concatenate([q.nodes[:, :2].min(0), q.nodes[:, :2].max(0)]) * 1e3).buffer(1.0, join_style=2)
    nets = _nets(board)
    layout = crop_layout(board, outline, nets, set())
    with open(os.path.join(out, f"{name}_board.json"), 'w', encoding='utf-8') as fh:
        json.dump(layout, fh)
    exc = os.path.join(out, name, f"exc_{port}")
    os.makedirs(exc, exist_ok=True)
    x, y = grid_of(outline.bounds, cells=CELLS)
    X, Y = np.meshgrid(x, y, indexing='ij')
    cu_z = {c['name']: c['z'] for c in board['stack']['copper']}
    density = [q.current_density(res, i) for i in idx]            # per frequency {layer: (centroids, J A/m)}
    dumps, files = [], []
    for lay in sorted((l for l in q.layer_tris if '#' not in str(l)), key=lambda l: -cu_z.get(l, 0.0)):
        tris = q.tris[q.layer_tris[lay]]
        used = np.unique(tris)
        local = np.searchsorted(used, tris)
        finder = Triangulation(q.nodes[used, 0] * 1e3, q.nodes[used, 1] * 1e3, local).get_trifinder()
        hit = np.asarray(finder(X, Y))
        data = np.zeros((len(idx),) + X.shape + (1, 3), np.complex64)
        for n, (i, d) in enumerate(zip(idx, density)):
            j = np.asarray(d[lay][1]) * 1e-3 / entered[i]        # A/mm per ampere into the port
            data[n, :, :, 0, :2] = np.where((hit >= 0)[..., None], j[np.clip(hit, 0, None)], 0)
        z = cu_z.get(lay, 0.0)
        dm = Dump(f"J@{lay}", 'J', (x[0], y[0], z), (x[-1], y[-1], z), [float(f[i]) for i in idx], 2, lay)
        write_npz(os.path.join(exc, dm.name + '.npz'), dict(x=x, y=y, z=np.array([z])), dm.freqs, data)
        dumps.append(dm)
        files.append(dict(layer=lay, file=os.path.join(name, f"exc_{port}", dm.name + '.npz')))
    m = SimpleNamespace(outline=outline, cu_z=cu_z, copper=_copper(board, nets, outline), ports=[],
                        vias=[(v['x'], v['y'], v['r'], v['z0'], v['z1']) for v in board.get('vias', [])],
                        thickness=board['stack'].get('thickness', max(cu_z.values())))
    write_info(exc, dumps, m)
    path = os.path.join(exc, 'dump_info.json')
    with open(path, encoding='utf-8') as fh:
        info = json.load(fh)
    for d in info['dumps']:
        d['unit'] = 'A/mm per A'
    info['ports'] = [dict(name=p['name'], at=[float(p['c'][0]), float(p['c'][1])]) for p in r['ports']]
    info.update(geometry(layout))
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(info, fh)
    log(f"sparams: current maps with {port} driven at {', '.join(f'{f[i]:.3g}' for i in idx)} Hz")
    return dict(board=f"{name}_board.json", dumps=files, port=port, frequencies_Hz=[float(f[i]) for i in idx])


def describe():
    from ._cli import describe as d
    return d(OPTIONS, features=['viewer'])


def main(argv=None):
    from ._cli import add_options, load_board, print_describe
    ap = argparse.ArgumentParser(prog='python -m pcbqs.sparams', description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('board', nargs='?', help='a .kicad_pcb (its P<n> footprints the ports) or a kicad2ems board.json')
    ap.add_argument('--out', default='sparams-results', help='the results folder (default: sparams-results)')
    ap.add_argument('--describe', action='store_true', help='the options (with --json: for a front end)')
    ap.add_argument('--json', action='store_true')
    add_options(ap, OPTIONS)
    a = ap.parse_args(argv)
    if a.describe:
        print_describe(describe(), a.json)
        return 0
    if not a.board:
        ap.error('the board is required')
    if not (0 < a.fmin <= a.fmax) or a.points < 1:
        ap.error('--fmin and --fmax: positive, fmin not above fmax; --points at least 1')
    say = lambda s: print(s if s.startswith(('sparams:', 'mqs:')) else f"sparams: {s}", flush=True)
    try:
        say(f"reading {os.path.basename(a.board)}")
        nets = {s.strip() for s in a.nets.split(',') if s.strip()}
        board = load_board(a.board, nets or None)
        f = np.geomspace(a.fmin, a.fmax, a.points) if a.points > 1 else np.array([a.fmin])
        r = solve(board, f, z0=a.z0, dmax=a.dmax, fine=a.fine, backend=a.backend, capacitance=a.capacitance,
                  thickness=a.thickness, log=say)
    except (ValueError, RuntimeError) as exc:
        print(f"sparams: error: {exc}", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(a.board))[0]
    name = name[:-len('_board')] if name.endswith('_board') else name
    n = len(r['names'])
    ts = f"{name}_pcbqs.s{n}p"
    info = r['info']
    write_touchstone(os.path.join(a.out, ts), r['f'], r['S'], r['zref'], r['names'], 'quasi-static: ' + info['model'])
    notes = [(f"The copper's currents and charge, without radiation: to about {info['valid_to_Hz']:.3g} Hz, where "
              "the ports' copper is a tenth of a wavelength long." if info['capacitance'] else
              f"Resistance and inductance of the copper only: above about {info['valid_to_Hz']:.3g} Hz the copper's "
              f"capacitance ({info['capacitance_F'] * 1e12:.3g} pF, estimated) moves S by more than 0.01 "
              "(--capacitance on, or auto with a band that reaches there, solves the charge too).")]
    over = int((r['f'] > info['valid_to_Hz']).sum())
    if over:
        notes.append(f"{over} of the {len(r['f'])} frequencies are above that.")
    for p in paths(r['f'], r['S'], r['names'], r['zref']):
        if not p['joined']:
            notes.append(apart_note(p))
    viewer = None
    if a.maps.strip().lower() != 'none':
        at = [float(v) for v in a.maps.split(',') if v.strip()] or list(np.geomspace(a.fmin, a.fmax, 3))
        port = a.map_port or r['names'][0]
        if port not in r['names']:
            print(f"sparams: error: --map-port {port!r}: one of {', '.join(r['names'])}", file=sys.stderr)
            return 2
        try:
            viewer = write_maps(a.out, board, r, port, at, log=say)
        except Exception as exc:                            # noqa: BLE001 (the maps are extra: the answer stands)
            notes.append(f"current maps not written: {exc}")
    out = dict(format=1, tool='pcbqs.sparams', board=os.path.abspath(a.board), touchstone=ts, names=r['names'],
               zref=r['zref'], f_Hz=r['f'].tolist(), S_re=r['S'].real.tolist(), S_im=r['S'].imag.tolist(),
               paths=paths(r['f'], r['S'], r['names'], r['zref']), settings={k: getattr(a, k) for k, *_ in OPTIONS},
               viewer=viewer, notes=notes, **info)
    with open(os.path.join(a.out, 'sparams.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh)
    for p in out['paths']:
        if not p['joined']:
            say(apart_note(p))
            continue
        say(f"{p['a']} to {p['b']}: {p['R_ohm'][0] * 1e3:.3f} mOhm at {r['f'][0]:.3g} Hz, "
            f"{p['L_H'][len(r['f']) // 2] * 1e9:.3f} nH at {r['f'][len(r['f']) // 2]:.3g} Hz")
    say(f"{info['triangles']:,} triangles ({info['backend']}, {'R, L, C' if info['capacitance'] else 'R, L'}"
        f"{', two faces' if info['thickness'] else ''}), "
        f"{info['seconds']} s; valid to about {info['valid_to_Hz']:.3g} Hz; wrote {os.path.join(a.out, ts)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
