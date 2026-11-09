"""Capacitance: pcbqs's PEEC electrostatics (MQS(capacitance=True): constant charge per triangle on zero-thickness
copper sheets; homogeneous er, or dielectric='stack' with laterally infinite slabs by images) against FasterCap
(3D boundary elements, adaptive refinement, explicit dielectric interfaces).

    python validation/cap_fastercap.py [plates traces_air traces_fr4]

FasterCap: `FASTERCAP` or ~/opt/src/FasterCap-build/FasterCap (headless). Its copper is boxes of the real 35 um
thickness: a conductor names one surrounding permittivity, so each box's FR-4 face and its air faces are separate
'C' files merged with '+'. Its FR-4 is a finite slab (the board outline plus a margin), its interfaces the slab's
faces less the copper footprints. Compared: the Maxwell capacitance matrix between the conductors.
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import OUT, write_json  # noqa: E402

FASTERCAP = os.environ.get('FASTERCAP', str(Path.home() / 'opt/src/FasterCap-build/FasterCap'))
EPS0 = 8.8541878128e-12
T = 0.035


def plates():
    """Two 10 x 10 mm plates 0.5 mm apart in air (the lower on B.Cu at z=0, the upper on F.Cu at z=0.5)."""
    return dict(copper={'F.Cu': [('A', box(0, 0, 10, 10))], 'B.Cu': [('B', box(0, 0, 10, 10))]}, h=0.5, er=1.0,
                extent=box(0, 0, 10, 10))


def traces(er):
    """Two 0.3 mm traces 20 mm long, 0.2 mm apart, on F.Cu 0.2 mm over a 10 x 24 mm B.Cu plane; slab er."""
    return dict(copper={'F.Cu': [('T1', box(-0.35, -10, -0.05, 10)), ('T2', box(0.05, -10, 0.35, 10))],
                        'B.Cu': [('G', box(-5, -12, 5, 12))]}, h=0.2, er=er, extent=box(-5, -12, 5, 12))


def microstrip(er):
    """One 0.3 mm trace, 20 mm long, 0.2 mm over a 10 x 24 mm plane: Hammerstad-Jensen gives C per length."""
    return dict(copper={'F.Cu': [('T', box(-0.15, -10, 0.15, 10))], 'B.Cu': [('G', box(-5, -12, 5, 12))]}, h=0.2,
                er=er, extent=box(-5, -12, 5, 12))


def hammerstad(w, h, er, t=0.0):
    """Microstrip C per metre (Hammerstad-Jensen, zero thickness): sqrt(eeff) / (c Z0)."""
    u = w / h
    a = 1 + np.log((u ** 4 + (u / 52) ** 2) / (u ** 4 + 0.432)) / 49 + np.log(1 + (u / 18.1) ** 3) / 18.7
    b = 0.564 * ((er - 0.9) / (er + 3)) ** 0.053
    eeff = (er + 1) / 2 + (er - 1) / 2 * (1 + 10 / u) ** (-a * b)
    f = 6 + (2 * np.pi - 6) * np.exp(-(30.666 / u) ** 0.7528)
    z01 = 60 * np.log(f / u + np.sqrt(1 + (2 / u) ** 2))      # air
    return np.sqrt(eeff) / (299792458.0 * z01 / np.sqrt(eeff))


CASES = dict(plates=plates, traces_air=lambda: traces(1.0), traces_fr4=lambda: traces(4.4),
             microstrip_air=lambda: microstrip(1.0), microstrip_fr4=lambda: microstrip(4.4))


def board_of(c):
    h = c['h']
    return dict(stack=dict(copper=[dict(name='B.Cu', z=0.0, t=T), dict(name='F.Cu', z=h, t=T)],
                           layers=[dict(name='core', z0=0.0, z1=h, er=c['er'], tand=0.0)], thickness=h),
                copper={l: [dict(net=n, polygon=mapping(p)) for n, p in ps] for l, ps in c['copper'].items()},
                vias=[], outline=mapping(c['extent'].buffer(5)), prims=dict(footprints=[]))


def pcbqs_c(c, dmax):
    """Maxwell C (conductors in the order of their nets' first appearance) from pcbqs, and the time."""
    from pcbqs.mqs import MQS
    b = board_of(c)
    nets = [n for ps in c['copper'].values() for n, _ in ps]
    # one small terminal per conductor, only to name it in the capacitance matrix: a terminal's triangles share one
    # charge density, so a whole-conductor terminal would forbid the edge charge (a 0.3 mm trace lost half its C)
    from shapely.geometry import Point
    terms = {}
    for lay, ps in c['copper'].items():
        for n, p in ps:
            terms.setdefault(n, [(lay, Point(p.representative_point()).buffer(0.05, quad_segs=4))])
    t = time.perf_counter()
    kw = dict(er=c['er']) if c['er'] == 1.0 else dict(dielectric='stack')
    q = MQS(b, set(nets), terminals=terms, dmax=dmax, capacitance=True, log=lambda *a: None, **kw)
    C, node_labels, term_labels = q.capacitance_matrix()
    order = [term_labels[n] for n in nets]
    return dict(C=np.real(np.asarray(C)[np.ix_(order, order)]), names=nets, tris=len(q.tris), seconds=time.perf_counter() - t)


def _quads_of(poly, z):
    """Coarse panels of a polygon at height z: Triangle's constrained triangulation, no extra points."""
    import triangle as tr
    from common import polygons
    out = []
    for p in polygons(poly):
        rings = [np.asarray(p.exterior.coords)[:-1]] + [np.asarray(h.coords)[:-1] for h in p.interiors]
        verts = np.vstack(rings)
        segs, off = [], 0
        for r in rings:
            n = len(r)
            segs += [(off + k, off + (k + 1) % n) for k in range(n)]
            off += n
        A = dict(vertices=verts, segments=np.array(segs))
        if p.interiors:
            A['holes'] = np.array([list(h.representative_point().coords)[0] for h in map(type(p), p.interiors)])
        T_ = tr.triangulate(A, 'pq25Q')                 # a minimum angle: FasterCap struggles with slivers
        out += [[(*T_['vertices'][i], z) for i in tri] for tri in T_['triangles']]
    return out


def _panel(kind, name, pts, scale=1e-3):
    return f'{kind} {name} ' + '  '.join(' '.join(f'{v * scale:.9g}' for v in p) for p in pts)


def _box_faces(poly, z0, z1):
    """(the faces at z0, z1, the side quads) of a polygon extruded from z0 to z1."""
    sides = []
    for ring in [poly.exterior, *poly.interiors]:
        pts = list(ring.coords)
        sides += [[(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)] for (x0, y0), (x1, y1) in zip(pts, pts[1:])]
    return _quads_of(poly, z0), _quads_of(poly, z1), sides


def fastercap_input(c, t=T):
    """The single-file FasterCap input (metres) for a case, copper boxes t mm thick."""
    h, er = c['h'], c['er']
    main, files = ['* pcbqs validation'], {}
    if not t:                                       # zero-thickness sheets, as pcbqs's copper (air only)
        if er != 1.0:
            raise ValueError('zero-thickness copper only in air: a sheet on an interface has two permittivities')
        for lay, items in c['copper'].items():
            for name, poly in items:
                files[name] = [_panel('T', name, p) for p in _quads_of(poly, h if lay == 'F.Cu' else 0.0)]
                main.append(f'C {name} 1.0 0 0 0')
        return '\n'.join(main) + '\nEnd\n' + ''.join(f'File {n}\n* {n}\n' + '\n'.join(v) + '\nEnd\n'
                                                     for n, v in files.items())
    for lay, items in c['copper'].items():
        top = lay == 'F.Cu'
        for name, poly in items:
            z0, z1 = (h, h + t) if top else (-t, 0.0)
            low, high, sides = _box_faces(poly, z0, z1)
            in_fr4 = low if top else high                      # the face against the slab
            in_air = (high if top else low)
            files[f'{name}_air'] = [_panel('T', name, p) for p in in_air] + [_panel('Q', name, p) for p in sides]
            files[f'{name}_fr4'] = [_panel('T', name, p) for p in in_fr4]
            main += [f'C {name}_air 1.0 0 0 0 +', f'C {name}_fr4 {er:g} 0 0 0']
    if er != 1.0:                                              # the slab's faces, less the copper on them
        slab = c['extent']           # the plane's footprint, as in FastFieldSolvers' WP130527 example B: the plane
                                     # covers the slab's bottom; a wider slab meant ~0.2 mm panels over much more area
        from shapely.ops import unary_union
        topcu = unary_union([p for _, p in c['copper'].get('F.Cu', [])])
        botcu = unary_union([p for _, p in c['copper'].get('B.Cu', [])])
        top = _quads_of(slab.difference(topcu), h)
        bot = _quads_of(slab.difference(botcu), 0.0)
        pts = list(slab.exterior.coords)
        sides = [[(x0, y0, 0), (x1, y1, 0), (x1, y1, h), (x0, y0, h)] for (x0, y0), (x1, y1) in zip(pts, pts[1:])]
        files['slab'] = [_panel('T', 'd', p) for p in top + bot] + [_panel('Q', 'd', p) for p in sides]
        cx, cy = slab.centroid.x * 1e-3, slab.centroid.y * 1e-3
        main.append(f'D slab 1.0 {er:g} 0 0 0 {cx:.9g} {cy:.9g} {h * 0.5e-3:.9g} -')     # reference point inside
    text = '\n'.join(main) + '\nEnd\n'
    for name, lines in files.items():
        text += f'File {name}\n* {name}\n' + '\n'.join(lines) + '\nEnd\n'
    return text


def fastercap_c(c, folder, rel=0.002, t=T):        # 0.5% stopped ~3% low on a microstrip
    """Maxwell C from FasterCap (auto refinement to `rel`), the conductor names in its order, and the time."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'case.lst').write_text(fastercap_input(c, t))
    t = time.perf_counter()
    # its out-of-core files go to TMPDIR (default /tmp: RAM here, so big runs aborted with "File error"): the run's
    # own folder on disk, emptied after; -f1: out of core only when the links exceed the free memory
    tmp = folder / 'tmp'
    tmp.mkdir(exist_ok=True)
    with open(folder / 'fastercap.log', 'w') as log:              # streamed: a long refinement shows progress
        p = subprocess.run([FASTERCAP, '-b', 'case.lst', f'-a{rel}', '-ap', '-f1'], cwd=folder, stdout=log,
                           stderr=subprocess.STDOUT, text=True, timeout=14400, env=dict(os.environ, TMPDIR=str(tmp)))
    for stale in tmp.glob('frc*.tmp'):
        stale.unlink()
    out = (folder / 'fastercap.log').read_text()
    blocks = out.split('Capacitance matrix is:')
    if p.returncode or len(blocks) < 2:
        raise RuntimeError('FasterCap failed; see ' + str(folder / 'fastercap.log'))
    lines = blocks[-1].strip().splitlines()
    n = int(re.search(r'Dimension (\d+) x', lines[0])[1])
    names = [re.sub(r'^g\d+_', '', l.split()[0]) for l in lines[1:1 + n]]
    C = np.array([[float(x) for x in l.split()[1:1 + n]] for l in lines[1:1 + n]])
    return dict(C=C, names=names, seconds=time.perf_counter() - t,
                panels=int(re.findall(r'Number of panels after refinement: (\d+)', out)[-1]))


def run(name, dmaxes=(0.5, 0.25), thin=False):        # 0.125 on the 20 mm traces: dense PEEC beyond 32 GB
    c = CASES[name]()
    rows = []
    for dm in dmaxes:
        p = pcbqs_c(c, dm)
        rows.append(dict(tool='pcbqs', mesh=dm, C=p['C'], names=p['names'], n=p['tris'], seconds=p['seconds']))
        print(f'{name}: pcbqs dmax {dm}: {p["tris"]:,} triangles, {p["seconds"]:.1f} s\n{np.round(p["C"] * 1e12, 4)} pF', flush=True)
    f = fastercap_c(c, OUT / 'fastercap' / (name + ('_thin' if thin else '')), t=0.0 if thin else T)
    order = [f['names'].index(n) for n in rows[-1]['names']]
    Cf = f['C'][np.ix_(order, order)]
    rows.append(dict(tool='fastercap', C=Cf, names=rows[-1]['names'], n=f['panels'], seconds=f['seconds']))
    print(f'{name}: FasterCap: {f["panels"]:,} panels, {f["seconds"]:.1f} s\n{np.round(Cf * 1e12, 4)} pF', flush=True)
    d = rows[-2]['C'] / Cf - 1
    print(f'{name}: pcbqs / FasterCap - 1 (%):\n{np.round(d * 100, 2)}', flush=True)
    write_json(OUT / 'fastercap' / (name + ('_thin' if thin else '')) / 'result.json', dict(case=name, thin=thin, rows=rows))
    return rows


if __name__ == '__main__':
    thin = '--thin' in sys.argv                    # zero-thickness sheets in FasterCap (air): isolates the thickness
    for name in [a for a in sys.argv[1:] if not a.startswith('--')] or list(CASES):
        run(name, thin=thin)
