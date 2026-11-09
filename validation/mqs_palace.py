"""Copper loop inductance and near field: pcbqs.mqs.MQS (PEEC on the copper sheets) against Palace (3D FEM, copper as
zero-thickness PEC sheets, vias as PEC-walled holes, an air box) and Grover's closed form where there is one.

    python validation/mqs_palace.py [case ...]       # cases: loop, plane_return

The port is a flat lumped-port sheet across a gap in the copper (Palace) and pcbqs's source between the two pads at
part height 0. Palace's PEC copper is the high-frequency limit; pcbqs is compared at 10 MHz, where 35 um copper is
~1.7 skin depths thick and the current has mostly left the bulk for the surfaces facing the loop.
"""
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Point, box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import OUT, IMAGES, write_json  # noqa: E402
import palace  # noqa: E402

MU0 = 4e-7 * np.pi


def grover_rect_loop(a, b, w):
    """Grover's inductance (H) of a rectangular loop a x b (centre lines, m) of thin strip of width w: the round-wire
    formula with the strip's geometric mean distance radius 0.2235 w."""
    r = 0.2235 * w
    d = np.hypot(a, b)
    return MU0 / np.pi * (a * np.log(2 * a * b / (r * (a + d))) + b * np.log(2 * a * b / (r * (b + d)))
                          - 2 * (a + b) + 2 * d + 0.25 * (a + b))


def board(copper, vias=(), h=1.6, er=4.4):
    """board.json-shaped: F.Cu at z=h over B.Cu at z=0, copper {layer: [(net, polygon)]}, vias [(x, y, r, net)]."""
    return dict(stack=dict(copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='F.Cu', z=h, t=0.035)],
                           layers=[dict(name='core', z0=0.0, z1=h, er=er, tand=0.0)], thickness=h),
                copper={l: [dict(net=n, polygon=mapping(p)) for n, p in ps] for l, ps in copper.items()},
                vias=[dict(x=x, y=y, r=r, z0=0.0, z1=h, net=n) for x, y, r, n in vias],
                outline=mapping(box(-50, -50, 50, 50)), prims=dict(footprints=[]))


def case_loop():
    """A 20 x 10 mm rectangular loop (centre lines) of 1 mm strip on F.Cu, cut by a 0.5 mm gap: the port."""
    outer, inner = box(-10.5, -5.5, 10.5, 5.5), box(-9.5, -4.5, 9.5, 4.5)
    ring = outer.difference(inner).difference(box(-0.25, -5.5, 0.25, -4.5))
    b = board({'F.Cu': [('L', ring)]})
    # terminals: 0.1 mm slivers at the gap's edges (a pcbqs terminal is one equipotential node: a big pad would
    # short part of the loop that Palace's gap port does not)
    pads = (('F.Cu', box(-0.35, -5.5, -0.25, -4.5)), ('F.Cu', box(0.25, -5.5, 0.35, -4.5)))
    return b, {'L'}, pads, box(-0.25, -5.5, 0.25, -4.5), 'x', grover_rect_loop(20e-3, 10e-3, 1e-3)


def case_plane_return():
    """A 1 mm trace on F.Cu, 20 mm long, down a via (r 0.15) to a 30 x 16 mm B.Cu plane and back up a second via to
    a pad; the port across the 0.5 mm gap between that pad and the trace's start. No closed form: the plane's
    return current spreads under the trace."""
    trace = box(-10, -0.5, 10, 0.5).union(Point(10, 0).buffer(0.6))
    pad = box(-12.1, -0.6, -10.5, 0.6)
    b = board({'F.Cu': [('S', trace), ('S', pad)], 'B.Cu': [('S', box(-15, -8, 15, 8))]},
              vias=[(10, 0, 0.15, 'S'), (-11.5, 0, 0.15, 'S')])
    pads = (('F.Cu', box(-10.6, -0.5, -10.5, 0.5)), ('F.Cu', box(-10, -0.5, -9.9, 0.5)))
    return b, {'S'}, pads, box(-10.5, -0.5, -10, 0.5), 'x', None


CASES = dict(loop=case_loop, plane_return=case_plane_return)
LADDERS = dict(loop=((0.5, 0.25, 0.125), (0.25, 0.12)), plane_return=((0.7, 0.5, 0.35), (0.45, 0.3, 0.22)))


def pcbqs_loop(b, nets, pads, f, dmax=0.5, part_h=0.0, probes=None):
    from pcbqs.mqs import MQS
    t = time.perf_counter()
    terms = {'A': [pads[0]], 'B': [pads[1]]}
    q = MQS(b, nets, terminals=terms, dmax=dmax, conductor_loss='slab', conductor_faces=2, log=lambda *a: None)
    q.source('port', 'A', 'B', part_h=part_h)
    r = q.solve(np.asarray(f, float))
    out = dict(Z=np.asarray(r.zc['port']), tris=len(q.tris), seconds=time.perf_counter() - t)
    if probes is not None:                          # |H| per amp of port current (the source is 1 A)
        out['H'] = np.linalg.norm(q.h_field(r, list(range(len(f))), np.asarray(probes) * 1e-3), axis=1)   # (n, nf)
    return out


def palace_mesh(b, nets, gap, path, margin=25.0, h_cu=0.25, h_air=6.0, extra_sheets=()):
    """gmsh model for Palace, everything air: each layer's copper of `nets` a PEC sheet at its z (attribute 2), each
    via a PEC rod (its faces attribute 2, its inside not meshed), the port sheet `gap` (a polygon on F.Cu's plane,
    attribute 3), the air box's faces absorbing (attribute 4), volume attribute 1."""
    import gmsh
    from pcbqs.pdn.mesh import copper_of
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    occ = gmsh.model.occ
    zs = {c['name']: c['z'] for c in b['stack']['copper']}

    def disc(ring, z):
        pts = [occ.addPoint(x, y, z) for x, y in list(ring.coords)[:-1]]
        return occ.addPlaneSurface([occ.addCurveLoop([occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                                                      for i in range(len(pts))])])

    def surfaces(poly, z):
        outer = [(2, disc(poly.exterior, z))]
        if poly.interiors:
            outer, _ = occ.cut(outer, [(2, disc(h, z)) for h in poly.interiors])
        return outer
    sheets = []
    for lay, z in zs.items():
        g = copper_of(b, nets, lay)
        for p in ([] if g is None else getattr(g, 'geoms', [g])):
            sheets += surfaces(p.simplify(0.005), z)
    port = surfaces(gap, zs['F.Cu'])
    vias = [v for v in b['vias'] if v['net'] in nets]
    rods = [(3, occ.addCylinder(v['x'], v['y'], v['z0'], 0, 0, v['z1'] - v['z0'], v['r'])) for v in vias]
    x0, y0, x1, y1 = np.min([occ.getBoundingBox(dd, t)[:2] for dd, t in sheets], 0).tolist() + \
        np.max([occ.getBoundingBox(dd, t)[3:5] for dd, t in sheets], 0).tolist()
    zlo, zhi = min(zs.values()), max(zs.values())
    air = (3, occ.addBox(x0 - margin, y0 - margin, zlo - margin, x1 - x0 + 2 * margin, y1 - y0 + 2 * margin,
                         zhi - zlo + 2 * margin))
    out, mp = occ.fragment([air], sheets + port + rods)
    occ.synchronize()
    rod_vols = {t for m in mp[1 + len(sheets) + len(port):] for dd, t in m if dd == 3}
    if rod_vols:                                    # the rods' insides are not air: removed, their faces PEC
        occ.remove([(3, t) for t in rod_vols])
        occ.synchronize()
    air_vols = [t for dd, t in gmsh.model.getEntities(3)]
    port_faces = {t for m in mp[1 + len(sheets):1 + len(sheets) + len(port)] for dd, t in m if dd == 2}
    sheet_faces = {t for m in mp[1:1 + len(sheets)] for dd, t in m if dd == 2} - port_faces
    box_faces, rod_faces = set(), set()
    for dd, t in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(2, t)
        on_box = any(abs(bb[k] - v) < 1e-6 and abs(bb[k + 3] - v) < 1e-6 for k, v in
                     ((0, x0 - margin), (0, x1 + margin), (1, y0 - margin), (1, y1 + margin), (2, zlo - margin), (2, zhi + margin)))
        if on_box:
            box_faces.add(t)
        elif t not in sheet_faces and t not in port_faces:
            up = gmsh.model.getAdjacencies(2, t)[0]
            if len(up) < 2:                        # a face of one air volume only: a rod's wall or cap
                rod_faces.add(t)
    # sheet pieces inside a rod were removed with it; keep those still bounding air
    live = {t for dd, t in gmsh.model.getEntities(2)}
    gmsh.model.addPhysicalGroup(3, air_vols, 1)
    gmsh.model.addPhysicalGroup(2, sorted((sheet_faces | rod_faces) & live), 2)
    gmsh.model.addPhysicalGroup(2, sorted(port_faces & live), 3)
    gmsh.model.addPhysicalGroup(2, sorted(box_faces & live), 4)
    f = gmsh.model.mesh.field.add('Distance')
    gmsh.model.mesh.field.setNumbers(f, 'SurfacesList', sorted((sheet_faces | port_faces | rod_faces) & live))
    th = gmsh.model.mesh.field.add('Threshold')
    for k, v in dict(InField=f, SizeMin=h_cu, SizeMax=h_air, DistMin=0.5, DistMax=margin).items():
        gmsh.model.mesh.field.setNumber(th, k, v)
    gmsh.model.mesh.field.setAsBackgroundMesh(th)
    gmsh.option.setNumber('Mesh.MeshSizeMax', h_air)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
    gmsh.model.mesh.generate(3)
    n = sum(len(t) for t in gmsh.model.mesh.getElements(3)[1])
    gmsh.write(str(path))
    gmsh.finalize()
    return n


def palace_loop(b, nets, gap, axis, f, folder, order=2, h_cu=0.25, margin=25.0, probes=()):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    n = palace_mesh(b, nets, gap, folder / 'mesh.msh', h_cu=h_cu, margin=margin)
    cfg = {
        'Problem': dict(Type='Driven', Verbose=2, Output='postpro'),
        'Model': dict(Mesh='mesh.msh', L0=1e-3),
        'Domains': dict(Materials=[dict(Attributes=[1], Permittivity=1.0)],
                        Postprocessing=dict(Probe=[dict(Index=k + 1, Center=list(map(float, c))) for k, c in enumerate(probes)])),
        'Boundaries': dict(PEC=dict(Attributes=[2]), Absorbing=dict(Attributes=[4], Order=1),
                           LumpedPort=[dict(Index=1, Attributes=[3], Direction='+' + axis.upper(), R=50.0, Excitation=1)]),
        'Solver': dict(Order=order, Device='CPU',
                       Driven=dict(Samples=[dict(Type='Point', Freq=[float(x) / 1e9 for x in f])], Save=[]),
                       Linear=dict(Type='Default', KSPType='GMRES', Tol=1e-10, MaxIts=2000)),
    }
    palace.run(folder, cfg, np_=8 if n < 300_000 else 4 if n < 700_000 else 2)     # big meshes: fewer ranks
    fz, Z = palace.port_z(folder, 1)
    return dict(f=fz, Z=Z[:, 0, 0], tets=n, seconds=time.perf_counter() - t)


def palace_probe_h(folder, f):
    """|H| (n, nf) per amp into the copper, from Palace's probe-B.csv: the copper's current is the incident wave's
    I_inc (1 - S11) (port-I.csv's I_inc[1], port-S.csv's S11)."""
    post = Path(folder) / 'postpro'
    head, B = palace._table(post / 'probe-B.csv')
    hi, I = palace._table(post / 'port-I.csv')
    hs, S = palace._table(post / 'port-S.csv')
    k = hs.index('|S[1][1]| (dB)')
    s11 = 10 ** (S[:, k] / 20) * np.exp(1j * np.radians(S[:, k + 1]))
    i_port = np.abs(I[:, hi.index('I_inc[1] (A)')] * (1 - s11))
    n = sum(1 for h in head if h.startswith('Re{B_x['))
    H = np.zeros((n, len(B)))
    for q in range(n):
        comp = [B[:, head.index(f'Re{{B_{c}[{q + 1}]}} (Wb/m²)')] + 1j * B[:, head.index(f'Im{{B_{c}[{q + 1}]}} (Wb/m²)')]
                for c in 'xyz']
        H[q] = np.sqrt(sum(abs(c) ** 2 for c in comp)) / MU0 / i_port
    return H


def run(name, f=(1e6, 1e7, 1e8), dmaxes=None, hs=None):
    b, nets, pads, gap, axis, L0 = CASES[name]()
    f = np.asarray(f, float)
    dmaxes, hs = dmaxes or LADDERS[name][0], hs or LADDERS[name][1]   # pcbqs's dense PEEC: ~12k triangles in 32 GB
    top = max(c['z'] for c in b['stack']['copper'])
    xs = np.linspace(-18, 18, 37)
    probes = np.array([(x, 0.0, top + h) for h in (1.0, 3.0) for x in xs])
    rows = []
    for dm in dmaxes:
        p = pcbqs_loop(b, nets, pads, f, dmax=dm, probes=probes)
        rows.append(dict(tool='pcbqs', mesh=dm, L=p['Z'].imag / (2 * np.pi * f), R=p['Z'].real, H=p['H'], n=p['tris'],
                         seconds=p['seconds']))
        print(f'{name}: pcbqs dmax {dm}: L {np.round(rows[-1]["L"] * 1e9, 3)} nH, R {np.round(p["Z"].real * 1e3, 2)} mOhm '
              f'({p["tris"]:,} triangles, {p["seconds"]:.0f} s)', flush=True)
    for h in hs:
        folder = OUT / 'mqs' / f'{name}_h{h:g}'
        q = palace_loop(b, nets, gap, axis, f, folder, h_cu=h, probes=probes)
        rows.append(dict(tool='palace', mesh=h, L=q['Z'].imag / (2 * np.pi * f), H=palace_probe_h(folder, f), n=q['tets'],
                         seconds=q['seconds']))
        print(f'{name}: Palace h {h}: L {np.round(rows[-1]["L"] * 1e9, 3)} nH ({q["tets"]:,} tets, {q["seconds"]:.0f} s)', flush=True)
    write_json(OUT / 'mqs' / name / 'result.json', dict(case=name, f=f, probes=probes, grover=L0, rows=rows))
    figure(name)
    return rows


def figure(name):
    """mqs_<case>.png from the saved result: L against mesh size at 10 MHz; |H| along the probe line at 1 and 3 mm."""
    import json
    from plots import style, mesh_axis, PCBQS, REFERENCE, ANALYTIC
    r = json.loads((OUT / 'mqs' / name / 'result.json').read_text())
    f, rows, L0 = np.array(r['f']), r['rows'], r['grover']
    for row in rows:
        row['L'], row['H'] = np.array(row['L']), np.array(row['H'])
    xs = np.array(r['probes'])[:len(r['probes']) // 2, 0]
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    k = int(np.argmin(abs(f - 1e7)))
    for tool, c, mk in (('pcbqs', PCBQS, 'o'), ('palace', REFERENCE, 's')):
        rr = [r for r in rows if r['tool'] == tool]
        ax[0].plot([r['mesh'] for r in rr], [r['L'][k] * 1e9 for r in rr], mk + '-', color=c,
                   label='pcbqs PEEC, 10 MHz (dmax)' if tool == 'pcbqs' else 'Palace PEC (h at copper)')
    if L0:
        ax[0].axhline(L0 * 1e9 - 4e-7 * 0.03 / 4 * 1e9, color=ANALYTIC, ls='--', lw=1, label='Grover, uniform current')
    mesh_axis(ax[0], [r['mesh'] for r in rows])
    ax[0].set(xlabel='mesh size (mm): pcbqs dmax, Palace h', ylabel='loop L (nH)', title=f'{name}: inductance at the port')
    ax[0].legend(frameon=False, fontsize=8)
    best = {t: [r for r in rows if r['tool'] == t][-1] for t in ('pcbqs', 'palace')}
    for j, (hh, ls) in enumerate(((1.0, '-'), (3.0, '--'))):
        sl = slice(j * len(xs), (j + 1) * len(xs))
        ax[1].plot(xs, best['pcbqs']['H'][sl, k], ls, color=PCBQS, label=f'pcbqs, {hh:g} mm')
        ax[1].plot(xs, best['palace']['H'][sl, k], ls, color=REFERENCE, marker='.', ms=4, lw=0.8, label=f'Palace, {hh:g} mm')
    ax[1].set(xlabel='x (mm), y = 0', ylabel='|H| per amp (A/m)', title='near field above the top copper, 10 MHz')
    ax[1].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(IMAGES / f'mqs_{name}.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--plot']:
        for name in sys.argv[2:] or list(CASES):
            figure(name)
    else:
        for name in sys.argv[1:] or list(CASES):
            run(name)
