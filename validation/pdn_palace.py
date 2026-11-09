"""PDN plane-pair impedance: pcbqs.pdn.cavity.PlanePair (2D cavity, ports as discs) against Palace (3D FEM: the
dielectric between PEC planes, open sides, vertical lumped ports between the planes) and the analytic modal sum.

    python validation/pdn_palace.py [case ...]       # cases: rect, lshape

Both use lossless (PEC) planes and the dielectric's loss tangent, so the comparison is of the cavity itself:
the planes' inductance and capacitance, and the frequencies and heights of the resonances.
"""
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import OUT, IMAGES, write_json  # noqa: E402
import palace  # noqa: E402

ER, TAND = 4.4, 0.02


def mesh(geom, d, ports, path, h=1.0, h_port=0.1):
    """gmsh: the dielectric slab of outline `geom` (mm) and thickness d, PEC top and bottom (attribute 2), a vertical
    w-wide port rectangle at each (x, y, w) (attributes 3, 4, ...), volume attribute 1."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    occ = gmsh.model.occ
    def disc(ring):
        pts = [occ.addPoint(x, y, 0) for x, y in list(ring.coords)[:-1]]
        return occ.addPlaneSurface([occ.addCurveLoop([occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                                                      for i in range(len(pts))])])
    base = [(2, disc(geom.exterior))]
    if geom.interiors:
        base, _ = occ.cut(base, [(2, disc(h)) for h in geom.interiors])
    vol = occ.extrude(base, 0, 0, d)
    slab = [t for dd, t in vol if dd == 3]
    rects = [occ.addRectangle(x - w / 2, y, 0, w, d) for x, y, w in ports]
    for r in rects:
        occ.rotate([(2, r)], 0, 0, 0, 1, 0, 0, np.pi / 2)
    # addRectangle is in the xy plane at z; rotated about the x axis it stands in the xz plane at y: move it back
    for r, (x, y, w) in zip(rects, ports):
        cx, cy, cz = occ.getCenterOfMass(2, r)
        occ.translate([(2, r)], 0, y - cy, d / 2 - cz)
    out, mp = occ.fragment([(3, t) for t in slab], [(2, r) for r in rects])
    occ.synchronize()
    vols = [t for dd, t in gmsh.model.getEntities(3)]
    gmsh.model.addPhysicalGroup(3, vols, 1)
    pec = []
    port_faces = [[t for dd, t in m] for m in mp[1:]]
    for dd, t in gmsh.model.getEntities(2):
        bb = gmsh.model.getBoundingBox(2, t)
        z = occ.getCenterOfMass(2, t)[2]                # OCC pads bounding boxes: test flatness loosely
        if abs(bb[2] - bb[5]) < 1e-5 and (abs(z) < 1e-6 or abs(z - d) < 1e-6):
            pec.append(t)
    if not pec:
        raise RuntimeError('no PEC plane faces found')
    gmsh.model.addPhysicalGroup(2, pec, 2)
    for k, faces in enumerate(port_faces):
        gmsh.model.addPhysicalGroup(2, faces, 3 + k)
    f = gmsh.model.mesh.field.add('Distance')
    gmsh.model.mesh.field.setNumbers(f, 'SurfacesList', [t for faces in port_faces for t in faces])
    th = gmsh.model.mesh.field.add('Threshold')
    for k, v in dict(InField=f, SizeMin=h_port, SizeMax=h, DistMin=0.2, DistMax=3.0).items():
        gmsh.model.mesh.field.setNumber(th, k, v)
    gmsh.model.mesh.field.setAsBackgroundMesh(th)
    gmsh.option.setNumber('Mesh.MeshSizeMax', h)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
    gmsh.model.mesh.generate(3)
    n = sum(len(t) for t in gmsh.model.mesh.getElements(3)[1])
    gmsh.write(str(path))
    gmsh.finalize()
    return n


def palace_cavity(geom, d, ports, f, folder, h=1.0, order=2, er=ER):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    n = mesh(geom, d, ports, folder / 'mesh.msh', h=h)
    cfg = {
        'Problem': dict(Type='Driven', Verbose=2, Output='postpro'),
        'Model': dict(Mesh='mesh.msh', L0=1e-3),
        'Domains': dict(Materials=[dict(Attributes=[1], Permittivity=er, LossTan=TAND)]),
        'Boundaries': dict(PEC=dict(Attributes=[2]),
                           LumpedPort=[dict(Index=k + 1, Attributes=[3 + k], Direction='+Z', R=50.0, Excitation=k + 1)
                                       for k in range(len(ports))]),
        'Solver': dict(Order=order, Device='CPU',
                       Driven=dict(Samples=[dict(Type='Point', Freq=[float(x) / 1e9 for x in f])], Save=[]),
                       Linear=dict(Type='Default', KSPType='GMRES', Tol=1e-9, MaxIts=1000)),
    }
    palace.run(folder, cfg)
    fz, Z = palace.port_z(folder, len(ports))
    return dict(f=fz, Z=Z, tets=n, seconds=time.perf_counter() - t)


def pcbqs_cavity(geom, d, ports, f, dmax=0.5, er=ER):
    from pcbqs.pdn.cavity import PlanePair
    t = time.perf_counter()
    # the port's w-wide strip as a disc of the same perimeter... and a lossless plane: sigma huge
    pp = PlanePair(geom=geom, d=d, er=er, tand=TAND, sigma=1e15,
                   ports=[dict(name=f'P{k + 1}', x=x, y=y, r=w / np.pi) for k, (x, y, w) in enumerate(ports)],
                   dmax=dmax, log=lambda *a: None)
    return dict(f=np.asarray(f), Z=pp.z(f, bare=True), seconds=time.perf_counter() - t)


def jetson_cavity():
    """Jetson AGX Thor +1V15: In4.Cu over In3.Cu as pcbqs's PlanePair sees it (antipads filled), ports at a U21 via
    and two decap vias."""
    from pcbqs._cli import load_board
    from pcbqs.pdn.cavity import PlanePair
    b = load_board('/usr/share/kicad/demos/jetson-agx-thor-baseboard/jetson-agx-thor-baseboard.kicad_pcb', {'+1V15', 'GND'})
    pp = PlanePair(b, power='+1V15', ground='GND', power_layer='In4.Cu', ground_layer='In3.Cu', ports=[], dmax=0.5,
                   log=lambda *a: None)
    geom = max(getattr(pp.geom, 'geoms', [pp.geom]), key=lambda g: g.area).simplify(0.01)
    vias = [v for v in b['vias'] if v['net'] == '+1V15']
    from shapely.geometry import Point
    inside = [v for v in vias if geom.buffer(-0.5).contains(Point(v['x'], v['y']))]
    picks = [inside[0], inside[len(inside) // 2], inside[-1]]
    return geom, pp.d, [(v['x'], v['y'], 0.3) for v in picks], pp.er
CASES = dict(
    rect=lambda: (box(0, 0, 40, 30), 0.1, [(10.0, 8.0, 0.4), (31.0, 21.0, 0.4)], ER),
    lshape=lambda: (Polygon([(0, 0), (40, 0), (40, 15), (15, 15), (15, 35), (0, 35)]), 0.1,
                    [(5.0, 30.0, 0.4), (35.0, 5.0, 0.4)], ER),
    jetson_1v15=jetson_cavity,
)


def run(name, f=None):
    geom, d, ports, er = CASES[name]()
    f = np.asarray(f if f is not None else np.r_[np.logspace(7, 9, 9), np.linspace(1.2e9, 4e9, 15)])
    folder = OUT / 'pdn' / name
    p = pcbqs_cavity(geom, d, ports, f, er=er)
    q = palace_cavity(geom, d, ports, f, folder, er=er)
    out = dict(case=name, f=f, pcbqs=p['Z'], palace=q['Z'], tets=q['tets'], seconds=dict(pcbqs=p['seconds'], palace=q['seconds']))
    if name == 'rect':
        from pcbqs.pdn.cavity import rect_cavity_z
        out['analytic'] = rect_cavity_z(40, 30, d, ER, TAND, [(x, y, w) for x, y, w in ports], f, modes=200, sigma=1e15)
    write_json(folder / 'result.json', out)
    for i, fi in enumerate(f):
        line = f'{fi / 1e6:9.1f} MHz  |Z11| pcbqs {abs(p["Z"][i, 0, 0]):9.4g}  Palace {abs(q["Z"][i, 0, 0]):9.4g}'
        if 'analytic' in out:
            line += f'  analytic {abs(out["analytic"][i, 0, 0]):9.4g}'
        line += f'   |Z21| {abs(p["Z"][i, 1, 0]):9.4g} {abs(q["Z"][i, 1, 0]):9.4g}'
        print(line, flush=True)
    return out



def figure(names=('rect', 'lshape', 'jetson_1v15')):
    """pdn_cavities.png: |Z11| and |Z21| against frequency for each case, pcbqs and Palace (and the analytic sum)."""
    import json
    from plots import style, PCBQS, REFERENCE, ANALYTIC
    plt = style()
    fig, axs = plt.subplots(1, len(names), figsize=(4.3 * len(names), 3.5))
    for ax, name in zip(np.atleast_1d(axs), names):
        r = json.loads((OUT / 'pdn' / name / 'result.json').read_text())
        f = np.array(r['f'])
        cz = lambda key: (lambda a: a[..., 0] + 1j * a[..., 1])(np.array(r[key]))
        for key, c, lab in (('pcbqs', PCBQS, 'pcbqs'), ('palace', REFERENCE, 'Palace')):
            Z = cz(key)
            ax.loglog(f, abs(Z[:, 0, 0]), '-', color=c, lw=1.4, label=f'{lab} Z11')
            ax.loglog(f, abs(Z[:, 1, 0]), '--', color=c, lw=1.0, label=f'{lab} Z21')
        if 'analytic' in r:
            ax.loglog(f, abs(cz('analytic')[:, 1, 0]), ':', color=ANALYTIC, lw=1.2, label='modal sum Z21')
        ax.set(xlabel='Hz', ylabel='|Z| (ohm)', title=name)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(IMAGES / 'pdn_cavities.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    for name in sys.argv[1:] or list(CASES):
        run(name)
    figure()
