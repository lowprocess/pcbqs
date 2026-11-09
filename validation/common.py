"""Shared pieces of the validation suite: paths, a 3D copper model of a board's nets in gmsh, Elmer runs.

The copper of each layer is board.json's sheet extruded by its thickness (outward for the outer layers, upward for
inner ones); each via of the nets is a cylinder through the board. Terminals (pads) become their own faces on the
copper surface, so a solver can hold them at one potential or draw a current from them.
"""
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from shapely.geometry import Point, Polygon, box

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get('PCBQS_VALIDATION_OUT', Path.home() / 'pcbqs-bench' / 'validation'))   # run data
IMAGES = HERE / 'images'                                                                         # figures (committed)
SIGMA_CU = 5.8e7


def polygons(g, tol=0.0):
    """The polygons of a shapely geometry; tol > 0 simplifies them (mm) and drops tiny holes and slivers."""
    if g is None or g.is_empty:
        return []
    if tol:
        g = g.simplify(tol).buffer(0)
    out = []
    for p in (g.geoms if hasattr(g, 'geoms') else [g]):
        if isinstance(p, Polygon) and p.area > 1e-4:
            out.append(Polygon(p.exterior, [h for h in p.interiors if Polygon(h).area > 4 * tol * tol + 1e-4]) if tol else p)
    return out


def slab(board, layer):
    """(bottom, top) z of a copper layer's slab, mm: outer layers grow outward, inner ones upward."""
    zs = {c['name']: (c['z'], c['t']) for c in board['stack']['copper']}
    z, t = zs[layer]
    if z <= min(v[0] for v in zs.values()):
        return z - t, z
    return z, z + t


def copper_model(board, nets, terminals, path, crop=None, h=0.4, h_fine=0.12, plating=0.025, log=print, tol=0.01):
    """Mesh the copper of `nets` (and their vias) in 3D with gmsh, written to `path` (.msh, version 2).

    terminals: {name: [(layer, shapely polygon)]}: the surface triangles of a layer's outer copper face whose centres
    lie in one of its pads become that terminal's boundary (tagged on the mesh, not split in the geometry: real
    copper's slivers make split pads fail to mesh); the mesh is refined to h_fine along the pad edges and at vias.
    Returns dict(bodies={'copper': 1, 'barrel': 2 or None}, boundaries={terminal: tag}, barrel_sigma, vias, tets).
    The barrels are solid cylinders whose conductivity gives each its plated annulus's conductance (pcbqs's model)."""
    import gmsh
    from pcbqs.pdn.mesh import copper_of
    nets = set(nets)
    layers = [c['name'] for c in board['stack']['copper']]
    zs = {c['name']: c['z'] for c in board['stack']['copper']}
    zmin = min(slab(board, l)[0] for l in layers)
    zmax = max(slab(board, l)[1] for l in layers)
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    gmsh.model.add('copper')
    occ = gmsh.model.occ

    def disc(ring, z):
        tags = [occ.addPoint(x, y, z) for x, y in list(ring.coords)[:-1]]
        return occ.addPlaneSurface([occ.addCurveLoop([occ.addLine(tags[i], tags[(i + 1) % len(tags)])
                                                      for i in range(len(tags))])])

    def surfaces(poly, z):
        # holes cut out by a boolean: as curve loops of one face, OCC silently dropped some on real planes
        outer = disc(poly.exterior, z)
        if not poly.interiors:
            return [outer]
        res, _ = occ.cut([(2, outer)], [(2, disc(h, z)) for h in poly.interiors])
        return [t for d, t in res if d == 2]

    volumes = []
    for lay in layers:
        g = copper_of(board, nets, lay)
        if g is None:
            continue
        if crop is not None:
            g = g.intersection(crop)
        z0, z1 = slab(board, lay)
        ext = occ.extrude([(2, s) for p in polygons(g, tol) for s in surfaces(p, z0)], 0, 0, z1 - z0)
        volumes += [t for d, t in ext if d == 3]

    def end(z, side):                    # a via ending at a layer's sheet goes through that layer's slab
        hit = [slab(board, l) for l in layers if abs(zs[l] - z) < 1e-6]
        return (hit[0][0] if side < 0 else hit[0][1]) if hit else z
    vias = [v for v in board['vias'] if v.get('net') in nets and
            (crop is None or crop.contains(box(v['x'] - v['r'], v['y'] - v['r'], v['x'] + v['r'], v['y'] + v['r'])))]
    cyl = [occ.addCylinder(v['x'], v['y'], end(v['z0'], -1), 0, 0, end(v['z1'], 1) - end(v['z0'], -1), v['r'])
           for v in vias]
    if cyl:                                         # copper against the barrels (layers never touch each other)
        occ.fragment([(3, t) for t in volumes], [(3, t) for t in cyl])
    occ.synchronize()
    slabs = [slab(board, l) for l in layers]
    copper, barrel = [], []
    for d, t in gmsh.model.getEntities(3):
        z = occ.getCenterOfMass(3, t)[2]
        (copper if any(a - 1e-6 <= z <= b + 1e-6 for a, b in slabs) else barrel).append(t)
    gmsh.model.addPhysicalGroup(3, copper, 1, 'copper')
    if barrel:
        gmsh.model.addPhysicalGroup(3, barrel, 2, 'barrel')
    gmsh.model.addPhysicalGroup(2, [t for _, t in gmsh.model.getEntities(2)], 3, 'surface')
    # mesh size: h, h_fine at vias and along the pad edges
    fields = []

    def threshold(field, dist):
        th = gmsh.model.mesh.field.add('Threshold')
        gmsh.model.mesh.field.setNumber(th, 'InField', field)
        gmsh.model.mesh.field.setNumber(th, 'SizeMin', h_fine)
        gmsh.model.mesh.field.setNumber(th, 'SizeMax', h)
        gmsh.model.mesh.field.setNumber(th, 'DistMin', dist)
        gmsh.model.mesh.field.setNumber(th, 'DistMax', dist + 3 * h)
        fields.append(th)
    marks, pts = [(v['x'], v['y'], 0.5 * (zmin + zmax)) for v in vias], []
    for lay, poly in [x for ts in terminals.values() for x in ts]:
        ring = poly.exterior.segmentize(h_fine)
        zf = slab(board, lay)[0] if slab(board, lay)[0] < zmin + 1e-9 else slab(board, lay)[1]
        marks += [(x, y, zf) for x, y in ring.coords]
    if marks:
        pts = [occ.addPoint(*m) for m in marks]
        occ.synchronize()
        f = gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(f, 'PointsList', pts)
        threshold(f, max([2 * v['r'] for v in vias], default=h_fine))
        m = gmsh.model.mesh.field.add('Min')
        gmsh.model.mesh.field.setNumbers(m, 'FieldsList', fields)
        gmsh.model.mesh.field.setAsBackgroundMesh(m)
    gmsh.option.setNumber('Mesh.MeshSizeMax', h)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)
    for algorithm in (10, 1):                       # HXT, then Delaunay
        gmsh.option.setNumber('Mesh.Algorithm3D', algorithm)
        try:
            gmsh.model.mesh.generate(3)
            break
        except Exception as exc:
            if algorithm == 1:
                raise
            log(f'gmsh: HXT failed ({str(exc).splitlines()[0]}); trying Delaunay')
            gmsh.model.mesh.clear()
    if pts:
        gmsh.model.mesh.clear([(0, t) for t in pts])    # the size-field points: no orphan nodes in the mesh
    n_tets = sum(len(t) for t in gmsh.model.mesh.getElements(3)[1])
    if not n_tets:
        gmsh.finalize()
        raise RuntimeError('gmsh produced no tetrahedra')
    gmsh.write(str(path))
    gmsh.finalize()
    tags = tag_terminals(path, board, terminals, zmin)
    r = np.array([v['r'] for v in vias]) if vias else np.array([0.15])
    factor = float(np.mean((r ** 2 - np.maximum(r - plating, 0) ** 2) / r ** 2))
    log(f'gmsh: {n_tets:,} tetrahedra, {len(vias)} vias, {len(tags)} terminals')
    return dict(bodies=dict(copper=1, barrel=2 if barrel else None), boundaries=tags, barrel_sigma=SIGMA_CU * factor,
                vias=len(vias), tets=n_tets)


def tag_terminals(path, board, terminals, zmin):
    """Re-tag the .msh's surface triangles that lie on a terminal's pads (outer copper face of the pad's layer) with
    the terminal's physical tag (10, 11, ...); returns {terminal: tag}."""
    text = Path(path).read_text().split('\n')
    i = text.index('$Nodes')
    n = int(text[i + 1])
    xyz = {int(l.split()[0]): tuple(map(float, l.split()[1:4])) for l in text[i + 2:i + 2 + n]}
    j = text.index('$Elements')
    m = int(text[j + 1])
    faces = []
    for lay_terms in terminals.values():
        faces.append([(poly.buffer(1e-3), (slab(board, lay)[0] if slab(board, lay)[0] < zmin + 1e-9 else slab(board, lay)[1]))
                      for lay, poly in lay_terms])
    counts = [0] * len(faces)
    for k in range(j + 2, j + 2 + m):
        f = text[k].split()
        if f[1] != '2':
            continue
        nodes = [xyz[int(q)] for q in f[3 + int(f[2]):]]
        c = np.mean(nodes, axis=0)
        if max(abs(p[2] - c[2]) for p in nodes) > 1e-9:
            continue                                # not a horizontal triangle
        for t, pads in enumerate(faces):
            if any(abs(c[2] - zf) < 1e-6 and poly.contains(Point(c[0], c[1])) for poly, zf in pads):
                f[3] = str(10 + t)
                text[k] = ' '.join(f)
                counts[t] += 1
                break
    for name, c in zip(terminals, counts):
        if not c:
            raise ValueError(f'terminal {name}: no copper surface under its pads')
    Path(path).write_text('\n'.join(text))
    return {name: 10 + t for t, name in enumerate(terminals)}


def elmergrid(msh, folder):
    """ElmerGrid's mesh folder from a gmsh .msh; returns it."""
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    subprocess.run(['ElmerGrid', '14', '2', str(msh), '-out', str(folder)], check=True,       # physical tags kept
                   capture_output=True, text=True)
    return folder


def elmer(case, sif_text, timeout=3600):
    """Run ElmerSolver on `sif_text` in `case`; returns its log."""
    case = Path(case)
    (case / 'case.sif').write_text(sif_text)
    (case / 'ELMERSOLVER_STARTINFO').write_text('case.sif\n1\n')
    p = subprocess.run(['ElmerSolver', 'case.sif'], cwd=case, capture_output=True, text=True, timeout=timeout)
    (case / 'elmer.log').write_text(p.stdout + p.stderr)
    if p.returncode or 'ALL DONE' not in p.stdout:
        raise RuntimeError('ElmerSolver failed; see ' + str(case / 'elmer.log'))
    return p.stdout


def save_scalars(path):
    """{name: value} of a SaveScalars file and its .names."""
    names = Path(str(path) + '.names').read_text().splitlines()
    cols = [re.sub(r'^\s*\d+:\s*', '', l).strip() for l in names if re.match(r'^\s*\d+:', l)]
    vals = np.loadtxt(path, ndmin=2)[-1]
    return dict(zip(cols, vals))


def _plain(o):
    a = np.asarray(o)
    if np.iscomplexobj(a):
        return np.stack([a.real, a.imag], -1).tolist()       # complex: [re, im] pairs
    return a.tolist() if a.ndim else a.item()


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=1, default=_plain))
