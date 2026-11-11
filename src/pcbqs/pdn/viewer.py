"""
The DC solve where kicad2ems's viewer reads results, beside the figures: the board, the mesh and, per layer, the
drop and the current on a grid, so the maps can be turned, cut and probed instead of only looked at.

    write_viewer(out, board, dc, volts, nets, ground, refs)    the files' names, for pdn.json

It writes
    <out>/pdn_board.json            the board read for the solve, the rail tree's and the return's copper marked
    <out>/pdn_mesh_pdn.npz          the DC mesh (triangles, each sheet at its layer's z)
    <out>/pdn/exc_power/            the rail's copper: per layer, dump_V_drop_<layer>_fd.npz (mV below the source)
                                    and dump_J_<layer>_fd.npz (the sheet current, A per mm of width, a vector
                                    as large as the solve's own J)
    <out>/pdn/exc_ground/           the return's: dump_V_rise_<layer>_fd.npz (mV above the source's ground), and J
with a dump_info.json in each, as the near-field tool's H dumps have: plane dumps at 0 Hz, zero off the copper.
A triangle's current is constant over it and the voltage linear, so the grid holds what the mesh does down to its
cell (the board's longer side over 600, at least 0.05 mm).
"""
import json
import os
from types import SimpleNamespace

import numpy as np

CELLS = 600                 # along the board's longer side
MIN_CELL = 0.05             # mm


def grid_of(bounds, cells=CELLS, min_cell=MIN_CELL):
    """The x and y lines (mm) of a grid over a box, square cells."""
    x0, y0, x1, y1 = bounds
    h = max(max(x1 - x0, y1 - y0) / cells, min_cell)
    return (np.linspace(x0, x1, max(2, int(round((x1 - x0) / h)) + 1)),
            np.linspace(y0, y1, max(2, int(round((y1 - y0) / h)) + 1)))


def on_grid(sh, x, y):
    """A sheet's voltage (V; NaN off its copper) and its sheet current (A/mm, x and y) at the grid's points:
    (nx, ny), (nx, ny, 2). The voltage is interpolated over the triangle a point lies in, the current that
    triangle's own; where matplotlib cannot search the mesh (a degenerate triangle), the nearest vertex and
    triangle of the points on the copper."""
    from .mesh import gradient
    V = np.nan_to_num(np.asarray(sh.V, float))
    g = np.asarray(gradient(sh.nodes, sh.tris, V), float)[:, :2]            # V/mm: the current runs down it,
    mag = np.hypot(g[:, 0], g[:, 1])                                        # as large as the solve's own J
    j = -g * (np.asarray(sh.J, float) / np.where(mag > 0, mag, 1.0))[:, None]
    X, Y = np.meshgrid(x, y, indexing='ij')
    try:
        from matplotlib.tri import Triangulation
        tri = Triangulation(sh.nodes[:, 0], sh.nodes[:, 1], sh.tris)
        k = np.asarray(tri.get_trifinder()(X, Y))
        inside = k >= 0
        n = sh.tris[k[inside]]                                             # the three vertices of each point's
        p = np.column_stack([X[inside], Y[inside]])
        a, b, c = sh.nodes[n[:, 0]], sh.nodes[n[:, 1]], sh.nodes[n[:, 2]]
        det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        det = np.where(det == 0, 1.0, det)
        wb = ((p[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (p[:, 1] - a[:, 1])) / det
        wc = ((b[:, 0] - a[:, 0]) * (p[:, 1] - a[:, 1]) - (p[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) / det
        v = (1 - wb - wc) * V[n[:, 0]] + wb * V[n[:, 1]] + wc * V[n[:, 2]]
        jj = j[k[inside]]
    except (RuntimeError, ValueError):
        import shapely
        from scipy.spatial import cKDTree
        inside = shapely.contains_xy(sh.geom, X, Y)
        p = np.column_stack([X[inside], Y[inside]])
        v = V[cKDTree(sh.nodes[:, :2]).query(p)[1]] if len(p) else np.zeros(0)
        jj = j[cKDTree(sh.nodes[sh.tris][:, :, :2].mean(axis=1)).query(p)[1]] if len(p) else np.zeros((0, 2))
    vg = np.full(X.shape, np.nan)
    jg = np.zeros(X.shape + (2,))
    vg[inside] = v
    jg[inside] = jj
    return vg, jg


def _copper(board, nets, outline):
    """{layer: [polygons]} of the nets' copper, for the viewer's outlines."""
    from shapely.geometry import shape
    out = {}
    for lay, records in board['copper'].items():
        polys = []
        for r in records:
            if r['net'] in nets:
                g = shape(r['polygon']).intersection(outline)
                polys += [p for p in getattr(g, 'geoms', [g]) if p.geom_type == 'Polygon' and not p.is_empty]
        out[lay] = polys
    return out


def write_viewer(out, board, dc, volts, nets, ground, refs=()):
    """Write the files above; returns dict(board=, mesh=, dumps=[{side, layer, field, file}]) for pdn.json.
    nets: the rail tree's; ground: the return's; refs: the parts to show as the solve's own (source, loads,
    series parts, decaps)."""
    from shapely.geometry import shape
    from .._artifacts import Dump, write_info, write_npz
    from ..mqs_layout import crop_layout, geometry
    outline = shape(board['outline'])
    nets, ground = set(nets), set(ground)
    layout = crop_layout(board, outline, nets | ground, set(refs))
    with open(os.path.join(out, 'pdn_board.json'), 'w', encoding='utf-8') as fh:
        json.dump(layout, fh)
    nodes, tris, off = [], [], 0
    for side in ('power', 'ground'):                    # one mesh: every sheet at its layer's height
        for sh in dc[side].sheets.values():
            nodes.append(np.column_stack([sh.nodes[:, :2], np.full(len(sh.nodes), sh.z)]))
            tris.append(np.asarray(sh.tris) + off)
            off += len(sh.nodes)
    if nodes:
        np.savez_compressed(os.path.join(out, 'pdn_mesh_pdn.npz'), kind='tris', nodes=np.vstack(nodes),
                            tris=np.vstack(tris))
    x, y = grid_of(outline.bounds)
    cu_z = {c['name']: c['z'] for c in board['stack']['copper']}
    top = max(cu_z.values())
    files = []
    for side, mine, what in (('power', nets, 'drop'), ('ground', ground, 'rise')):
        exc = os.path.join(out, 'pdn', f"exc_{side}")
        os.makedirs(exc, exist_ok=True)
        dumps, units = [], {}
        for lay, sh in sorted(dc[side].sheets.items(), key=lambda kv: -kv[1].z):
            vg, jg = on_grid(sh, x, y)
            mv = np.nan_to_num((volts - vg if side == 'power' else vg) * 1e3)      # off the copper: 0
            start, stop = (x[0], y[0], sh.z), (x[-1], y[-1], sh.z)
            for field, unit, label, data in (
                    ('V', 'mV', f"{what} {lay}", np.stack([mv, 0 * mv, 0 * mv], axis=-1)),
                    ('J', 'A/mm', lay, np.concatenate([jg, np.zeros(jg.shape[:2] + (1,))], axis=-1))):
                dm = Dump(f"{field}@{lay}", field, start, stop, [0.0], 2, label)
                write_npz(os.path.join(exc, dm.name + '.npz'), dict(x=x, y=y, z=np.array([sh.z])), [0.0],
                          data[None, :, :, None, :])
                dumps.append(dm)
                units[dm.name] = unit
                files.append(dict(side=side, layer=lay, field=field, unit=unit,
                                  file=os.path.join('pdn', f"exc_{side}", dm.name + '.npz')))
        for stale in os.listdir(exc):                   # a layer of an earlier solve in this folder
            if stale.startswith('dump_') and stale.endswith('.npz') and stale[:-4] not in units:
                os.remove(os.path.join(exc, stale))
        vias = [v for v in board['vias'] if v['net'] in mine]
        m = SimpleNamespace(outline=outline, cu_z=cu_z, copper=_copper(board, mine, outline), ports=[],
                            vias=[(v['x'], v['y'], v['r'], v['z0'], v['z1']) for v in vias],
                            thickness=board['stack'].get('thickness', top))
        write_info(exc, dumps, m)
        path = os.path.join(exc, 'dump_info.json')
        with open(path, encoding='utf-8') as fh:
            info = json.load(fh)
        for d in info['dumps']:
            d['unit'] = units[d['name']]
        info.update(geometry(layout))                   # the whole layout and the parts, as the near field's
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(info, fh)
    return dict(board='pdn_board.json', mesh='pdn_mesh_pdn.npz' if nodes else None, dumps=files)
