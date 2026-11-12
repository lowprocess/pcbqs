"""
The temperatures where kicad2ems's viewer reads results: the board and, per layer, each solver's map on a grid, to
turn, cut and probe in its 2D and 3D views (as the DC drop's maps: pdn/viewer.py).

    write_viewer(out, board, problem, results)      the files' names, for thermal.json

It writes
    <out>/thermal_board.json        the board: all its copper (it all conducts), the parts that dissipate marked
    <out>/thermal/exc_board/        dump_T_<layer>_fd.npz per layer and solver: the temperature, C (NaN off the
                                    board); with both solvers also mesh minus grid, K
with a dump_info.json, the dumps at 0 Hz as plane dumps (a temperature is one number: the vector's x). Each says
its unit and that its colours span its own values ("range": "data"), so a map runs from the board's coolest point
to its hottest and not from zero.
"""
import json
import os
from types import SimpleNamespace

import numpy as np

GROUP, RUN = 'thermal', 'exc_board'


def _maps(r):
    """A result's temperatures per layer on a grid: (x, y lines mm, [(ny, nx) C, NaN off the board])."""
    p = r.problem
    if r.method == 'grid':
        ny, nx = r.T[0].shape
        return r.x0 + (np.arange(nx) + 0.5) * r.size, r.y0 + (np.arange(ny) + 0.5) * r.size, list(r.T)
    from ..pdn.viewer import grid_of
    x0, y0, x1, y1 = p.outline.bounds
    x, y = grid_of((x0, y0, x1, y1), cells=400)
    h = x[1] - x[0]
    x, y = x[:-1] + 0.5 * h, y[:-1] + 0.5 * (y[1] - y[0])           # cell centres: none on the outline itself
    X, Y = np.meshgrid(x, y)
    return x, y, [r.sample(l, X, Y) for l in range(len(p.layers))]


def write_viewer(out, board, problem, results):
    """Write the files above for results {method: Result}; returns dict(board=, dumps=[{method, layer, file}])."""
    from .._artifacts import Dump, write_info, write_npz
    from ..mqs_layout import crop_layout, geometry
    p = problem
    nets = {r['net'] for records in board['copper'].values() for r in records}
    layout = crop_layout(board, p.outline, nets, set(p.powers))
    with open(os.path.join(out, 'thermal_board.json'), 'w', encoding='utf-8') as fh:
        json.dump(layout, fh)
    exc = os.path.join(out, GROUP, RUN)
    os.makedirs(exc, exist_ok=True)
    dumps, meta, files = [], {}, []

    def add(method, layer, z, x, y, t, unit='°C', what=''):
        label = ' '.join(w for w in (what, layer, f"({method})" if len(results) > 1 else '') if w)
        dm = Dump(f"T@{layer}", 'T', (x[0], y[0], z), (x[-1], y[-1], z), [0.0], 2, label)
        data = np.zeros((len(x), len(y), 3))
        data[:, :, 0] = t.T                                         # (ny, nx) -> (nx, ny)
        write_npz(os.path.join(exc, dm.name + '.npz'), dict(x=x, y=y, z=np.array([z])), [0.0], data[None, :, :, None, :])
        dumps.append(dm)
        meta[dm.name] = dict(unit=unit, **({'range': 'data'} if unit == '°C' else {}))
        files.append(dict(method=method, layer=layer, unit=unit, file=os.path.join(GROUP, RUN, dm.name + '.npz')))
    order = sorted(range(len(p.layers)), key=lambda l: -p.layers[l].z)          # the top layer first
    for method, r in results.items():
        x, y, maps = _maps(r)
        for l in order:
            add(method, p.layers[l].name, p.layers[l].z, x, y, maps[l])
    if 'grid' in results and 'mesh' in results:
        from .picture import difference
        g = results['grid']
        x, y, _ = _maps(g)
        for l in order:
            add('mesh − grid', p.layers[l].name, p.layers[l].z, x, y, difference(g, results['mesh'], l), unit='K',
                what='difference')
    for stale in os.listdir(exc):                                   # a layer of an earlier solve in this folder
        if stale.startswith('dump_') and stale.endswith('.npz') and stale[:-4] not in meta:
            os.remove(os.path.join(exc, stale))
    cu_z = {l.name: l.z for l in p.layers}
    m = SimpleNamespace(outline=p.outline, cu_z=cu_z, copper={}, ports=[],
                        vias=[], thickness=board['stack'].get('thickness', max(cu_z.values())))
    write_info(exc, dumps, m)
    path = os.path.join(exc, 'dump_info.json')
    with open(path, encoding='utf-8') as fh:
        info = json.load(fh)
    for d in info['dumps']:
        d.update(meta[d['name']])
    info.update(geometry(layout))                                   # the copper, the vias and the parts
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(info, fh)
    return dict(board='thermal_board.json', dumps=files)
