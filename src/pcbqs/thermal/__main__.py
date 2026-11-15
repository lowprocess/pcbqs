"""python -m pcbqs.thermal: the steady temperature of a board from what its parts dissipate.

    python -m pcbqs.thermal board.kicad_pcb --loss Q1=0.35 --loss D1=0.5
    python -m pcbqs.thermal --circuit circuit-results            # the board and the losses of a circuit run
    python -m pcbqs.thermal --circuit circuit-results --method both
    python -m pcbqs.thermal --describe --json                     # the options, for kicad2ems-gui's form

It writes <out>/thermal.json (each part's temperature, each layer's hottest and mean, the heat in and out),
thermal.npz (the maps), thermal.png and, where kicad2ems's viewer reads results, the board and each layer's
temperature as field dumps (thermal/viewer.py).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import Problem, solve, losses_of, FILM, AMBIENT


OPTIONS = [            # (key, kind, default, help, group): _cli's table, which --describe gives the window
    ('loss', 'list', [], "A footprint's loss, REF=W (repeatable); with a circuit run, these replace its figures", 'Heat'),
    ('circuit', 'str', '', "A circuit run's results folder (python -m pcbqs.circuit): its board and its parts' "
     'losses', 'Heat'),
    ('case', 'str', '', "The circuit run's case, by name; empty: the one with the most loss; all: a map for each "
     "(in <out>/<case>) and a table of each part at each (thermal-cases.json)", 'Heat'),
    ('theta', 'list', [], "From a part's pads to the part itself, REF=K/W (repeatable); a footprint's field "
     "thermal.theta gives it too", 'Parts'),
    ('tmax', 'list', [], "The most a part may reach, REF=C (repeatable: its junction's limit); a footprint's field "
     "thermal.tmax gives it too", 'Parts'),
    ('ambient', 'float', AMBIENT, 'The air, C', 'Air'),
    ('film', 'float', FILM, 'Convection and radiation from the faces, W/m2/K (the top face, with film-bottom)', 'Air'),
    ('film_bottom', 'float', None, 'The bottom face, W/m2/K, when it differs', 'Air'),
    ('film_edge', 'float', None, "The board's edges and cut-outs, W/m2/K; empty: the faces' mean, 0: insulated", 'Air'),
    ('method', 'str', 'grid', 'grid, mesh, or both (and their difference)', 'Solver'),
    ('size', 'float', None, 'mm: the grid pitch (0.5) or the mesh\'s longest edge (1)', 'Solver'),
]


def describe():
    from .._cli import describe as describe_options
    return describe_options(OPTIONS)


COPPER_BIN = 1.0                # mm: the copper's loss map gathered into squares this size, a heat source each


def add_copper_loss(p, case, board):
    """The board copper's own loss into the problem, where the circuit's copper currents put it (the case's
    copper-loss.npz, which the circuit's --fields writes), gathered into COPPER_BIN squares; without that map, said
    (the copper's loss had reached no thermal map: 3.6 % of the buck's)."""
    total = float(((case.get('copper') or {}).get('loss_W')) or 0.0)
    if total <= 0:
        return
    path = Path(case.get('folder', '')) / 'copper-loss.npz'
    if not path.is_file():
        p.notes.append(f"the board copper's own loss ({1e3 * total:.3g} mW) is left out: run the circuit with --fields "
                       f"for where it is")
        return
    from shapely.geometry import box
    with np.load(path, allow_pickle=False) as m:
        layer, x, y, W = m['layer'], m['x_mm'], m['y_mm'], m['W']
    names = {l.name for l in p.layers}
    bins = {}
    for lay, xi, yi, w in zip(layer, x, y, W):
        if lay in names and w > 0:
            k = (str(lay), int(np.floor(xi / COPPER_BIN)), int(np.floor(yi / COPPER_BIN)))
            bins[k] = bins.get(k, 0.0) + float(w)
    for (lay, i, j), w in bins.items():
        p.heat('board copper', [(lay, box(i * COPPER_BIN, j * COPPER_BIN, (i + 1) * COPPER_BIN, (j + 1) * COPPER_BIN))], w)


def part_fields(board):
    """({ref: K/W}, {ref: C}) from the footprints' fields thermal.theta and thermal.tmax (as pdn.load gives a part's
    current): what the design holds of its parts' thermal limits."""
    from ..pdn.rail import field_value
    theta, tmax = {}, {}
    for fp in (board.get('prims') or {}).get('footprints', []):
        t, m = field_value(fp, 'thermal.theta'), field_value(fp, 'thermal.tmax')
        if t is not None:
            theta[fp['ref']] = t
        if m is not None:
            tmax[fp['ref']] = m
    return theta, tmax


def _pairs(items, what, ap):
    out = {}
    for it in items:
        ref, _, val = it.partition('=')
        try:
            out[ref.strip()] = float(val)
        except ValueError:
            ap.error(f"{what} {it!r}: REF=NUMBER")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m pcbqs.thermal', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    from .._cli import add_options, load_board, print_describe
    ap.add_argument('board', nargs='?', help='a .kicad_pcb or a kicad2ems board.json')
    ap.add_argument('--out', default='thermal-results', help='the results folder (default: thermal-results)')
    ap.add_argument('--describe', action='store_true', help='the options (with --json: for kicad2ems-gui)')
    ap.add_argument('--json', action='store_true')
    add_options(ap, OPTIONS)
    a = ap.parse_args(argv)
    if a.describe:
        print_describe(describe(), a.json)
        return 0
    if a.method not in ('grid', 'mesh', 'both'):
        ap.error(f"--method {a.method!r}: grid, mesh or both")
    if a.circuit and a.case == 'all':                               # (R-E11: a sweep had one map, its worst point)
        return every_case(a, argv)
    losses, left = _pairs(a.loss, '--loss', ap), []
    board_file, case_name, case = a.board, None, None
    if a.circuit:
        folder = Path(a.circuit)
        try:
            model = json.loads((folder / 'model.json').read_text())
            cases = json.loads((folder / 'circuit.json').read_text())['cases']
        except (OSError, KeyError, ValueError) as e:
            ap.error(f'{folder}: not the results of a circuit run ({e})')
        if a.case:
            pick = [c for c in cases if c['name'] == a.case]
            if not pick:
                ap.error(f"case {a.case!r}: one of {', '.join(c['name'] for c in cases)}")
            case = pick[0]
        else:
            case = max(cases, key=lambda c: c.get('loss_W', 0.0))
        from_run, left = losses_of(case, model)
        losses = {**from_run, **losses}
        board_file = board_file or (model.get('board') or {}).get('file')
        case_name = case['name']
        print(f"losses of case {case_name}")
    if not board_file:
        ap.error('a board, or --circuit with a model that names one')
    if not losses:
        ap.error('no loss: --loss REF=W or --circuit FOLDER')
    try:
        board = load_board(board_file)
    except (OSError, RuntimeError, ValueError) as e:
        ap.error(str(e))
    theta, tmax = part_fields(board)                                # the footprints' own, the options over them
    theta.update(_pairs(a.theta, '--theta', ap))
    tmax.update(_pairs(a.tmax, '--tmax', ap))
    p = Problem.of(board, losses, film=(a.film if a.film_bottom is None else a.film_bottom, a.film),
                   ambient=a.ambient, theta=theta, edge_film=a.film_edge)
    p.tmax = tmax
    p.notes += left
    if case is not None:
        add_copper_loss(p, case, board)
    if not p.sources:
        for n in p.notes:
            print(n, file=sys.stderr)
        ap.error('none of the losses is on the board')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from .picture import picture, compare, difference
    methods = ('grid', 'mesh') if a.method == 'both' else (a.method,)
    res = {m: solve(p, m, a.size if len(methods) == 1 else None) for m in methods}
    summary = {m: dict(r.summary(), board=str(Path(board_file).resolve()), circuit=str(a.circuit or ''),
                       case=case_name, figure=f'thermal-{m}.png' if len(res) > 1 else 'thermal.png')
               for m, r in res.items()}
    arrays = {}
    for m, r in res.items():
        arrays[f'{m}_T'] = r.T
        if m == 'mesh':
            arrays.update(mesh_nodes=r.nodes, mesh_tris=r.tris)
        else:
            arrays.update(grid_origin=np.array([r.x0, r.y0]), grid_pitch=np.array(r.size))
        picture(r, out / (f'thermal-{m}.png' if len(res) > 1 else 'thermal.png'))
    first = res[methods[0]]
    print(f"{p.watts:.3f} W into the board, {first.out_W:.3f} W out through the films ({first.edge_W:.3f} W of it "
          f"through the edges); ambient {p.ambient:g} C")
    print(f"{'part':8s} {'W':>7s} " + ' '.join(f"{m + ' C':>9s}" for m in methods))
    for ref in sorted(first.parts, key=lambda k: -first.parts[k]['part_C']):
        print(f"{ref:8s} {first.parts[ref]['power_W']:7.3f} " +
              ' '.join(f"{res[m].parts[ref]['part_C']:9.1f}" for m in methods))
    print(f"{'hottest copper':16s} " + ' '.join(f"{res[m].hottest:9.1f}" for m in methods))
    if len(res) == 2:
        g, m = res['grid'], res['mesh']
        d = [difference(g, m, l) for l in range(len(p.layers))]
        summary['difference'] = dict(
            largest_K=float(max(np.nanmax(np.abs(x)) for x in d)),
            parts_K={k: m.parts[k]['board_C'] - g.parts[k]['board_C'] for k in g.parts})
        compare(g, m, out / 'thermal-compare.png')
        print(f"mesh against grid: parts within {max(abs(v) for v in summary['difference']['parts_K'].values()):.2f} K, "
              f"the maps within {summary['difference']['largest_K']:.2f} K")
    for n in p.notes:
        print('note:', n)
    np.savez_compressed(out / 'thermal.npz', layers=np.array([l.name for l in p.layers]), **arrays)
    try:                                                # the same maps for the viewer; a failure costs only them
        from .viewer import write_viewer
        shown = write_viewer(str(out), board, p, res)
    except Exception as e:
        shown = None
        print(f"note: the viewer's maps were not written: {type(e).__name__}: {e}")
    for m in summary:
        if m in res:
            summary[m]['viewer'] = shown
    (out / 'thermal.json').write_text(json.dumps(summary if len(res) > 1 else summary[methods[0]], indent=1))
    print(f"results in {out}")
    return 0


def every_case(a, argv):
    """--case all: each case of the circuit run its own map (<out>/<case>), the one with the most loss also at the
    top (what the report and the Viewer read, as before), and thermal-cases.json: each part's temperature, margin
    and loss at each case, and each case's hottest copper."""
    import shutil
    folder = Path(a.circuit)
    cases = json.loads((folder / 'circuit.json').read_text())['cases']
    argv = list(sys.argv[1:] if argv is None else argv)
    keep = []                                                       # the arguments but --case and --out
    skip = False
    for i, x in enumerate(argv):
        if skip:
            skip = False
            continue
        if x in ('--case', '--out'):
            skip = True
            continue
        if x.startswith(('--case=', '--out=')):
            continue
        keep.append(x)
    out = Path(a.out)
    table = dict(format=1, circuit=str(folder.resolve()), cases=[], parts={})
    for c in cases:
        sub = out / c['name']
        main(keep + ['--case', c['name'], '--out', str(sub)])
        r = json.loads((sub / 'thermal.json').read_text())
        r = r if 'parts' in r else next(iter(v for v in r.values() if isinstance(v, dict) and 'parts' in v))
        table['cases'].append(dict(name=c['name'], loss_W=c.get('loss_W'), hottest_C=r.get('hottest_C')))
        for ref, q in r['parts'].items():
            table['parts'].setdefault(ref, {})[c['name']] = {k: q.get(k) for k in ('part_C', 'power_W', 'margin_K')}
    worst = max(cases, key=lambda c: c.get('loss_W', 0.0))['name']
    for f in (out / worst).iterdir():                              # the worst case at the top, as before
        dest = out / f.name
        (shutil.copytree(f, dest, dirs_exist_ok=True) if f.is_dir() else shutil.copy2(f, dest))
    table['top'] = worst
    (out / 'thermal-cases.json').write_text(json.dumps(table, indent=1))
    refs = sorted(table['parts'], key=lambda k: -max(v['part_C'] or 0 for v in table['parts'][k].values()))
    print(f"{'part':8s} " + ' '.join(f"{c['name'][:14]:>14s}" for c in table['cases']))
    for ref in refs:
        print(f"{ref:8s} " + ' '.join(f"{(table['parts'][ref].get(c['name']) or {}).get('part_C') or float('nan'):14.1f}"
                                      for c in table['cases']))
    print(f"{len(cases)} cases; the top-level map is {worst}'s (the most loss); thermal-cases.json has the table")
    return 0


if __name__ == '__main__':
    sys.exit(main())
