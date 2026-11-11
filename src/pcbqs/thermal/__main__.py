"""python -m pcbqs.thermal: the steady temperature of a board from what its parts dissipate.

    python -m pcbqs.thermal board.kicad_pcb --loss Q1=0.35 --loss D1=0.5
    python -m pcbqs.thermal --circuit circuit-results            # the board and the losses of a circuit run
    python -m pcbqs.thermal --circuit circuit-results --method both
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import Problem, solve, losses_of, FILM, AMBIENT


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
    ap.add_argument('board', nargs='?', help='a .kicad_pcb or a kicad2ems board.json')
    ap.add_argument('--loss', action='append', default=[], metavar='REF=W', help="a footprint's loss, W (repeatable)")
    ap.add_argument('--circuit', metavar='FOLDER', help='the results folder of a circuit run (python -m pcbqs.circuit): '
                    'its board and the losses of its parts')
    ap.add_argument('--case', help="the circuit run's case, by name (default: the one with the most loss)")
    ap.add_argument('--method', choices=('grid', 'mesh', 'both'), default='grid')
    ap.add_argument('--size', type=float, help='mm: the pitch of the grid (default 0.5) or the longest edge of the '
                    'mesh (default 1)')
    ap.add_argument('--film', type=float, default=FILM, help=f'W/m2/K to the air on both faces (default {FILM:g})')
    ap.add_argument('--film-bottom', type=float, help='W/m2/K on the bottom face, when it differs')
    ap.add_argument('--film-edge', type=float, help="W/m2/K at the board's edges and cut-outs (default: the mean "
                    "of the faces'; 0: insulated)")
    ap.add_argument('--ambient', type=float, default=AMBIENT, help=f'C (default {AMBIENT:g})')
    ap.add_argument('--theta', action='append', default=[], metavar='REF=K/W',
                    help="from a part's pads to the part itself, K/W (repeatable)")
    ap.add_argument('--out', default='thermal-results', help='the results folder (default: thermal-results)')
    a = ap.parse_args(argv)
    from .._cli import load_board
    losses, left = _pairs(a.loss, '--loss', ap), []
    board_file = a.board
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
        print(f"losses of case {case['name']}")
    if not board_file:
        ap.error('a board, or --circuit with a model that names one')
    if not losses:
        ap.error('no loss: --loss REF=W or --circuit FOLDER')
    try:
        board = load_board(board_file)
    except (OSError, RuntimeError, ValueError) as e:
        ap.error(str(e))
    p = Problem.of(board, losses, film=(a.film if a.film_bottom is None else a.film_bottom, a.film),
                   ambient=a.ambient, theta=_pairs(a.theta, '--theta', ap), edge_film=a.film_edge)
    p.notes += left
    if not p.sources:
        for n in p.notes:
            print(n, file=sys.stderr)
        ap.error('none of the losses is on the board')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from .picture import picture, compare, difference
    methods = ('grid', 'mesh') if a.method == 'both' else (a.method,)
    res = {m: solve(p, m, a.size if len(methods) == 1 else None) for m in methods}
    summary = {m: r.summary() for m, r in res.items()}
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
    (out / 'thermal.json').write_text(json.dumps(summary if len(res) > 1 else summary[methods[0]], indent=1))
    print(f"results in {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
