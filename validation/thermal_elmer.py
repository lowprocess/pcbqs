"""Steady temperature: pcbqs.thermal (grid and mesh, layered sheets) against Elmer (kicad2ems-fem's elmer/thermal.py:
3D heat conduction in the copper, the barrels and an anisotropic laminate, a film on every exposed face).

    python validation/thermal_elmer.py [example ...]      # circuit examples, e.g. 05_buck_synchronous (default: all)
    python validation/thermal_elmer.py --wpthb            # the as-built 24 V bridge, 4 layers, Q1 Q2 L3

Each example's circuit is run (python -m pcbqs.circuit) and its lossiest case's part losses go to both tools, with
the same film (10 W/m2/K), ambient (25 C) and conductivities. Twice: as each tool models a board by default (a film
on the faces and on the edges), and with the edges insulated (pcbqs edge_film=0, Elmer's side faces given no
boundary condition), which isolates the edges from the rest. Compared: each part's temperature (the mean under its
pads). Needs kicad2ems-fem on the path and ElmerSolver.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import OUT, write_json  # noqa: E402

from pcbqs.thermal import Problem, solve, losses_of  # noqa: E402
from pcbqs._cli import load_board  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / 'src' / 'pcbqs' / 'circuit' / 'examples'
WPTHB = Path.home() / 'Desktop' / 'wpthb' / 'wpthb.kicad_pcb'
FILM, AMBIENT = 10.0, 25.0
quiet = lambda *a, **k: None


def elmer_module():
    from kicad2ems_fem.elmer import thermal
    return thermal


def insulated_sides(layered):
    """kicad2ems-fem's layered_mesh with the side faces (the 4-node boundary elements) moved to a tag of their own
    that gets no boundary condition: no heat leaves through them."""
    def mesh(*a, **k):
        m = layered(*a, **k)
        f = Path(a[1]) / 'mesh.boundary'
        rows = []
        for line in f.read_text().splitlines():
            w = line.split()
            if w[4] == '404':
                w[1] = '9'
            rows.append(' '.join(w))
        f.write_text('\n'.join(rows) + '\n')
        m['boundaries'][1] = (m['boundaries'][1][0], m['boundaries'][1][1] - float(m['side_area'].sum()))
        m['side_area'] = m['side_area'] * 0.0
        return m
    return mesh


def circuit_losses(name):
    run = OUT / 'thermal' / 'circuits' / name
    if not (run / 'circuit.json').exists():
        subprocess.run([sys.executable, '-m', 'pcbqs.circuit', str(EXAMPLES / f'{name}.json'), '--out', str(run)],
                       check=True, stdout=subprocess.DEVNULL)
    model = json.loads((run / 'model.json').read_text())
    cases = json.loads((run / 'circuit.json').read_text())['cases']
    losses, _ = losses_of(max(cases, key=lambda c: c.get('loss_W', 0.0)), model)
    return model['board']['file'], losses


def compare(name, board_file, losses, pitch=0.5, dmax=0.5, h=0.3, hmax=2.0):
    et = elmer_module()
    layered = et.layered_mesh
    board = load_board(board_file)
    row = dict(name=name, board=str(board_file))
    try:
        for edges in (True, False):
            key = 'edges' if edges else 'no_edges'
            p = Problem.of(board, losses, film=FILM, ambient=AMBIENT, edge_film=None if edges else 0.0)
            heat = {k: w for k, w in p.powers.items() if w > 0}
            r = row[key] = dict(watts=p.watts, layers=len(p.layers))
            for method, size in (('grid', pitch), ('mesh', dmax)):
                t = time.perf_counter()
                s = solve(p, method, size, log=quiet)
                r[method] = dict(seconds=time.perf_counter() - t, unknowns=s.unknowns, edge_W=s.edge_W,
                                 parts={k: v['board_C'] for k, v in s.parts.items()})
            et.layered_mesh = layered if edges else insulated_sides(layered)
            t = time.perf_counter()
            e = et.solve(board, str(OUT / 'thermal' / 'elmer' / f'{name}_{key}'), heat=heat, htc=FILM,
                         ambient=AMBIENT, h=h, hmax=hmax, log=quiet)
            r['elmer'] = dict(seconds=time.perf_counter() - t, wedges=e['wedges'], parts=e['parts'],
                              balance=e['balance']['error'])
    finally:
        et.layered_mesh = layered
    return row


def report(rows):
    for row in rows:
        r = row['edges']
        print(f"\n{row['name']}: {r['layers']} layers, {r['watts']:.3f} W; elmer {r['elmer']['wedges']:,} wedges, "
              f"{r['elmer']['seconds']:.1f} s; grid {r['grid']['seconds']:.2f} s")
        print(f"  {'part':6s} {'elmer':>7s} {'grid':>7s} {'mesh':>7s} {'grid-elmer':>11s}   insulated edges: grid-elmer")
        for ref, te in sorted(r['elmer']['parts'].items(), key=lambda kv: -kv[1]):
            g, m = r['grid']['parts'][ref], r['mesh']['parts'][ref]
            n = row['no_edges']
            d0 = n['grid']['parts'][ref] - n['elmer']['parts'][ref]
            print(f"  {ref:6s} {te:7.2f} {g:7.2f} {m:7.2f} {g - te:+6.2f} K {100 * (g - te) / (te - AMBIENT):+5.1f}%"
                  f"   {d0:+6.2f} K")
    worst = max(((abs(r['edges']['grid']['parts'][k] - v), r['name'], k) for r in rows
                 for k, v in r['edges']['elmer']['parts'].items()), default=None)
    if worst:
        print(f"\nworst part: {worst[2]} on {worst[1]}, {worst[0]:.2f} K")


def main(argv):
    names = [a for a in argv if not a.startswith('-')] or sorted(f.stem for f in EXAMPLES.glob('*.json'))
    rows = []
    if '--wpthb' in argv:
        rows.append(compare('wpthb', WPTHB, {'Q1': 0.3, 'Q2': 0.3, 'L3': 0.4}))
    else:
        for name in names:
            board_file, losses = circuit_losses(name)
            rows.append(compare(name, board_file, losses))
            print(f'{name}: done', flush=True)
    write_json(OUT / 'thermal' / ('wpthb.json' if '--wpthb' in argv else 'examples.json'), rows)
    report(rows)


if __name__ == '__main__':
    main(sys.argv[1:])
