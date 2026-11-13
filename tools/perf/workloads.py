"""cProfile of one pcbqs workload: where a tool's time goes, in its own Python, in the compiled core and in the
libraries under it.

    python tools/perf/workloads.py CASE [--out DIR] [--cm5 PCB] [--wpthb PCB] [--top N]

Writes DIR/CASE.txt (wall time, peak memory, the functions with the most time of their own, then pcbqs's own by
cumulative time) and DIR/CASE.prof (for pstats or snakeviz). One case per process: the peak memory is the
process's. Cases:

    sparams_rl      low-frequency S-parameters of mom25d's demo board, the magnetic solve alone (7 frequencies)
    sparams_charge  the same with the copper's charge (the PEEC solve)
    sparams_faces   the same on the copper's two faces (--thickness): the largest dense solve
    capmatrix       the capacitance matrix of two traces on FR4 (validation/cap_fastercap.py)
    mqs             a converter's near field (KiCad's CM5 demo, U702)
    pdn_cm5         a rail's drop and impedance (CM5, +5V)
    pdn_wpthb       the same on the 24 V bridge (--wpthb)
    thermal         the bridge's temperature, grid and mesh
    ports1, ports2  the bridge's copper ports for the circuit tool, one sheet and two faces
    circuit_copper  a synchronous buck on its board's copper (ngspice)

The boards are KiCad's CM5 demo (/usr/share/kicad/demos) and the wireless-power bridge (not in this repository:
--wpthb, default ~/Desktop/wpthb/wpthb.kicad_pcb), as tools/benchmark_ports.py uses.
"""
import argparse
import cProfile
import io
import json
import os
import pathlib
import pstats
import resource
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
CASES = ('sparams_rl', 'sparams_charge', 'sparams_faces', 'capmatrix', 'mqs', 'pdn_cm5', 'pdn_wpthb', 'thermal',
         'ports1', 'ports2', 'circuit_copper')


def demo_board():
    """mom25d's demo board (its tests' JSON), beside this checkout or installed with mom25d's sources."""
    for p in (REPO.parent / 'mom25d/tests/boards/demo.json',):
        if p.exists():
            return json.loads(p.read_text())
    raise SystemExit('the sparams cases need mom25d checked out beside pcbqs (its tests/boards/demo.json)')


def work(case, a, out):
    quiet = lambda *args: None
    if case == 'mqs':
        from pcbqs.mqs_cli import main
        return main([a.cm5, '--converter', 'U702', '--vin', '+5V', '--harmonics', '1,2,3,5,7', '--heights', '1',
                     '--out', out])
    if case == 'pdn_wpthb':
        from pcbqs.pdn.cli import main
        caps = [f'CA{k}=47e-6,0.02,1e-9' for k in range(1, 5)]
        return main([a.wpthb, '--rail', 'VIN', '--volts', '24', '--source', 'TP2', '--ground-source', 'TP1',
                     '--load', 'Q1=0.67', '--load', 'Q2=0.67', '--through', 'L3', '--dcr', 'L3=0.03',
                     *[x for c in caps for x in ('--cap', c)], '--dmax', '0.4', '--out', out])
    if case == 'pdn_cm5':
        from pcbqs.pdn.cli import main
        return main([a.cm5, '--rail', '+5V', '--volts', '5', '--source', 'J101', '--ground-source', 'J101',
                     '--load', 'U702=1.4', '--load', 'U405=0.5', '--out', out])
    if case.startswith('sparams_'):
        from pcbqs import sparams
        kw = dict(sparams_charge=dict(capacitance='on'), sparams_faces=dict(capacitance='off', thickness=True),
                  sparams_rl=dict(capacitance='off'))[case]
        return sparams.solve(demo_board(), np.geomspace(1e6, 1e9, 7), log=quiet, **kw)
    if case in ('ports1', 'ports2'):
        src = (REPO / 'tools/benchmark_ports.py').read_text()
        sys.argv = ['benchmark_ports.py', case[-1], 'auto', '--board', a.wpthb]
        return exec(compile(src, 'benchmark_ports.py', 'exec'), {'__name__': '__main__'})
    if case == 'thermal':
        from pcbqs.thermal.__main__ import main
        return main([a.wpthb, *[x for q in ('Q1', 'Q2', 'Q3', 'Q4') for x in ('--loss', f'{q}=0.4')],
                     '--loss', 'L3=0.6', '--method', 'both', '--out', out])
    if case == 'capmatrix':
        sys.path.insert(0, str(REPO / 'validation'))
        import cap_fastercap as V
        return V.pcbqs_c(V.CASES['traces_fr4'](), 0.25, edge_rings=6, edge_width=0.15)
    if case == 'circuit_copper':
        from pcbqs.circuit.engine import run
        ex = REPO / 'src/pcbqs/circuit/examples'
        model = json.loads((ex / '05_buck_synchronous.json').read_text())
        model['board']['file'] = str(ex / '05_buck_synchronous.kicad_pcb')
        return run(model, out, log=quiet, copper=True)
    raise SystemExit('unknown case ' + case)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case', choices=CASES)
    ap.add_argument('--out', default=str(HERE / 'results' / 'workloads'), help='where the reports and the runs go')
    ap.add_argument('--cm5', default='/usr/share/kicad/demos/cm5_minima/CM5_MINIMA_3.kicad_pcb')
    ap.add_argument('--wpthb', default=str(pathlib.Path.home() / 'Desktop/wpthb/wpthb.kicad_pcb'))
    ap.add_argument('--top', type=int, default=28, help='how many functions each list has')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pr = cProfile.Profile()
    t = time.time()
    pr.enable()
    try:
        work(a.case, a, os.path.join(a.out, 'out_' + a.case))
    except SystemExit as e:                             # (a command line's exit)
        print('exit', e)
    pr.disable()
    wall = time.time() - t
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf)
    buf.write(f"CASE {a.case}: {wall:.1f} s wall, peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB\n")
    st.sort_stats('tottime').print_stats(a.top)
    st.sort_stats('cumtime').print_stats(r'pcbqs', a.top + 17)
    with open(os.path.join(a.out, a.case + '.txt'), 'w') as f:
        f.write(buf.getvalue())
    st.dump_stats(os.path.join(a.out, a.case + '.prof'))
    print(f"{a.case}: {wall:.1f} s")


if __name__ == '__main__':
    main()
