"""Copper-port extraction on a real board (the circuit tool's 11 frequencies): time, peak memory, GMRES iterations.

    python tools/benchmark_ports.py SHEETS [SOLVER [PRECONDITIONER]] [--board PCB] [--save Z.npy] [--compare Z.npy]

SHEETS 1 or 2 (face sheets); SOLVER auto | direct | iterative | dense (the dense backend); PRECONDITIONER auto |
graph | faces (mqs_iterative.port_method). One run per process: peak memory is the process's. Default board:
~/Desktop/wpthb with pcbqs's wpthb-power.json.
"""
import argparse
import json
import pathlib
import resource
import time

import numpy as np

import pcbqs
from pcbqs._cli import load_board
from pcbqs.mqs_ports import CopperPorts
from pcbqs.power_bridge import load_config, make_board_ports

FREQUENCIES = [0.0, 1e3, 1e4, 3e4, 9e4, 1e5, 3e5, 9e5, 3e6, 7e6, 2e7]

a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
a.add_argument('sheets', type=int, choices=(1, 2))
a.add_argument('solver', nargs='?', default='auto', choices=('auto', 'direct', 'iterative', 'dense'))
a.add_argument('preconditioner', nargs='?', default='auto', choices=('auto', 'graph', 'faces'))
a.add_argument('--board', default=str(pathlib.Path.home() / 'Desktop/wpthb/wpthb.kicad_pcb'))
a.add_argument('--config', default=str(pathlib.Path(pcbqs.__file__).parent / 'data/wpthb-power.json'))
a.add_argument('--save')
a.add_argument('--compare')
a = a.parse_args()

cfg = load_config(a.config)
cfg['mesh']['matrix_backend'] = 'dense' if a.solver == 'dense' else 'hierarchical'
cfg['mesh']['thickness_sheets'] = a.sheets
t0 = time.time()
q, ports, _ = make_board_ports(load_board(a.board, set(cfg['mesh']['nets'])), cfg, print)
if a.solver != 'dense':
    q.port_solver, q.port_preconditioner = a.solver, a.preconditioner
q._assemble()
t1 = time.time()
r = CopperPorts(q).extract(FREQUENCIES, log=print)
t2 = time.time()
out = dict(sheets=a.sheets, solver=a.solver, preconditioner=a.preconditioner, triangles=len(q.tris), ports=len(ports),
           loops=int(q._mats['C'].shape[1]), model_s=round(t1 - t0, 1), extract_s=round(t2 - t1, 1),
           peak_MB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024,
           iterations=[i.get('iterations', 0) for i in q.solve_info],
           method=sorted({f"{i.get('method', 'dense')}/{i.get('preconditioner', '-')}" for i in q.solve_info} or {'dense/-'}))
if a.compare:
    ref = np.load(a.compare)
    out['max_rel_difference'] = float(abs(r.impedance - ref).max() / abs(ref).max())
if a.save:
    np.save(a.save, r.impedance)
print(json.dumps(out))
