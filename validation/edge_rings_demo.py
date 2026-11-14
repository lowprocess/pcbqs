"""demo_transition (mom25d's test board): pcbqs with the charge, with and without edge rings, against mom25d solved
at each frequency: the CHANGELOG's 0.040 to 0.019 in |S| at 1 GHz (R-C10). Needs mom25d (beside pcbqs in the
project's checkout, or MOM25D_DEMO for its tests/boards/demo.json).

    python validation/edge_rings_demo.py
"""
import json, os, time
from pathlib import Path
import numpy as np
from pcbqs import sparams as SP
from mom25d.api import build_board
from mom25d.run import renorm, _zc, out_refs
d = json.load(open(os.environ.get('MOM25D_DEMO') or Path(__file__).resolve().parents[2] / 'mom25d' / 'tests' / 'boards' / 'demo.json'))
f = np.array([5e7, 1e8, 2e8, 5e8, 1e9])
quiet = lambda *a, **k: None
model, sim = build_board(d, f[0], f[-1], port='via', log=quiet)
ref = np.array([renorm(sim.sparams(x), _zc(sim, x), out_refs(sim, model.port_z0)) for x in f])
for rings in (1, None):
    t = time.perf_counter()
    r = SP.solve(d, f, capacitance='on', edge_rings=rings, log=quiet)
    D = np.abs(r['S'] - ref)
    print(f"edge rings {r['info']['edge_rings']}: {r['info']['triangles']} tri (cells {r['info']['dmax_mm']:.2f} / {r['info']['fine_mm']:.2f} mm), {time.perf_counter() - t:.0f} s; max |dS| to mom25d " + ' '.join(f"{x.max():.4f}" for x in D)
          + '  |S11| ' + ' '.join(f"{abs(x[0, 0]):.4f}" for x in r['S']) + '  mom25d ' + ' '.join(f"{abs(x[0, 0]):.4f}" for x in ref), flush=True)
