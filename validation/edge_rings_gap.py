"""Two traces 0.1 mm apart over a plane (or a microstrip, plates), in air, zero thickness: pcbqs's capacitance with
graded edge rings against FasterCap's stored answer (results/fastercap). The CHANGELOG's edge-ring figures (R-C10).

    python validation/edge_rings_gap.py [traces_air|microstrip_air|plates] [dmax:edge_rings=6,edge_width=0.5 ...]
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import json, time
import numpy as np
import cap_fastercap as V
from pcbqs.mqs import MQS
from shapely.geometry import Point
name = sys.argv[1] if len(sys.argv) > 1 else 'traces_air'
c = V.CASES[name]()
ref = json.load(open(HERE / 'results' / 'fastercap' / f'{name}_thin.json'))['rows'][-1]    # FasterCap's, stored
Cf, names = np.array(ref['C']), ref['names']
b = V.board_of(c)
nets = [n for ps in c['copper'].values() for n, _ in ps]
terms = {}
for lay, ps in c['copper'].items():
    for n, p in ps:
        terms.setdefault(n, [(lay, Point(p.representative_point()).buffer(0.05, quad_segs=4))])
def run(dmax, **kw):
    t = time.perf_counter()
    q = MQS(b, set(nets), terminals=terms, dmax=dmax, capacitance=True, er=1.0, log=lambda *a: None, **kw)
    C, node_labels, term_labels = q.capacitance_matrix()
    order = [term_labels[n] for n in names]
    C = np.real(np.asarray(C)[np.ix_(order, order)])
    d = (C / Cf - 1) * 100
    print(f"dmax {dmax:5.3f} {str(kw):50s} {len(q.tris):6d} tri {time.perf_counter() - t:6.1f} s   C pF {np.round(C[0] * 1e12, 4).tolist()}  against FasterCap, %: {np.round(d[0], 2).tolist()}", flush=True)
print(names, np.round(Cf * 1e12, 4).tolist())
import ast
for spec in sys.argv[2:]:
    dmax, kw = spec.split(':', 1)
    try:
        run(float(dmax), **ast.literal_eval('{' + ','.join(f"'{x.split('=')[0]}':{x.split('=')[1]}" for x in kw.split(',') if x) + '}'))
    except Exception as ex:
        print(dmax, kw, 'failed:', repr(ex)[:200])
