"""face_study's trace over a plane with graded edge rings: AC resistance against FastHenry's stored answers
(results/face_study). The CHANGELOG's edge-ring figures for --thickness (R-C10).

    python validation/edge_rings_faces.py [h,tp ...]      # h: the trace's height, tp: the plane's thickness (mm)
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
import json, time
import numpy as np
import face_study as FS
import pcbqs.mqs as M
rows = json.loads((HERE / 'results' / 'face_study' / 'result.json').read_text())['rows']   # FastHenry's, stored
orig = M.MQS
cases = [tuple(float(x) for x in a.split(',')) for a in sys.argv[1:]] or [(1.6, 0.035), (0.1, 0.035)]
for h, tp in cases:
    fh = {round(np.log10(r['f'][0])): r for r in rows if r['tool'] == 'fasthenry' and r['h'] == h and r['tp'] == tp}
    R0 = np.array([fh[k]['R'][0] for k in (6, 7, 8)]); L0 = np.array([fh[k]['L'][0] for k in (6, 7, 8)])
    print(f"h {h} mm, plane {tp * 1e3:g} um: FastHenry R {np.round(R0 * 1e3, 2)} mOhm, L {np.round(L0 * 1e9, 3)} nH (1, 10, 100 MHz)")
    for sheets in (1, 2):
        for dmax, kw in ((0.5, {}), (0.5, dict(edge_rings=6, edge_width=0.5)), (0.5, dict(edge_rings=10, edge_width=0.5)), (0.35, dict(edge_rings=10, edge_width=0.5))):
            class Q(orig):
                def __init__(self, *a, **k):
                    k.update(kw); super().__init__(*a, **k)
            M.MQS = Q
            t = time.perf_counter()
            try:
                r = FS.pcbqs(h, tp, dmax, sheets)
            except Exception as ex:
                print(f"   sheets {sheets} dmax {dmax} {kw}: failed {repr(ex)[:120]}"); continue
            finally:
                M.MQS = orig
            dR, dL = (np.array(r['R']) / R0 - 1) * 100, (np.array(r['L']) / L0 - 1) * 100
            print(f"   sheets {sheets} dmax {dmax:<5g} {str(kw):42s} {r['n']:6d} tri {time.perf_counter() - t:5.1f} s  R {np.round(dR, 1)} %  L {np.round(dL, 2)} %", flush=True)
