"""Two face sheets near a plane: pcbqs.mqs (one sheet, two faces) against FastHenry on a trace over a plane, the plane
close or far and thin or thick (the CM5 loop's In1 case, without its port-region convergence).

    python validation/face_study.py fasthenry H TP FREQ      # one FastHenry run (single-threaded: run side by side)
    python validation/face_study.py pcbqs [DMAX ...]           # every case, one sheet and two faces
    python validation/face_study.py report

A 1 mm trace 20 mm long on top (35 um), down a via (r 0.15) to a 26 x 6 mm plane, back up a via to a pad; the port
across the 0.5 mm gap between the pad and the trace's start. H is the dielectric between the plane's top face and the
trace's bottom face (KiCad's convention), the plane TP thick; FastHenry places the copper as pcbqs's thick models do
(pcbqs.pdn.mesh.copper_centres), one sheet sits at the interfaces. FastHenry: the trace as one segment split
into 31 x 5 filaments, the plane a uniform 0.125 mm grid with 5 filaments through its thickness, -p diag.
"""
import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Point, box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import OUT, write_json  # noqa: E402

DIR = OUT / 'face_study'
F = np.array([1e6, 1e7, 1e8])
TT = 0.035
PLANE = (-13.0, -3.0, 13.0, 3.0)
CASES = [(h, tp) for h in (1.6, 0.4, 0.1) for tp in (0.015, 0.035)]


def board(h, tp):
    trace = box(-10, -0.5, 10, 0.5).union(Point(10, 0).buffer(0.6))
    pad = box(-12.1, -0.6, -10.5, 0.6)
    return dict(stack=dict(copper=[dict(name='B.Cu', z=0.0, t=tp), dict(name='F.Cu', z=h, t=TT)],
                           layers=[dict(name='core', z0=0.0, z1=h, er=4.4, tand=0.0)], thickness=h),
                copper={'F.Cu': [dict(net='S', polygon=mapping(trace)), dict(net='S', polygon=mapping(pad))],
                        'B.Cu': [dict(net='S', polygon=mapping(box(*PLANE)))]},
                vias=[dict(x=x, y=0.0, r=0.15, z0=0.0, z1=h, net='S') for x in (10.0, -11.5)],
                outline=mapping(box(-20, -10, 20, 10)), prims=dict(footprints=[]))


PADS = (('F.Cu', box(-10.6, -0.5, -10.5, 0.5)), ('F.Cu', box(-10, -0.5, -9.9, 0.5)))


def pcbqs(h, tp, dmax, sheets, face='overlap'):
    from pcbqs.mqs import MQS
    q = MQS(board(h, tp), {'S'}, terminals={'A': [PADS[0]], 'B': [PADS[1]]}, dmax=dmax, conductor_loss='slab',
            thickness_sheets=sheets, log=lambda *a: None)
    q.face_correction = face
    q.source('port', 'A', 'B', part_h=0.0)
    Z = np.asarray(q.solve(F).zc['port'])
    return dict(tool='pcbqs', h=h, tp=tp, mesh=dmax, sheets=sheets, face=face if sheets == 2 else None,
                n=len(q.tris), R=Z.real, L=Z.imag / (2 * np.pi * F))


def fasthenry_inp(h, tp, cell=0.125, nw=31, nh=5):
    from mqs_fasthenry import via_bars
    x0, y0, x1, y1 = PLANE
    s1, s2 = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    from pcbqs.pdn.mesh import copper_centres
    c = copper_centres(board(h, tp))                # as pcbqs's thick models: the dielectric (h) between faces
    zt, zb = c['F.Cu'][0], c['B.Cu'][0]
    return f""".units mm
.default sigma=5.8e4 nhinc={nh} nwinc=1 rw=2 rh=2
G1 x1={x0} y1={y0} z1={zb} x2={x1} y2={y0} z2={zb} x3={x1} y3={y1} z3={zb} thick={tp} seg1={s1} seg2={s2}
+ nhinc={nh} relx=0 rely=0 relz=0
+ npa (10,0,{zb})
+ npb (-11.5,0,{zb})
NT0 x=-10 y=0 z={zt}
NT1 x=10 y=0 z={zt}
NP0 x=-10.5 y=0 z={zt}
NP1 x=-11.5 y=0 z={zt}
ET NT0 NT1 w=1 h={TT} nwinc={nw}
EP NP0 NP1 w=1.2 h={TT} nwinc=9
{via_bars('VA', 10, 0, 'NT1', 'npa', zt, zb)}
{via_bars('VB', -11.5, 0, 'NP1', 'npb', zt, zb)}
.external NT0 NP0
"""


def fasthenry(h, tp, f):
    from mqs_fasthenry import fasthenry as run_fh
    tag = f'h{h:g}_tp{tp * 1e3:g}_f{f:.0e}'
    fz, R, L = run_fh('face', fasthenry_inp(h, tp), [f, f], DIR / f'fasthenry_{tag}', options=('-p', 'diag'))
    return dict(tool='fasthenry', h=h, tp=tp, f=[float(fz[0])], R=[float(R[0])], L=[float(L[0])])


def save(rows):
    import fcntl
    DIR.mkdir(parents=True, exist_ok=True)
    with open(DIR / 'result.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = DIR / 'result.json'
        old = json.loads(path.read_text())['rows'] if path.is_file() else []
        key = lambda r: (r['tool'], r['h'], r['tp'], r.get('mesh'), r.get('sheets'), r.get('face'), str(r.get('f')))
        merged = {key(r): r for r in old}
        merged.update({key(r): r for r in rows})
        write_json(path, dict(f=F, rows=list(merged.values())))


def report():
    rows = json.loads((DIR / 'result.json').read_text())['rows']
    for h, tp in CASES:
        fh = {round(np.log10(r['f'][0])): r for r in rows if r['tool'] == 'fasthenry' and r['h'] == h and r['tp'] == tp}
        if len(fh) < 3:
            continue
        R0 = np.array([fh[k]['R'][0] for k in (6, 7, 8)])
        L0 = np.array([fh[k]['L'][0] for k in (6, 7, 8)])
        print(f'h {h} mm, plane {tp * 1e3:g} um: FastHenry R {np.round(R0 * 1e3, 3)} mOhm, L {np.round(L0 * 1e9, 3)} nH')
        for r in sorted((r for r in rows if r['tool'] == 'pcbqs' and r['h'] == h and r['tp'] == tp),
                        key=lambda r: (r['sheets'], str(r['face']), -r['mesh'])):
            dR, dL = (np.array(r['R']) / R0 - 1) * 100, (np.array(r['L']) / L0 - 1) * 100
            print(f"   sheets {r['sheets']} {r['face'] or '':8s} dmax {r['mesh']:<5g} R {np.round(dR, 1)}%  L {np.round(dL, 2)}%")


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'report'
    if what == 'fasthenry':
        h, tp, f = map(float, sys.argv[2:5])
        save([fasthenry(h, tp, f)])
    elif what == 'pcbqs':
        for dmax in [float(a) for a in sys.argv[2:]] or [0.5, 0.35]:
            rows = []
            for h, tp in CASES:
                for sheets in (1, 2):
                    rows.append(pcbqs(h, tp, dmax, sheets))
            save(rows)
            print(f'pcbqs dmax {dmax} done', flush=True)
    else:
        report()
