"""Attempt the complete cropped +1V15 rail using the experimental backend.

Requires JETSON_PDN_DIR/board.json from the Jetson import study. Writes partial
stage results before solving so failed attempts remain inspectable.
"""
from pathlib import Path
import os,json,time,resource
import numpy as np
from shapely.ops import unary_union
from pcbqs.pdn.rail import Rail
from pcbqs.pdn.stack import RailPDN
from pcbqs.pdn.peec_trial import PEECPDN
from pcbqs.electrostatic import DielectricStack

ROOT=Path(os.environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'))
out=Path(os.environ.get('PEEC_RESULT',str(ROOT/'compressed-rail.json')))
start=time.perf_counter();result=dict(stages=[])
def stage(name,**data):
    result['stages'].append(dict(name=name,seconds=time.perf_counter()-start,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**data))
    out.write_text(json.dumps(result,indent=2)+'\n');print(name,json.dumps(result['stages'][-1]),flush=True)
b=json.loads((ROOT/'board.json').read_text())
rail=Rail(b,'+1V15',source='L5',ground_source='U69',loads={'U21':1.})
rail.parts['U21'].pads=[p for p in rail.parts['U21'].pads if p.net in {'+1V15','GND'}]
old=RailPDN(rail,dmax=3.,plane_min=50.,vrm=dict(R=.002,L=20e-9),log=lambda *_:None)
polys=[p.geom for info in old.parts.values() for p in info['part'].pads if p.net in set(rail.nets)|{'GND'}]
polys += [p.geom for s in rail.series for p in s.part.pads]
crop=unary_union(polys).envelope.buffer(1.,join_style=2)
try:
    adapter=PEECPDN(old,dmax=3.,region=crop,coarsen=4,matrix_backend='hierarchical',
                    compression_tol=1e-5,solver_tol=1e-7,solver_maxiter=1000,fill_workers=2,log=print)
    # This is a feasibility run. The relaxed image expansion needs a separate
    # tolerance convergence study before claiming numerical accuracy.
    adapter.q._dielectric_stack=DielectricStack.from_board(b,1e-4)
    model=adapter._model(False)
    stage('mesh',triangles=len(model.tris),edges=model.ne,charge_nodes=int(model.n_nodes),
          crop_bounds_mm=list(crop.bounds),dielectric_tol=1e-4,compression_tol=1e-5)
    model._assemble();stage('inductance',fill=model.fill_info)
    model._ensure_potential();stage('potential',fill=model.fill_info)
    r=model.solve([1e6]);z=r.z['load:U21'][0]
    stage('solved',frequency_Hz=1e6,Z_ohm=[float(z.real),float(z.imag)],solves=model.solve_info)
except Exception as e:
    stage('failed',error=type(e).__name__+': '+str(e));raise
