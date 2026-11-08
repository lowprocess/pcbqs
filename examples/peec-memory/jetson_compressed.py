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
from operator_cache import checkpoint

ROOT=Path(os.environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'))
out=Path(os.environ.get('PEEC_RESULT',str(ROOT/'compressed-rail.json')))
cache=Path(os.environ.get('PEEC_CACHE_DIR',str(ROOT/'operator-cache')))
dielectric_tol=float(os.environ.get('PEEC_DIELECTRIC_TOL','1e-4'))
frequencies=[float(f) for f in os.environ.get('PEEC_FREQUENCIES','1e6').split(',')]
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
                    compression_tol=1e-5,solver_tol=float(os.environ.get('PEEC_SOLVER_TOL','1e-7')),
                    solver_maxiter=int(os.environ.get('PEEC_SOLVER_MAXITER','1000')),fill_workers=2,
                    log=lambda msg:print(msg,flush=True))
    # This is a feasibility run. The relaxed image expansion needs a separate
    # tolerance convergence study before claiming numerical accuracy.
    adapter.q._dielectric_stack=DielectricStack.from_board(b,dielectric_tol)
    model=adapter._model(False)
    stage('mesh',triangles=len(model.tris),edges=model.ne,charge_nodes=int(model.n_nodes),
          crop_bounds_mm=list(crop.bounds),dielectric_tol=dielectric_tol,compression_tol=1e-5)
    cached=checkpoint(model,'inductance',cache,model._assemble)
    stage('inductance',fill=model.fill_info,cache=cached)
    cached=checkpoint(model,'potential',cache,model._ensure_potential)
    stage('potential',fill=model.fill_info,cache=cached)
    for frequency in frequencies:
        r=model.solve([frequency]);z=r.z['load:U21'][0]
        stage('solved',frequency_Hz=frequency,Z_ohm=[float(z.real),float(z.imag)],solves=model.solve_info)
except KeyboardInterrupt:
    stage('interrupted');raise
except Exception as e:
    stage('failed',error=type(e).__name__+': '+str(e),solves=model.solve_info if 'model' in globals() else []);raise
