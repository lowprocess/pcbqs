"""Compare dense/compressed PDN in separate processes (including peak RSS).

PYTHONPATH=src python tools/benchmark_peec_memory.py --out examples/peec-memory/backend-comparison.json
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np


def worker(args):
    from trial_pdn_peec import board,VRM
    from pcbqs.pdn.rail import Rail
    from pcbqs.pdn.stack import RailPDN
    from pcbqs.pdn.peec_trial import PEECPDN
    b=board(slot=args.case=='slotted')
    old=RailPDN(Rail(b,'P',source='U0',ground_source='U0',loads={'U1':1.}),dmax=1.25,vrm=VRM,log=lambda *_:None)
    t=time.perf_counter()
    q=PEECPDN(old,dmax=1.25,contact_spacing=1.25/12,matrix_backend=args.worker,
              fill_workers=2,compression_tol=args.tol,solver_tol=1e-8,solver_maxiter=2000,coarsen=args.coarsen,
              log=lambda *_:None)
    f=np.array([1e5,1e6,4e6,4.86e6,5e6,6e6,10e6,30e6])
    z=q.z(f)['U1'];model=q._model(False)
    return dict(backend=args.worker,case=args.case,triangles=len(model.tris),compression_tol=args.tol,
                coarsen=args.coarsen,frequencies_Hz=f.tolist(),real_ohm=z.real.tolist(),imag_ohm=z.imag.tolist(),
                seconds=time.perf_counter()-t,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                fill=model.fill_info,solves=model.solve_info)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',choices=['dense','hierarchical']);p.add_argument('--case',choices=['rectangular','slotted'],default='rectangular')
    p.add_argument('--tol',type=float,default=1e-6);p.add_argument('--coarsen',type=float)
    p.add_argument('--out',type=Path,default=Path('examples/peec-memory/backend-comparison.json'))
    args=p.parse_args()
    if args.worker:print(json.dumps(worker(args)));return
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1');result=dict(runs=[])
    for case in ['rectangular','slotted']:
        reference=None
        for backend in ['dense','hierarchical']:
            command=[sys.executable,__file__,'--worker',backend,'--case',case,'--tol',str(args.tol)]
            if args.coarsen is not None:command+=['--coarsen',str(args.coarsen)]
            run=subprocess.run(command,env=env,text=True,capture_output=True)
            if run.returncode:raise RuntimeError(run.stderr)
            data=json.loads(run.stdout);z=np.array(data['real_ohm'])+1j*np.array(data['imag_ohm'])
            if reference is None:reference=z
            data['max_complex_relative_error_vs_dense']=float(np.max(abs(z-reference)/abs(reference)))
            result['runs'].append(data);args.out.parent.mkdir(parents=True,exist_ok=True)
            args.out.write_text(json.dumps(result,indent=2)+'\n')
            print(case,backend,data['triangles'],f"{data['seconds']:.2f}s",f"{data['peak_RSS_MiB']:.1f} MiB",data['max_complex_relative_error_vs_dense'],flush=True)


if __name__=='__main__':main()
