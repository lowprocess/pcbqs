"""Compare opt-in bulk grading with the existing dense PEEC on small fixtures.

Run from the pcbqs checkout with PYTHONPATH=src. No production code is changed.
This is a mesh perturbation study, not an independent physical validation.
"""
from pathlib import Path
import importlib.util
import json, time
import numpy as np
import pcbqs.mqs as mqs
from pcbqs.pdn.rail import Rail
from pcbqs.pdn.stack import RailPDN
from pcbqs.pdn.peec_trial import PEECPDN

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fixture',HERE.parents[1]/'tools/trial_pdn_peec.py')
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
original=mqs.mesh_sheet
f=np.array([1e5,1e6,4e6,4.86e6,5e6,6e6,10e6,30e6])
out=dict(frequencies_Hz=f.tolist(),runs=[])
for slot in [False,True]:
    b=fixture.board(slot=slot)
    old=RailPDN(Rail(b,'P',source='U0',ground_source='U0',loads={'U1':1.}),dmax=1.25,vrm=fixture.VRM,log=lambda *_:None)
    baseline=None
    for coarsen in [None,4]:
        def mesh(*a,**kw):return original(*a,**kw,coarsen=coarsen)
        mqs.mesh_sheet=mesh
        t=time.perf_counter()
        model=PEECPDN(old,dmax=1.25,contact_spacing=1.25/12,fill_workers=2,log=lambda *_:None)
        z=model.z(f)['U1']
        if baseline is None:baseline=z
        entry=dict(slot=slot,coarsen=coarsen,triangles=len(model.q.tris),seconds=time.perf_counter()-t,
                   real_ohm=z.real.tolist(),imag_ohm=z.imag.tolist(),
                   max_complex_relative_change=float(np.max(abs(z-baseline)/abs(baseline))),
                   positive_real=bool(np.all(z.real>=0)))
        print(json.dumps(entry),flush=True);out['runs'].append(entry)
        (HERE/'graded-accuracy.json').write_text(json.dumps(out,indent=2)+'\n')
mqs.mesh_sheet=original
