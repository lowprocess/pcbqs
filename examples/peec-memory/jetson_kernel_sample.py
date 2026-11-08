"""Check on-demand layered entries against dense integration on real triangles.

This samples geometry/kernel integration only, not full-rail port accuracy.
"""
from pathlib import Path
from types import SimpleNamespace
import os,json,time
import numpy as np
from pcbqs.mqs import MQS
from pcbqs.static_backends import FillEngine
from pcbqs.compressed import _LayeredPairs
from pcbqs.electrostatic import potential_moments,DielectricStack

root=Path(os.environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'))
start=time.perf_counter();b=json.loads((root/'board.json').read_text())
q=MQS(b,{'+1V15','/DCDC/1V15_OUT','GND'},region=(190.37,-135.475,215.375,-116.025),
      capacitance=True,dmax=3.,coarsen=4,dielectric='stack',dielectric_tol=1e-4,fill_workers=2,log=lambda *_:None)
selected=[];layers={}
for name,ids in q.layer_tris.items():
    cen=q.nodes[q.tris[ids]].mean(1);area=q.rwg.area[ids]
    chosen=np.unique([0,int(np.argmin(area)),int(np.argmax(area)),
                      int(np.argmin(np.linalg.norm(cen-[211.,-121.],axis=1)))])
    layers[name]=np.arange(len(selected),len(selected)+len(chosen))
    selected.extend(ids[chosen])
selected=np.array(selected)
m=SimpleNamespace(nodes=q.nodes,tris=q.tris[selected],z_of=q.z_of[selected],layer_tris=layers,
                  near=3.,outer_levels=1,_fill_options=q._fill_options,
                  _dielectric_stack=DielectricStack.from_board(b,1e-4),fill_info={})
reference=potential_moments(m)
engine=FillEngine(m.nodes,m.tris,m.z_of,near=3.,outer_levels=1,**m._fill_options)
p,r=np.meshgrid(np.arange(len(selected)),np.arange(len(selected)),indexing='ij')
actual=_LayeredPairs(m,engine)(p.ravel(),r.ravel()).reshape(reference.shape)
relative=abs(actual-reference)/np.maximum(abs(reference),np.finfo(float).tiny)
result=dict(triangles=len(selected),layer_counts={k:len(v) for k,v in layers.items()},dielectric_tol=1e-4,
            max_entry_relative_error=float(relative.max()),relative_frobenius_error=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
            seconds=time.perf_counter()-start)
target=Path(__file__).with_name('jetson-kernel-sample.json');target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
