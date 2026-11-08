# Check the intended copper/capacitor loop before matrix filling.
from pathlib import Path
import json,time
import numpy as np
from pcbqs.pdn.rail import parts_of,terminal,part_value
from pcbqs.mqs import MQS
from shapely.ops import unary_union
ROOT=Path(__import__('os').environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'));b=json.loads((ROOT/'board.json').read_text());ps={p.ref:p for p in parts_of(b)};cap=ps['C90'];load=ps['U21'];layers=[c['name'] for c in b['stack']['copper']]
terms={k:terminal(p,layers) for k,p in [('cap_p',cap.on({'+1V15'})),('cap_g',cap.on({'GND'})),('load_p',[p for p in load.pads if p.name=='25']),('load_g',[p for p in load.pads if p.name=='24'])]}
extent=unary_union([p for vv in terms.values() for _,p in vv]).envelope;f=np.array([1e6,1e7,30e6]);rows=[]
for margin,dmax in [(.6,1.),(1.,1.),(1.5,1.),(1.,.5)]:
 t=time.perf_counter();crop=extent.buffer(margin,join_style=2)
 q=MQS(b,{'+1V15','GND'},terminals=terms,region=crop,dmax=dmax,conductor_loss='slab',conductor_faces=1,fill_workers=2,log=lambda *_:None)
 q.source('I','load_p','load_g',part_h=0);q.element('C90','cap_p','cap_g',C=part_value(cap),ESR=.005,ESL=.35e-9,part_h=0);q._loops()
 if len(q.tris)>3000:raise RuntimeError('Dense limit exceeded')
 r=q.solve(f);z=r.z['I'];row=dict(margin_mm=margin,dmax_mm=dmax,triangles=len(q.tris),real_ohm=z.real.tolist(),imag_ohm=z.imag.tolist(),seconds=time.perf_counter()-t);rows.append(row)
 (ROOT/'local-crop-check.json').write_text(json.dumps(dict(frequencies_Hz=f.tolist(),runs=rows),indent=2)+'\n');print(row,flush=True)
