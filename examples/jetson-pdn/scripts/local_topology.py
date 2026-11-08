import json
from pathlib import Path
from shapely.ops import unary_union
from pcbqs.pdn.rail import parts_of,terminal,part_value
from pcbqs.mqs import MQS
ROOT=Path(__import__('os').environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'));b=json.loads((ROOT/'board.json').read_text());ps={p.ref:p for p in parts_of(b)}
cap=ps['C90'];load=ps['U21'];layers=[c['name'] for c in b['stack']['copper']]
terms={k:terminal(p,layers) for k,p in [('cap_p',cap.on({'+1V15'})),('cap_g',cap.on({'GND'})),('load_p',[p for p in load.pads if p.name=='25']),('load_g',[p for p in load.pads if p.name=='24'])]}
extent=unary_union([p for vv in terms.values() for _,p in vv]).envelope;rows=[]
for margin in [.3,.6,1.,1.5,2.,3.]:
 crop=extent.buffer(margin,join_style=2)
 q=MQS(b,{'+1V15','GND'},terminals=terms,region=crop,dmax=1.,log=lambda *_:None)
 q.source('I','load_p','load_g',part_h=0);q.element('C90','cap_p','cap_g',C=part_value(cap),ESR=.005,ESL=.35e-9,part_h=0)
 row=dict(margin_mm=margin,triangles=len(q.tris),via_segments=q.nv,crop_bounds_mm=list(crop.bounds))
 try:q._loops();row['closed_loop']=True
 except Exception as e:row.update(closed_loop=False,error=str(e))
 rows.append(row);print(row,flush=True)
(ROOT/'local-topology.json').write_text(json.dumps(rows,indent=2)+'\n')
