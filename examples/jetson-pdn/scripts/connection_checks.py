import json,time
from pathlib import Path
import numpy as np
from pcbqs.pdn.rail import Rail,terminal
from pcbqs.pdn.dc import Network
ROOT=Path(__import__('os').environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'));b=json.loads((ROOT/'board.json').read_text());layers=[c['name'] for c in b['stack']['copper']];rows=[]
for net,source,gnd,load in [('+1V8','L7','U7','U78'),('+12V','L6','U8','TP98')]:
 t=time.perf_counter();r=Rail(b,net,source=source,ground_source=gnd,loads={load:1.})
 terms={'source':terminal(r.source_pads,layers),'load':terminal(r.parts[load].on({net}),layers)}
 for s in r.series:
  terms[s.part.ref+':a']=terminal([p for p in s.part.pads if p.net==s.a],layers)
  terms[s.part.ref+':b']=terminal([p for p in s.part.pads if p.net==s.b],layers)
 nw=Network(b,r.nets,terms,dmax=3.,log=lambda *_:None,fine=['source','load'])
 for s in r.series:nw.series(s.part.ref,s.part.ref+':a',s.part.ref+':b',s.R)
 result=nw.solve({'source':1.},{'load':-1.})
 row=dict(net=net,load=load,power_path_only=True,source=source,drop_V_at_1A=1-result.terminals['load'],power_path_R_ohm=1-result.terminals['load'],loss_W=result.loss,notes=result.notes,mesh_nodes=result.nodes,seconds=time.perf_counter()-t)
 rows.append(row);print(row,flush=True)
(ROOT/'connection-checks.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
