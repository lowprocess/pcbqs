from pathlib import Path
import json,time
from shapely.ops import unary_union
from pcbqs.pdn.rail import Rail,parts_of
from pcbqs.pdn.stack import RailPDN
from pcbqs.pdn.peec_trial import PEECPDN
from pcbqs.electrostatic import DielectricStack
ROOT=Path(__import__('os').environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'));b=json.loads((ROOT/'board.json').read_text())
rail=Rail(b,'+1V15',source='L5',ground_source='U69',loads={'U21':1.})
rail.parts['U21'].pads=[p for p in rail.parts['U21'].pads if p.net in {'+1V15','GND'}]
print('Build legacy metadata',flush=True)
old=RailPDN(rail,dmax=3.,plane_min=50.,vrm=dict(R=.002,L=20e-9),log=lambda *_:None)
polys=[p.geom for info in old.parts.values() for p in info['part'].pads if p.net in set(rail.nets)|{'GND'}]
polys += [p.geom for s in rail.series for p in s.part.pads]
crop=unary_union(polys).envelope.buffer(1.,join_style=2)
entry=dict(net='+1V15',crop_bounds_mm=list(crop.bounds),dmax_mm=3.,dense_triangle_limit=3000)
t=time.perf_counter()
try:
 q=PEECPDN(old,dmax=3.,region=crop,max_triangles=3000,fill_workers=2,log=print)
 entry.update(triangles=len(q.q.tris),layers={k:len(v) for k,v in q.q.layer_tris.items()})
except Exception as e:
 entry['error']=type(e).__name__+': '+str(e)
entry['seconds']=time.perf_counter()-t
(ROOT/'peec-feasibility.json').write_text(json.dumps(entry,indent=2)+'\n');print(json.dumps(entry,indent=2),flush=True)
