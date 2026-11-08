import json,time,copy,gc,traceback
from pathlib import Path
import numpy as np
from shapely.geometry import box
from pcbqs.pdn.rail import parts_of,part_value,terminal
from pcbqs.mqs import MQS
ROOT=Path(__import__('os').environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'));b=json.loads((ROOT/'board.json').read_text());ps={p.ref:p for p in parts_of(b)}
cap=ps['C90'];load=ps['U21'];cp=cap.on({'+1V15'});cg=cap.on({'GND'})
lp=[p for p in load.pads if p.name=='25'];lg=[p for p in load.pads if p.name=='24']
layers=[c['name'] for c in b['stack']['copper']]
terms={k:terminal(pads,layers) for k,pads in [('cap_p',cp),('cap_g',cg),('load_p',lp),('load_g',lg)]}
from shapely.ops import unary_union
extent=unary_union([p for pads in terms.values() for _,p in pads]).envelope
f=np.geomspace(1e5,30e6,13);result=dict(net='+1V15',load='U21',power_pin='25',ground_pin='24',capacitor='C90',C_F=part_value(cap),assumed_ESR_ohm=.005,assumed_ESL_H=.35e-9,frequencies_Hz=f.tolist(),runs=[])
for margin in [.3,.6]:
 crop=extent.buffer(margin,join_style=2);entry=dict(margin_mm=margin,crop_bounds_mm=list(crop.bounds),models={});shared=None
 for name,capacitance,loss in [('peec_skin',True,'slab'),('mqs_skin',False,'slab'),('peec_dc',True,'dc')]:
  t=time.perf_counter();print('LOCAL',margin,name,flush=True)
  try:
   q=MQS(b,{'+1V15','GND'},terminals=terms,region=crop,dmax=1.,capacitance=capacitance,dielectric='stack',dielectric_tol=1e-6,conductor_loss=loss,conductor_faces=1,fill_workers=2,factor_cache_bytes=0,log=lambda msg:print('  '+msg,flush=True))
   print('MESH',len(q.tris),flush=True)
   if len(q.tris)>3000:raise RuntimeError('local mesh exceeds 3000-triangle dense limit: '+str(len(q.tris)))
   if shared is not None:
    for attr in ['nodes','tris','node_of','z_of','t_of']:
     if not np.array_equal(getattr(q,attr),getattr(shared,attr)):raise RuntimeError('mesh differs')
    q._copper,q._potential,q._sheet_terms=shared._copper,shared._potential,shared._sheet_terms
   q.source('I','load_p','load_g',part_h=0)
   q.element('C90','cap_p','cap_g',C=part_value(cap),ESR=.005,ESL=.35e-9,part_h=0)
   q._loops(allow_open=False) # Intended local decap loop must survive the crop.
   r=q.solve(f);z=r.z['I'];Rcu=[]
   for i,ff in enumerate(f):Rcu.append(float(np.vdot(r.i[i],(q._mats['R']+q._resistance_delta(ff))@r.i[i]).real))
   entry['models'][name]=dict(real_ohm=z.real.tolist(),imag_ohm=z.imag.tolist(),effective_copper_R_ohm=Rcu,triangles=len(q.tris),fill=q.fill_info,seconds=time.perf_counter()-t)
   if shared is None:shared=q
   print('DONE',name,entry['models'][name]['seconds'],flush=True)
  except Exception as e:
   entry['models'][name]=dict(error=type(e).__name__+': '+str(e));traceback.print_exc()
   if shared is None:break
 result['runs'].append(entry);(ROOT/'local-peec.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 del shared;gc.collect()
