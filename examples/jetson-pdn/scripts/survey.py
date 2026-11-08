import json,time,gc,traceback,sys
from pathlib import Path
import numpy as np
from pcbqs.pdn.rail import Rail,parts_of
from pcbqs.pdn.stack import RailPDN
ROOT=Path(__import__('os').environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'))
b=json.loads((ROOT/'board.json').read_text());parts={p.ref:p for p in parts_of(b)}
f=np.geomspace(1e3,30e6,41)
configs=[('SYS_VIN_HV','L3','U11',['J1']),('+1V15','L5','U69',['U21']),('+5V','L1','U12',['U69','U7','U30']),('+3V3','L2','U13',['J13','U21','J12']),('+1V8','L7','U7',['U43','U78']),('+5V_AON','L8','U75',['U15','U70']),('+20V','L4','U65',['U63','U64']),('+12V','L6','U8',['TP98'])]
results=[]
for net,source,ground,users in configs:
 print('START',net,flush=True);t=time.perf_counter();entry=dict(net=net,source=source,ground_source=ground,loads=users,load_A_each=1.,VRM_assumption=dict(R_ohm=.002,L_H=20e-9),resolutions=[])
 try:
  rail=Rail(b,net,source=source,ground_source=ground,loads={u:1. for u in users})
  # Observe only actual named supply pads, not logic strap pins reachable through zero-ohm links.
  from copy import copy
  rail.parts=dict(rail.parts)
  for ref in users:
   p=copy(rail.parts[ref]);p.pads=[pad for pad in p.pads if pad.net in {net,'GND','/M.2/3V3_M2'}];rail.parts[ref]=p
  entry.update(nets=sorted(rail.nets),series=[dict(ref=s.part.ref,kind=s.kind,R_ohm=s.R,a=s.a,b=s.b) for s in rail.series],decaps=len(rail.decaps))
  plane_min = None if net in {'SYS_VIN_HV','+5V','+3V3','+1V8'} else 50.
  entry['plane_selection'] = 'default' if plane_min is None else 'explicit 50 mm2 small-rail diagnostic'
  for dmax in [3.,2.]:
   print('BUILD',net,dmax,flush=True);tb=time.perf_counter();model=RailPDN(rail,dmax=dmax,plane_min=plane_min,vrm=dict(R=.002,L=20e-9),log=lambda msg:print('  '+msg,flush=True))
   z=model.z(f,observe=users);bare=model.z(f,observe=users,bare=True)
   rec=dict(dmax_mm=dmax,plane_min_mm2=plane_min,vertices=model.nv,seconds=time.perf_counter()-tb,
       cavities=[dict(power=c.rail_layer,ground=c.ground_layer,area_mm2=c.geom.area,d_mm=c.d,er=c.er) for c in model.cavities],notes=model.notes,
       port_connections={ref:dict(role=v['role'],power=v['rail'].why,ground=v['gnd'].why,mount_H=v['mount'],dof=model.dof(ref),element=v['el']) for ref,v in model.parts.items()},
       impedance={ref:dict(real_ohm=zz.real.tolist(),imag_ohm=zz.imag.tolist(),bare_abs_ohm=abs(bare[ref]).tolist()) for ref,zz in z.items()})
   entry['resolutions'].append(rec)
   print('DONE',net,dmax,'seconds',rec['seconds'],'ports',list(z),flush=True)
   (ROOT/'survey.json').write_text(json.dumps(dict(frequencies_Hz=f.tolist(),rails=results+[entry]),indent=2,allow_nan=False))
   del model;gc.collect()
 except Exception as exc:
  entry['error']=type(exc).__name__+': '+str(exc);print('ERROR',net,entry['error'],flush=True);traceback.print_exc()
 entry['seconds']=time.perf_counter()-t;results.append(entry)
 (ROOT/'survey.json').write_text(json.dumps(dict(frequencies_Hz=f.tolist(),rails=results),indent=2,allow_nan=False))
 print('FINISH',net,entry['seconds'],flush=True)
