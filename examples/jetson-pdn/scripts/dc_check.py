import json,time,traceback,gc
from pathlib import Path
from pcbqs.pdn.rail import Rail
from pcbqs.pdn.dc import solve_rail
ROOT=Path(__import__('os').environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'));b=json.loads((ROOT/'board.json').read_text());rows=[]
for net,source,ground,load,volts in [('SYS_VIN_HV','L3','U11','J1',16.),('+1V15','L5','U69','U21',1.15)]:
 entry=dict(net=net,source=source,ground_source=ground,load=load,normalized_A=1.,source_volts=volts,resolutions=[])
 rail=Rail(b,net,source=source,ground_source=ground,loads={load:1.})
 rail.parts[load].pads=[p for p in rail.parts[load].pads if p.net in {net,'GND'}]
 for dmax in [3.,2.]:
  print('DC',net,dmax,flush=True);t=time.perf_counter()
  try:
   r=solve_rail(rail,volts,dmax=dmax,log=lambda msg:print('  '+msg,flush=True))
   rec=dict(dmax_mm=dmax,seconds=time.perf_counter()-t,load=r['loads'][load],series=r['series'],notes=r['notes'],loss_W=r['power'].loss+r['ground'].loss,
            power_nodes=r['power'].nodes,ground_nodes=r['ground'].nodes)
   entry['resolutions'].append(rec);print(json.dumps(rec),flush=True)
   del r;gc.collect()
  except Exception as e:
   entry['error']=str(e);traceback.print_exc();break
  (ROOT/'dc.json').write_text(json.dumps(rows+[entry],indent=2,allow_nan=False))
 rows.append(entry);(ROOT/'dc.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
