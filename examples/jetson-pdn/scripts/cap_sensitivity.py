import json,time
from pathlib import Path
import numpy as np
from pcbqs.pdn.rail import Rail
from pcbqs.pdn.stack import RailPDN
ROOT=Path(__import__('os').environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'));b=json.loads((ROOT/'board.json').read_text());r=Rail(b,'+1V15',source='L5',ground_source='U69',loads={'U21':1.})
r.parts['U21'].pads=[p for p in r.parts['U21'].pads if p.net in {'+1V15','GND'}]
p=RailPDN(r,dmax=2.,plane_min=50.,vrm=dict(R=.002,L=20e-9),log=lambda *_:None)
f=np.geomspace(1e3,30e6,161);original={n:i['el']['C'] for n,i in p.parts.items() if i['el'] and 'C' in i['el']};rows=[]
for scale in [1.,.5,.2]:
 for name,C in original.items():p.parts[name]['el']['C']=C*scale
 z=p.z(f,observe=['U21'])['U21'];k=abs(z).argmax();rows.append(dict(C_scale=scale,peak_ohm=float(abs(z[k])),peak_frequency_Hz=float(f[k]),real_ohm=z.real.tolist(),imag_ohm=z.imag.tolist()))
(ROOT/'cap-sensitivity.json').write_text(json.dumps(dict(net='+1V15',frequencies_Hz=f.tolist(),runs=rows,note='Uniform capacitance scaling is a sensitivity study, not measured MLCC bias data.'),indent=2)+'\n')
print([(r['C_scale'],r['peak_ohm'],r['peak_frequency_Hz']) for r in rows],flush=True)
