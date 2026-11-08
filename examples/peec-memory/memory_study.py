"""Mesh-only grading and sampled kernel ranks; never allocates a full rail matrix.

Run with the pcbqs/kicad2ems sources on PYTHONPATH. Monkeypatches are confined
to this process and do not change the solver's public API or defaults.
"""
from pathlib import Path
import os
import json, time, gc
import numpy as np
from shapely.ops import unary_union
from pcbqs.pdn.rail import Rail
from pcbqs.pdn.stack import RailPDN
import pcbqs.pdn.peec_trial as trial
import pcbqs.mqs as mqs
from pcbqs.static_backends import FillEngine

ROOT=Path(os.environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'))
b=json.loads((ROOT/'board.json').read_text())
rail=Rail(b,'+1V15',source='L5',ground_source='U69',loads={'U21':1.})
rail.parts['U21'].pads=[p for p in rail.parts['U21'].pads if p.net in {'+1V15','GND'}]
old=RailPDN(rail,dmax=3.,plane_min=50.,vrm=dict(R=.002,L=20e-9),log=lambda *_:None)
polys=[p.geom for info in old.parts.values() for p in info['part'].pads if p.net in set(rail.nets)|{'GND'}]
polys += [p.geom for s in rail.series for p in s.part.pads]
crop=unary_union(polys).envelope.buffer(1.,join_style=2)
out=dict(crop_bounds_mm=list(crop.bounds),mesh_runs=[],blocks=[])
captured=[]
def capture(*a,**kw):
    q=mqs.MQS(*a,**kw);captured.append(q);return q
trial.MQS=capture
original_copper=mqs.copper_of
for coarsen,simplify in [(None,0),(2,0),(4,0),(8,0),(4,.002),(4,.005),(4,.01),(4,.025)]:
    def copper(*a,**kw):
        g=original_copper(*a,**kw)
        return g.simplify(simplify,preserve_topology=True) if simplify and g is not None else g
    mqs.copper_of=copper
    t=time.perf_counter()
    try:
        trial.PEECPDN(old,dmax=3.,region=crop,max_triangles=1,coarsen=coarsen,log=lambda *_:None)
    except RuntimeError as e:
        if 'dense-solver limit' not in str(e):raise
    q=captured.pop();n=len(q.tris)
    entry=dict(coarsen=coarsen,simplify_mm=simplify,triangles=n,rwg_edges=len(q.rwg.tp),charge_nodes=int(q.n_nodes),via_segments=len(q.vseg),terminal_count=len(q.term_node),four_moments_GB=32*n*n/1e9,seconds=time.perf_counter()-t,layers={k:len(v) for k,v in q.layer_tris.items()})
    out['mesh_runs'].append(entry);print(json.dumps(entry),flush=True)
    (ROOT/'memory-study.json').write_text(json.dumps(out,indent=2)+'\n')
    if coarsen is None:base=q
    del q;gc.collect()
mqs.copper_of=original_copper

# The scalar potential kernel and all six reciprocal current moments share
# the same geometric far field. SVD here measures compressibility; it does
# not implement a scalable compression algorithm (ACA would avoid the block).
q=base
engine=FillEngine(q.nodes,q.tris,q.z_of,workers=2)
cent=engine.geometry.cent
layers=sorted(q.layer_tris,key=lambda l:len(q.layer_tris[l]),reverse=True)
la=layers[0]
lb=min(layers[1:],key=lambda l:abs(q.z_of[q.layer_tris[l][0]]-q.z_of[q.layer_tris[la][0]]))
ids=q.layer_tris[la];other=q.layer_tris[lb]
rng=np.random.default_rng(42)
left=ids[cent[ids,0]<np.quantile(cent[ids,0],.25)]
right=ids[cent[ids,0]>np.quantile(cent[ids,0],.75)]
pick=lambda a:rng.choice(a,min(384,len(a)),replace=False)
cases=[('same_layer_separated',pick(left),pick(right)),('same_layer_touching',pick(ids[cent[ids,0]<=np.median(cent[ids,0])]),pick(ids[cent[ids,0]>np.median(cent[ids,0])])),('overlapping_planes',pick(ids),pick(other))]
for name,a,c in cases:
    p,r=np.meshgrid(a,c,indexing='ij');p=p.ravel();r=r.ravel()
    t=time.perf_counter();values=[]
    for start in range(0,len(p),4096):
        pp,rr=p[start:start+4096],r[start:start+4096]
        values.append(engine.evaluate(pp,rr,q.z_of[pp]-q.z_of[rr]))
    moments=np.vstack(values).reshape(len(a),len(c),6)
    metrics=[]
    for k in [0,1,2,5]:
        mat=moments[:,:,k];s=np.linalg.svd(mat,compute_uv=False)
        tail=np.r_[np.sqrt(np.cumsum(s[::-1]**2))[::-1],0.]/np.linalg.norm(s)
        metrics.append(dict(moment=['M0','Mrx','Mry','Mrr'][[0,1,2,5].index(k)],ranks={str(tol):int(np.flatnonzero(tail<=tol)[0]) for tol in [1e-3,1e-4,1e-6]}))
    mat=moments[:,:,0]/(q.rwg.area[a,None]*q.rwg.area[None,c])
    s=np.linalg.svd(mat,compute_uv=False)
    tail=np.r_[np.sqrt(np.cumsum(s[::-1]**2))[::-1],0.]/np.linalg.norm(s)
    metrics.append(dict(moment='M0_area_normalized',ranks={str(tol):int(np.flatnonzero(tail<=tol)[0]) for tol in [1e-3,1e-4,1e-6]}))
    entry=dict(case=name,layer_a=la,layer_b=la if name!='overlapping_planes' else lb,separation_mm=float(abs(q.z_of[a[0]]-q.z_of[c[0]])*1000),shape=[len(a),len(c)],seconds=time.perf_counter()-t,moments=metrics)
    out['blocks'].append(entry);print(json.dumps(entry),flush=True)
    (ROOT/'memory-study.json').write_text(json.dumps(out,indent=2)+'\n')
