"""Check small blocks from the cached full-rail operators against dense fill.

Only loads completed checkpoints; it never starts a missing full-rail build.
"""
import copy,json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from jetson_compressed import build_model
from operator_cache import checkpoint,sampled_block
from pcbqs._numerics.base import RWG,combine,MU0,EPS0
from pcbqs.static import static_moments
from pcbqs.electrostatic import potential_moments


def error(actual,reference):
    floor=1e-12*np.max(abs(reference));difference=abs(actual-reference)
    return dict(relative_frobenius_error=float(np.linalg.norm(actual-reference)/np.linalg.norm(reference)),
                max_entry_relative_error=float(np.max(difference/np.maximum(abs(reference),floor))),
                relative_entry_floor=float(floor),max_absolute_error=float(difference.max()))


def main():
    root=Path(os.environ.get('JETSON_PDN_DIR','/workspace/jetson-pdn'))
    cache=Path(os.environ.get('PEEC_CACHE_DIR',str(root/'operator-cache')))
    start=time.perf_counter();q,_=build_model(root,float(os.environ.get('PEEC_DIELECTRIC_TOL','1e-4')))
    def missing():raise RuntimeError('required operator checkpoint is missing')
    checkpoint(q,'inductance',cache,missing)
    rng=np.random.default_rng(318);panels=[];edges=[];layers={}
    for name,ids in q.layer_tris.items():
        a=q.rwg.area[ids]
        chosen=np.unique([ids[np.argmin(a)],ids[np.argmax(a)],*rng.choice(ids,2,replace=False)])
        layers[name]=np.arange(len(panels),len(panels)+len(chosen));panels.extend(chosen)
        on=np.flatnonzero(q.layer_of[q.rwg.tp]==name)
        edges.extend(rng.choice(on,min(4,len(on)),replace=False))
    panels=np.asarray(panels);edges=np.unique(edges)
    result={}
    support=np.unique(np.r_[q.rwg.tp[edges],q.rwg.tm[edges]])
    nodes,M0,Mx,My,Mrr=static_moments(q.nodes,q.tris[support],q.z_of[support],near=q.near,
                                    outer_levels=q.outer_levels,**q._fill_options)
    r=RWG(nodes,q.tris)
    basis=SimpleNamespace(**{key:getattr(r,key)[support][:,edges] for key in ('Aal','Bx','By')})
    reference=MU0/(4*np.pi)*combine(basis,Mrr,Mx,My,M0)/np.outer(r.l[edges],r.l[edges])
    actual=sampled_block(q._copper[1],edges,edges)
    result['inductance']=dict(edges=len(edges),**error(actual,reference))
    if os.environ.get('PEEC_CHECK_ONLY')!='inductance':
        checkpoint(q,'potential',cache,missing)
        m=copy.copy(q);m.tris=q.tris[panels];m.z_of=q.z_of[panels];m.layer_tris=layers;m.fill_info={}
        raw=potential_moments(m);areas=q.rwg.area[panels]
        reference=raw/(areas[:,None]*areas[None,:]*4*np.pi*EPS0)
        actual=sampled_block(q._potential.kernel,panels,panels)
        result['potential']=dict(panels=len(panels),**error(actual,reference))
    result['seconds']=time.perf_counter()-start
    out=Path(os.environ.get('PEEC_CHECK_RESULT',str(Path(__file__).with_name('jetson-operator-check.json'))))
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
