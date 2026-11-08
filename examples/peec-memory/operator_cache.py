"""Atomic checkpoints for this benchmark's locally constructed operators.

The fingerprint includes geometry, materials, kernel sources and tolerances;
solver changes can reuse the expensive kernels. Files are local pickle caches,
not interchange artifacts, and must only be read from a trusted cache folder.
"""
import hashlib
import json
import os
import pickle
from pathlib import Path
import numpy as np
import pcbqs


def fingerprint(model,stage):
    digest=hashlib.sha256()
    for name in ('nodes','tris','z_of','t_of','node_of'):
        a=np.ascontiguousarray(getattr(model,name))
        digest.update(name.encode());digest.update(str((a.shape,a.dtype.str)).encode());digest.update(a.tobytes())
    settings=dict(stage=stage,sigma=model.sigma,plating=model.plating,conductor_loss=model.conductor_loss,
                  conductor_faces=model.conductor_faces,near=str(model.near),outer_levels=model.outer_levels,
                  compression=model._compression_options,dielectric=model.dielectric,er=model.er,
                  stack=model.board['stack'],numpy=np.__version__,vias=model.vseg)
    if model._dielectric_stack is not None:settings['dielectric_tol']=model._dielectric_stack.tol
    digest.update(json.dumps(settings,sort_keys=True).encode())
    root=Path(pcbqs.__file__).parent
    paths=['compressed.py','hmatrix.py','static_backends.py','electrostatic.py','mqs.py',
           '_numerics/base.py','_numerics/sing.py']
    paths+=sorted(str(p.relative_to(root)) for p in (root/'core').glob('*.hpp'))
    paths+=sorted(str(p.relative_to(root)) for p in (root/'core').glob('*.cpp'))
    for name in paths:digest.update(name.encode());digest.update((root/name).read_bytes())
    return digest.hexdigest()


def checkpoint(model,stage,folder,build):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    key=fingerprint(model,stage);path=folder/(stage+'-'+key+'.pickle')
    if path.exists():
        with path.open('rb') as stream:payload=pickle.load(stream)
        if payload['fingerprint']!=key:raise ValueError('operator cache fingerprint mismatch')
        if stage=='inductance':
            model._copper=payload['copper'];model._sheet_terms=payload['sheet_terms']
        else:model._potential=payload['potential']
        model.fill_info.update(payload['fill'])
        # Rebuild topology from the currently configured parts, not cache data.
        if stage=='inductance':model._assemble()
        return dict(hit=True,path=str(path),file_bytes=path.stat().st_size)
    build()
    payload=dict(fingerprint=key,fill=dict(model.fill_info))
    if stage=='inductance':payload.update(copper=model._copper,sheet_terms=model._sheet_terms)
    else:payload['potential']=model._potential
    temporary=path.with_suffix('.tmp')
    try:
        with temporary.open('wb') as stream:pickle.dump(payload,stream,protocol=5)
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(hit=False,path=str(path),file_bytes=path.stat().st_size)


def sampled_block(operator,rows,cols):
    """Extract a small H-matrix block without applying N dense unit vectors."""
    rows,cols=np.asarray(rows,int),np.asarray(cols,int)
    if len(np.unique(rows))!=len(rows) or len(np.unique(cols))!=len(cols):
        raise ValueError('sample indices must be unique')
    if (rows<0).any() or (cols<0).any() or (rows>=operator.shape[0]).any() or (cols>=operator.shape[1]).any():
        raise ValueError('sample indices out of range')
    rm=np.full(operator.shape[0],-1,int);cm=np.full(operator.shape[1],-1,int)
    rm[rows]=np.arange(len(rows));cm[cols]=np.arange(len(cols))
    out=np.zeros((len(rows),len(cols)),dtype=operator.dtype)
    for a,b,u,v,same in operator.blocks:
        ra=np.flatnonzero(rm[a]>=0);cb=np.flatnonzero(cm[b]>=0)
        if len(ra) and len(cb):
            value=u[np.ix_(ra,cb)] if v is None else u[ra]@v[:,cb]
            out[np.ix_(rm[a[ra]],cm[b[cb]])]=value
        if not same:
            rb=np.flatnonzero(rm[b]>=0);ca=np.flatnonzero(cm[a]>=0)
            if len(rb) and len(ca):
                value=u[np.ix_(ca,rb)].T if v is None else v[:,rb].T@u[ca].T
                out[np.ix_(rm[b[rb]],cm[a[ca]])]=value
    return out
