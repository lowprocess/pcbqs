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
