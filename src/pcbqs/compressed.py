"""On-demand inductance and potential operators for experimental PEEC.

No triangle-square moments or edge-square inductance are materialized. The
same close-pair integration and reciprocal ordering as the dense path apply.
"""
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.interpolate import CubicSpline

from .hmatrix import HMatrix
from .static_backends import FillEngine
from ._numerics.base import RWG, MU0, EPS0


def _engine(model):
    return FillEngine(model.nodes,model.tris,model.z_of,model.near,model.outer_levels,**model._fill_options)


def _moments(engine,p,q,scalar=False):
    """Canonical triangle ordering enforces reciprocity of quadrature too."""
    flip=p>q;a=np.minimum(p,q);b=np.maximum(p,q)
    result=np.empty((len(p),1 if scalar else 6))
    for start in range(0,len(p),engine.chunk):
        aa,bb=a[start:start+engine.chunk],b[start:start+engine.chunk]
        result[start:start+len(aa)]=engine.evaluate(aa,bb,engine.geometry.z[aa]-engine.geometry.z[bb],scalar_only=scalar)
    if not scalar:
        result[flip,1:5]=result[flip][:,[3,4,1,2]]
        same=p==q
        x=(result[same,1:3]+result[same,3:5])/2
        result[same,1:3]=x;result[same,3:5]=x
    return result


def _boxes(engine):
    g=engine.geometry;v=g.nodes[g.tris]
    lo=np.c_[v.min(1),g.z];hi=np.c_[v.max(1),g.z]
    return lo,hi


def edge_inductance(model):
    started=time.perf_counter();engine=_engine(model);g=engine.geometry
    r=RWG(g.nodes,g.tris);lo,hi=_boxes(engine)
    support=np.stack((r.tp,r.tm),axis=1)
    vertices=np.stack((r.vp,r.vm),axis=1)
    alpha=np.stack((1/(2*r.area[r.tp]),-1/(2*r.area[r.tm])),axis=1)

    def entry(rows,cols):
        a,b=np.meshgrid(rows,cols,indexing='ij');a=a.ravel();b=b.ravel()
        out=np.zeros(len(a))
        for i in range(2):
            for j in range(2):
                p,q=support[a,i],support[b,j];m=_moments(engine,p,q)
                vp,vq=vertices[a,i],vertices[b,j]
                val=m[:,5]-(vq*m[:,1:3]).sum(1)-(vp*m[:,3:5]).sum(1)+(vp*vq).sum(1)*m[:,0]
                out+=alpha[a,i]*alpha[b,j]*val
        return (MU0/(4*np.pi)*out).reshape(len(rows),len(cols))

    L=HMatrix(entry,lo[support].min(1),hi[support].max(1),**model._compression_options)
    model.fill_info['magnetic']=dict(engine.info,**L.info,seconds=time.perf_counter()-started,matrix_backend='hierarchical')
    model.log(f'peec: compressed inductance {L.storage_bytes/1024**2:.2f} MiB')
    return L,r


class _LayeredPairs:
    """The dense dielectric image/spline integration, evaluated at chosen pairs."""
    def __init__(self,model,engine):
        self.model,self.engine=model,engine;self.cache={}
        self.layer=np.empty(len(model.tris),int)
        self.interfaces=[]
        for k,ids in enumerate(model.layer_tris.values()):
            self.layer[ids]=k;self.interfaces.append(model._dielectric_stack.interface(model.z_of[ids[0]]))

    def images(self,a,b):
        key=(a,b)
        if key not in self.cache:
            m=self.model;g=self.engine.geometry
            images=m._dielectric_stack.images(self.interfaces[b],self.interfaces[a])
            positive=[(d,c) for d,c in images if d>0];smooth=None
            if len(positive)>8 and np.isfinite(m.near):
                rho_max=np.linalg.norm(np.ptp(g.nodes,axis=0))
                count=max(257,int(np.ceil(32*np.sqrt(rho_max/min(d for d,_ in positive)))))
                rho=rho_max*np.linspace(0,1,count)**2;kernel=np.zeros(count,complex)
                for d,c in positive:kernel+=c/np.hypot(rho,d)
                smooth=CubicSpline(rho,kernel)
            self.cache[key]=images,smooth,np.asarray([d for d,_ in images])
        return self.cache[key]

    def __call__(self,p,q):
        p,q=np.minimum(p,q),np.maximum(p,q)
        out=np.zeros(len(p),complex);g=self.engine.geometry;m=self.model
        nl=len(self.interfaces);code=self.layer[p]*nl+self.layer[q]
        for k in np.unique(code):
            ids=np.flatnonzero(code==k);images,smooth,distances=self.images(int(k)//nl,int(k)%nl)
            for start in range(0,len(ids),self.engine.chunk):
                pick=ids[start:start+self.engine.chunk];a,b=p[pick],q[pick]
                rho2=((g.cent[a]-g.cent[b])**2).sum(1)
                delta=g.points[a,:,None]-g.points[b,None,:]
                quad=(delta*delta).sum(-1);area=g.area[a]*g.area[b]
                value=np.zeros(len(a),complex)
                if smooth is not None:value+=area*smooth(np.sqrt(quad)).mean((1,2))
                # The spline already includes all positive images. Those
                # beyond every pair's near radius cannot need a correction.
                limit=m.near*np.max(g.radius[a]+g.radius[b])
                active=images[:np.searchsorted(distances,limit,side='right')] if smooth is not None else images
                for d,c in active:
                    dist=np.sqrt(rho2+d*d);size=g.radius[a]+g.radius[b]
                    close=dist<=m.near*size
                    if smooth is not None and d>0:
                        if not close.any():continue
                        value[close]-=c*area[close]*(1/np.sqrt(quad[close]+d*d)).mean((1,2))
                    refine=close & (dist<=size)
                    for mask,level in ((close & ~refine,0),(refine,m.outer_levels)):
                        if mask.any():
                            value[mask]+=c*self.engine.evaluate(a[mask],b[mask],d,np.full(mask.sum(),level,np.int32),scalar_only=True)[:,0]
                    far=~close if d==0 or smooth is None else np.zeros(len(a),bool)
                    if far.any():value[far]+=c*self.engine.evaluate(a[far],b[far],d,np.full(far.sum(),-1,np.int32),scalar_only=True)[:,0]
                out[pick]=value
        return out


def potential(model):
    started=time.perf_counter();engine=_engine(model);g=engine.geometry
    pairs=_LayeredPairs(model,engine) if model.dielectric=='stack' else None

    def entry(rows,cols):
        p,q=np.meshgrid(rows,cols,indexing='ij');p,q=p.ravel(),q.ravel()
        vals=pairs(p,q) if pairs is not None else _moments(engine,p,q,True)[:,0]/model.er
        vals=vals/(g.area[p]*g.area[q]*4*np.pi*EPS0)
        return vals.reshape(len(rows),len(cols))

    lo,hi=_boxes(engine)
    H=HMatrix(entry,lo,hi,dtype=complex if pairs is not None else float,**model._compression_options)
    area=np.bincount(model.node_of,weights=g.area,minlength=model.n_nodes)
    W=coo_matrix((g.area/area[model.node_of],(np.arange(len(g.area)),model.node_of)),shape=(len(g.area),model.n_nodes)).tocsr()
    def apply(x):return W.T@(H@(W@x))
    P=LinearOperator((model.n_nodes,model.n_nodes),matvec=apply,matmat=apply,dtype=H.dtype)
    # Exact grouped-contact self terms, without computing a node-square array.
    diagonal=np.zeros(model.n_nodes,dtype=H.dtype)
    order=np.argsort(model.node_of,kind='stable');offset=np.r_[0,np.cumsum(np.bincount(model.node_of))]
    for node in range(model.n_nodes):
        ids=order[offset[node]:offset[node+1]]
        if len(ids)==1:diagonal[node]=H.diagonal[ids[0]]
        else:
            weights=g.area[ids]/area[node];total=0.
            for start in range(0,len(ids),64):
                for stop in range(0,len(ids),64):
                    a,b=ids[start:start+64],ids[stop:stop+64]
                    total+=weights[start:start+len(a)]@entry(a,b)@weights[stop:stop+len(b)]
            diagonal[node]=total
    P.diagonal=diagonal;P.kernel=H
    P.storage_bytes=H.storage_bytes+W.data.nbytes+W.indices.nbytes+W.indptr.nbytes+diagonal.nbytes
    model.fill_info['capacitance']=dict(engine.info,**H.info,seconds=time.perf_counter()-started,matrix_backend='hierarchical')
    model.fill_info['capacitance']['bytes']=P.storage_bytes
    model.log(f'peec: compressed potential {P.storage_bytes/1024**2:.2f} MiB')
    return P
