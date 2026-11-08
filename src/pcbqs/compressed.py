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
    n=len(engine.geometry.tris)
    keys,inverse=np.unique(a.astype(np.int64)*n+b,return_inverse=True)
    unique=np.empty((len(keys),1 if scalar else 6))
    for start in range(0,len(keys),engine.chunk):
        key=keys[start:start+engine.chunk];aa,bb=key//n,key%n
        unique[start:start+len(key)]=engine.evaluate(aa,bb,engine.geometry.z[aa]-engine.geometry.z[bb],scalar_only=scalar)
    result=unique[inverse]
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
        # Evaluate the four support combinations together, deduplicating the
        # triangle pairs shared by neighboring RWGs. Scratch stays tile-sized.
        side=max(1,engine.chunk//4)
        for start in range(0,len(a),side):
            aa,bb=a[start:start+side],b[start:start+side]
            p=np.broadcast_to(support[aa,:,None],(len(aa),2,2)).ravel()
            q=np.broadcast_to(support[bb,None,:],(len(bb),2,2)).ravel()
            m=_moments(engine,p,q).reshape(len(aa),2,2,6)
            vp,vq=vertices[aa,:,None,:],vertices[bb,None,:,:]
            val=m[...,5]-(vq*m[...,1:3]).sum(-1)-(vp*m[...,3:5]).sum(-1)+(vp*vq).sum(-1)*m[...,0]
            out[start:start+len(aa)]=(alpha[aa,:,None]*alpha[bb,None,:]*val).sum((1,2))
        return (MU0/(4*np.pi)*out).reshape(len(rows),len(cols))

    L=HMatrix(entry,lo[support].min(1),hi[support].max(1),log=model.log,**model._compression_options)
    model.fill_info['magnetic']=dict(engine.info,**L.info,seconds=time.perf_counter()-started,matrix_backend='hierarchical')
    model.log(f'peec: compressed inductance {L.storage_bytes/1024**2:.2f} MiB')
    return L,r


class _LayeredPairs:
    """The dense dielectric image/spline integration, evaluated at chosen pairs."""
    def __init__(self,model,engine):
        self.model,self.engine=model,engine;self.cache={}
        self.info=dict(pair_requests=0,image_interactions=0,integration_calls=0)
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
        self.info['pair_requests']+=len(p)
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
                size=g.radius[a]+g.radius[b]
                coefficients=np.asarray([c for _,c in images])
                if smooth is not None:
                    # All positive far images are already in the spline.
                    zero=distances==0
                    if zero.any():
                        base=self.engine.evaluate(a,b,0.,scalar_only=True)[:,0]
                        self.info['integration_calls']+=1;self.info['image_interactions']+=len(a)
                        value+=coefficients[zero].sum()*base
                    ds,cs=distances[~zero],coefficients[~zero]
                    threshold2=(m.near*size)**2-rho2
                    counts=np.searchsorted(ds,np.sqrt(np.maximum(threshold2,0)),side='right')
                    counts[threshold2<0]=0
                else:
                    ds,cs=distances,coefficients
                    counts=np.full(len(a),len(ds),int)
                # Flatten only the required (panel pair, image) interactions.
                # One integration call handles many offsets and layer images.
                offsets=np.r_[0,np.cumsum(counts)];begin=0
                while begin<len(a):
                    end=max(begin+1,int(np.searchsorted(offsets,offsets[begin]+self.engine.chunk,side='right')-1))
                    end=min(end,len(a))
                    ii=np.repeat(np.arange(begin,end),counts[begin:end])
                    jj=np.arange(offsets[end]-offsets[begin])-np.repeat(offsets[begin:end]-offsets[begin],counts[begin:end])
                    for first in range(0,len(ii),self.engine.chunk):
                        ix,jx=ii[first:first+self.engine.chunk],jj[first:first+self.engine.chunk]
                        d=ds[jx];dist=np.sqrt(rho2[ix]+d*d)
                        kind=np.full(len(ix),-1,np.int32)
                        close=dist<=m.near*size[ix];kind[close]=0
                        kind[close & (dist<=size[ix])]=m.outer_levels
                        exact=self.engine.evaluate(a[ix],b[ix],d,kind,scalar_only=True)[:,0]
                        self.info['integration_calls']+=1;self.info['image_interactions']+=len(ix)
                        if smooth is not None:
                            exact-=area[ix]*(1/np.sqrt(quad[ix]+d[:,None,None]**2)).mean((1,2))
                        np.add.at(value,ix,cs[jx]*exact)
                    begin=end
                out[pick]=value
        return out


class PotentialOperator(LinearOperator):
    """Serializable sparse aggregation around a reciprocal panel kernel."""
    def __init__(self,kernel,weights,diagonal):
        self.kernel,self.weights,self.diagonal=kernel,weights,diagonal
        n=weights.shape[1]
        super().__init__(kernel.dtype,(n,n))
        self.storage_bytes=kernel.storage_bytes+weights.data.nbytes+weights.indices.nbytes+weights.indptr.nbytes+diagonal.nbytes

    def _matvec(self,x):return self.weights.T@(self.kernel@(self.weights@x))
    def _matmat(self,x):return self._matvec(x)
    def _rmatvec(self,x):return self._matvec(x.conj()).conj()


def potential(model):
    started=time.perf_counter();engine=_engine(model);g=engine.geometry
    pairs=_LayeredPairs(model,engine) if model.dielectric=='stack' else None

    def entry(rows,cols):
        p,q=np.meshgrid(rows,cols,indexing='ij');p,q=p.ravel(),q.ravel()
        vals=pairs(p,q) if pairs is not None else _moments(engine,p,q,True)[:,0]/model.er
        vals=vals/(g.area[p]*g.area[q]*4*np.pi*EPS0)
        return vals.reshape(len(rows),len(cols))

    lo,hi=_boxes(engine)
    H=HMatrix(entry,lo,hi,dtype=complex if pairs is not None else float,log=model.log,**model._compression_options)
    area=np.bincount(model.node_of,weights=g.area,minlength=model.n_nodes)
    W=coo_matrix((g.area/area[model.node_of],(np.arange(len(g.area)),model.node_of)),shape=(len(g.area),model.n_nodes)).tocsr()
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
    P=PotentialOperator(H,W,diagonal)
    model.fill_info['capacitance']=dict(engine.info,**H.info,seconds=time.perf_counter()-started,matrix_backend='hierarchical')
    model.fill_info['capacitance']['bytes']=P.storage_bytes
    if pairs is not None:model.fill_info['capacitance']['image_fill']=pairs.info
    model.log(f'peec: compressed potential {P.storage_bytes/1024**2:.2f} MiB')
    return P
