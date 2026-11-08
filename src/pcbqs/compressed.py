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
from .static_backends import FillEngine, layered_spline_args, native_layered
from ._numerics.base import RWG, MU0, EPS0


def _engine(model):
    return FillEngine(model.nodes,model.tris,model.z_of,model.near,model.outer_levels,**model._fill_options)


def _native(engine):
    """OpenMP-backed engines also assemble whole entry blocks in the compiled core.

    Returns (core, args) for static_block/rwg_inductance, or None to use the
    NumPy assembly (other fill backends, or a core that predates the blocks).
    """
    backend=engine.backend
    core=getattr(backend,'core',None)
    if engine.info['backend']!='openmp' or core is None or not hasattr(core,'rwg_inductance'):
        return None
    g=engine.geometry
    near=float('inf') if np.isinf(engine.near) else float(engine.near)
    return core,(backend.data,g.z,g.cent,g.radius,near,int(engine.outer_levels)),backend.workers


def _moments(engine,p,q,scalar=False):
    """Canonical triangle ordering enforces reciprocity of quadrature too."""
    flip=p>q;a=np.minimum(p,q);b=np.maximum(p,q)
    n=len(engine.geometry.tris)
    keys,inverse=np.unique(a.astype(np.int64)*n+b,return_inverse=True)
    unique=np.empty((len(keys),1 if scalar else 6))
    for start in range(0,len(keys),engine.pair_batch):
        key=keys[start:start+engine.pair_batch];aa,bb=key//n,key%n
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
    native=_native(engine)
    if native is not None:
        core,common,workers=native
        support64=np.ascontiguousarray(support,np.int64)
        vertices=np.ascontiguousarray(vertices,float);alpha=np.ascontiguousarray(alpha,float)

        provider=core.RwgEntries(*common,support64,vertices,alpha,MU0/(4*np.pi)) if hasattr(core,'RwgEntries') else None

        def entry(rows,cols):
            return core.rwg_inductance(*common,support64,vertices,alpha,np.asarray(rows,np.int64),
                                       np.asarray(cols,np.int64),MU0/(4*np.pi),workers)
        return _compressed_inductance(model,engine,entry,lo,hi,support,started,r,provider,engine.info['workers'])

    def entry(rows,cols):
        a,b=np.meshgrid(rows,cols,indexing='ij');a=a.ravel();b=b.ravel()
        out=np.zeros(len(a))
        # Evaluate the four support combinations together, deduplicating the
        # triangle pairs shared by neighboring RWGs. Scratch stays tile-sized.
        side=max(1,engine.pair_batch//4)
        for start in range(0,len(a),side):
            aa,bb=a[start:start+side],b[start:start+side]
            p=np.broadcast_to(support[aa,:,None],(len(aa),2,2)).ravel()
            q=np.broadcast_to(support[bb,None,:],(len(bb),2,2)).ravel()
            m=_moments(engine,p,q).reshape(len(aa),2,2,6)
            vp,vq=vertices[aa,:,None,:],vertices[bb,None,:,:]
            val=m[...,5]-(vq*m[...,1:3]).sum(-1)-(vp*m[...,3:5]).sum(-1)+(vp*vq).sum(-1)*m[...,0]
            out[start:start+len(aa)]=(alpha[aa,:,None]*alpha[bb,None,:]*val).sum((1,2))
        return (MU0/(4*np.pi)*out).reshape(len(rows),len(cols))

    return _compressed_inductance(model,engine,entry,lo,hi,support,started,r)


def _compressed_inductance(model,engine,entry,lo,hi,support,started,r,provider=None,workers=None):
    L=HMatrix(entry,lo[support].min(1),hi[support].max(1),log=model.log,native=provider,workers=workers,
              **model._compression_options)
    model.fill_info['magnetic']=dict(engine.info,**L.info,seconds=time.perf_counter()-started,matrix_backend='hierarchical',
                                     native_entries=_native(engine) is not None)
    model.log(f'peec: compressed inductance {L.storage_bytes/1024**2:.2f} MiB')
    return L,r


class _LayeredPairs:
    """The dense dielectric image/spline integration, evaluated at chosen pairs."""
    def __init__(self,model,engine):
        self.model,self.engine=model,engine;self.cache={}
        self.native=native_layered(engine)
        self.info=dict(pair_requests=0,image_interactions=0,integration_calls=0,native=self.native is not None)
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
            self.cache[key]=images,smooth,np.asarray([d for d,_ in images]),layered_spline_args(images,smooth)
        return self.cache[key]

    def __call__(self,p,q):
        self.info['pair_requests']+=len(p)
        p,q=np.minimum(p,q),np.maximum(p,q)
        out=np.zeros(len(p),complex);g=self.engine.geometry;m=self.model
        nl=len(self.interfaces);code=self.layer[p]*nl+self.layer[q]
        for k in np.unique(code):
            ids=np.flatnonzero(code==k);images,smooth,distances,args=self.images(int(k)//nl,int(k)%nl)
            if self.native is not None:
                # One compiled call integrates every image of every pair here.
                core,common,workers=self.native
                out[ids]=core.layered_pairs(*common,np.ascontiguousarray(p[ids],np.int64),
                                            np.ascontiguousarray(q[ids],np.int64),*args,workers)
                self.info['integration_calls']+=1
                continue
            for start in range(0,len(ids),self.engine.pair_batch):
                pick=ids[start:start+self.engine.pair_batch];a,b=p[pick],q[pick]
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
                    end=max(begin+1,int(np.searchsorted(offsets,offsets[begin]+self.engine.pair_batch,side='right')-1))
                    end=min(end,len(a))
                    ii=np.repeat(np.arange(begin,end),counts[begin:end])
                    jj=np.arange(offsets[end]-offsets[begin])-np.repeat(offsets[begin:end]-offsets[begin],counts[begin:end])
                    for first in range(0,len(ii),self.engine.pair_batch):
                        ix,jx=ii[first:first+self.engine.pair_batch],jj[first:first+self.engine.pair_batch]
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
    native=_native(engine) if pairs is None else None

    def entry(rows,cols):
        if native is not None:
            core,common,workers=native
            rows,cols=np.asarray(rows,np.int64),np.asarray(cols,np.int64)
            vals=core.static_block(*common,rows,cols,True,workers)[:,:,0]/model.er
            return vals/(g.area[rows][:,None]*g.area[cols][None,:]*4*np.pi*EPS0)
        p,q=np.meshgrid(rows,cols,indexing='ij');p,q=p.ravel(),q.ravel()
        vals=pairs(p,q) if pairs is not None else _moments(engine,p,q,True)[:,0]/model.er
        vals=vals/(g.area[p]*g.area[q]*4*np.pi*EPS0)
        return vals.reshape(len(rows),len(cols))

    provider=None
    if native is not None and hasattr(native[0],'PotentialEntries'):
        core,common,_=native
        provider=core.PotentialEntries(*common,float(model.er),np.pi,EPS0)
    elif pairs is not None and pairs.native is not None and hasattr(pairs.native[0],'LayeredEntries'):
        core,common,_=pairs.native;nl=len(pairs.interfaces)
        provider=core.LayeredEntries(*common,np.ascontiguousarray(pairs.layer,np.int64),
                                     [pairs.images(a,b)[3] for a in range(nl) for b in range(nl)],np.pi,EPS0)
    lo,hi=_boxes(engine)
    H=HMatrix(entry,lo,hi,dtype=complex if pairs is not None else float,log=model.log,native=provider,
              workers=engine.info['workers'],**model._compression_options)
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
