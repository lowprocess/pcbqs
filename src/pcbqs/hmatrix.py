"""Reciprocal geometric H-matrix from entries, without a global dense array.

Near blocks are exact. Admissible far blocks use checked partial-pivot ACA;
failed or uneconomical approximations are subdivided, never silently truncated.
Transpose symmetry (not Hermitian symmetry) also applies to lossy kernels.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator


@dataclass
class _Cluster:
    ids: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    children: tuple = ()


class HMatrix(LinearOperator):
    def __init__(self, entry, lo, hi, *, dtype=float, tol=1e-6, leaf=64, eta=1.,log=None):
        lo,hi=np.asarray(lo,float),np.asarray(hi,float)
        if lo.shape != hi.shape or lo.ndim != 2 or not np.isfinite([lo,hi]).all() or (lo>hi).any():
            raise ValueError('finite support bounding boxes required')
        if not np.isfinite(tol) or not 0 < tol < 1 or not np.isfinite(eta) or eta <= 0:
            raise ValueError('invalid compression tolerance or admissibility')
        if not isinstance(leaf,(int,np.integer)) or leaf < 1:
            raise ValueError('leaf must be a positive integer')
        super().__init__(np.dtype(dtype),(len(lo),len(lo)))
        self.blocks=[];self.entries=0;self.diagonal=np.zeros(len(lo),dtype=dtype)
        rng=np.random.default_rng(0)
        next_log=time.monotonic()+15;payload=0;row_hits=0

        def save(a,b,u,v,same):
            nonlocal next_log,payload
            self.blocks.append((a,b,u,v,same))
            payload+=a.nbytes+b.nbytes+u.nbytes+(v.nbytes if v is not None else 0)
            if log is not None and time.monotonic()>=next_log:
                log(f'hmatrix: {len(self.blocks):,} blocks, {payload/1024**2:.1f} MiB, {self.entries:,} entries')
                next_log=time.monotonic()+15

        def tree(ids):
            node=_Cluster(ids,lo[ids].min(0),hi[ids].max(0))
            if len(ids)>leaf:
                centers=(lo[ids]+hi[ids])/2;axis=np.argmax(np.ptp(centers,axis=0))
                order=ids[np.argsort(centers[:,axis],kind='stable')];mid=len(ids)//2
                node.children=(tree(order[:mid]),tree(order[mid:]))
            return node

        def evaluate(rows,cols):
            self.entries+=len(rows)*len(cols)
            values=np.asarray(entry(rows,cols),dtype=dtype)
            if values.shape != (len(rows),len(cols)) or not np.isfinite(values).all():
                raise ValueError('invalid H-matrix entry block')
            return values

        def aca(a,b):
            nonlocal row_hits
            m,n=len(a),len(b)
            cap=min(128,int(.7*m*n/(m+n)))
            if cap < 1:return None
            U=np.empty((m,0),dtype=dtype);V=np.empty((0,n),dtype=dtype)
            used=np.zeros(m,bool);i=0
            cache={}
            def rows(indices):
                nonlocal row_hits
                missing=np.array([int(k) for k in indices if int(k) not in cache],int)
                row_hits+=len(indices)-len(missing)
                if len(missing):
                    values=evaluate(a[missing],b)
                    for k,row in zip(missing,values):cache[int(k)]=row
                return np.asarray([cache[int(k)] for k in indices])
            # Checking rows independently of the pivots catches premature ACA
            # convergence. This is an estimator, not a certified operator bound.
            check=rng.choice(m,min(m,16),replace=False)
            reference=rows(check)
            for _ in range(cap):
                row=rows(np.array([i]))[0]-U[i]@V
                j=int(np.argmax(abs(row)));pivot=row[j]
                if abs(pivot)>np.finfo(float).tiny:
                    col=np.empty(m,dtype=dtype)
                    known=np.array(list(cache),int);unknown=np.ones(m,bool);unknown[known]=False
                    col[known]=[cache[int(k)][j] for k in known]
                    col[unknown]=evaluate(a[unknown],b[j:j+1])[:,0]
                    col-=U@V[:,j]
                    U=np.column_stack((U,col));V=np.vstack((V,row/pivot))
                used[i]=True
                residual=reference-U[check]@V
                scale=max(np.linalg.norm(reference),np.finfo(float).tiny)
                if np.linalg.norm(residual)<=tol*scale:
                    # A second independent set is checked before accepting.
                    extra=rng.choice(m,min(m,16),replace=False)
                    ref=rows(extra);err=ref-U[extra]@V
                    if np.linalg.norm(err)<=tol*max(np.linalg.norm(ref),np.finfo(float).tiny):
                        return U,V
                    check=np.r_[check,extra];reference=np.vstack((reference,ref))
                    residual=reference-U[check]@V
                norms=np.linalg.norm(residual,axis=1)
                order=np.argsort(norms)[::-1]
                available=[int(check[k]) for k in order if not used[check[k]]]
                if available:i=available[0]
                elif np.any(~used):i=int(np.flatnonzero(~used)[np.argmax(abs(U[~used,-1]))]) if U.shape[1] else int(np.flatnonzero(~used)[0])
                else:break
            return None

        def build(a,b):
            same=a is b
            gap=np.linalg.norm(np.maximum(0,np.maximum(a.lo-b.hi,b.lo-a.hi)))
            diameter=max(np.linalg.norm(a.hi-a.lo),np.linalg.norm(b.hi-b.lo))
            if not same and gap>eta*diameter:
                factors=aca(a.ids,b.ids)
                if factors is not None:
                    save(a.ids,b.ids,*factors,False);return
            if not a.children and not b.children:
                block=evaluate(a.ids,b.ids)
                if same:
                    block=(block+block.T)/2;self.diagonal[a.ids]=np.diag(block)
                save(a.ids,b.ids,block,None,same);return
            if same:
                x,y=a.children;build(x,x);build(x,y);build(y,y)
            elif a.children and (not b.children or len(a.ids)>=len(b.ids)):
                for child in a.children:build(child,b)
            else:
                for child in b.children:build(a,child)

        if len(lo):
            root=tree(np.arange(len(lo)));build(root,root)
        self.storage_bytes=sum(a.nbytes+b.nbytes+u.nbytes+(v.nbytes if v is not None else 0)
                               for a,b,u,v,_ in self.blocks)+self.diagonal.nbytes
        self.info=dict(bytes=self.storage_bytes,dense_bytes=len(lo)**2*self.dtype.itemsize,
                       blocks=len(self.blocks),low_rank_blocks=sum(v is not None for _,_,_,v,_ in self.blocks),
                       evaluated_entries=self.entries,cached_rows=row_hits,tolerance=tol,leaf=leaf,eta=eta)

    def _matmat(self,x):
        out=np.zeros((self.shape[0],x.shape[1]),dtype=np.result_type(self.dtype,x.dtype))
        for a,b,u,v,same in self.blocks:
            out[a]+=u@x[b] if v is None else u@(v@x[b])
            if not same:out[b]+=u.T@x[a] if v is None else v.T@(u.T@x[a])
        return out

    def _matvec(self,x):return self._matmat(np.asarray(x).reshape(-1,1))[:,0]
    def _rmatvec(self,x):return self._matvec(x.conj()).conj()
