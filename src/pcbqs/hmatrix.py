"""Reciprocal geometric H-matrix from entries, without a global dense array.

Near blocks are exact. Admissible far blocks use checked partial-pivot ACA;
failed or uneconomical approximations are subdivided, never silently truncated.
Transpose symmetry (not Hermitian symmetry) also applies to lossy kernels.
"""
from dataclasses import dataclass
from importlib import import_module
import os
import threading
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator


@dataclass
class _Cluster:
    ids: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    children: tuple = ()


def _native():
    """The compiled block product, unless PCBQS_NO_CORE is set or the core predates it."""
    if os.environ.get('PCBQS_NO_CORE'):
        return None
    try:
        core=import_module('pcbqs._core')
    except ImportError:
        return None
    return core if hasattr(core,'HBlocks') else None


def _trim():
    try:
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError,AttributeError):
        pass


def _build_core():
    """The compiled ACA/dense-leaf builder, unless PCBQS_NO_CORE is set or the core predates it."""
    core=_native()
    return core if core is not None and hasattr(core,'aca_blocks') else None


def _workers():
    env=os.environ.get('PCBQS_HMATRIX_THREADS')
    if env:
        value=int(env)
        if value<1:raise ValueError('PCBQS_HMATRIX_THREADS must be a positive integer')
        return value
    from .static_backends import default_workers
    return default_workers()


class HMatrix(LinearOperator):
    def __init__(self, entry, lo, hi, *, dtype=float, tol=1e-6, leaf=64, eta=1.,log=None,native=None,workers=None):
        """`native` (optional) is a compiled entry provider from pcbqs._core with the
        same entries as `entry`; ACA and dense leaves then run natively, one tree
        level at a time and in parallel over blocks (`workers` threads). Native
        ACA draws check rows from a per-block stream, so its factors are
        deterministic but differ from the NumPy path within `tol`.
        """
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

        def children(a,b):
            if a is b:
                x,y=a.children;return [(x,x),(x,y),(y,y)]
            if a.children and (not b.children or len(a.ids)>=len(b.ids)):
                return [(child,b) for child in a.children]
            return [(a,child) for child in b.children]

        def build_native(root):
            nonlocal row_hits
            count=workers or _workers();pending=[(root,root)]
            while pending:
                admissible,dense,following=[],[],[]
                for a,b in pending:
                    gap=np.linalg.norm(np.maximum(0,np.maximum(a.lo-b.hi,b.lo-a.hi)))
                    diameter=max(np.linalg.norm(a.hi-a.lo),np.linalg.norm(b.hi-b.lo))
                    if a is not b and gap>eta*diameter:admissible.append((a,b))
                    elif not a.children and not b.children:dense.append((a,b))
                    else:following.extend(children(a,b))
                if admissible:
                    factors,entries,hits=core.aca_blocks(native,[np.asarray(a.ids,np.int64) for a,_ in admissible],
                                                         [np.asarray(b.ids,np.int64) for _,b in admissible],float(tol),count)
                    self.entries+=entries;row_hits+=hits
                    for (a,b),f in zip(admissible,factors):
                        if f is not None:save(a.ids,b.ids,f[0],f[1],False)
                        elif not a.children and not b.children:dense.append((a,b))
                        else:following.extend(children(a,b))
                if dense:
                    blocks=core.dense_blocks(native,[np.asarray(a.ids,np.int64) for a,_ in dense],
                                             [np.asarray(b.ids,np.int64) for _,b in dense],count)
                    for (a,b),block in zip(dense,blocks):
                        self.entries+=block.size
                        block=block.astype(dtype,copy=False)
                        if a is b:
                            block=(block+block.T)/2;self.diagonal[a.ids]=np.diag(block)
                        save(a.ids,b.ids,block,None,a is b)
                pending=following

        core=_build_core() if native is not None else None
        if len(lo):
            root=tree(np.arange(len(lo)))
            if core is not None:build_native(root)
            else:build(root,root)
        self.native_build=core is not None
        self._compact()
        self.storage_bytes=sum(a.nbytes+b.nbytes+u.nbytes+(v.nbytes if v is not None else 0)
                               for a,b,u,v,_ in self.blocks)+self.diagonal.nbytes
        self.info=dict(bytes=self.storage_bytes,dense_bytes=len(lo)**2*self.dtype.itemsize,
                       blocks=len(self.blocks),low_rank_blocks=sum(v is not None for _,_,_,v,_ in self.blocks),
                       evaluated_entries=self.entries,cached_rows=row_hits,tolerance=tol,leaf=leaf,eta=eta,
                       native_build=self.native_build)

    def _matmat(self,x):
        x=np.asarray(x)
        core=_native()
        if core is not None and self.blocks:
            return self._native_matmat(core,x)
        out=np.zeros((self.shape[0],x.shape[1]),dtype=np.result_type(self.dtype,x.dtype))
        for a,b,u,v,same in self.blocks:
            out[a]+=u@x[b] if v is None else u@(v@x[b])
            if not same:out[b]+=u.T@x[a] if v is None else v.T@(u.T@x[a])
        return out

    def _pack(self):
        """Index the blocks once for the compiled product, without copying them.

        Block index vectors are gathered into one array (blocks are rebound to
        views of it); factor arrays are used in place, converted only when their
        dtype or layout differs. Contribution slots follow block order (forward
        rows, then transposed rows), and each output row sums its slots in that
        order, so native products do not depend on thread count or scheduling.
        """
        core=_native()
        complex_=self.dtype.kind=='c' or any(u.dtype.kind=='c' or (v is not None and v.dtype.kind=='c')
                                             for _,_,u,v,_ in self.blocks)
        dtype=np.complex128 if complex_ else np.float64
        nb=len(self.blocks);meta=np.zeros((nb,10),np.int64)
        idx=np.empty(sum(len(a)+len(b) for a,b,*_ in self.blocks),np.int64);targets=[];factors=[]
        io=so=0;blocks=self.blocks
        for k in range(nb):
            a,b,u,v,same=blocks[k]
            la,lb=len(a),len(b)
            idx[io:io+la]=a;idx[io+la:io+la+lb]=b
            av,bv=idx[io:io+la],idx[io+la:io+la+lb]
            u=np.ascontiguousarray(u,dtype=dtype)
            v=None if v is None else np.ascontiguousarray(v,dtype=dtype)
            rank=0 if v is None else u.shape[1]
            meta[k]=(io,la,io+la,lb,0,-1 if v is None else 0,rank,int(bool(same)),so,-1 if same else so+la)
            targets.append(av)
            so+=la
            if not same:targets.append(bv);so+=lb
            io+=la+lb
            blocks[k]=(av,bv,u,v,same);factors.append((u,v))
        targets=np.concatenate(targets) if targets else np.empty(0,np.int64)
        slots=np.argsort(targets,kind='stable').astype(np.int64)
        row_ptr=np.zeros(self.shape[0]+1,np.int64);np.cumsum(np.bincount(targets,minlength=self.shape[0]),out=row_ptr[1:])
        self._packed=dict(blocks=core.HBlocks(meta,idx,row_ptr,slots,so,self.shape[0],factors,complex_),
                          dtype=np.dtype(dtype),nslots=so,work=np.empty(0,dtype),lock=threading.Lock())
        return self._packed

    def _native_matmat(self,core,x):
        packed=getattr(self,'_packed',None) or self._pack()
        dtype=packed['dtype']
        if dtype.kind=='c':
            xx=np.ascontiguousarray(x,dtype=np.complex128)
        elif x.dtype.kind=='c':
            # A real operator acts on real and imaginary parts as independent columns.
            xx=np.ascontiguousarray(x,dtype=np.complex128).view(np.float64)
        else:
            xx=np.ascontiguousarray(x,dtype=np.float64)
        workers=getattr(self,'workers',None) or _workers()
        size=packed['nslots']*xx.shape[1]
        # Reusing scratch avoids page-faulting ~100 contribution rows per
        # unknown on every product; concurrent callers get their own buffer.
        lock=packed['lock'];owned=lock.acquire(blocking=False)
        try:
            if owned:
                if packed['work'].size<size:packed['work']=np.empty(size,dtype=dtype)
                work=packed['work']
            else:
                work=np.empty(size,dtype=dtype)
            y=packed['blocks'].apply(xx,work,workers)
        finally:
            if owned:lock.release()
        if dtype.kind!='c' and x.dtype.kind=='c':
            y=y.view(np.complex128)
        return y.astype(np.result_type(self.dtype,x.dtype),copy=False)

    def _compact(self):
        """Index blocks for the native product now, and return freed heap pages.

        Indexing copies only index vectors; any dtype/layout conversions free
        the originals, and malloc_trim returns those pages (glibc only).
        """
        if self.blocks and _native() is not None and getattr(self,'_packed',None) is None:
            self._pack();_trim()

    def __setstate__(self,state):
        super().__setstate__(state)
        self._compact()

    def __getstate__(self):
        # The packed copy is a cache; it is rebuilt from blocks after loading.
        state=super().__getstate__();state.pop('_packed',None)
        return state

    def _matvec(self,x):return self._matmat(np.asarray(x).reshape(-1,1))[:,0]
    def _rmatvec(self,x):return self._matvec(x.conj()).conj()
