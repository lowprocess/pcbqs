"""Experimental compressed PEEC with implicit low-frequency loop/tree scaling.

Uses the dense reference's equilibrium-charge split, but never forms R/L/P
projections, PT, electrostatic or frequency matrices densely. A sparse local
impedance approximation preconditions restarted GMRES. Nonconvergence raises.
"""
import inspect
import numpy as np
from scipy.sparse import coo_matrix, hstack, eye, diags, block_diag
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, gmres, splu, spilu

# Conservative upper bound for assembling a projected sparse surrogate. Large
# cases use an original-graph nodal preconditioner, avoiding cycle-pair fill.
_PROJECTED_PRECONDITIONER_BYTES=128*1024**2


def _tree_preconditioner(m,diagonal,zscale):
    """Bounded-memory local-potential/tree and loop-diagonal surrogate.

    Inverting a branch-space Woodbury approximation loses tree-charge accuracy
    through current cancellation at low frequency. Solve in charge coordinates
    instead, retaining the equilibrium constraint in every passive component.
    """
    K,T,B=m['K'],m['T'],m['B'];nl=m['nl'];keep=m['keep']
    loop=np.asarray(K[:,:nl].power(2).T@diagonal).ravel()/zscale
    loop[abs(loop)<np.finfo(float).tiny]=1
    inverse=1/m['P'].diagonal
    total=np.asarray(B.T@inverse).ravel()
    def apply(y):
        out=np.empty(K.shape[1],complex);out[:nl]=y[:nl]/loop
        if T.shape[1]:
            v=np.zeros(T.shape[0],complex)
            v[keep]=m['tree_lu'].solve(y[nl:],trans='T')
            common=np.asarray(B.T@(inverse*v)).ravel()/total
            charge=inverse*(v-B@common)
            out[nl:]=m['pscale']*m['tree_lu'].solve(charge[keep])
        return out
    size=loop.nbytes+inverse.nbytes+total.nbytes
    return LinearOperator((K.shape[1],K.shape[1]),matvec=apply,dtype=complex),size


def _gmres(operator,rhs,preconditioner,tol,maxiter,history):
    # scipy 1.10/1.11 use tol; 1.12+ accept rtol.
    key='rtol' if 'rtol' in inspect.signature(gmres).parameters else 'tol'
    x,status=gmres(operator,rhs,M=preconditioner,atol=0.,restart=min(100,len(rhs)),
                   maxiter=maxiter,callback=lambda r:history.append(float(r)),callback_type='legacy',**{key:tol})
    residual=np.linalg.norm(operator@x-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny)
    if status or not np.isfinite(residual) or residual>max(tol*5,1e-13):
        raise RuntimeError(f'compressed PEEC GMRES failed: status={status}, relative residual={residual:.3g}, iterations={len(history)}')
    return x,residual


def potential_solve(model,rhs):
    P=model._ensure_potential()
    diagonal=np.maximum(abs(P.diagonal),np.finfo(float).tiny)
    M=LinearOperator(P.shape,matvec=lambda x:x/diagonal,dtype=complex)
    rhs=np.asarray(rhs);one=rhs.ndim==1
    columns=rhs[:,None] if one else rhs
    answer=np.empty(columns.shape,complex)
    for k in range(columns.shape[1]):
        answer[:,k],_=_gmres(P,columns[:,k],M,model.solver_tol,model.solver_maxiter,[])
    return answer[:,0] if one else answer


def prepare(model):
    if model._peec is not None:return model._peec
    m=model._assemble();P=model._ensure_potential()
    ends=model.br+[(model.term_node[p['a']],model.term_node[p['b']]) for p in model.parts]
    nb=len(ends);a,b=np.asarray(ends,int).reshape(-1,2).T;branch=np.arange(nb)
    A=coo_matrix((np.r_[np.ones(nb),-np.ones(nb)],(np.r_[a,b],np.r_[branch,branch])),shape=(model.n_nodes,nb)).tocsc()
    sources=np.flatnonzero(m['is_src']);passive=np.flatnonzero(~m['is_src'])
    cycles=[i for i in range(m['C'].shape[1]) if i not in m['src'].values()]
    K=hstack((m['C'][:,cycles],eye(nb,format='csc')[:,np.flatnonzero(m['tree'])]),format='csc')
    nl=len(cycles);T=(A@K[:,nl:]).tocsc()
    graph=coo_matrix((np.ones(len(passive)),(a[passive],b[passive])),shape=(model.n_nodes,model.n_nodes))
    nc,labels=connected_components(graph,directed=False)
    # Sparse component indicators avoid a potentially node-square identity.
    B=coo_matrix((np.ones(model.n_nodes),(np.arange(model.n_nodes),labels)),shape=(model.n_nodes,nc)).tocsc()
    roots=np.unique(labels,return_index=True)[1];keep=np.ones(model.n_nodes,bool);keep[roots]=False
    tree_lu=splu(T[keep].astype(complex)) if T.shape[1] else None
    names=[model.parts[k-len(model.br)]['name'] for k in sources]
    pscale=max(float(np.max(abs(P.diagonal))),np.finfo(float).tiny)
    # Approximate only for preconditioning; sparse R and skin corrections stay
    # exact in operator products. No dense projection is used for the solution.
    Le,Lv=m['L'];vdiag=Lv.diagonal() if callable(Lv.diagonal) else Lv.diagonal
    ldiag=np.r_[Le.diagonal,vdiag,np.zeros(len(model.parts))]
    model._peec=dict(A=A,K=K,T=T,P=P,nl=nl,B=B,keep=keep,tree_lu=tree_lu,
                     sources=sources,passive=passive,names=names,pscale=pscale,
                     source_components=(A[:,sources].T@B).toarray(),ldiag=ldiag,
                     backend='hierarchical',factorizations=0,factor_bytes=0)
    return model._peec


def solve(model,freqs,amps=None):
    from .mqs import Result
    f=np.atleast_1d(np.asarray(freqs,float))
    if f.ndim!=1 or not np.isfinite(f).all() or (f<=0).any() or not len(f):
        raise ValueError('frequencies must be finite and positive')
    m=prepare(model);A,K,T,P,B=[m[k] for k in ('A','K','T','P','B')]
    nl=m['nl'];sources=m['sources'];names=m['names'];nb=A.shape[1];n=K.shape[1]
    excitation=np.array([(amps or {}).get(name,1.) for name in names],complex)
    drive=np.asarray(A[:,sources]@excitation).ravel();totals=m['source_components'].T@excitation
    qpart=np.zeros(model.n_nodes,complex);vpart=np.zeros(B.shape[1],complex)
    if np.any(totals):
        if 'PinvB' not in m:
            m['PinvB']=potential_solve(model,B.toarray())
            m['component']=B.T@m['PinvB']
        vpart=np.linalg.solve(m['component'],-totals);qpart=m['PinvB']@vpart
    y0=m['tree_lu'].solve((-drive-qpart)[m['keep']]) if m['tree_lu'] is not None else np.zeros(0,complex)
    baseline=K[:,nl:]@y0;source_voltage=-m['source_components']@vpart
    currents=np.zeros((len(f),nb),complex);charge=np.zeros((len(f),model.n_nodes),complex)
    potential=np.zeros_like(charge);z={name:np.zeros(len(f),complex) for name in names}
    model.solve_info=[]
    for fi,freq in enumerate(f):
        jw=2j*np.pi*freq
        zp=np.zeros(nb,complex)
        for k,part in enumerate(model.parts,len(model.br)):
            if part['kind']!='I':
                zp[k]=part['R']+jw*part['L']+(1/(jw*part['C']) if part.get('C') else 0)
        resistance=model._mats['R']+model._resistance_delta(freq)
        def impedance(x):return resistance@x+jw*model._inductive_voltage(x)+zp*x
        diagonal=resistance.diagonal()+jw*m['ldiag']+zp
        zscale=max(float(np.max(abs(diagonal))),np.finfo(float).tiny);alpha=zscale/m['pscale']
        def product(x):
            branch=K[:,:nl]@x[:nl]+jw*alpha*(K[:,nl:]@x[nl:])
            out=np.asarray(K.T@impedance(branch)).ravel()/zscale
            out[nl:]+=np.asarray(T.T@(P@(T@x[nl:]))).ravel()/m['pscale']
            return out
        rhs=-np.asarray(K.T@impedance(baseline)).ravel()/zscale
        history=[];preconditioner_bytes=0
        kind='none'
        if n:
            # A sparse surrogate in the same scaled coordinates. Fundamental
            # cycle fill remains a limitation on very large meshes.
            counts=np.diff(K.tocsr().indptr).astype(float)
            electric_counts=np.diff(T.tocsr().indptr).astype(float)
            estimated_bytes=min(n*n,float(counts@counts+electric_counts@electric_counts))*24
            if estimated_bytes>_PROJECTED_PRECONDITIONER_BYTES:
                M,preconditioner_bytes=_tree_preconditioner(m,diagonal,zscale)
                kind='tree_diagonal'
            else:
                approx=(K.T@diags(diagonal)@K).tocsc()/zscale
                scaling=np.r_[np.ones(nl),np.full(n-nl,jw*alpha)]
                approx=approx@diags(scaling)
                electric=(T.T@diags(P.diagonal)@T).tocsc()/m['pscale']
                approx=approx+block_diag((coo_matrix((nl,nl)),electric),format='csc')
                try:
                    lu=spilu(approx.tocsc(),drop_tol=1e-4,fill_factor=10)
                    M=LinearOperator((n,n),matvec=lu.solve,dtype=complex)
                    preconditioner_bytes=sum(mat.data.nbytes+mat.indices.nbytes+mat.indptr.nbytes for mat in (lu.L,lu.U))
                    kind='projected_ilu'
                except RuntimeError:
                    d=approx.diagonal();d[abs(d)<np.finfo(float).tiny]=1
                    M=LinearOperator((n,n),matvec=lambda x:x/d,dtype=complex)
                    kind='diagonal'
            op=LinearOperator((n,n),matvec=product,dtype=complex)
            x,residual=_gmres(op,rhs,M,model.solver_tol,model.solver_maxiter,history)
            correction=alpha*x[nl:]
            currents[fi]=baseline+K[:,:nl]@x[:nl]+jw*(K[:,nl:]@correction)
            charge[fi]=qpart/jw-T@correction
            pcorrection=P@(T@correction)
            potential[fi]=np.asarray(B@vpart).ravel()/jw-pcorrection
            voltage=source_voltage/jw+A[:,sources].T@pcorrection
        else:
            residual=0.;charge[fi]=qpart/jw;potential[fi]=np.asarray(B@vpart).ravel()/jw;voltage=source_voltage/jw
        currents[fi,sources]=excitation
        for k,name in enumerate(names):z[name][fi]=voltage[k]/excitation[k] if excitation[k] else np.nan
        continuity=A@currents[fi]+jw*charge[fi]
        model.solve_info.append(dict(frequency_Hz=float(freq),iterations=len(history),relative_residual=float(residual),
                                     continuity_residual_A=float(np.max(abs(continuity))),preconditioner_bytes=preconditioner_bytes,
                                     preconditioner=kind))
    return Result(f,currents,z,[p['name'] for p in model.parts],charge=charge,potential=potential)
