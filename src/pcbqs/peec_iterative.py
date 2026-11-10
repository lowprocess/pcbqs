"""Experimental compressed PEEC with implicit low-frequency loop/tree scaling.

Uses the dense reference's equilibrium-charge split, but never forms R/L/P
projections, PT, electrostatic or frequency matrices densely. A sparse local
impedance approximation preconditions restarted GMRES. Nonconvergence raises.
"""
import inspect
from functools import partial
import numpy as np
from scipy.sparse import coo_matrix, hstack, eye, diags, block_diag
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, gmres, splu, spilu

# Conservative upper bound for assembling a projected sparse surrogate. Large
# cases use a local-potential tree preconditioner, avoiding cycle-pair fill.
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


def _graph_preconditioner(m,diagonal,zscale,resistance=None,jw=0):
    """Graph solve for loop impedance, followed by a stable charge-tree solve.

    The graph Laplacian applies the inverse of Kc.T diag(Z) Kc without forming
    it. Recover only loop currents from that operation; tree charge is solved
    separately to avoid subtracting conduction currents and dividing by jw.
    """
    K,A,T=m['K'],m['A'],m['T'];nl=m['nl'];passive=m['passive'];keep=m['keep']
    if not nl:return _tree_preconditioner(m,diagonal,zscale)
    Kc,Kt=K[:,:nl],K[:,nl:];cotree=m['cotree']
    position=np.full(A.shape[1],-1,int);position[passive]=np.arange(len(passive))
    d=diagonal[passive].copy();d[abs(d)<zscale*1e-12]=zscale*1e-12
    Ap=A[:,passive]
    # the passive branches' impedance inverse: diagonal, with each face-sheet edge pair's 2x2 block inverted
    e,g=m['pairs'] if resistance is not None else (np.zeros(0,int),np.zeros(0,int))
    dinv=1/d;rows,cols,vals=[np.arange(len(d))],[np.arange(len(d))],[dinv]
    if len(e):
        pe,pg=position[e],position[g];ok=(pe>=0)&(pg>=0);pe,pg=pe[ok],pg[ok]
        zpair=(np.asarray(resistance[e,g]).ravel()+jw*m['lpair'])[ok]
        det=d[pe]*d[pg]-zpair**2
        dinv=dinv.copy();dinv[pe]=d[pg]/det;dinv[pg]=d[pe]/det
        rows,cols,vals=[np.arange(len(d)),pe,pg],[np.arange(len(d)),pg,pe],[dinv,-zpair/det,-zpair/det]
    Dinv=coo_matrix((np.concatenate(vals),(np.concatenate(rows),np.concatenate(cols))),shape=(len(d),len(d))).tocsr()
    Zb=_branch_surrogate(m,diagonal,resistance,jw) if resistance is not None else diags(diagonal).tocsc()
    graph=(Ap@Dinv@Ap.T).tocsc()
    lu=spilu(graph[keep][:,keep].tocsc(),drop_tol=1e-6,fill_factor=20)
    # Reuse the charge-coordinate inverse, whose tree block is independent of
    # the loop-diagonal block discarded here.
    charge_inverse,base_bytes=_tree_preconditioner(m,diagonal,zscale)
    def apply(y):
        force=np.zeros(A.shape[1],complex);force[cotree]=zscale*y[:nl]
        trial=Dinv@force[passive]
        voltage=np.zeros(A.shape[0],complex)
        voltage[keep]=lu.solve(np.asarray(Ap@trial).ravel()[keep])
        loop=(trial-Dinv@(Ap.T@voltage))[position[cotree]]
        rhs=y.copy()
        rhs[nl:]-=np.asarray(Kt.T@(Zb@(Kc@loop))).ravel()/zscale
        result=charge_inverse@rhs;result[:nl]=loop
        return result
    size=base_bytes+sum(s.data.nbytes+s.indices.nbytes+s.indptr.nbytes for s in (lu.L,lu.U))
    return LinearOperator((K.shape[1],K.shape[1]),matvec=apply,dtype=complex),size


class ConvergenceError(RuntimeError):
    def __init__(self,status,residual,iterations):
        self.status,self.relative_residual,self.iterations=status,residual,iterations
        super().__init__(f'compressed PEEC GMRES failed: status={status}, relative residual={residual:.3g}, iterations={iterations}')


def _restart(model):
    value=getattr(model,'solver_restart',100)
    if not isinstance(value,(int,np.integer)) or isinstance(value,bool) or value<1:
        raise ValueError('solver_restart must be a positive integer')
    return value


def _gmres(operator,rhs,preconditioner,tol,maxiter,history,log=None,restart=100):
    # scipy 1.10/1.11 use tol; 1.12+ accept rtol.
    key='rtol' if 'rtol' in inspect.signature(gmres).parameters else 'tol'
    def observe(r):
        history.append(float(r))
        if log is not None and len(history)%25==0:
            log(f'peec GMRES: {len(history)} iterations, preconditioned residual {float(r):.3g}')
    x,status=gmres(operator,rhs,M=preconditioner,atol=0.,restart=min(restart,len(rhs),maxiter),
                   maxiter=maxiter,callback=observe,callback_type='legacy',**{key:tol})
    residual=np.linalg.norm(operator@x-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny)
    if status or not np.isfinite(residual) or residual>max(tol*5,1e-13):
        raise ConvergenceError(status,float(residual),len(history))
    return x,residual


def potential_solve(model,rhs):
    P=model._ensure_potential()
    diagonal=np.maximum(abs(P.diagonal),np.finfo(float).tiny)
    M=LinearOperator(P.shape,matvec=lambda x:x/diagonal,dtype=complex)
    rhs=np.asarray(rhs);one=rhs.ndim==1
    columns=rhs[:,None] if one else rhs
    answer=np.empty(columns.shape,complex)
    for k in range(columns.shape[1]):
        answer[:,k],_=_gmres(P,columns[:,k],M,model.solver_tol,model.solver_maxiter,[],restart=_restart(model))
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
    non_tree=np.flatnonzero(~m['tree']);identity=m['C'][non_tree][:,cycles].tocoo()
    cotree=np.empty(nl,int);cotree[identity.col]=non_tree[identity.row]
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
    from .mqs_iterative import face_pairs
    pairs,lpair=face_pairs(model,Le)                 # face sheets: each edge pair's 2x2 block in the surrogates
    copies=_charge_copies(model,P)
    if copies is not None:
        P=copies['node_P'];pscale=max(float(np.max(abs(copies['Pc'].diagonal))),np.finfo(float).tiny)
    model._peec=dict(A=A,K=K,T=T,P=P,nl=nl,B=B,keep=keep,tree_lu=tree_lu,
                     sources=sources,passive=passive,names=names,pscale=pscale,
                     source_components=(A[:,sources].T@B).toarray(),ldiag=ldiag,
                     backend='hierarchical',factorizations=0,factor_bytes=0,cotree=cotree,copies=copies,
                     pairs=pairs,lpair=lpair)
    return model._peec


def _near_field(Pc,nc):
    """The dense leaves of a PotentialOperator's compressed kernel as a sparse charge-node matrix (W^T H_near W),
    its exact diagonal kept."""
    H,W=Pc.kernel,Pc.weights
    rows,cols,vals=[],[],[]
    for a,b,u,v,same in H.blocks:
        if v is not None:continue
        r,c=np.meshgrid(a,b,indexing='ij')
        rows.append(r.ravel());cols.append(c.ravel());vals.append(np.asarray(u).ravel())
        if not same:rows.append(c.ravel());cols.append(r.ravel());vals.append(np.asarray(u).ravel())
    nt=H.shape[0]
    Hn=coo_matrix((np.concatenate(vals) if vals else np.zeros(0),(np.concatenate(rows) if rows else np.zeros(0,int),
                   np.concatenate(cols) if cols else np.zeros(0,int))),shape=(nt,nt)).tocsr()
    Ps=(W.T@Hn@W).tocsr().astype(complex)
    Ps.setdiag(np.asarray(Pc.diagonal,complex))
    return Ps.tocsc()


def _branch_surrogate(m,diagonal,resistance,jw):
    """The branches' impedance for the preconditioner: each branch's own, and for face-sheet edge pairs their
    coupling (resistance + jw mutual inductance, nearly the self terms: current circulating between the faces is
    cheap, which a diagonal misses)."""
    nb=len(diagonal);e,g=m['pairs']
    if not len(e):return diags(diagonal).tocsc()
    zpair=np.asarray(resistance[e,g]).ravel()+jw*m['lpair']
    return (diags(diagonal)+coo_matrix((np.r_[zpair,zpair],(np.r_[e,g],np.r_[g,e])),shape=(nb,nb))).tocsc()


class _NodeDiagonal:
    """Face sheets: the node-level potential's diagonal, for the preconditioners only (each node's copies'
    diagonal, area-weighted). Products go through the copies (_charge_copies)."""
    def __init__(self,diagonal):self.diagonal=diagonal


def _charge_copies(model,Pc):
    """Face sheets with capacitance: the copies' charges q_c = D0 q + N u, D0 spreading a node's charge over its
    copies by area and N's columns (copy i minus the node's first copy) moving it between them; u are extra
    unknowns, held by N^T Pc q_c = 0 (a node's copies at one potential). None without face sheets."""
    from .peec import charge_nodes
    Bn=charge_nodes(model)
    if Bn is None:return None
    nc=Bn.shape[0];node=np.asarray(Bn.tocsr().indices)
    area_c=np.asarray(getattr(Pc,'node_area',None) if getattr(Pc,'node_area',None) is not None else
                      np.bincount(model.charge_of,weights=model.rwg.area,minlength=nc),float)
    area_n=np.bincount(node,weights=area_c,minlength=model.n_nodes)
    D0=coo_matrix((area_c/area_n[node],(np.arange(nc),node)),shape=(nc,model.n_nodes)).tocsc()
    order=np.argsort(node,kind='stable');first=np.full(model.n_nodes,-1)
    seen=np.zeros(model.n_nodes,bool);rows,cols,vals=[],[],[];npair=0
    for c in order:
        k=node[c]
        if not seen[k]:seen[k]=True;first[k]=c;continue
        rows+=[c,first[k]];cols+=[npair,npair];vals+=[1.,-1.];npair+=1
    N=coo_matrix((vals,(rows,cols)),shape=(nc,npair)).tocsc()
    # a sparse surrogate of Pc for the preconditioner: the compressed matrix's dense (near-field) leaves, exact
    # for every close pair. The redistribution block N^T Pc N is differences of near-equal terms whose neighbour
    # couplings are as large as its diagonal, so a diagonal surrogate converges slowly. Its diagonal (a copy and
    # the node's first copy, always near) comes from it exactly
    Ps=_near_field(Pc,nc)
    du=np.asarray((N.T@Ps@N).diagonal(),complex)
    scale=float(np.max(abs(np.asarray(Pc.diagonal)))) if nc else 1.
    du[abs(du)<=1e-12*scale]=scale                     # (never expected: a guard against a zero pivot)
    node_diag=np.asarray(D0.T@Pc.diagonal).ravel()
    return dict(Bn=Bn,D0=D0,N=N,du=du,Pc=Pc,Ps=Ps,node_P=_NodeDiagonal(node_diag),nu=N.shape[1])


def solve(model,freqs,amps=None):
    from .mqs import Result
    f=np.atleast_1d(np.asarray(freqs,float))
    if f.ndim!=1 or not np.isfinite(f).all() or (f<=0).any() or not len(f):
        raise ValueError('frequencies must be finite and positive')
    preconditioner=getattr(model,'solver_preconditioner','auto')
    if preconditioner not in ('auto','graph','tree'):
        raise ValueError('solver_preconditioner must be auto, graph or tree')
    restart=_restart(model)
    m=prepare(model);A,K,T,P,B=[m[k] for k in ('A','K','T','P','B')]
    nl=m['nl'];sources=m['sources'];names=m['names'];nb=A.shape[1];n=K.shape[1]
    excitation=np.array([(amps or {}).get(name,1.) for name in names],complex)
    drive=np.asarray(A[:,sources]@excitation).ravel();totals=m['source_components'].T@excitation
    qpart=np.zeros(model.n_nodes,complex);vpart=np.zeros(B.shape[1],complex)
    if np.any(totals):
        if 'PinvB' not in m:
            cp=m['copies']
            if cp is None:m['PinvB']=potential_solve(model,B.toarray())
            else:m['PinvB']=np.asarray(cp['Bn'].T@potential_solve(model,(cp['Bn']@B).toarray()))
            m['component']=B.T@m['PinvB']
        vpart=np.linalg.solve(m['component'],-totals);qpart=m['PinvB']@vpart
    y0=m['tree_lu'].solve((-drive-qpart)[m['keep']]) if m['tree_lu'] is not None else np.zeros(0,complex)
    baseline=K[:,nl:]@y0;source_voltage=-m['source_components']@vpart
    currents=np.zeros((len(f),nb),complex);charge=np.zeros((len(f),model.n_nodes),complex)
    potential=np.zeros_like(charge);z={name:np.zeros(len(f),complex) for name in names}
    model.solve_info=[]
    log=partial(print,flush=True) if model.log is print else model.log
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
        cp=m['copies'];nu=cp['nu'] if cp is not None else 0
        def product(y):
            x=y[:n]
            branch=K[:,:nl]@x[:nl]+jw*alpha*(K[:,nl:]@x[nl:])
            out=np.zeros(n+nu,complex)
            out[:n]=np.asarray(K.T@impedance(branch)).ravel()/zscale
            if cp is None:
                out[nl:n]+=np.asarray(T.T@(P@(T@x[nl:]))).ravel()/m['pscale']
            else:                                        # through the copies: q_c = D0 T x + N u
                vc=cp['Pc']@(cp['D0']@(T@x[nl:])+cp['N']@y[n:])
                out[nl:n]+=np.asarray(T.T@(cp['D0'].T@vc)).ravel()/m['pscale']
                out[n:]=np.asarray(cp['N'].T@vc).ravel()/m['pscale']
            return out
        rhs=np.r_[-np.asarray(K.T@impedance(baseline)).ravel()/zscale,np.zeros(nu,complex)]
        history=[];preconditioner_bytes=0
        kind='none'
        if n:
            # A sparse surrogate in the same scaled coordinates. Fundamental
            # cycle fill remains a limitation on very large meshes.
            counts=np.diff(K.tocsr().indptr).astype(float)
            electric_counts=np.diff(T.tocsr().indptr).astype(float)
            estimated_bytes=min(n*n,float(counts@counts+electric_counts@electric_counts))*24
            if preconditioner=='tree':
                M,preconditioner_bytes=_tree_preconditioner(m,diagonal,zscale)
                kind='tree_diagonal'
            elif preconditioner=='graph' or estimated_bytes>_PROJECTED_PRECONDITIONER_BYTES:
                try:
                    M,preconditioner_bytes=_graph_preconditioner(m,diagonal,zscale,resistance,jw)
                    kind='graph_ilu'
                except RuntimeError:
                    M,preconditioner_bytes=_tree_preconditioner(m,diagonal,zscale)
                    kind='tree_diagonal'
            else:
                approx=(K.T@_branch_surrogate(m,diagonal,resistance,jw)@K).tocsc()/zscale
                scaling=np.r_[np.ones(nl),np.full(n-nl,jw*alpha)]
                approx=approx@diags(scaling)
                if cp is None:
                    electric=(T.T@diags(P.diagonal)@T).tocsc()/m['pscale']
                    approx=approx+block_diag((coo_matrix((nl,nl)),electric),format='csc')
                else:                                    # the copies' unknowns in the same surrogate
                    G=hstack((cp['D0']@T,cp['N']),format='csc')
                    electric=(G.T@cp['Ps']@G).tocsc()/m['pscale']
                    approx=block_diag((approx,coo_matrix((nu,nu))),format='csc')
                    approx=approx+block_diag((coo_matrix((nl,nl)),electric),format='csc')
                try:
                    lu=spilu(approx.tocsc(),drop_tol=1e-4,fill_factor=10)
                    M=LinearOperator(approx.shape,matvec=lu.solve,dtype=complex)
                    preconditioner_bytes=sum(mat.data.nbytes+mat.indices.nbytes+mat.indptr.nbytes for mat in (lu.L,lu.U))
                    kind='projected_ilu'
                except RuntimeError:
                    d=approx.diagonal();d[abs(d)<np.finfo(float).tiny]=1
                    M=LinearOperator(approx.shape,matvec=lambda x:x/d,dtype=complex)
                    kind='diagonal'
            if cp is not None and M.shape[0]==n:         # tree/graph preconditioners: the copies' block diagonal
                Mx,du=M,cp['du']/m['pscale']
                M=LinearOperator((n+nu,n+nu),matvec=lambda y,Mx=Mx,du=du:np.r_[Mx@y[:n],y[n:]/du],dtype=complex)
            op=LinearOperator((n+nu,n+nu),matvec=product,dtype=complex)
            log(f'peec: {freq:g} Hz, {n:,} unknowns, {kind}, {preconditioner_bytes/1024**2:.2f} MiB preconditioner')
            try:
                x,residual=_gmres(op,rhs,M,model.solver_tol,model.solver_maxiter,history,log,restart)
            except ConvergenceError as error:
                model.solve_info.append(dict(frequency_Hz=float(freq),iterations=error.iterations,
                    relative_residual=error.relative_residual,preconditioner=kind,converged=False,
                    preconditioner_bytes=preconditioner_bytes,residual_history=history))
                raise
            correction=alpha*x[nl:n]
            currents[fi]=baseline+K[:,:nl]@x[:nl]+jw*(K[:,nl:n]@correction)
            charge[fi]=qpart/jw-T@correction
            if cp is None:pcorrection=P@(T@correction)
            else:pcorrection=np.asarray(cp['D0'].T@(cp['Pc']@(cp['D0']@(T@correction)+cp['N']@(alpha*x[n:])))).ravel()
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
