"""Compressed MQS (inductance only, capacitance=False): loop currents by GMRES on the compressed copper inductance.

The dense path projects R and L onto the loop basis C and factors C^T Z C at each frequency: O(n^2) memory, which
stops a 32 GB machine near 12k triangles. Here Z is applied as products (sparse R and its skin correction exactly,
L through its hierarchical matrix) and C^T Z C is never formed. The preconditioner inverts C^T D C exactly for a
diagonal D (each branch's resistance, self inductance and part): with the loops' cotree branches carrying the
force, that is one sparse graph-Laplacian solve on the nodes, A D^-1 A^T, factored once per frequency.
"""
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, splu

from .peec_iterative import ConvergenceError, _gmres, _restart

__all__ = ['solve', 'ConvergenceError']


def _diagonal(L):
    d = L.diagonal
    return np.asarray(d() if callable(d) else d, float)


def _prepare(model):
    if getattr(model, '_mqs_iter', None) is not None:
        return model._mqs_iter
    m = model._assemble()
    C = m['C'].tocsc()
    nb, ne, nv = m['nb'], model.ne, model.nv
    src = {model.parts[k - ne - nv]['name']: c for k, c in m['src'].items()}
    sc = np.array(list(src.values()), int)
    free = np.setdiff1d(np.arange(C.shape[1]), sc)
    ends = list(model.br) + [(model.term_node[p['a']], model.term_node[p['b']]) for p in model.parts]
    a, b = np.asarray(ends, int).reshape(-1, 2).T
    branch = np.arange(nb)
    A = coo_matrix((np.r_[np.ones(nb), -np.ones(nb)], (np.r_[a, b], np.r_[branch, branch])),
                   shape=(model.n_nodes, nb)).tocsc()
    passive = np.flatnonzero(~m['is_src'])
    # each free loop's cotree branch: the non-tree branch with coefficient 1 in its column
    non_tree = np.flatnonzero(~m['tree'])
    ident = C[non_tree][:, free].tocoo()
    one = ident.data == 1
    cotree = np.empty(len(free), int)
    cotree[ident.col[one]] = non_tree[ident.row[one]]
    graph = coo_matrix((np.ones(len(passive)), (a[passive], b[passive])), shape=(model.n_nodes, model.n_nodes))
    _, labels = connected_components(graph, directed=False)
    roots = np.unique(labels, return_index=True)[1]
    keep = np.ones(model.n_nodes, bool)
    keep[roots] = False
    Le, Lv = m['L']
    ld = _diagonal(Le)
    if getattr(model, 'thick_kernel', False):
        ld = ld + np.r_[model._thick_correction().diagonal(), np.zeros(len(ld) - model.n_rwg)]
    ldiag = np.r_[ld, _diagonal(Lv) if model.nv else np.zeros(0), np.zeros(len(model.parts))]
    # face sheets (thickness_sheets=2): an edge's two copies are nearly one conductor (35 um apart), so the
    # preconditioner keeps each pair's 2x2 block: their mutual inductance, exactly, in chunks
    pairs, lpair = face_pairs(model, Le)
    model._mqs_iter = dict(C=C, Cf=C[:, free], Cs=C[:, sc], src=src, sc=sc, free=free, A=A, passive=passive,
                           cotree=cotree, keep=keep, ldiag=ldiag, pairs=pairs, lpair=lpair)
    return model._mqs_iter


def face_pairs(model, Le):
    """Face sheets: the (2, P) edge pairs (an edge on one face, the same edge on the other) and their mutual
    partial inductance, exactly, for the preconditioners' 2x2 blocks. Empty without face sheets."""
    pairs = np.hstack(getattr(model, '_face_pairs', [])) if getattr(model, '_face_pairs', None) else np.zeros((2, 0), int)
    lpair = np.zeros(pairs.shape[1])
    for k in range(0, pairs.shape[1], 256):
        e, g = pairs[0, k:k + 256], pairs[1, k:k + 256]
        if hasattr(Le, 'entry') and Le.entry is not None:
            lpair[k:k + 256] = np.diagonal(np.asarray(Le.entry(e, g)))
        elif hasattr(Le, 'entry'):                       # a cached, reloaded operator keeps no exact entries
            lpair[k:k + 256] = _pair_mutual(model, e, g)
        else:
            lpair[k:k + 256] = np.asarray(Le)[e, g]
    if getattr(model, 'thick_kernel', False) and pairs.shape[1]:
        lpair = lpair + np.asarray(model._thick_correction()[pairs[0], pairs[1]]).ravel()
    return pairs, lpair


def _pair_mutual(model, e, g):
    """Mutual partial inductance (branch units, as MQS's L) of RWG edges e and g, from their support triangles'
    analytic moments: the face pairs' preconditioner blocks when the compressed operator keeps no exact entries."""
    from ._numerics.sing import offplane_pairs
    r, z = model.rwg, model.z_of
    out = np.zeros(len(e))
    for ta, aa, va in ((r.tp[e], r.l[e] / (2 * r.area[r.tp[e]]), r.vp[e]), (r.tm[e], -r.l[e] / (2 * r.area[r.tm[e]]), r.vm[e])):
        for tb, ab, vb in ((r.tp[g], r.l[g] / (2 * r.area[r.tp[g]]), r.vp[g]), (r.tm[g], -r.l[g] / (2 * r.area[r.tm[g]]), r.vm[g])):
            m0, mro, mrs, mrr = offplane_pairs(r.nodes, r.tris, ta, tb, z[ta] - z[tb], outer_levels=1)
            out += aa * ab * (mrr - (vb * mro).sum(1) - (va * mrs).sum(1) + (va * vb).sum(1) * m0)
    return 1e-7 * out / (r.l[e] * r.l[g])                   # mu0 / 4 pi


def _graph_preconditioner(p, d, zpair):
    """(C_f^T D C_f)^-1 applied exactly for a block-diagonal D: each branch's own impedance d, and for face-sheet
    pairs the 2x2 block with their coupling zpair. The loop currents under a force y on the loops' cotree branches
    are i = D^-1 (f - A^T v) with A i = 0, i.e. (A D^-1 A^T) v = A D^-1 f on the passive branches (one node grounded
    per connected piece of copper); both copies of an edge join the same nodes, so A D^-1 A^T stays sparse."""
    A, passive, keep, cotree, pairs = p['A'], p['passive'], p['keep'], p['cotree'], p['pairs']
    nb = len(d)
    inv = np.zeros(nb, complex)
    inv[passive] = 1 / d[passive]
    rows, cols, vals = [np.arange(nb)], [np.arange(nb)], [inv]
    if pairs.shape[1]:
        e, g = pairs
        det = d[e] * d[g] - zpair ** 2
        inv[e], inv[g] = d[g] / det, d[e] / det
        rows += [e, g]
        cols += [g, e]
        vals += [-zpair / det, -zpair / det]
    Dinv = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nb, nb)).tocsr()
    Dinv = diags(np.where(np.isin(np.arange(nb), passive), 1.0, 0.0)) @ Dinv
    lap = (A @ Dinv @ A.T).tocsc()[keep][:, keep]
    lu = splu(lap.tocsc())

    def apply(y):                                        # y: (loops,) or (loops, k) columns
        force = np.zeros((nb,) + np.shape(y)[1:], complex)
        force[cotree] = y
        trial = Dinv @ force
        v = np.zeros((A.shape[0],) + np.shape(y)[1:], complex)
        v[keep] = lu.solve(np.ascontiguousarray(np.asarray(A @ trial)[keep]))
        return (trial - Dinv @ np.asarray(A.T @ v))[cotree]
    size = sum(s.data.nbytes + s.indices.nbytes + s.indptr.nbytes for s in (lu.L, lu.U))
    M = LinearOperator((len(cotree), len(cotree)), matvec=apply, matmat=apply, dtype=complex)
    M.block = apply
    return M, size


def _lockstep_gmres(op, M, B, X0, tol, maxiter, restart):
    """GMRES on k right-hand sides at once (B: (n, k)), right-preconditioned, each column its own Krylov space
    but every step one operator product on all the active columns (a compressed matrix read once for k columns).
    Columns that converge drop out at the next restart. Returns (X, iterations per column, relative residuals)."""
    n, k = B.shape
    X = np.array(X0, complex)
    bn = np.maximum(np.linalg.norm(B, axis=0), np.finfo(float).tiny)
    its = np.zeros(k, int)
    R = B - op(X)
    active = np.flatnonzero(np.linalg.norm(R, axis=0) / bn > tol)
    while len(active) and its[active].max() < maxiter:
        m = int(min(restart, maxiter - its[active].max()))
        r = R[:, active]
        beta = np.linalg.norm(r, axis=0)
        ka = len(active)
        V = np.zeros((m + 1, n, ka), complex)
        V[0] = r / beta
        H = np.zeros((m + 1, m, ka), complex)
        cs = np.zeros((m, ka), complex)
        sn = np.zeros((m, ka), complex)
        g = np.zeros((m + 1, ka), complex)
        g[0] = beta
        Z = np.zeros((m, n, ka), complex)
        done = np.zeros(ka, bool)
        last = np.full(ka, m - 1)
        for j in range(m):
            Z[j] = M(V[j])
            w = op(Z[j])
            for i in range(j + 1):                       # modified Gram-Schmidt, per column
                h = np.einsum('nk,nk->k', V[i].conj(), w)
                H[i, j] = h
                w -= V[i] * h
            hn = np.linalg.norm(w, axis=0)
            H[j + 1, j] = hn
            V[j + 1] = w / np.where(hn > 0, hn, 1)
            for i in range(j):                           # the earlier Givens rotations (copies: H[i, j] is a view)
                a, b = H[i, j].copy(), H[i + 1, j].copy()
                H[i, j] = cs[i].conj() * a + sn[i].conj() * b
                H[i + 1, j] = -sn[i] * a + cs[i] * b
            a, b = H[j, j].copy(), H[j + 1, j].copy()
            den = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
            den = np.where(den > 0, den, 1)
            cs[j], sn[j] = a / den, b / den
            H[j, j] = cs[j].conj() * a + sn[j].conj() * b
            H[j + 1, j] = 0
            g[j + 1] = -sn[j] * g[j]
            g[j] = cs[j].conj() * g[j]
            newly = (~done) & (abs(g[j + 1]) / bn[active] <= tol)
            last[newly] = j
            done |= newly
            its[active[~done | newly]] += 1
            if done.all():
                break
        for c in range(ka):                              # each column's update from its own Hessenberg
            jj = last[c] + 1 if done[c] else j + 1
            y = np.linalg.solve(np.triu(H[:jj, :jj, c]), g[:jj, c]) if jj else np.zeros(0)
            X[:, active[c]] += np.einsum('j,jn->n', y, Z[:jj, :, c])
        R[:, active] = B[:, active] - op(X[:, active])
        res = np.linalg.norm(R[:, active], axis=0) / bn[active]
        active = active[res > tol]
    res = np.linalg.norm(R, axis=0) / bn
    return X, its, res


def solve(model, freqs, amps=None):
    """MQS.solve for matrix_backend='hierarchical' without capacitance: the same Result (z, zc) as the dense path."""
    from .mqs import Result
    f = np.atleast_1d(np.asarray(freqs, float))
    if f.ndim != 1 or not np.isfinite(f).all() or (f <= 0).any() or not len(f):
        raise ValueError('frequencies must be finite and positive')
    p = _prepare(model)
    m = model._mats
    Cf, Cs, src, free = p['Cf'], p['Cs'], p['src'], p['free']
    nb, ne, nv = m['nb'], model.ne, model.nv
    names = list(src)
    xs = np.array([(amps or {}).get(n, 1.0) for n in names], complex)
    I = np.zeros((len(f), nb), complex)
    Z = {n: np.zeros(len(f), complex) for n in names}
    Zc = {n: np.zeros(len(f), complex) for n in names}
    restart = _restart(model)
    model.solve_info = []
    for i, fi in enumerate(f):
        w = 2 * np.pi * fi
        zp = np.zeros(nb, complex)
        for k, part in enumerate(model.parts):
            if part['kind'] == 'Z':
                zp[ne + nv + k] = part['R'] + 1j * w * part['L'] + (1 / (1j * w * part['C']) if part.get('C') else 0)
        R = (m['R'] + model._resistance_delta(fi)).tocsr()

        def copper(x):                                   # branch voltages of the copper (R, skin, L)
            return R @ x + 1j * w * model._inductive_voltage(x)

        def total(x):
            return copper(x) + zp * x
        d = R.diagonal() + 1j * w * p['ldiag'] + zp
        scale = max(float(np.max(abs(d))), np.finfo(float).tiny)
        d = np.where(abs(d) < scale * 1e-12, scale * 1e-12, d)
        base = Cs @ xs                                   # the sources' loops, fixed
        rhs = -np.asarray(Cf.T @ total(base)).ravel()
        history = []
        if len(free):
            e, g = p['pairs']
            zpair = np.asarray(R[e, g]).ravel() + 1j * w * p['lpair'] if len(e) else np.zeros(0, complex)
            M, mbytes = _graph_preconditioner(p, d, zpair)
            op = LinearOperator((len(free), len(free)), matvec=lambda x: np.asarray(Cf.T @ total(Cf @ x)).ravel(),
                                dtype=complex)
            try:
                x, residual = _gmres(op, rhs, M, model.solver_tol, model.solver_maxiter, history, model.log, restart)
            except ConvergenceError as error:
                model.solve_info.append(dict(frequency_Hz=float(fi), iterations=error.iterations, converged=False,
                                             relative_residual=error.relative_residual))
                raise
        else:
            x, residual, mbytes = np.zeros(0, complex), 0.0, 0
        I[i] = base + Cf @ x
        loop_v = np.asarray(Cs.T @ total(I[i])).ravel()  # KVL round each source's loop
        copper_v = np.asarray(Cs.T @ copper(I[i])).ravel()
        for j, n in enumerate(names):
            Z[n][i] = loop_v[j] / xs[j] if xs[j] else np.nan
            Zc[n][i] = copper_v[j] / xs[j] if xs[j] else np.nan
        model.solve_info.append(dict(frequency_Hz=float(fi), iterations=len(history), relative_residual=float(residual),
                                     preconditioner='graph_lu', preconditioner_bytes=int(mbytes), converged=True))
    return Result(f, I, Z, [part['name'] for part in model.parts], Zc)


def port_matrix(model, freqs, log=None):
    """Copper ports on the compressed backend (mqs_ports.CopperPorts): for each frequency (0 allowed: DC) the port
    impedance matrix (P, P) and the branch currents (nb, P) of a unit current into each port in turn, the others
    open. One GMRES per port, sharing each frequency's preconditioner; no dense loop matrix is formed, so face
    sheets fit in memory. Same conventions as the dense extraction: branch = -(the source loops' currents)."""
    p = _prepare(model)
    m = model._mats
    Cf, Cs, src, free = p['Cf'], p['Cs'], p['src'], p['free']
    nb = m['nb']
    if any(part['kind'] != 'I' for part in model.parts):
        raise ValueError('copper ports take current sources only')
    restart = _restart(model)
    Zs, Bs = [], []
    model.solve_info = []
    warm = None
    for fi in np.atleast_1d(np.asarray(freqs, float)):
        w = 2 * np.pi * fi
        R = (m['R'] + (model._resistance_delta(fi) if fi else 0 * m['R'])).tocsr()

        def total(x):
            return R @ x + (1j * w * model._inductive_voltage(x) if fi else 0)
        d = (R.diagonal() + 1j * w * p['ldiag']).astype(complex)
        scale = max(float(np.max(abs(d))), np.finfo(float).tiny)
        d = np.where(abs(d) < scale * 1e-12, scale * 1e-12, d)
        e, g = p['pairs']
        zpair = (np.asarray(R[e, g]).ravel() + 1j * w * p['lpair']).astype(complex) if len(e) else np.zeros(0, complex)
        M, _ = _graph_preconditioner(p, d, zpair) if len(free) else (None, 0)
        base = Cs.toarray().astype(complex)              # every port's own loop, at once
        iterations = 0
        if len(free):
            rhs = -np.asarray(Cf.T @ total(base))
            block = lambda X: np.asarray(Cf.T @ total(Cf @ X))
            X0 = warm if warm is not None else np.zeros_like(rhs)
            X, its, res = _lockstep_gmres(block, M.block, rhs, X0, model.solver_tol, model.solver_maxiter,
                                          min(restart, 60))
            if (res > max(5 * model.solver_tol, 1e-13)).any():
                raise ConvergenceError(1, float(res.max()), int(its.max()))
            warm = X                                     # the next frequency starts here
            iterations = int(its.sum())
            I = base + Cf @ X
        else:
            I = base
        Zp = np.asarray(Cs.T @ total(I))
        branch = -I
        Zs.append(0.5 * (Zp + Zp.T))                     # reciprocal (GMRES leaves ~tol asymmetry)
        Bs.append(branch)
        model.solve_info.append(dict(frequency_Hz=float(fi), iterations=iterations, ports=Cs.shape[1]))
        if log:
            log(f'Copper ports: {fi:g} Hz ({iterations:,} GMRES iterations over {Cs.shape[1]} ports)')
    return np.asarray(Zs), np.asarray(Bs)
