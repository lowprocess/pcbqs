"""Trial: two face sheets, port GMRES preconditioned by an exact common-mode coarse solve plus the graph smoother."""
import sys, time, pathlib, resource
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csr_matrix, identity
import pcbqs
from pcbqs._cli import load_board
from pcbqs.power_bridge import load_config, make_board_ports
from pcbqs import mqs_iterative as mi
variant = sys.argv[1]
fs = [float(v) for v in sys.argv[2].split(',')]
cfg = load_config(pathlib.Path(pcbqs.__file__).parent / 'data/wpthb-power.json')
cfg['mesh']['matrix_backend'] = 'hierarchical'; cfg['mesh']['thickness_sheets'] = 2
board = load_board('/home/bt/Desktop/wpthb/wpthb.kicad_pcb', set(cfg['mesh']['nets']))
q, ports, crop = make_board_ports(board, cfg, lambda *a: None)
m = q._assemble(); p = mi._prepare(q)
C, Cf, Cs, free, cotree = m['C'].tocsc(), p['Cf'], p['Cs'], p['free'], p['cotree']
nb = m['nb']; e, g = p['pairs']
t0 = time.time()
perm = np.arange(nb); perm[e], perm[g] = g, e
S = 0.5 * (identity(nb, format='csr') + csr_matrix((np.ones(nb), (np.arange(nb), perm)), shape=(nb, nb)))
B = (S @ Cf).tocsc()                                    # symmetrised free loops, as branch currents
partner_first = np.ones(nb, bool); partner_first[g] = False      # keep one of each pair's two loops
norms = np.sqrt(np.asarray(B.multiply(B).sum(0)).ravel())
sel = np.flatnonzero(partner_first[cotree] & (norms > 1e-9))
# drop duplicates (a pair with one copy in the tree gives its partner's symmetrised loop = 0; both cotree: equal columns)
B = B[:, sel]; W = B[cotree]                             # W: free-loop coefficients (cotree identity)
nc = B.shape[1]; print('free loops', len(free), 'coarse', nc, 'prep', round(time.time() - t0, 1))
t0 = time.time(); Bd = B.toarray(); LB = np.empty((nb, nc))
for k in range(0, nc, 48):
    LB[:, k:k + 48] = q._inductive_voltage(Bd[:, k:k + 48].astype(complex)).real
Lc = Bd.T @ LB; Lc = (Lc + Lc.T) / 2; del Bd
CfL = np.asarray(Cf.T @ LB)                               # A W's inductive part, free loops x coarse
del LB
print('coarse L', round(time.time() - t0, 1), 's')
tot = 0; t_all = time.time(); warm = None
for fi in fs:
    w = 2 * np.pi * fi
    R = (m['R'] + (q._resistance_delta(fi) if fi else 0 * m['R'])).tocsr()
    total = lambda x: R @ x + (1j * w * q._inductive_voltage(x) if fi else 0)
    d = (R.diagonal() + 1j * w * p['ldiag']).astype(complex)
    sc_ = max(float(np.max(abs(d))), 1e-300); d = np.where(abs(d) < sc_ * 1e-12, sc_ * 1e-12, d)
    zpair = (np.asarray(R[e, g]).ravel() + 1j * w * p['lpair']).astype(complex)
    G, _ = mi._graph_preconditioner(p, d, zpair)
    RB = R @ B
    Ac = (B.T @ RB).toarray() + 1j * w * Lc
    lu = lu_factor(Ac, check_finite=False)
    AW = np.asarray((Cf.T @ RB).toarray()) + 1j * w * CfL   # A W, dense
    Wd = W
    def coarse(r):
        return Wd @ lu_solve(lu, np.asarray(Wd.T @ r), check_finite=False)
    if variant == 'add':
        M = lambda r: coarse(r) + G.block(r)
    elif variant == 'mult':                                 # coarse first, smoother on the remaining residual
        def M(r):
            y = lu_solve(lu, np.asarray(Wd.T @ r), check_finite=False)
            return Wd @ y + G.block(r - AW @ y)
    else:
        M = G.block
    base = Cs.toarray().astype(complex)
    rhs = -np.asarray(Cf.T @ total(base))
    block = lambda X: np.asarray(Cf.T @ total(Cf @ X))
    t0 = time.time()
    X, its, res = mi._lockstep_gmres(block, M, rhs, warm if warm is not None else np.zeros_like(rhs), q.solver_tol, q.solver_maxiter, 60)
    warm = X
    I = base + Cf @ X; Zp = np.asarray(Cs.T @ total(I))
    print(f'{fi:g} Hz: {its.sum()} iterations (max {its.max()}), {time.time() - t0:.1f} s, res {res.max():.1e}, Z00 {Zp[0,0]:.6g}', flush=True)
    tot += its.sum()
print(variant, 'total iterations', tot, 'time', round(time.time() - t_all, 1), 'peak MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
