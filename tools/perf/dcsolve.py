import time, sys, numpy as np, scipy.sparse.linalg as sla
import pcbqs.pdn.dc as dc
from pcbqs.pdn.cli import main
orig = dc.spsolve
def trial(A, b):
    print('DC system', A.shape, 'nnz', A.nnz, 'rhs', np.shape(b), flush=True)
    x = None
    if 'mmd' in sys.argv:
        t = time.time(); x = sla.splu(A, permc_spec='MMD_AT_PLUS_A', diag_pivot_thresh=0, options=dict(SymmetricMode=True)).solve(np.asarray(b).ravel())
        print('   MMD_AT_PLUS_A symmetric mode', round(time.time() - t, 2), flush=True)
    if 'default' in sys.argv:
        t = time.time(); x0 = orig(A, b); print('  spsolve default (COLAMD)', round(time.time() - t, 2), 'diff', None if x is None else float(abs(x - x0).max() / abs(x0).max()), flush=True); x = x0
    return x
dc.spsolve = trial
caps = [f'CA{k}=47e-6,0.02,1e-9' for k in range(1, 5)]
main(['/home/bt/Desktop/wpthb/wpthb.kicad_pcb', '--rail', 'VIN', '--volts', '24', '--source', 'TP2', '--ground-source', 'TP1',
      '--load', 'Q1=0.67', '--load', 'Q2=0.67', '--through', 'L3', '--dcr', 'L3=0.03', *[x for c in caps for x in ('--cap', c)], '--dmax', '0.4', '--out', 'out_pdn2'])
