"""Profile copper-port extraction on wpthb: lockstep GMRES against a dense loop matrix built from the H-matrix."""
import sys, time, json, resource
import numpy as np
from scipy.linalg import solve
from pcbqs._cli import load_board
from pcbqs.power_bridge import load_config, make_board_ports
from pcbqs import mqs_iterative as mi
import pcbqs, pathlib
sheets = int(sys.argv[1]); mode = sys.argv[2]
cfg = load_config(pathlib.Path(pcbqs.__file__).parent / 'data/wpthb-power.json')
cfg['mesh']['matrix_backend'] = 'hierarchical'; cfg['mesh']['thickness_sheets'] = sheets
board = load_board('/home/bt/Desktop/wpthb/wpthb.kicad_pcb', set(cfg['mesh']['nets']))
t = time.time(); q, ports, crop = make_board_ports(board, cfg, print); print('model', time.time() - t)
t = time.time(); m = q._assemble(); p = mi._prepare(q); print('assemble+prepare', time.time() - t)
fs = [0.0, 1e3, 1e4, 3e4, 9e4, 1e5, 3e5, 9e5, 3e6, 7e6, 2e7]
Cf, Cs = p['Cf'], p['Cs']; C = m['C'].tocsc(); print('loops', C.shape, 'free', Cf.shape[1], 'nb', m['nb'])
x = np.random.default_rng(0).standard_normal((m['nb'], 24)) + 0j
t = time.time(); q._inductive_voltage(x); dt = time.time() - t; print(f'product 24 cols {dt:.3f} s, {dt/24*1e3:.1f} ms/col')
if mode == 'lock':
    t = time.time(); Z, B = mi.port_matrix(q, fs, print); print('lockstep', time.time() - t); np.save(f'Z_lock{sheets}.npy', Z)
else:
    t = time.time()
    Cd = C.toarray()                                  # (nb, loops)
    LC = np.empty(Cd.shape)
    for k in range(0, Cd.shape[1], 48):              # real products: the operator is real
        LC[:, k:k + 48] = q._inductive_voltage(Cd[:, k:k + 48].astype(complex)).real
    Ll = Cd.T @ LC; Ll = (Ll + Ll.T) / 2; del LC
    Rl = (C.T @ m['R'] @ C).toarray()
    print('loop matrices from H', time.time() - t); t = time.time()
    sc, free = p['sc'], p['free']; Zs = []
    for fi in fs:
        Zm = Rl + 2j * np.pi * fi * Ll
        d = q._resistance_delta(fi) if fi else None
        if d is not None and d.nnz: Zm += (C.T @ d @ C).toarray()
        X = solve(Zm[np.ix_(free, free)], -Zm[np.ix_(free, sc)], assume_a='sym', overwrite_a=True, check_finite=False)
        Zs.append(Zm[np.ix_(sc, sc)] + Zm[np.ix_(sc, free)] @ X)
    print('dense loop solves', time.time() - t); np.save(f'Z_hd{sheets}.npy', np.array(Zs))
print('peak MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
