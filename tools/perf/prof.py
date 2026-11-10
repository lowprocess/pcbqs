import sys, time, cProfile, pstats, pathlib
import numpy as np
import pcbqs
from pcbqs._cli import load_board
from pcbqs.power_bridge import load_config, make_board_ports
from pcbqs import mqs_iterative as mi
sheets = int(sys.argv[1])
cfg = load_config(pathlib.Path(pcbqs.__file__).parent / 'data/wpthb-power.json')
cfg['mesh']['matrix_backend'] = 'hierarchical'; cfg['mesh']['thickness_sheets'] = sheets
board = load_board('/home/bt/Desktop/wpthb/wpthb.kicad_pcb', set(cfg['mesh']['nets']))
q, ports, crop = make_board_ports(board, cfg, lambda *a: None)
m = q._assemble(); p = mi._prepare(q)
C = m['C']; print('C', C.shape, 'nnz', C.nnz, 'per loop', C.nnz / C.shape[1], 'max', np.diff(C.tocsc().indptr).max())
Le = m['L'][0]; print('H info', {k: v for k, v in Le.info.items()})
pr = cProfile.Profile(); pr.enable(); mi.port_matrix(q, [9e5], print); pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(14)
