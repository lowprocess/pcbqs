import time, sys, resource, numpy as np
import pcbqs.pdn.dc as dc
from pcbqs.pdn.cli import main
init = dc.Network.__init__
def patched(self, board, *a, **k):
    if 'fix' in sys.argv:
        board = dict(board); board['vias'] = [v for v in board['vias'] if not (abs(v['x'] - 153.0) < 1e-6 and abs(v['y'] + 24.0) < 1e-6)]
    init(self, board, *a, **k)
dc.Network.__init__ = patched
caps = [f'CA{k}=47e-6,0.02,1e-9' for k in range(1, 5)]
t = time.time()
main(['/home/bt/Desktop/wpthb/wpthb.kicad_pcb', '--rail', 'VIN', '--volts', '24', '--source', 'TP2', '--ground-source', 'TP1',
      '--load', 'Q1=0.67', '--load', 'Q2=0.67', '--through', 'L3', '--dcr', 'L3=0.03', *[x for c in caps for x in ('--cap', c)], '--dmax', sys.argv[1], '--out', 'out_pdn3'])
print('TOTAL', round(time.time() - t, 1), 's peak MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
