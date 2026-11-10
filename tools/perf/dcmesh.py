import time, sys, numpy as np
import pcbqs.pdn.dc as dc
from pcbqs.pdn.cli import main
orig = dc.net_sheets
def ns(board, nets, pts, **kw):
    t = time.time(); s = orig(board, nets, pts, **kw)
    print('net_sheets', sorted(nets)[:2], 'dmax', kw.get('dmax'), 'dmax_fine', kw.get('dmax_fine'), 'quality', kw.get('quality'),
          'fine area', None if kw.get('fine') is None else round(kw['fine'].area, 1), round(time.time() - t, 1), 's')
    for lay, sh in s.items():
        n, tr = sh.nodes, sh.tris
        a = 0.5 * abs(np.cross(n[tr[:, 1]] - n[tr[:, 0]], n[tr[:, 2]] - n[tr[:, 0]]))
        edge = np.sqrt(4 * a / np.sqrt(3))
        print(f'   {lay}: {len(n):,} vertices, {len(tr):,} triangles, area {a.sum():.0f} mm2, equivalent edge median {np.median(edge):.3f} mm, '
              f'share of triangles under 0.05 mm: {np.mean(edge < 0.05):.2f}, area share {a[edge < 0.05].sum() / a.sum():.3f}')
    if 'GND' in nets:
        for lay, sh in s.items():
            n, tr = sh.nodes, sh.tris
            a = 0.5 * abs(np.cross(n[tr[:, 1]] - n[tr[:, 0]], n[tr[:, 2]] - n[tr[:, 0]]))
            small = a < 0.05 ** 2 * 0.43
            c = n[tr[small]].mean(1)
            cells, counts = np.unique(np.round(c / 0.5).astype(int), axis=0, return_counts=True)
            o = np.argsort(counts)[::-1][:4]
            print('  ', lay, 'tiny triangles', small.sum(), 'in', len(cells), '0.5 mm cells; top:', [(tuple((cells[k] * 0.5).tolist()), int(counts[k])) for k in o], 'min area', a.min())
            reg = kw.get('regions', {}).get(lay, [])
            print('     regions here:', [(round(q.centroid.x, 2), round(q.centroid.y, 2), round(q.area, 3)) for q in reg][:6])
        import pickle; pickle.dump(dict(kw=kw, pts=pts), open('gnd_mesh_args.pkl', 'wb'))
        raise SystemExit
    return s
dc.net_sheets = ns
caps = [f'CA{k}=47e-6,0.02,1e-9' for k in range(1, 5)]
main(['/home/bt/Desktop/wpthb/wpthb.kicad_pcb', '--rail', 'VIN', '--volts', '24', '--source', 'TP2', '--ground-source', 'TP1',
      '--load', 'Q1=0.67', '--load', 'Q2=0.67', '--through', 'L3', '--dcr', 'L3=0.03', *[x for c in caps for x in ('--cap', c)], '--dmax', sys.argv[1], '--out', 'out_pdn2'])
