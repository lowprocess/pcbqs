"""A real converter loop: CM5 MINIMA's U702 input loop (VIN to GND through C704-C706) in pcbqs.mqs.MQS, Palace (PEC
sheets) and FastHenry (volume filaments), the three given the same copper.

    python validation/mqs_cm5.py [geometry|pcbqs|palace|fasthenry|figure|report] ...

The copper is the +5V and GND copper of all six layers inside a 9.4 x 13.7 mm box around the parts (pcbqs's own
crop for this loop), with the vias of both nets. Each capacitor is a copper bridge across its pads (an ideal short,
the same in every tool) and the port is across the 0.2 mm gap between U702's VIN pad and the GND pad beside it:
Palace's lumped port fills the gap; pcbqs's source and FastHenry's external pair join 0.1 mm slivers (equipotential
lines in FastHenry) along the gap's two edges, so no tool shorts pad copper the others don't (C13).
"""
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Point, box, mapping, shape
from shapely.ops import unary_union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import OUT, IMAGES, write_json  # noqa: E402

CM5 = '/usr/share/kicad/demos/cm5_minima/CM5_MINIMA_3.kicad_pcb'
NETS = ('+5V', 'GND')
REGION = box(122.28, -76.51, 131.69, -62.84)          # the parts' pads buffered 3 mm (cm5 loop's MQS region)
PORT_GAP = box(125.965, -71.71, 126.165, -71.21)      # U702: VIN pad (x 126.165-126.465) | GND pad (125.665-125.965)
SLIVERS = dict(A=box(126.165, -71.71, 126.265, -71.21), B=box(125.865, -71.71, 125.965, -71.21))
BRIDGES = dict(C704=box(126.79, -67.29, 127.69, -65.84), C705=box(126.79, -69.265, 127.69, -67.815),
               C706=box(125.84, -70.775, 126.24, -70.155))
FINE = box(124.28, -74.51, 129.69, -64.84)            # the pads' envelope + 1 mm: pcbqs's fine zone
F = np.array([1e6, 1e7, 1e8])
DIR = OUT / 'cm5'
LAYERS = None                                         # all six
THICK = {}                                            # layer: thickness (mm), overriding the board's
VARIANT = os.environ.get('CM5_VARIANT', '')
if VARIANT.startswith('lite') or VARIANT == 'top':
    # CM5-lite, for a FastHenry answer in minutes: F.Cu and In1 (the GND plane 0.1 mm below, the loop's main return)
    # only, the pads + 1.5 mm, the vias between those two layers. The same copper in every tool, so like for like.
    # Diagnostic variants: lite_t35 (In1 35 um instead of 15) and top (F.Cu alone, its own GND pour the return).
    REGION = box(123.78, -75.01, 130.19, -64.34)
    LAYERS = ('F.Cu',) if VARIANT == 'top' else ('F.Cu', 'In1.Cu')
    THICK = {'In1.Cu': 0.035} if VARIANT == 'lite_t35' else {}
    DIR = OUT / f'cm5_{VARIANT}'


def case():
    """board.json-shaped dict of the cropped copper (nets +5V, GND and CAP: the bridges), its vias and the stack."""
    cache = DIR / 'board.pickle'
    if cache.is_file():
        return pickle.loads(cache.read_bytes())
    from pcbqs._cli import load_board
    from pcbqs.pdn.mesh import copper_of
    b = load_board(CM5, set(NETS))
    copper = {}
    if LAYERS:
        b['stack'] = dict(b['stack'], copper=[dict(c, t=THICK.get(c['name'], c['t'])) for c in b['stack']['copper']
                                              if c['name'] in LAYERS])
    for c in b['stack']['copper']:
        lay = c['name']
        for n in NETS:
            g = copper_of(b, {n}, lay)
            g = None if g is None else g.intersection(REGION).buffer(0)
            if g is not None and not g.is_empty:
                copper.setdefault(lay, []).extend(dict(net=n, polygon=mapping(p)) for p in getattr(g, 'geoms', [g])
                                                  if p.geom_type == 'Polygon')
    top = unary_union([shape(e['polygon']) for e in copper['F.Cu']])
    for name, s in SLIVERS.items():
        assert top.buffer(1e-6).contains(s), f'sliver {name} is not on F.Cu copper'
    copper['F.Cu'] += [dict(net='CAP', polygon=mapping(p)) for p in BRIDGES.values()]
    vias = [v for v in b['vias'] if v.get('net') in NETS and REGION.contains(Point(v['x'], v['y']))]
    if LAYERS:                                        # a via spans the kept layers only
        z = [c['z'] for c in b['stack']['copper']]
        vias = [dict(v, z0=max(v['z0'], min(z)), z1=min(v['z1'], max(z))) for v in vias] if len(z) > 1 else []
    out = dict(stack=b['stack'], copper=copper, vias=vias, outline=mapping(REGION), prims=dict(footprints=[]))
    DIR.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(out))
    return out


def nets():
    return set(NETS) | {'CAP'}


def geometry():
    """cm5_geometry.png: each layer's copper, the bridges, the port and the vias."""
    from plots import style
    b = case()
    plt = style()
    layers = [c['name'] for c in sorted(b['stack']['copper'], key=lambda c: -c['z'])]
    fig, axs = plt.subplots(1, len(layers), figsize=(3.0 * len(layers), 4.4))
    colour = {'+5V': '#eb6834', 'GND': '#2a78d6', 'CAP': '#1baf7a'}
    for ax, lay in zip(axs, layers):
        for e in b['copper'].get(lay, []):
            p = shape(e['polygon'])
            ax.fill(*p.exterior.xy, color=colour[e['net']], alpha=0.55, lw=0)
            for h in p.interiors:
                ax.fill(*h.xy, color='white', lw=0)
        for v in b['vias']:
            ax.add_patch(plt.Circle((v['x'], v['y']), v['r'], fill=False, color='k', lw=0.6))
        if lay == 'F.Cu':
            ax.fill(*PORT_GAP.exterior.xy, color='k')
        ax.set(title=lay, aspect='equal', xlim=REGION.bounds[0::2], ylim=REGION.bounds[1::2])
        ax.tick_params(labelsize=7)
    fig.suptitle('CM5 U702 input loop: +5V (orange), GND (blue), capacitor bridges (green), port (black)', fontsize=10)
    fig.tight_layout()
    fig.savefig(IMAGES / f'{DIR.name}_geometry.png', dpi=130)
    plt.close(fig)


def pcbqs(dmax, sheets=1, backend='dense', face='pair', workers=16):
    from pcbqs.mqs import MQS
    t = time.perf_counter()
    q = MQS(case(), nets(), terminals={k: [('F.Cu', s)] for k, s in SLIVERS.items()}, dmax=dmax, fine=FINE,
            dmax_fine=dmax / 2, conductor_loss='slab', thickness_sheets=sheets, matrix_backend=backend,
            fill_workers=workers, log=lambda *a: None)
    q.face_correction = face
    q.source('port', 'A', 'B', part_h=0.0)
    r = q.solve(F)
    Z = np.asarray(r.zc['port'])
    row = dict(tool='pcbqs', sheets=sheets, backend=backend, face=face if sheets == 2 else None, mesh=dmax,
               n=len(q.tris), R=Z.real, L=Z.imag / (2 * np.pi * F),
               seconds=time.perf_counter() - t)
    print(f"cm5: pcbqs dmax {dmax} sheets {sheets} {backend} {row['face'] or ''}: L {np.round(row['L'] * 1e9, 4)} nH, "
          f"R {np.round(Z.real * 1e3, 3)} mOhm ({row['n']:,} triangles, {row['seconds']:.0f} s)", flush=True)
    return row


def palace(h):
    from mqs_palace import palace_loop
    q = palace_loop(case(), nets(), PORT_GAP, 'x', F, DIR / f'palace_h{h:g}', h_cu=h)
    row = dict(tool='palace', mesh=h, n=q['tets'], L=q['Z'].imag / (2 * np.pi * F), seconds=q['seconds'])
    print(f"cm5: Palace h {h}: L {np.round(row['L'] * 1e9, 4)} nH ({q['tets']:,} tets, {q['seconds']:.0f} s)",
          flush=True)
    return row


def fasthenry_inp(p, nh_outer=5, nh_inner=3):
    """The copper as FastHenry segments on a p-mm grid aligned to the port gap's edges: a node where the grid point is
    in a layer's copper, a p-wide segment between neighbours whose midpoint is. Each via is four bars (the plating's
    area) from layer to layer, joining the node nearest it on each layer whose copper of its net touches its barrel.
    The port's two edges are equipotential lines (.equiv) of nodes."""
    from mqs_fasthenry import via_bars
    b = case()
    zs = {c['name']: (c['z'], c['t']) for c in b['stack']['copper']}
    outer = {min(zs, key=lambda l: zs[l][0]), max(zs, key=lambda l: zs[l][0])}
    x0, y0 = PORT_GAP.bounds[0], 0.5 * (PORT_GAP.bounds[1] + PORT_GAP.bounds[3])
    xb, yb = REGION.bounds[0::2], REGION.bounds[1::2]
    ix = np.arange(np.ceil((xb[0] - x0) / p), np.floor((xb[1] - x0) / p) + 1)
    iy = np.arange(np.ceil((yb[0] - y0) / p), np.floor((yb[1] - y0) / p) + 1)
    X, Y = np.meshgrid(x0 + ix * p, y0 + iy * p, indexing='ij')
    import shapely
    lines, nodes, n_seg = [], {}, 0
    for lay, (z, t) in zs.items():
        polys = b['copper'].get(lay, [])
        if not polys:
            continue
        g = unary_union([shape(e['polygon']) for e in polys]).buffer(1e-6)
        inside = shapely.contains_xy(g, X, Y)
        tag = lay.replace('.', '')
        nh = nh_outer if lay in outer else nh_inner
        idx = {}
        for i, j in zip(*np.nonzero(inside)):
            name = f'N{tag}_{i}_{j}'
            idx[i, j] = name
            lines.append(f'{name} x={X[i, j]:.4f} y={Y[i, j]:.4f} z={z:.5f}')
        for (i, j), a in idx.items():
            for di, dj in ((1, 0), (0, 1)):
                c = idx.get((i + di, j + dj))
                if c and g.contains(Point(X[i, j] + di * p / 2, Y[i, j] + dj * p / 2)):
                    lines.append(f'E{tag}_{i}_{j}_{di} {a} {c} w={p:.4f} h={t:.5f} nhinc={nh}')
                    n_seg += 1
        nodes[lay] = (idx, polys)
    # vias: each joins, layer to layer, the node nearest it on every layer whose copper of its net touches it
    for k, v in enumerate(b['vias']):
        ring = Point(v['x'], v['y']).buffer(v['r'] * 1.05)
        hit = []
        for lay, (idx, polys) in sorted(nodes.items(), key=lambda kv: zs[kv[0]][0]):
            if any(e['net'] == v['net'] and shape(e['polygon']).intersects(ring) for e in polys):
                (i, j) = min(idx, key=lambda ij: (X[ij] - v['x']) ** 2 + (Y[ij] - v['y']) ** 2)
                hit.append((idx[i, j], zs[lay][0]))
        for m, ((a, za), (c, zc)) in enumerate(zip(hit[:-1], hit[1:])):
            lines.append(via_bars(f'V{k}_{m}', v['x'], v['y'], c, a, zc, za, r=v['r']))
    # the port: each edge of the gap an equipotential line of F.Cu nodes
    idx = nodes['F.Cu'][0]
    ends = {}
    for name, xe in (('A', PORT_GAP.bounds[2]), ('B', PORT_GAP.bounds[0])):
        ends[name] = [n for (i, j), n in idx.items() if abs(X[i, j] - xe) < 1e-6 and
                      PORT_GAP.bounds[1] - 1e-6 <= Y[i, j] <= PORT_GAP.bounds[3] + 1e-6]
        assert ends[name], f'no grid node on the port edge {name}'
        if len(ends[name]) > 1:
            lines.append('.equiv ' + ' '.join(ends[name]))
    head = '.units mm\n.default sigma=5.8e4 nwinc=1 rw=2 rh=2\n'
    return head + '\n'.join(lines) + f"\n.external {ends['A'][0]} {ends['B'][0]}\n", n_seg


def fasthenry(p, freq=None, nh_outer=5, nh_inner=3, options=()):
    """FastHenry on the p-mm raster at one frequency (FastHenry is single-threaded: run one process per frequency
    side by side) or all of F; its progress is in <run folder>/fasthenry.log (0.2 mm: 28k segments)."""
    from mqs_fasthenry import fasthenry as run_fh
    t = time.perf_counter()
    inp, n_seg = fasthenry_inp(p, nh_outer, nh_inner)
    band = [freq, freq] if freq else [F[0], F[-1]]
    tol = ''.join(options).replace('-', '').replace(' ', '')
    tag = f'p{p:g}_nh{nh_outer}{nh_inner}' + (f'_{tol}' if tol else '') + (f'_f{freq:.0e}' if freq else '')
    fz, R, L = run_fh('cm5', inp, band, DIR / f'fasthenry_{tag}', options=options)
    keep = np.isin(np.round(np.log10(fz)), np.log10(F))
    row = dict(tool='fasthenry', mesh=p, nh=[nh_outer, nh_inner], options=' '.join(options), n=n_seg,
               f=fz[keep], R=R[keep], L=L[keep],
               seconds=time.perf_counter() - t)
    print(f"cm5: FastHenry {tag}: L {np.round(row['L'] * 1e9, 4)} nH, R {np.round(row['R'] * 1e3, 3)} mOhm "
          f"({n_seg:,} segments, {row['seconds']:.0f} s)", flush=True)
    return row


def fasthenry_merged(rows):
    """FastHenry rows by (grid, filaments), each frequency's run merged into R and L over F (nan where missing)."""
    out = {}
    for x in rows:
        if x['tool'] != 'fasthenry':
            continue
        key = (x['mesh'], tuple(x.get('nh', (5, 3))), x.get('options', ''))
        m = out.setdefault(key, dict(mesh=x['mesh'], nh=key[1], options=key[2],
                                     R=np.full(len(F), np.nan), L=np.full(len(F), np.nan)))
        for fi, R, L in zip(x.get('f', F), x['R'], x['L']):
            k = int(np.argmin(abs(np.log10(F) - np.log10(fi))))
            m['R'][k], m['L'][k] = R, L
    return sorted(out.values(), key=lambda m: (m['mesh'], -sum(m['nh']), m['options']))


def save(rows):
    """Merge rows into result.json (locked: parallel runs finish independently)."""
    import fcntl
    DIR.mkdir(parents=True, exist_ok=True)
    with open(DIR / 'result.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _save(rows)


def _save(rows):
    path = DIR / 'result.json'
    old = json.loads(path.read_text())['rows'] if path.is_file() else []
    key = lambda r: (r['tool'], r['mesh'], r.get('sheets'), r.get('backend'), r.get('face'), str(r.get('nh')),
                     r.get('options', ''),
                     str(r.get('f')) if r['tool'] == 'fasthenry' else '')
    merged = {key(r): r for r in old}
    merged.update({key(r): r for r in rows})
    write_json(path, dict(f=F, rows=list(merged.values())))


def figure():
    """mqs_cm5.png: L at 100 MHz against mesh size (pcbqs one and two face sheets, Palace's PEC sheets, FastHenry's
    finest), and R against frequency (each pcbqs model's finest mesh, FastHenry)."""
    from plots import style, PCBQS, REFERENCE, ANALYTIC
    r = json.loads((DIR / 'result.json').read_text())
    rows = r['rows']
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    one = lambda x: x['tool'] == 'pcbqs' and x['sheets'] == 1 and x['backend'] == 'hierarchical'
    two = lambda x: x['tool'] == 'pcbqs' and x['sheets'] == 2 and x.get('face') == 'overlap'
    series = [('pcbqs, one sheet', PCBQS, 'o-', one), ('pcbqs, two face sheets', PCBQS, 's--', two),
              ('Palace, PEC sheets', REFERENCE, 'D-', lambda x: x['tool'] == 'palace')]
    for label, colour, style_, pick in series:
        rr = sorted((x for x in rows if pick(x)), key=lambda x: -x['mesh'])
        if rr:
            ax[0].plot([x['mesh'] for x in rr], [x['L'][-1] * 1e9 for x in rr], style_, color=colour, label=label)
    fh = fasthenry_merged(rows)
    if fh and np.isfinite(fh[0]['L'][-1]):
        ax[0].axhline(fh[0]['L'][-1] * 1e9, color=ANALYTIC, ls=':', label=f"FastHenry, {fh[0]['mesh']:g} mm grid")
    ax[0].invert_xaxis()
    ax[0].set(xlabel='mesh size (mm): pcbqs dmax, Palace h', ylabel='loop L at 100 MHz (nH)',
              title=f"CM5 U702 input loop{' (lite: F.Cu, In1)' if LAYERS else ''}: inductance")
    ax[0].legend(frameon=False, fontsize=8)
    f = np.array(r['f'])
    for label, colour, style_, pick in series[:2]:
        rr = sorted((x for x in rows if pick(x)), key=lambda x: x['mesh'])
        if rr:
            ax[1].loglog(f, np.array(rr[0]['R']) * 1e3, style_, color=colour, label=f"{label} (dmax {rr[0]['mesh']:g})")
    if fh:
        ax[1].loglog(f, np.array(fh[0]['R']) * 1e3, 'D:', color=ANALYTIC, label='FastHenry')
    ax[1].set(xlabel='Hz', ylabel='R (mOhm)', title='resistance at the port')
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES / f'mqs_{DIR.name}.png', dpi=140)
    plt.close(fig)


def report():
    r = json.loads((DIR / 'result.json').read_text())
    for row in sorted(r['rows'], key=lambda x: (x['tool'], str(x.get('sheets')), str(x.get('face')), x['mesh'])):
        L = np.array(row['L']) * 1e9
        R = np.array(row['R']) * 1e3 if 'R' in row else None
        what = row['tool']
        if row['tool'] == 'pcbqs':
            what += f" sheets {row['sheets']} {row['backend']} {row.get('face') or ''}"
        elif row['tool'] == 'fasthenry':
            what += f" nh {row.get('nh')} {row.get('options', '')} f {np.round(np.log10(row.get('f', F)), 1)}"
        print(f"{what:32s} mesh {row['mesh']:<5g} n {row['n']:>9,}  L {np.round(L, 4)} nH"
              + (f"  R {np.round(R, 3)} mOhm" if R is not None else '') + f"  {row['seconds']:.0f} s")


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'report'
    args = [float(a) for a in sys.argv[2:] if a[0].isdigit() or a[0] == '.']
    if what == 'geometry':
        case()
        geometry()
    elif what == 'pcbqs':                         # pcbqs DMAX [SHEETS] [hier] [overlap]
        save([pcbqs(args[0], int(args[1]) if len(args) > 1 else 1,
                    'hierarchical' if 'hier' in sys.argv else 'dense', 'overlap' if 'overlap' in sys.argv else 'pair')])
    elif what == 'palace':
        save([palace(h) for h in args])
    elif what == 'fasthenry':                     # fasthenry P [FREQ [NH_OUTER NH_INNER]]; FASTHENRY_OPTS='-t 1e-2'
        import os
        opts = tuple(os.environ.get('FASTHENRY_OPTS', '').split())
        save([fasthenry(args[0], *([args[1]] if len(args) > 1 else []), *map(int, args[2:4]), options=opts)])
    elif what == 'figure':
        figure()
    else:
        report()
