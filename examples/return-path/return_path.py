"""
Where the return current goes when a trace changes layers, as low-frequency S-parameters (pcbqs.sparams).

    python examples/return-path/return_path.py [--out DIR] [--figure docs/images/sparams-return.png]

A 0.35 mm trace runs 15 mm on F.Cu over the In1 ground plane, drops through a via and runs 15 mm on B.Cu over the
In2 ground plane: a 40 x 20 mm, four-layer board, a port at each end between the trace and the plane under it.
The two ground planes are joined by one via: beside the signal via, 8 mm away, at the board's corner, or not at
all. The script writes each board as a kicad2ems board file (DIR/<name>_board.json: run pcbqs.sparams on it
yourself), solves them, prints the path each makes between the ports and draws the figure of docs/sparams.md.
"""
import argparse
import json
import os
import tempfile

import numpy as np
from shapely.geometry import Point, box, mapping

from pcbqs import sparams

W, T = 0.35, 0.035
Z = {'B.Cu': 0.0, 'In2.Cu': 0.2, 'In1.Cu': 1.3, 'F.Cu': 1.5}
VIAS = {'beside': [(21.2, 10.0)], '8 mm away': [(20.0, 2.0)], 'at the corner': [(38.0, 18.0)], 'none': []}
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#6b6a63', '#e1e0d9', '#fcfcfb'
SERIES = {'beside': '#2a78d6', '8 mm away': '#eb6834', 'at the corner': '#1baf7a'}
RAMP = [SURFACE, '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


def board(ground_vias):
    plane = box(0, 0, 40, 20).difference(Point(20, 10).buffer(0.7, 16))
    copper = {'F.Cu': [dict(net='SIG', polygon=mapping(box(5, 10 - W / 2, 20.4, 10 + W / 2)))],
              'B.Cu': [dict(net='SIG', polygon=mapping(box(19.6, 10 - W / 2, 35, 10 + W / 2)))],
              'In1.Cu': [dict(net='GND', polygon=mapping(plane))], 'In2.Cu': [dict(net='GND', polygon=mapping(plane))]}
    vias = [dict(x=20.0, y=10.0, r=0.15, z0=0.0, z1=1.5, net='SIG')]
    vias += [dict(x=x, y=y, r=0.15, z0=0.2, z1=1.3, net='GND') for x, y in ground_vias]
    port = lambda name, layer, ref, x, sgn: dict(name=name, layer=layer, ref_layer=ref, c=[x, 10.0], axis=0, sgn=sgn,
                                                 w=W, z0=50.0, net='SIG', kind='z')
    return dict(stack=dict(copper=[dict(name=n, z=z, t=T) for n, z in sorted(Z.items(), key=lambda kv: kv[1])],
                           layers=[dict(name=n, z0=a, z1=b, er=4.3, tand=0.02, kind='diel')
                                   for n, a, b in (('d3', 0, 0.2), ('d2', 0.2, 1.3), ('d1', 1.3, 1.5))], thickness=1.5),
                copper=copper, vias=vias, outline=mapping(box(0, 0, 40, 20)), ground_nets=['GND'],
                prims=dict(footprints=[]), ports=[port('P1', 'F.Cu', 'In1.Cu', 5.0, 1), port('P2', 'B.Cu', 'In2.Cu', 35.0, -1)])


def plane_current(b, r, layer, f):
    """|J| on a layer with P1 driven at the solve's frequency nearest f: (x, y, A/mm per ampere)."""
    with tempfile.TemporaryDirectory() as tmp:
        v = sparams.write_maps(tmp, b, r, 'P1', [f], log=lambda *a: None)
        with np.load(os.path.join(tmp, next(d['file'] for d in v['dumps'] if d['layer'] == layer))) as d:
            return d['x'], d['y'], np.linalg.norm(np.abs(d['data'][0, :, :, 0, :2]), axis=-1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--out', default='return-path-out')
    ap.add_argument('--figure', default='')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    quiet = dict(log=lambda *x: None)
    f = np.geomspace(1e3, 1e7, 25)
    L, maps = {}, {}
    print(f"{'ground via':<16s}{'R at 1 kHz':>12s}{'L at 1 kHz':>12s}{'L at 1 MHz':>12s}{'L at 10 MHz':>13s}")
    for name, vias in VIAS.items():
        b = board(vias)
        with open(os.path.join(a.out, f"{name.replace(' ', '_')}_board.json"), 'w', encoding='utf-8') as fh:
            json.dump(b, fh)
        r = sparams.solve(b, f, **quiet)
        p, = sparams.paths(r['f'], r['S'], r['names'], r['zref'])
        if not p['joined']:
            print(f"{name:<16s}  {sparams.apart_note(p)}")
            continue
        L[name] = np.array(p['L_H']) * 1e9
        at = lambda hz: L[name][np.argmin(np.abs(f - hz))]
        print(f"{name:<16s}{p['R_ohm'][0] * 1e3:9.1f} mΩ{at(1e3):9.1f} nH{at(1e6):9.1f} nH{at(1e7):10.1f} nH")
        if name in ('beside', 'at the corner') and a.figure:              # the maps: one frequency, a finer mesh
            maps[name] = plane_current(b, sparams.solve(b, [1e6], dmax=0.8, **quiet), 'In1.Cu', 1e6)
    fc = np.geomspace(1e3, 1e9, 25)
    open_ = sparams.solve(board([]), fc, dmax=3.0, fine=0.7, **quiet)
    s21 = 20 * np.log10(np.abs(open_['S'][:, 1, 0]))
    p, = sparams.paths(open_['f'], open_['S'], open_['names'], open_['zref'])
    print(f"{'none, with C':<16s}  |S21| {s21[0]:.0f} dB at 1 kHz, {s21[np.argmin(np.abs(fc - 1e6))]:.0f} dB at 1 MHz, "
          f"{s21[-1]:.1f} dB at 1 GHz; {p['C_F'] * 1e12:.0f} pF closes the path")
    if a.figure:
        draw(a.figure, f, L, fc, s21, p['C_F'], maps)
        print(f"wrote {a.figure}")
    return 0


def draw(path, f, L, fc, s21, C, maps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': MUTED, 'text.color': INK,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                         'axes.titlelocation': 'left', 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE})
    fig, ax = plt.subplot_mosaic([['L', 'S'], ['beside', 'at the corner']], figsize=(10.5, 7.0), layout='constrained',
                                 gridspec_kw=dict(height_ratios=[1.0, 1.0]))
    for x in (ax['L'], ax['S']):
        x.set_xscale('log')
        x.grid(True, which='major', color=GRID, lw=0.6)
        x.spines[['top', 'right']].set_visible(False)
        x.set_xlabel('frequency (Hz)')
        x.minorticks_off()
    x = ax['L']
    for name, col in SERIES.items():
        x.plot(f, L[name], color=col, lw=2, label=name)
        x.annotate(f"{name}  {L[name][-1]:.1f} nH", (f[-1], L[name][-1]), xytext=(6, 0), textcoords='offset points',
                   va='center', color=INK, fontsize=8.5)
    x.set_xlim(f[0], f[-1] * 40)
    x.set_xticks([1e3, 1e4, 1e5, 1e6, 1e7])
    x.set_ylabel('loop inductance between the ports (nH)')
    x.set_title('One ground via: its distance is loop inductance')
    x.legend(title='the ground via', frameon=False, loc='lower left', fontsize=8.5, title_fontsize=8.5)
    x = ax['S']
    x.plot(fc, s21, color=SERIES['beside'], lw=2)
    x.set_xticks([1e3, 1e5, 1e7, 1e9])
    x.set_ylabel('|S21| (dB)')
    x.set_title('No ground via: the planes’ capacitance closes the path')
    k = int(np.argmin(np.abs(fc - 1e6)))
    x.annotate(f"{C * 1e12:.0f} pF between the two planes\ncarries the return: {s21[k]:.0f} dB at 1 MHz", (fc[k], s21[k]),
               xytext=(18, -34), textcoords='offset points', color=INK, fontsize=8.5,
               arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    cmap = LinearSegmentedColormap.from_list('blue', RAMP)
    top = max(m[2].max() for m in maps.values())
    norm = LogNorm(top / 300, top / 3, clip=True)
    for name, (gx, gy, j) in maps.items():
        x = ax[name]
        im = x.pcolormesh(gx, gy, np.maximum(j.T, top / 1e4), cmap=cmap, norm=norm, shading='auto', rasterized=True)
        x.set_aspect('equal')
        x.set_xticks([]), x.set_yticks([])
        for sp in x.spines.values():
            sp.set_color('#c3c2b7')
        x.plot([5, 20], [10, 10], color=INK, lw=0.8, ls=(0, (4, 3)))
        x.plot([20, 35], [10, 10], color=MUTED, lw=0.8, ls=(0, (1, 2)))
        gv = VIAS[name][0]
        x.plot(*gv, 'o', ms=6, mfc='none', mec=INK, mew=1.2)
        x.annotate('ground via', gv, xytext=(0, -13), textcoords='offset points', ha='center', fontsize=8, color=INK)
        x.annotate('trace on F.Cu', (6, 10), xytext=(0, 6), textcoords='offset points', fontsize=8, color=INK)
        x.annotate('on B.Cu', (29, 10), xytext=(0, 6), textcoords='offset points', fontsize=8, color=MUTED)
        x.set_title(f"Via {name}: the In1 plane’s current, 1 MHz")
    cb = fig.colorbar(im, ax=[ax['beside'], ax['at the corner']], location='bottom', shrink=0.45, aspect=40, pad=0.04)
    cb.set_label('sheet current in the plane (A/mm per ampere into the port)', color=MUTED)
    cb.outline.set_visible(False)
    fig.savefig(path, dpi=130, facecolor=SURFACE)


if __name__ == '__main__':
    raise SystemExit(main())
