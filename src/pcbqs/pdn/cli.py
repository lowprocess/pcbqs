"""
python -m pcbqs.pdn: a power rail's DC drop and PDN impedance from a KiCad board (or a kicad2ems board.json).

    python -m pcbqs.pdn board.kicad_pcb --rail +3V3 --volts 3.3 --load U21=2 --load U14=1 --out pdn_3v3
    python -m pcbqs.pdn board.kicad_pcb --rail +3V3 --volts 3.3 --spread-load 5 --out pdn_3v3   # 5 A shared by pad count
    python -m pcbqs.pdn --describe --json          # the options, for the GUI's PDN page

It writes <out>/pdn.json (the rail tree, each load's voltage and drop, the series parts, the hottest vias, the plane
pairs, each part's connection and mount inductance, the impedance each load sees and where it is over the target,
the notes), <out>/z.csv, and the figures <out>/dc_power_<layer>.png, dc_ground_<layer>.png (drop or rise, current
per mm, vias by current) and pdn_z.png. Its progress goes to stdout, one line per step ('pdn: ...').
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from .._cli import add_options, describe as describe_options, load_board, print_describe

G_RAIL, G_DC, G_PDN, G_MESH = 'Rail', 'DC', 'PDN impedance', 'Mesh'
OPTIONS = [
    # key, kind, default, help, group (pcbqs._cli: the flag is --key)
    ('rail', 'str', '', "the rail's net (the tree grows from it through series parts)", G_RAIL),
    ('ground', 'str', 'GND', "its ground net(s), comma-separated", G_RAIL),
    ('volts', 'float', 3.3, "the rail's voltage at its source", G_RAIL),
    ('source', 'str', '', "the part it is fed from (default: the power inductor feeding the tree)", G_RAIL),
    ('ground_source', 'str', '', "the part whose ground pads its return goes back to (default: "
     "the parts on that inductor's switch node)", G_RAIL),
    ('load', 'list', [], "a load: REF=AMPS (repeat)", G_RAIL),
    ('spread_load', 'float', 0.0, "amps shared by +rail pad count over every part but R, C, L "
     "(an assumption, for a first look when the loads are not known)", G_RAIL),
    ('max_ohms', 'float', 1.0, "the largest resistor the tree follows (sense and 0 ohm links)", G_RAIL),
    ('through', 'str', '', "parts the tree follows whatever they are, comma-separated (an inductor "
     "named here is a series filter inductor: its inductance and DCR stay in the model)", G_RAIL),
    ('dcr', 'list', [], "a series part's DC resistance: REF=OHMS (repeat); overrides its value or the "
     "assumed default", G_RAIL),
    ('cap', 'list', [], "a capacitor's datasheet values: REF=C[,ESR[,ESL]] in F, ohms, H (repeat); overrides "
     "its value field and the package defaults", G_PDN),
    ('stop', 'str', '', "parts the tree never follows, comma-separated", G_RAIL),
    ('plating', 'float', 0.025, "via plating thickness, mm", G_DC),
    ('dmax', 'float', 1.0, "the largest mesh cell, mm (finer at the source and the loads)", G_MESH),
    ('plane_min', 'float', 0.0, "smallest rail or ground copper on a layer counted as a plane, mm^2 "
     "(0: a tenth of the board's area; small rails need less)", G_PDN),
    ('vrm_r', 'float', 1e-3, "the regulator's output resistance, ohms", G_PDN),
    ('vrm_l', 'float', 30e-9, "the regulator's output inductance (its loop's bandwidth), H", G_PDN),
    ('fmin', 'float', 1e4, "lowest frequency, Hz", G_PDN),
    ('fmax', 'float', 2e9, "highest frequency, Hz", G_PDN),
    ('points', 'int', 160, "frequencies (log-spaced)", G_PDN),
    ('target', 'float', 0.0, "the target impedance, ohms (0: volts x ripple / step)", G_PDN),
    ('ripple', 'float', 0.05, "the allowed ripple, a fraction of the voltage", G_PDN),
    ('step', 'float', 0.0, "the load step, A (0: half the loads' current)", G_PDN),
    ('no_pdn', 'bool', False, "the DC drop only", G_PDN),
]


def describe():
    return describe_options(OPTIONS)


def parser():
    ap = argparse.ArgumentParser(prog='python -m pcbqs.pdn', description=__doc__.split('\n\n')[0].strip())
    ap.add_argument('board', nargs='?', help='.kicad_pcb, or a kicad2ems <name>_board.json')
    ap.add_argument('--out', default='pdn', help='output folder')
    ap.add_argument('--describe', action='store_true', help='print the options and exit')
    ap.add_argument('--json', action='store_true', help='with --describe: as JSON')
    add_options(ap, OPTIONS)
    return ap


def say(*a):
    print('pdn:', *a, flush=True)



def spread(board, rail_net, amps):
    users = {f['ref']: sum(p.get('net') == rail_net for p in f['pads']) for f in board['prims']['footprints']
             if not f.get('dnp') and any(p.get('net') == rail_net for p in f['pads'])
             and not f['ref'].upper().startswith(('C', 'R', 'L', 'FB'))}
    tot = sum(users.values())
    return {r: amps * n / tot for r, n in users.items()} if tot else {}


def ref_values(items, flag, form, most=1):
    """{REF: [values]} of REF=V[,V...] options (`most` values, each positive); SystemExit naming the bad one."""
    out = {}
    for item in items:
        ref, _, rest = item.partition('=')
        try:
            vals = [float(x) for x in rest.split(',') if x.strip()]
        except ValueError:
            vals = []
        if not ref.strip() or not 1 <= len(vals) <= most or any(not np.isfinite(v) or v <= 0 for v in vals):
            raise SystemExit(f'pdn: error: {flag} {item!r}: expected {form}, positive')
        out[ref.strip()] = vals
    return out


def bands(f, z, target):
    over = np.abs(z) > target
    edges = np.flatnonzero(np.diff(np.r_[0, over.astype(int), 0]))
    return [[float(f[a]), float(f[min(b, len(f)) - 1])] for a, b in zip(edges[::2], edges[1::2])]


def run(a):
    from .rail import Rail
    from .dc import solve_rail
    t0 = time.time()
    os.makedirs(a.out, exist_ok=True)
    ground = tuple(x.strip() for x in a.ground.split(',') if x.strip())
    if not a.rail:
        raise SystemExit('pdn: --rail is required')
    if a.plane_min < 0:                                 # every option checked before the (slow) board import
        raise SystemExit('pdn: error: --plane-min must be nonnegative')
    loads = {r: v[0] for r, v in ref_values(a.load, '--load', 'REF=AMPS').items()}
    dcr = {r: v[0] for r, v in ref_values(a.dcr, '--dcr', 'REF=OHMS').items()}
    caps = {r: dict(zip(('C', 'ESR', 'ESL'), v)) for r, v in ref_values(a.cap, '--cap', 'REF=C[,ESR[,ESL]]', 3).items()}
    say(f'reading {os.path.basename(a.board)}')
    board = load_board(a.board, {a.rail, *ground})
    assumed = False
    if not loads and a.spread_load > 0:
        loads, assumed = spread(board, a.rail, a.spread_load), True
    if not loads:
        raise SystemExit('pdn: no loads: --load REF=AMPS, or --spread-load AMPS')
    kw = dict(ground=ground, loads=loads, source=a.source or None, ground_source=a.ground_source or None,
              max_ohms=a.max_ohms, through=tuple(x for x in a.through.split(',') if x),
              stop=tuple(x for x in a.stop.split(',') if x), dcr=dcr)
    rail = Rail(board, a.rail, **kw)
    if not a.board.endswith('.json') and not rail.nets <= {a.rail}:      # the tree's other nets: read with it
        say(f'rail tree: {len(rail.nets)} nets; reading their copper')
        board = load_board(a.board, set(rail.nets) | set(ground))
        rail = Rail(board, a.rail, **kw)
    for line in rail.describe():
        say(line.strip())
    say('DC: meshing and solving')
    t = time.time()
    dc = solve_rail(rail, a.volts, dmax=a.dmax, plating=a.plating, log=say)
    t_dc = time.time() - t
    out = dict(format=1, rail=a.rail, nets=sorted(rail.nets), ground=list(ground), volts=a.volts, assumed_loads=assumed,
               settings={k: getattr(a, k) for k, *_ in OPTIONS},
               tree=[dict(ref=s.part.ref, kind=s.kind, R=s.R, L=s.L, a=s.a, b=s.b, defaulted=s.defaulted) for s in rail.series],
               source=[p.ref for p in rail.source_parts],
               loads={r: dict(I=v['I'], V=v['V'], drop=v['drop'], power_drop=v['power_drop'], ground_rise=v['ground_rise'])
                      for r, v in dc['loads'].items()},
               series=dict(dc['series']),
               loss_W=dc['power'].loss + dc['ground'].loss,
               vias={side: sorted([dict(x=v['x'], y=v['y'], I=v['I']) for v in dc[side].vias], key=lambda v: -v['I'])[:20]
                     for side in ('power', 'ground')},
               vertices=dict(power=dc['power'].nodes, ground=dc['ground'].nodes), seconds=dict(dc=round(t_dc, 1)),
               notes=list(rail.notes) + dc['notes'])
    say(f"DC done in {t_dc:.1f} s: worst drop {max(v['drop'] for v in out['loads'].values()) * 1e3:.2f} mV")
    figs = dc_figures(board, dc, a.volts, a.out)
    out['figures'] = figs
    if not a.no_pdn:
        from .stack import RailPDN
        from .cavity import capacitance
        say('PDN: plane pairs, parts, mesh')
        t = time.time()
        pdn = RailPDN(rail, dmax=a.dmax, plane_min=a.plane_min or None, vrm=dict(R=a.vrm_r, L=a.vrm_l), caps=caps, log=say)
        t_b = time.time() - t
        f = np.logspace(np.log10(a.fmin), np.log10(a.fmax), a.points)
        say(f'PDN: {len(f)} frequencies')
        t = time.time()
        obs = [r for r in loads if pdn.dof(r) is not None]
        Z = pdn.z(f, observe=obs)
        Z0 = pdn.z(f, observe=obs, bare=True)
        t_z = time.time() - t
        step = a.step or 0.5 * sum(loads.values())
        target = a.target or a.volts * a.ripple / step
        out['pdn'] = dict(f=f.tolist(), target=target,
                          z={r: dict(mag=np.abs(Z[r]).tolist(), deg=np.degrees(np.angle(Z[r])).tolist(),
                                     bare=np.abs(Z0[r]).tolist(), over=bands(f, Z[r], target)) for r in obs},
                          cavities=[dict(rail=c.rail_layer, ground=c.ground_layer, d=c.d, er=c.er, area=c.geom.area,
                                         C=capacitance(c.geom.area, c.d, c.er)) for c in pdn.cavities],
                          parts={n: dict(role=i['role'], mount_nH=i['mount'] * 1e9, node=i['rail'].node,
                                         rail=i['rail'].why, gnd=i['gnd'].why,
                                         element={k: v for k, v in (i['el'] or {}).items() if k != 'defaulted'},
                                         defaulted=bool((i['el'] or {}).get('defaulted')))
                                 for n, i in pdn.parts.items()},
                          vertices=pdn.nv)
        out['notes'] += pdn.notes
        out['seconds'].update(pdn_build=round(t_b, 1), pdn_sweep=round(t_z, 1))
        with open(os.path.join(a.out, 'z.csv'), 'w') as fh:
            fh.write('f_Hz,' + ','.join(f'{r}_mag,{r}_deg' for r in obs) + '\n')
            for i in range(len(f)):
                fh.write(f'{f[i]:.6g},' + ','.join(f'{abs(Z[r][i]):.6g},{np.degrees(np.angle(Z[r][i])):.3f}' for r in obs) + '\n')
        figs.append(z_figure(f, Z, Z0, target, a.out))
        worst = max(obs, key=lambda r: np.abs(Z[r]).max()) if obs else None
        if worst:
            say(f'PDN done in {t_b + t_z:.1f} s; target {target * 1e3:.1f} mOhm; {worst} over it at '
                f'{len(out["pdn"]["z"][worst]["over"])} band(s)')
    out['seconds']['total'] = round(time.time() - t0, 1)
    with open(os.path.join(a.out, 'pdn.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    say(f'wrote {os.path.join(a.out, "pdn.json")}')
    return out


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def dc_figures(board, dc, volts, out):
    """dc_power_<layer>.png and dc_ground_<layer>.png for the layers carrying most of the current: the drop (or the
    rise), the current per mm, the vias by their current."""
    from matplotlib.tri import Triangulation
    from shapely.geometry import shape
    plt = _plt()
    x0, y0, x1, y1 = shape(board['outline']).bounds
    files = []
    for side, res in (('power', dc['power']), ('ground', dc['ground'])):
        carry = sorted(res.sheets.items(), key=lambda kv: -float(np.nansum(kv[1].J)))
        for lay, sh in carry[:2]:
            tri = Triangulation(sh.nodes[:, 0], sh.nodes[:, 1], sh.tris)
            fig, ax = plt.subplots(1, 2, figsize=(14, 4.6 * (y1 - y0) / max(x1 - x0, 1) + 1.2))
            v = (volts - sh.V) * 1e3 if side == 'power' else sh.V * 1e3
            m = ax[0].tripcolor(tri, np.nan_to_num(v), shading='gouraud', cmap='viridis' if side == 'power' else 'cividis')
            fig.colorbar(m, ax=ax[0], label='mV')
            ax[0].set_title(f"{lay}: {'drop from the source' if side == 'power' else 'rise above the source'}, mV", fontsize=10)
            j = sh.J
            n = ax[1].tripcolor(tri, facecolors=j, cmap='magma', vmax=float(np.percentile(j, 99.7)) if len(j) else 1)
            fig.colorbar(n, ax=ax[1], label='A/mm')
            I = np.array([q['I'] for q in res.vias])
            if len(I) and I.max() > 0:
                XY = np.array([(q['x'], q['y']) for q in res.vias])
                ax[1].scatter(XY[:, 0], XY[:, 1], s=2 + 50 * I / I.max(), facecolors='none', edgecolors='c', lw=0.5)
            ax[1].set_title(f"{lay}: current per mm; vias by current (max {I.max() if len(I) else 0:.2f} A)", fontsize=10)
            for q in ax:
                q.set_aspect('equal'); q.set_xlim(x0, x1); q.set_ylim(y0, y1); q.set_xticks([]); q.set_yticks([])
            fig.tight_layout()
            name = f"dc_{side}_{lay.replace('.', '_')}.png"
            fig.savefig(os.path.join(out, name), dpi=110)
            plt.close(fig)
            files.append(dict(file=name, side=side, layer=lay))
    return files


def z_figure(f, Z, Z0, target, out):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k, (r, z) in enumerate(Z.items()):
        ax.loglog(f, np.abs(z), c=f'C{k % 10}', lw=1.8, label=r)
        ax.loglog(f, np.abs(Z0[r]), ':', c=f'C{k % 10}', lw=0.8)
    ax.axhline(target, c='r', lw=1, label=f'target {target * 1e3:.1f} mOhm')
    ax.set_xlabel('Hz'); ax.set_ylabel('|Z| (Ohm)'); ax.grid(True, which='both', alpha=0.3)
    ax.set_title('impedance each load sees (dotted: the planes alone)', fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'pdn_z.png'), dpi=110); plt.close(fig)
    return dict(file='pdn_z.png', side='pdn', layer='')


def main(argv=None):
    a = parser().parse_args(argv)
    if a.describe:
        print_describe(describe(), a.json)
        return 0
    if not a.board:
        parser().error('a board is required')
    try:
        run(a)
    except (ValueError, KeyError) as e:
        print(f'pdn: error: {e}', flush=True)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
