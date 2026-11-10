"""Copper R(f) and L(f): pcbqs.mqs.MQS (sheet PEEC, slab loss) against FastHenry (volume-filament PEEC: skin and
proximity effect resolved across width and thickness), on mqs_palace.py's cases.

    python validation/mqs_fasthenry.py [loop plane_return]

FastHenry: `FASTHENRY` or ~/opt/src/FastHenry2/bin/fasthenry (MIT FastHenry 3.0, built with gcc -std=gnu89 -fcommon).
pcbqs's results are read from mqs_palace.py's saved run (its finest dmax), so run that first. The geometry is
written as FastHenry segments: strips w x 35 um, each via four bars on its circle (the plated barrel's area),
the plane a uniform grid.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import OUT, IMAGES, write_json  # noqa: E402

FASTHENRY = os.environ.get('FASTHENRY', str(Path.home() / 'opt/src/FastHenry2/bin/fasthenry'))
T = 0.035


def loop_inp(nw, nh):
    """mqs_palace.case_loop: a 20 x 10 mm (centre lines) loop of 1 mm strip, port across the 0.5 mm gap."""
    pts = [(-0.25, -5), (-10, -5), (-10, 5), (10, 5), (10, -5), (0.25, -5)]
    lines = [f'N{k + 1} x={x} y={y}' for k, (x, y) in enumerate(pts)]
    lines += [f'E{k + 1} N{k + 1} N{k + 2}' for k in range(len(pts) - 1)]
    return (f'.units mm\n.default z={1.6 + T / 2} sigma=5.8e4 w=1 h={T} nhinc={nh} nwinc={nw} rw=2 rh=2\n'
            + '\n'.join(lines) + f'\n.external N1 N{len(pts)}\n')


def via_bars(name, x, y, top, bottom, zt, zb, r=0.15, plating=0.025):
    """A plated via as four square bars on its circle (together the annulus's area, at the barrel's radius: one bar
    of that area is too thin and has too much inductance), their ends shorted to `top` and `bottom`."""
    area = np.pi * (r ** 2 - (r - plating) ** 2) / 4
    s, rc = np.sqrt(area), r - plating / 2
    out, ends_t, ends_b = [], [], []
    for k, (dx, dy) in enumerate(((rc, 0), (-rc, 0), (0, rc), (0, -rc))):
        t, b = f'N{name}T{k}', f'N{name}B{k}'          # FastHenry: node names start with N
        out += [f'{t} x={x + dx:.4f} y={y + dy:.4f} z={zt}', f'{b} x={x + dx:.4f} y={y + dy:.4f} z={zb}',
                f'E{name}{k} {t} {b} w={s:.4f} h={s:.4f}']
        ends_t.append(t)
        ends_b.append(b)
    out += [f'.equiv {top} ' + ' '.join(ends_t), f'.equiv {bottom} ' + ' '.join(ends_b)]
    return '\n'.join(out)


def plane_return_inp(nw, nh, seg=(120, 64)):
    """mqs_palace.case_plane_return: a 1 mm trace from x=-10 to 10 on F.Cu (z 1.6), a via (r 0.15) down to a
    30 x 16 mm B.Cu plane (z 0), back up a via at x=-11.5 to a pad; the port across the 0.5 mm gap at x=-10.25."""
    zt, zb = 1.6 + T / 2, -T / 2
    return f""".units mm
.default sigma=5.8e4 nhinc={nh} nwinc={nw} rw=2 rh=2
G1 x1=-15 y1=-8 z1={zb} x2=15 y2=-8 z2={zb} x3=15 y3=8 z3={zb} thick={T} seg1={seg[0]} seg2={seg[1]}
+ nhinc={nh} relx=0 rely=0 relz=0
+ npa (10,0,{zb})
+ npb (-11.5,0,{zb})
NT0 x=-10 y=0 z={zt}
NT1 x=10 y=0 z={zt}
NP0 x=-10.5 y=0 z={zt}
NP1 x=-11.5 y=0 z={zt}
ET NT0 NT1 w=1 h={T}
EP NP0 NP1 w=1.2 h={T}
{via_bars('VA', 10, 0, 'NT1', 'npa', zt, zb)}
{via_bars('VB', -11.5, 0, 'NP1', 'npb', zt, zb)}
.external NT0 NP0
"""


CASES = dict(loop=loop_inp, plane_return=plane_return_inp)


def fasthenry(name, inp, f, folder, timeout=None, options=()):
    """Run FastHenry on `inp` in `folder`; its console output goes to fasthenry.log as it runs (a board-sized model
    takes hours: watch it there). `options`: extra command-line flags (e.g. -t 1e-2 -b 1e-1, looser GMRES
    tolerances than the defaults 1e-3 and 1e-2). Returns (f, R, L) from Zc.mat."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'Zc.mat').unlink(missing_ok=True)      # never read a previous run's result
    # FastHenry reads the first line as the title: without one it swallows `.units mm` (and solves in metres)
    (folder / f'{name}.inp').write_text(f'* pcbqs validation: {name}\n' + inp
                                        + f'.freq fmin={f[0]:g} fmax={f[-1]:g} ndec=1\n.end\n')
    with open(folder / 'fasthenry.log', 'w') as log:
        line = ['stdbuf', '-oL'] if shutil.which('stdbuf') else []      # its stdio buffers 4 KB when not a tty
        subprocess.run([*line, FASTHENRY, *options, f'{name}.inp'], cwd=folder, stdout=log, stderr=subprocess.STDOUT,
                       timeout=timeout)
    text = (folder / 'Zc.mat').read_text()           # FastHenry exits non-zero even when it has finished
    rows = [tuple(map(float, m)) for m in
            re.findall(r'frequency = ([0-9.e+-]+) 1 x 1\s+([-0-9.e+]+)\s+([-0-9.e+]+)j', text)]
    if not rows:
        raise RuntimeError('FastHenry gave no impedance; see ' + str(folder / 'fasthenry.log'))
    fz, R, X = map(np.array, zip(*rows))
    return fz, R, X / (2 * np.pi * fz)


def run(name, levels=((9, 3), (17, 5), (31, 9))):
    f = np.array([1e6, 1e7, 1e8])
    saved = json.loads((OUT / 'mqs' / name / 'result.json').read_text())
    pc = [r for r in saved['rows'] if r['tool'] == 'pcbqs'][-1]
    pal = [r for r in saved['rows'] if r['tool'] == 'palace'][-1]
    rows = []
    for nw, nh in levels:
        fz, R, L = fasthenry(name, CASES[name](nw, nh), [1e3, 1e8], OUT / 'fasthenry' / f'{name}_{nw}x{nh}')
        keep = np.isin(np.round(np.log10(fz)), np.log10(f))
        rows.append(dict(nwinc=nw, nhinc=nh, f=fz[keep], R=R[keep], L=L[keep], f_all=fz, R_all=R, L_all=L))
        print(f'{name}: FastHenry {nw}x{nh}: R {np.round(R[keep] * 1e3, 2)} mOhm, L {np.round(L[keep] * 1e9, 3)} nH', flush=True)
    fh = rows[-1]
    print(f'{name}: pcbqs dmax {pc["mesh"]}: R {np.round(np.array(pc["R"]) * 1e3, 2)} mOhm, L {np.round(np.array(pc["L"]) * 1e9, 3)} nH; '
          f'Palace (PEC) L {np.round(np.array(pal["L"]) * 1e9, 3)} nH', flush=True)
    print(f'{name}: pcbqs vs FastHenry: R {np.round((np.array(pc["R"]) / fh["R"] - 1) * 100, 1)}%, '
          f'L {np.round((np.array(pc["L"]) / fh["L"] - 1) * 100, 2)}%', flush=True)
    write_json(OUT / 'fasthenry' / name / 'result.json', dict(case=name, f=f, fasthenry=rows, pcbqs=pc, palace_L=pal['L']))
    return f, pc, pal, rows


def figure(results):
    from plots import style, PCBQS, REFERENCE, ANALYTIC
    plt = style()
    fig, axs = plt.subplots(2, len(results), figsize=(4.6 * len(results), 6.2), squeeze=False)
    for j, (name, (f, pc, pal, rows)) in enumerate(results.items()):
        fh = rows[-1]
        axs[0, j].loglog(fh['f_all'], fh['R_all'] * 1e3, 's-', color=REFERENCE, label=f'FastHenry {fh["nwinc"]}x{fh["nhinc"]}')
        axs[0, j].loglog(f, np.array(pc['R']) * 1e3, 'o-', color=PCBQS, label=f'pcbqs (dmax {pc["mesh"]:g})')
        axs[0, j].set(ylabel='R (mOhm)', title=f'{name}: resistance at the port')
        axs[1, j].semilogx(fh['f_all'], fh['L_all'] * 1e9, 's-', color=REFERENCE, label='FastHenry')
        axs[1, j].semilogx(f, np.array(pc['L']) * 1e9, 'o-', color=PCBQS, label='pcbqs')
        axs[1, j].axhline(pal['L'][-1] * 1e9, color=ANALYTIC, ls='--', lw=1, label='Palace, PEC sheets')
        axs[1, j].set(xlabel='Hz', ylabel='L (nH)', title=f'{name}: inductance at the port')
        for ax in axs[:, j]:
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES / 'mqs_fasthenry.png', dpi=140)
    plt.close(fig)


def saved(name):
    """run()'s result for `name`, read back from its result.json."""
    r = json.loads((OUT / 'fasthenry' / name / 'result.json').read_text())
    rows = [{k: np.array(v) if isinstance(v, list) else v for k, v in row.items()} for row in r['fasthenry']]
    return np.array(r['f']), r['pcbqs'], dict(L=r['palace_L']), rows


if __name__ == '__main__':
    if sys.argv[1:2] == ['--plot']:
        figure({name: saved(name) for name in sys.argv[2:] or list(CASES)})
    else:
        names = sys.argv[1:] or list(CASES)
        res = {name: run(name) for name in names}
        figure({name: res.get(name) or saved(name) for name in CASES if name in res or (OUT / 'fasthenry' / name).is_dir()})
