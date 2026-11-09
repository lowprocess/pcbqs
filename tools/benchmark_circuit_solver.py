"""Switching-converter transient benchmark: ngspice vs Xyce, solver options, accuracy vs a tight reference.

The circuit mirrors the wpthb power path: 24 V source (50 mOhm, 100 nH), R9 5 mOhm, VBUS polymer bulk, L3 5.6 uH,
VHB ceramics, a phase-shifted full bridge (dead time 100 ns, 20 ns edges), 99 nF + 24 uH series tank into a
reflected-load resistor. Switches are either pcbqs's ngspice VDMOS card or a smooth gate-controlled conductance
(behavioural, runs in both simulators) with a body diode and Coss.

    python tools/benchmark_circuit_solver.py --out bench [case-filter ...]

Results (wall time, accepted/rejected steps, Newton iterations, last-cycle input power, coil RMS, VHB p-p) go to
<out>/results.json. Xyce cases are skipped when Xyce is not installed.
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pcbqs.power_spice import options_line  # noqa: E402

HERE = Path('circuit-solver-bench')
F = 110e3
T = 1 / F
PHASE = 90.0
DEAD = 100e-9
EDGE = 20e-9


def passives(sim):
    """Common netlist body; returns lines. Nodes: vin, vbus, vhb, a, b, gnd=0."""
    L = []
    add = L.append
    add('VSUP src 0 DC 24')
    add('RSRC src s1 0.05')
    add('LSRC s1 vin 100n')
    add('R9 vin vbus 5m')
    for k in range(4):                                   # VBUS polymer bulk
        add(f'RCA{k} vbus ca{k} 20m'); add(f'LCA{k} ca{k} cb{k} 1n'); add(f'CCA{k} cb{k} 0 47u IC=24')
    add('RL3 vbus l3 30m'); add('LL3 l3 vhb 5.6u')
    for k, c in enumerate(('10u', '10u', '10u', '10u', '1u', '1u')):   # VHB ceramics
        add(f'RCV{k} vhb cv{k} 5m'); add(f'LCV{k} cv{k} cw{k} 0.8n'); add(f'CCV{k} cw{k} 0 {c} IC=24')
    add('RT1 a t1 5m'); add('CT t1 t2 99n')              # series tank: 99 nF, coil 24 uH + 110 mOhm + Cp, load
    add('RCOIL t2 t3 0.11'); add('LCOIL t3 t4 24u'); add('CPAR t2 t4 17.8p')
    add('RLOAD t4 b 2.2')
    return L


def gates():
    """(name, drain, source, delay) for the four switches; legs A (a) and B (b), B shifted by PHASE."""
    half = T / 2
    shift = PHASE / 360 * T
    on = half - DEAD - EDGE
    out = []
    for leg, node, d0 in (('A', 'a', 0.0), ('B', 'b', shift)):
        out.append((f'H{leg}', 'vhb', node, d0 + DEAD, on))
        out.append((f'L{leg}', node, '0', (d0 + half + DEAD) % T, on))
    return out


def netlist(sim, model, opts, cycles, out):
    L = [f'* bench {sim} {model}'] + passives(sim)
    for name, d, s, delay, on in gates():
        g = f'g{name}'
        L.append(f'VG{name} {g} {s} PULSE(0 10 {delay:.12g} {EDGE:g} {EDGE:g} {on:.12g} {T:.12g})')
        if model == 'vdmos':
            L.append(f'M{name} {d} {g} {s} NMQ')
        else:                                            # smooth switch: g(Vgs) = goff + gon / (1 + exp(-(Vgs-4)/0.3))
            expr = f'V({d},{s})*(1e-9 + 125/(1+exp(-(V({g},{s})-4)/0.3)))'
            L.append(f'B{name} {d} {s} I={{{expr}}}' if sim == 'xyce' else f'B{name} {d} {s} I={expr}')
            L.append(f'D{name} {s} {d} DBODY')
            L.append(f'C{name} {d} {s} 500p')
            L.append(f'RG{name} {g} {s} 1e6')
    if model == 'vdmos':
        L.append('.model NMQ VDMOS(VTO=3 KP=17.857142857142858 RDS=1e10 CGDMIN=5e-11 CGDMAX=5e-11 CGS=1e-9 '
                 'CJO=5e-10 TT=2e-08 RB=0.02 IS=1e-12)')
    L.append('.model DBODY D(IS=1e-12 RS=0.02 TT=20n CJO=10p)')
    stop = cycles * T
    tmax = T / opts.get('samples', 256)
    if sim == 'ngspice':
        L.append('.options ' + opts['options'])
        L += ['.control', *(['set klu'] if opts.get('klu', True) else []), 'set num_threads=1',
              'save v(vhb) i(VSUP) i(LCOIL) v(a) v(b)',
              f"tran {tmax:.12g} {stop:.12g} 0 {tmax:.12g}" + (' uic' if opts.get('uic', True) else ''),
              f'meas tran iavg AVG i(VSUP) from={stop - T:.12g} to={stop:.12g}',
              f'meas tran irms RMS i(LCOIL) from={stop - T:.12g} to={stop:.12g}',
              f'meas tran vpp PP v(vhb) from={stop - T:.12g} to={stop:.12g}',
              f'meas tran ipk MAX i(LCOIL) from={stop - T:.12g} to={stop:.12g}',
              'rusage all', 'quit', '.endc']
    else:
        L.append(f'.tran {tmax:.12g} {stop:.12g}' + (' uic' if opts.get('uic', True) else ''))
        L.append(f'.options timeint {opts["options"]}')
        if opts.get('nonlin'):
            L.append(f'.options nonlin-tran {opts["nonlin"]}')
        L.append(f'.measure tran iavg AVG I(VSUP) from={stop - T:.12g} to={stop:.12g}')
        L.append(f'.measure tran irms RMS I(LCOIL) from={stop - T:.12g} to={stop:.12g}')
        L.append(f'.measure tran vpp PP V(vhb) from={stop - T:.12g} to={stop:.12g}')
        L.append(f'.measure tran ipk MAX I(LCOIL) from={stop - T:.12g} to={stop:.12g}')
    L.append('.end')
    out.write_text('\n'.join(L) + '\n')


def run(sim, model, opts, cycles=40, tag=''):
    work = HERE / 'runs' / f'{sim}-{model}-{tag}'
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    cir = work / 'bench.cir'
    netlist(sim, model, opts, cycles, cir)
    exe = shutil.which('ngspice') if sim == 'ngspice' else shutil.which('Xyce')
    t0 = time.perf_counter()
    p = subprocess.run([exe, '-b', 'bench.cir'] if sim == 'ngspice' else [exe, 'bench.cir'], cwd=work,
                       capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - t0
    log = p.stdout + p.stderr
    (work / 'log.txt').write_text(log)
    res = dict(sim=sim, model=model, tag=tag, cycles=cycles, wall_s=wall, ok=p.returncode == 0)
    if sim == 'ngspice':
        for k in ('iavg', 'irms', 'vpp', 'ipk'):
            m = re.search(rf'^{k}\s*=\s*([-+0-9.eE]+)', log, re.M)
            res[k] = float(m.group(1)) if m else None
        for key, pat in (('accepted', r'Accepted timepoints\s*=\s*(\d+)'), ('rejected', r'Rejected timepoints\s*=\s*(\d+)'),
                         ('iterations', r'Transient iterations\s*=\s*(\d+)')):
            m = re.search(pat, log)
            res[key] = int(m.group(1)) if m else None
    else:
        mt = work / 'bench.cir.mt0'
        txt = mt.read_text() if mt.exists() else log
        for k in ('iavg', 'irms', 'vpp', 'ipk'):
            m = re.search(rf'^{k.upper()}\s*=\s*([-+0-9.eE]+)', txt, re.M | re.I)
            res[k] = float(m.group(1)) if m else None
        for key, pat in (('accepted', r'Successful Step\s+(\d+)'), ('rejected', r'Failed Steps\s+(\d+)'),
                         ('iterations', r'Residual\s+(\d+)')):
            m = re.search(pat, log)
            res[key] = int(m.group(1)) if m else None
    if res.get('iavg') is not None:
        res['pin'] = 24 * abs(res['iavg'])
    return res


NG = lambda **changes: options_line(changes)    # pcbqs's SOLVER_DEFAULTS with these changes
CASES = [
    # ngspice, pcbqs's options and variations; the references also tighten abstol
    ('ngspice', 'vdmos', dict(options=NG(abstol=1e-9), samples=2048), 'reference'),       # reltol 1e-5/1e-6 do not converge
    ('ngspice', 'vdmos', dict(options=NG()), 'pcbqs-default'),
    ('ngspice', 'vdmos', dict(options=NG(), klu=False), 'sparse'),
    ('ngspice', 'vdmos', dict(options=NG(reltol=1e-3)), 'reltol1e-3'),
    ('ngspice', 'vdmos', dict(options=NG(chgtol=1e-14)), 'chgtol-default'),
    ('ngspice', 'vdmos', dict(options=NG(method='trap')), 'trap'),
    ('ngspice', 'vdmos', dict(options=NG() + ' trtol=7'), 'trtol7'),
    ('ngspice', 'vdmos', dict(options=NG(reltol=1e-3) + ' trtol=7'), 'reltol1e-3-trtol7'),
    ('ngspice', 'smooth', dict(options=NG(reltol=1e-6, abstol=1e-9), samples=1024), 'reference'),
    ('ngspice', 'smooth', dict(options=NG()), 'pcbqs-default'),
    ('ngspice', 'smooth', dict(options=NG(reltol=1e-3)), 'reltol1e-3'),
    # Xyce, smooth switch
    ('xyce', 'smooth', dict(options='reltol=1e-4 abstol=1e-9 method=gear maxord=2'), 'gear-1e-4'),
    ('xyce', 'smooth', dict(options='reltol=1e-3 abstol=1e-9 method=gear maxord=2'), 'gear-1e-3'),
    ('xyce', 'smooth', dict(options='reltol=1e-4 abstol=1e-9 method=trap'), 'trap-1e-4'),
    ('xyce', 'smooth', dict(options='reltol=1e-3 abstol=1e-9'), 'default-1e-3'),
]


def main():
    global HERE
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('filters', nargs='*', help='run only cases whose sim-model-tag contains one of these')
    ap.add_argument('--out', default='circuit-solver-bench')
    args = ap.parse_args()
    HERE = Path(args.out); HERE.mkdir(parents=True, exist_ok=True)
    only = args.filters
    results = []
    for sim, model, opts, tag in CASES:
        if only and not any(o in f'{sim}-{model}-{tag}' for o in only):
            continue
        if not shutil.which('ngspice' if sim == 'ngspice' else 'Xyce'):
            print(f'skip {sim}: not installed'); continue
        r = run(sim, model, opts, tag=tag)
        results.append(r)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    json.dump(results, open(HERE / 'results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
