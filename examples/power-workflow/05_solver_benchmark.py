"""Circuit-solver settings on a small wpthb-like converter (tools/benchmark_circuit_solver.py): time against error.

Error is the worst of input power and coil RMS current against each switch model's own tight reference: ngspice
VDMOS at max step T/2048, and for the behavioural (smooth) switch Xyce trap at reltol 1e-4 (which agreed with an
ngspice reltol 1e-6 run to 0.007%). ngspice could not converge the smooth switch at reltol 1e-4 or 1e-3.
"""
import json
import subprocess
import sys
from _common import HERE, OUT, keep, style, grid, INK, MUTED, SERIES

out = OUT / '05-solver-benchmark'
subprocess.run([sys.executable, str(HERE.parents[1] / 'tools' / 'benchmark_circuit_solver.py'), '--out', str(out),
                'vdmos-reference', 'pcbqs-default', 'sparse', 'reltol1e-3', 'trap', 'xyce-smooth'], check=True)
rows = [r for r in json.loads((out / 'results.json').read_text()) if r.get('pin')]
refs = {'vdmos': next(r for r in rows if r['tag'] == 'reference'), 'smooth': next(r for r in rows if r['tag'] == 'trap-1e-4')}
plt = style(); fig, ax = plt.subplots(figsize=(8.5, 4.8))
points = []
for r in rows:
    ref = refs[r['model']]
    if r is ref:
        continue
    err = max(abs(r['pin'] / ref['pin'] - 1), abs(r['irms'] / ref['irms'] - 1)) * 100
    near = next((q for q in points if q['sim'] == r['sim'] and abs(q['err'] / err - 1) < .02 and 0.85 < q['wall'] / r['wall_s'] < 1.18), None)
    if near:  # same result, similar time: one point, joint label
        near['tags'].append(r['tag'])
        continue
    points.append(dict(sim=r['sim'], model=r['model'], tags=[r['tag']], wall=r['wall_s'], err=err))
for q in points:
    ng = q['sim'] == 'ngspice'
    ax.scatter(q['wall'], q['err'], s=60, color=SERIES[0] if ng else SERIES[1], edgecolor='white', zorder=3)
    # ngspice labels above-right, Xyce below-left, so the two families never collide
    ax.annotate(f"{q['sim']} {q['model']} {' / '.join(q['tags'])}", (q['wall'], q['err']), (6, 5) if ng else (-6, -12),
                textcoords='offset points', fontsize=8, color=INK, ha='left' if ng else 'right')
ax.set_xscale('log'); ax.set_yscale('log'); grid(ax)
ax.set_xlabel('Wall time, 40 switching cycles (s)'); ax.set_ylabel('Worst error: input power, coil RMS (%)')
ax.set_title('Solver settings: blue ngspice, orange Xyce', loc='left', color=INK, fontweight='bold')
fig.tight_layout(); fig.savefig(out / 'speed-accuracy.png', dpi=160); keep(out / 'speed-accuracy.png', '05-solver-speed-accuracy.png')
