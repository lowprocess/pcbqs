"""A small rail's DC drop and PDN impedance (pcbqs.pdn): Jetson AGX Thor +1V15 from L5 to U21.

U21 draws 1 A; the regulator is modelled as 2 mOhm + 20 nH. +1V15's planes are small, so --plane-min 50.
"""
import csv
import numpy as np
from _common import JETSON, OUT, need, run, keep, load, style, grid, INK, MUTED, SERIES

out = OUT / '02-jetson-1v15'
run('pcbqs.pdn', need(JETSON, 'Jetson AGX Thor board (KICAD_DEMOS)'), '--out', out, '--rail', '+1V15', '--volts', 1.15,
    '--source', 'L5', '--ground-source', 'U69', '--load', 'U21=1', '--vrm-r', 0.002, '--vrm-l', 20e-9,
    '--plane-min', 50, '--dmax', 3)
res = load(out / 'pdn.json')
for fig in res['figures']:
    if fig['side'] != 'pdn':
        keep(out / fig['file'], f"02-jetson-1v15-{fig['file']}")
p = res['pdn']; f = np.array(p['f']); z = np.array(p['z']['U21']['mag'])
plt = style(); fig, ax = plt.subplots(figsize=(9, 4.6))
ax.loglog(f, z, color=SERIES[0], lw=2, label='|Z| seen by U21')
ax.axhline(p['target'], color='#c0392b', lw=1, ls='--', label=f"target {p['target']*1e3:.0f} mΩ")
cross = f[np.argmax(z > p['target'])] if (z > p['target']).any() else None
if cross:
    ax.axvline(cross, color=MUTED, lw=.8, ls=':'); ax.text(cross * 1.1, p['target'] * 1.5, f'crosses at {cross/1e6:.0f} MHz', color=MUTED, fontsize=9)
ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('|Z| (Ω)'); grid(ax); ax.legend(frameon=False, loc='upper left')
drop = res['loads']['U21']['drop'] * 1e3
ax.set_title(f'Jetson +1V15 → U21: {drop:.2f} mV DC drop at 1 A', loc='left', color=INK, fontweight='bold')
fig.tight_layout(); fig.savefig(out / 'impedance.png', dpi=160); keep(out / 'impedance.png', '02-jetson-1v15-impedance.png')
