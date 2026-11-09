"""The switching circuit with PCB copper (pcbqs.power_circuit): wpthb's bridge into the 24 uH coil, three builds.

Template model (python -m pcbqs.power_circuit --template); CA1-CA4 as 47 uF polymers. Builds: as built, a damped
47 uF polymer on VHB (C19), and L3 replaced by 10 mOhm. 30 W to the k = 0.4 receiver at 90 and 110 kHz.
"""
import json
import numpy as np
from _common import WPTHB, OUT, need, run, keep, style, grid, INK, SERIES

board = need(WPTHB, 'wpthb board (WPTHB_BOARD)')
out = OUT / '04-wpthb-circuit'; out.mkdir(parents=True, exist_ok=True)
from pcbqs.power_circuit import template
base = template()
for c in base['components']:
    if c['ref'].startswith('CA'):
        c['value'] = 47e-6
builds = {'as-built': lambda c: None,
          'damped-polymer': lambda c: c.update(value=47e-6, ESR_ohm=0.49, ESL_H=1.5e-9) if c['ref'] == 'C19' else None,
          'L3-10mOhm': lambda c: (c.clear(), c.update(ref='L3', a='NET:VBUS', b='L3:VHB', type='resistor', value=0.01)) if c['ref'] == 'L3' else None}
labels = {'as-built': 'As built: L3 5.6 µH', 'damped-polymer': 'L3 + damped polymer on VHB', 'L3-10mOhm': 'L3 → 10 mΩ resistor'}
for name, change in builds.items():
    cfg = json.loads(json.dumps(base))
    for c in cfg['components']:
        change(c)
    (out / f'{name}.json').write_text(json.dumps(cfg, indent=1))
    run('pcbqs.power_circuit', out / f'{name}.json', '--board', board, '--out', out / name, '--copper', '--powers', 30, '--jobs', 4)

plt = style(); fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharex='col')
rows = []
for col, fsw in enumerate((90e3, 110e3)):
    for (name, label), color in zip(labels.items(), SERIES):
        case = out / name / f'{int(fsw/1e3)}kHz_30W'
        d = np.load(case / 'transient.npz'); t = d['time_s']; T = 1 / fsw; m = t >= t[-1] - 2 * T
        x = (t[m] - (t[-1] - 2 * T)) / T
        axes[0, col].plot(x, d['bus.V'][m], color=color, lw=1.3, label=label)
        axes[1, col].plot(x, -d['supply.I'][m], color=color, lw=1.3, label=label)
        r = next(c for c in json.loads((out / name / 'power-circuit.json').read_text())['cases'] if c['frequency_Hz'] == fsw)
        rows.append((fsw, label, r['input_power_W'], r['efficiency'], r['bus_ripple_V'], float(np.std(-d['supply.I'][t >= t[-1] - T])), r['runtime_s']))
    axes[0, col].set_title(f'{fsw/1e3:.0f} kHz, 30 W', loc='left', color=INK)
axes[0, 0].set_ylabel('VHB (V)'); axes[1, 0].set_ylabel('Supply current (A)')
for ax in axes.flat:
    grid(ax)
for ax in axes[1]:
    ax.set_xlabel('Switching periods (last two; samples every T/256)')
axes[0, 1].legend(frameon=False, fontsize=8.5)
fig.suptitle('wpthb bridge, settled: VHB and the current drawn from the 24 V supply', x=.06, ha='left', color=INK, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(out / 'builds.png', dpi=160); keep(out / 'builds.png', '04-wpthb-circuit-builds.png')
for fsw, label, pin, eff, spike, ripple, secs in rows:
    print(f'{fsw/1e3:.0f} kHz | {label:28s} | Pin {pin:.2f} W | eff {eff*100:.1f}% | VHB p-p {spike:.2f} V | supply ripple {ripple*1e3:.1f} mA rms | {secs:.0f} s')
