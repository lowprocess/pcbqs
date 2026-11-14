"""The switching circuit with PCB copper (pcbqs.circuit): wpthb's bridge into the 24 uH coil, three builds.

The model is pcbqs's data/wpthb_bridge.json (the bridge as a circuit model: leg B's phase set for the load's power);
CA1-CA4 as 47 uF polymers. Builds: as built, a damped 47 uF polymer on VHB (C19), and L3 replaced by 10 mOhm. 30 W
to the k = 0.4 receiver at 90 and 110 kHz.
"""
import json
import numpy as np
import pcbqs
from pathlib import Path
from _common import WPTHB, OUT, need, run, keep, style, grid, INK, SERIES

board = need(WPTHB, 'wpthb board (WPTHB_BOARD)')
out = OUT / '04-wpthb-circuit'; out.mkdir(parents=True, exist_ok=True)
base = json.loads((Path(pcbqs.__file__).parent / 'data' / 'wpthb_bridge.json').read_text())
base['sweep'] = {'parameters.f': [90e3, 110e3], 'parameters.P': [30]}
for p in base['parts']:
    if p['ref'].startswith('CA'):
        p['value'] = 47e-6


def l3_resistor(p):
    if p['ref'] == 'L3':
        nodes = p['nodes']
        p.clear()
        p.update(ref='L3', type='resistor', nodes=nodes, value=0.01)


builds = {'as-built': lambda p: None,
          'damped-polymer': lambda p: p.update(value=47e-6, ESR_ohm=0.49, ESL_H=1.5e-9) if p['ref'] == 'C19' else None,
          'L3-10mOhm': l3_resistor}
labels = {'as-built': 'As built: L3 5.6 µH', 'damped-polymer': 'L3 + damped polymer on VHB', 'L3-10mOhm': 'L3 → 10 mΩ resistor'}
for name, change in builds.items():
    model = json.loads(json.dumps(base))
    for p in model['parts']:
        change(p)
    (out / f'{name}.json').write_text(json.dumps(model, indent=1))
    run('pcbqs.circuit', out / f'{name}.json', '--board', board, '--out', out / name, '--copper', '--jobs', 4)

plt = style(); fig, axes = plt.subplots(2, 2, figsize=(12, 6.6), sharex='col')
rows = []
for col, fsw in enumerate((90e3, 110e3)):
    for (name, label), color in zip(labels.items(), SERIES):
        r = next(c for c in json.loads((out / name / 'circuit.json').read_text())['cases']
                 if c['sweep']['parameters.f'] == fsw)
        d = np.load(Path(r['folder']) / 'transient.npz'); t = d['time_s']; T = 1 / fsw; m = t >= t[-1] - 2 * T
        x = (t[m] - (t[-1] - 2 * T)) / T
        axes[0, col].plot(x, d['vhb'][m], color=color, lw=1.3, label=label)
        axes[1, col].plot(x, -d['VSUP.I'][m], color=color, lw=1.3, label=label)
        rows.append((fsw, label, r['source_power_W'], r['efficiency'], r['measures']['bus_ripple_V'],
                     float(np.std(-d['VSUP.I'][t >= t[-1] - T])), r['runtime_s']))
    axes[0, col].set_title(f'{fsw/1e3:.0f} kHz, 30 W', loc='left', color=INK)
axes[0, 0].set_ylabel('VHB (V)'); axes[1, 0].set_ylabel('Supply current (A)')
for ax in axes.flat:
    grid(ax)
for ax in axes[1]:
    ax.set_xlabel('Switching periods (the last two)')
axes[0, 1].legend(frameon=False, fontsize=8.5)
fig.suptitle('wpthb bridge, settled: VHB and the current drawn from the 24 V supply', x=.06, ha='left', color=INK, fontweight='bold')
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(out / 'builds.png', dpi=160); keep(out / 'builds.png', '04-wpthb-circuit-builds.png')
for fsw, label, pin, eff, spike, ripple, secs in rows:
    print(f'{fsw/1e3:.0f} kHz | {label:28s} | Pin {pin:.2f} W | eff {eff*100:.1f}% | VHB p-p {spike:.2f} V | supply ripple {ripple*1e3:.1f} mA rms | {secs:.0f} s')
