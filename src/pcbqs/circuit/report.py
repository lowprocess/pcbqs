"""A picture of a finished point: its probes (else its capacitor voltages and inductor currents) over the last two
periods, voltages over currents, and each part's share of the loss."""
from pathlib import Path

import numpy as np


def waveforms(out, row, model, deck):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with np.load(Path(out) / 'transient.npz') as data:
        res = {k: data[k] for k in data.files}
    t = res['time_s']
    period = row['period_s']
    keep = t >= t[-1] - 2 * period
    x = (t[keep] - t[keep][0]) / period
    probes = list(model.get('probes', {}))
    volts = [p for p in probes if 'voltage' in model['probes'][p]]
    amps = [p for p in probes if 'current' in model['probes'][p]]
    if not probes:
        volts = [r + '.V' for r, k in deck.kinds.items() if k == 'capacitor']
        amps = [r + '.I' for r, k in deck.kinds.items() if k == 'inductor']
    losses = {r: p['power_W'] for r, p in row['parts'].items() if p['type'] not in ('source', 'load')
              and not p.get('coupled_with')}
    losses.update({'+'.join(g['inductors']): g['power_W'] for g in row.get('coupled', [])})     # a transformer: its windings together
    losses = {r: w for r, w in losses.items() if w > 0}
    fig, axs = plt.subplots(3 if losses else 2, 1, figsize=(8, 7.5 if losses else 5.5), constrained_layout=True)
    for ax, keys, unit in ((axs[0], volts, 'V'), (axs[1], amps, 'A')):
        for k in keys:
            if k in res:
                ax.plot(x, res[k][keep], label=k, lw=1.2)
        ax.set(ylabel=unit, xlim=(0, 2))
        ax.grid(alpha=0.3)
        if keys:
            ax.legend(fontsize=8, ncol=4, loc='upper right')
    axs[1].set_xlabel('time / period')
    title = model.get('name', '') + (f"  ({row['name']})" if row['sweep'] else '')
    eff = row.get('efficiency')
    axs[0].set_title(title + (f"   efficiency {100 * eff:.1f} %" if eff else ''), fontsize=10)
    if losses:
        order = sorted(losses, key=losses.get, reverse=True)[:12]
        axs[2].bar(order, [losses[r] for r in order], color='#9a4f22')
        axs[2].set(ylabel='loss (W)')
        axs[2].grid(alpha=0.3, axis='y')
    fig.savefig(Path(out) / 'waveforms.png', dpi=110)
    plt.close(fig)
