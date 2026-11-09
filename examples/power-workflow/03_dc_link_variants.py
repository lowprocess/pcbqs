"""A converter's DC link through an input filter (pcbqs.pdn): wpthb, VIN -> R9 -> L3 -> VHB, five variants.

L3 is a series filter inductor (--through L3 --dcr L3=0.03); CA1-CA4 are 47 uF polymers (--cap). The
variants: as built, a 47 uF polymer on VHB, the same with 0.47 ohm damping, and L3 replaced by 10 mOhm / 0 ohm.
"""
import copy
import json
import numpy as np
from _common import WPTHB, OUT, need, run, keep, style, grid, INK, SERIES

board = need(WPTHB, 'wpthb board (WPTHB_BOARD)')
out = OUT / '03-wpthb-dc-link'; out.mkdir(parents=True, exist_ok=True)
caps = [x for k in range(1, 5) for x in ('--cap', f'CA{k}=47e-6,0.02,1e-9')]
common = ['--rail', 'VIN', '--volts', 24, '--source', 'TP2', '--ground-source', 'TP1', '--load', 'Q1=0.67', '--load', 'Q2=0.67',
          '--vrm-r', 0.05, '--vrm-l', 1e-7, '--fmin', 1e3, '--fmax', 1e8, '--points', 241, '--dmax', 0.8, *caps]


def resistor_board(value):
    """board.json with L3 replaced by a resistor R99 between the same pads."""
    from kicad2ems.board import Board
    from kicad2ems.importers.kicad import KiCadBoard
    b = Board(KiCadBoard(str(board)), need_ports=False, log=lambda *_: None).to_dict()
    fp = next(f for f in b['prims']['footprints'] if f['ref'] == 'L3')
    fp.update(ref='R99', value=value, lib='R_1206_3216Metric'); fp['props'] = dict(fp.get('props', {}), Value=value)
    path = out / f'board-R{value}.json'; path.write_text(json.dumps(b)); return path


variants = [('As built: L3 5.6 µH', board, ['--through', 'L3', '--dcr', 'L3=0.03']),
            ('L3 + 47 µF polymer on VHB', board, ['--through', 'L3', '--dcr', 'L3=0.03', '--cap', 'C19=47e-6,0.02,1.5e-9']),
            ('L3 + 47 µF polymer + 0.47 Ω damping', board, ['--through', 'L3', '--dcr', 'L3=0.03', '--cap', 'C19=47e-6,0.49,1.5e-9']),
            ('L3 → 10 mΩ resistor', resistor_board('10m'), []),
            ('L3 → 0 Ω link', resistor_board('0'), [])]
plt = style(); fig, ax = plt.subplots(figsize=(10, 5.2))
for lo, hi in ((90e3, 110e3), (180e3, 220e3)):
    ax.axvspan(lo, hi, color='#b7791f', alpha=.12, lw=0)
for (label, src, extra), color in zip(variants, SERIES):
    folder = out / label.split(':')[0].replace(' ', '_').replace('µ', 'u').replace('Ω', 'ohm').replace('→', 'to')
    run('pcbqs.pdn', src, '--out', folder, *common, *extra)
    p = json.loads((folder / 'pdn.json').read_text())['pdn']; f = np.array(p['f']); z = np.array(p['z']['Q1']['mag'])
    k = int(np.argmax(z * (f < 1e6)))
    ax.loglog(f, z, color=color, lw=2, label=f'{label} (peak {z[k]:.2f} Ω @ {f[k]/1e3:.1f} kHz)')
ax.axhline(p['target'], color='#c0392b', lw=1, ls='--', label=f"target {p['target']:.2f} Ω")
ax.set_xlim(1e3, 1e8); ax.set_ylim(1e-3, 5); ax.set_xlabel('Frequency (Hz)'); ax.set_ylabel('|Z| seen by the bridge (Ω)')
grid(ax); ax.legend(frameon=False, fontsize=8.5, loc='lower left')
ax.set_title('wpthb DC link (shaded: f_sw and 2·f_sw)', loc='left', color=INK, fontweight='bold')
fig.tight_layout(); fig.savefig(out / 'variants.png', dpi=160); keep(out / 'variants.png', '03-wpthb-dc-link-variants.png')
