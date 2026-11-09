"""Converter input loop: loop R/L and magnetic near field (pcbqs.mqs) for CM5 Minima's U702 buck.

Operating point (5 V -> 3.3 V, 1 MHz, 2 A) is an assumption; parts and copper come from the board.
"""
from _common import CM5, OUT, need, run, keep, load

out = OUT / '01-cm5-u702'
run('pcbqs.mqs', need(CM5, 'CM5 Minima board (KICAD_DEMOS)'), '--converter', 'U702', '--vin-volts', 5, '--vout', 3.3,
    '--fsw', 1e6, '--current', 2, '--out', out)
res = load(out / 'mqs.json')
for fig in res['figures']:
    keep(out / fig, f'01-cm5-u702-{fig}')
print(f"U702: {res['triangles']:,} triangles; copper L at {res['frequencies_Hz'][0]/1e6:g} MHz = {res['copper_L_H'][0]*1e9:.3f} nH; "
      f"peak |H| at 1 mm = {max(res['peak_H_Am'][0]):.0f} A/m")
