"""Predictions for a bench measurement: wpthb's VHB impedance at TP7 (VHB) / TP6 (GND), 0.3-100 MHz.

The six VHB ceramics (C19, C24: 1 uF 0805; C20-C23: 10 uF 1206) sit in a column between TP7 and TP6, so a VNA
shunt-through across the two test points (unpowered) sees the caps through a few mm of copper. Above their series
resonance (~1-3 MHz) the reading is inductive: the parts' own ESL plus the board's mounting and spreading
inductance, which is where pcbqs's copper model and its part contacts (C13) show.

    python validation/wpthb_vhb_port.py [dmax] [termination,pad,edge] [board]

Per contact mode: |Z| and R, L_eff = Im Z / w at the port, with the parts as placeholders (below), and the board
alone (each part an ideal short at its contacts): the copper's loop inductance seen from the port. Results go to
validation/results/wpthb_vhb_port.json.

Part values are placeholders until the vendor models are in (TDK CGA5L3X5R1H106M160AB, Murata GRM219R61H105KA73J):
unbiased capacitance (the measurement is unpowered), ESR 3 / 10 mOhm, ESL 0.40 / 0.35 nH. The bridge MOSFETs
(Q1-Q4, unpowered, Coss of a few nF in series pairs) and L3 (5.6 uH: > 30 ohm above 1 MHz) are left out.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from shapely.geometry import box

from pcbqs._cli import load_board
from pcbqs.contacts import contact_terminal
from pcbqs.mqs import MQS
from pcbqs.pdn.rail import parts_of

BOARD = Path.home() / 'Desktop' / 'wpthb' / 'wpthb.kicad_pcb'
CAPS = {'C19': (1e-6, 0.010, 0.35e-9), 'C24': (1e-6, 0.010, 0.35e-9),
        **{f'C{k}': (10e-6, 0.003, 0.40e-9) for k in (20, 21, 22, 23)}}
F = np.geomspace(3e5, 1e8, 12)
OUT = Path(__file__).parent / 'results' / 'wpthb_vhb_port.json'


def model(board, mode, dmax, ideal):
    parts = {p.ref: p for p in parts_of(board)}
    terms = {'TP7': [(l, p.geom) for p in parts['TP7'].pads for l in p.layers],
             'TP6': [(l, p.geom) for p in parts['TP6'].pads for l in p.layers]}
    for ref in CAPS:
        for net in ('VHB', 'GND'):
            terms[f'{ref}:{net}'] = contact_terminal(parts[ref], parts[ref].on({net}), mode)
    x = [c for ref in [*CAPS, 'TP6', 'TP7'] for p in parts[ref].pads for c in p.geom.bounds[::2]]
    y = [c for ref in [*CAPS, 'TP6', 'TP7'] for p in parts[ref].pads for c in p.geom.bounds[1::2]]
    region = box(min(x) - 3, min(y) - 3, max(x) + 3, max(y) + 3)
    q = MQS(board, {'VHB', 'GND'}, terminals=terms, region=region, dmax=dmax, conductor_loss='slab',
            log=lambda *a: None)
    q.source('port', 'TP7', 'TP6', part_h=0.0)
    for ref, (C, esr, esl) in CAPS.items():
        if ideal:
            q.element(ref, f'{ref}:VHB', f'{ref}:GND', R=1e-6, L=0.0, part_h=0.0)
        else:
            q.element(ref, f'{ref}:VHB', f'{ref}:GND', C=C, ESR=esr, ESL=esl, part_h=0.0)
    return q


def main(argv):
    dmax = float(argv[0]) if argv else 0.4
    modes = argv[1].split(',') if len(argv) > 1 else ['termination', 'pad', 'edge']
    board = load_board(str(Path(argv[2]) if len(argv) > 2 else BOARD))
    out = json.loads(OUT.read_text()) if OUT.is_file() else {}
    for mode in modes:
        t0 = time.perf_counter()
        res = {}
        for ideal in (False, True):
            q = model(board, mode, dmax, ideal)
            z = np.asarray(q.solve(F).zc['port'])
            res['board_only' if ideal else 'with_parts'] = dict(
                f_Hz=F.tolist(), z_mag_ohm=np.abs(z).tolist(), R_ohm=z.real.tolist(),
                L_eff_H=(z.imag / (2 * np.pi * F)).tolist())
        res.update(triangles=len(q.tris), dmax=dmax, seconds=time.perf_counter() - t0)
        out[f'{mode}@{dmax:g}'] = res
        b, w = res['board_only'], res['with_parts']
        print(f"{mode:12s} dmax {dmax:g}: board L {b['L_eff_H'][-1] * 1e12:6.1f} pH (100 MHz), "
              f"|Z| {w['z_mag_ohm'][0] * 1e3:6.2f} / {w['z_mag_ohm'][-1] * 1e3:6.1f} mOhm at 0.3 / 100 MHz "
              f"({res['triangles']:,} triangles, {res['seconds']:.0f} s)", flush=True)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main(sys.argv[1:])
