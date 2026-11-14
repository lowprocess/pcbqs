"""Real-board regressions: one coarse case each, checked against stored results so a change that moves real-board
answers is seen (C4-C6 were drift of this kind). Skipped where the board is not installed.

    CM5 U702 near field (python -m pcbqs.mqs)     /usr/share/kicad/demos/cm5_minima (KiCad's demos)
    wpthb DC drop and PDN (python -m pcbqs.pdn)   ~/Desktop/wpthb (the as-built 24 V bridge)
    wpthb circuit, 90 kHz 10 W, with copper        pcbqs's data/wpthb_bridge.json (python -m pcbqs.circuit)

A deliberate change to these answers: python tests/test_regression_boards.py --update, then review the diff of
tests/data/regression_boards.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).parent / 'data' / 'regression_boards.json'
CM5 = Path('/usr/share/kicad/demos/cm5_minima/CM5_MINIMA_3.kicad_pcb')
WPTHB = Path.home() / 'Desktop' / 'wpthb' / 'wpthb.kicad_pcb'
RTOL = 1e-3


def cm5_near_field(out):
    pytest.importorskip('kicad2ems')
    from pcbqs.mqs_cli import main
    assert main([str(CM5), '--converter', 'U702', '--vin', '+5V', '--harmonics', '1,2', '--heights', '1',
                 '--scan-points', '9', '--out', str(out)]) in (0, None)
    r = json.loads((Path(out) / 'mqs.json').read_text())
    return dict(triangles=r['triangles'], copper_R_ohm=r['copper_R_ohm'], copper_L_H=r['copper_L_H'],
                peak_H_Am=r['peak_H_Am'])


def wpthb_pdn(out):
    pytest.importorskip('kicad2ems')
    from pcbqs.pdn.cli import main
    caps = [f'CA{k}=47e-6,0.02,1e-9' for k in range(1, 5)]
    argv = [str(WPTHB), '--rail', 'VIN', '--volts', '24', '--source', 'TP2', '--ground-source', 'TP1',
            '--load', 'Q1=0.67', '--load', 'Q2=0.67', '--through', 'L3', '--dcr', 'L3=0.03',
            *[x for c in caps for x in ('--cap', c)], '--dmax', '0.8', '--points', '25', '--out', str(out)]
    assert main(argv) in (0, None)
    r = json.loads((Path(out) / 'pdn.json').read_text())
    return dict(drop_V={k: v['drop'] for k, v in sorted(r['loads'].items())}, loss_W=r['loss_W'],
                f_Hz=r['pdn']['f'], z_mag_Q1=r['pdn']['z']['Q1']['mag'])


def wpthb_circuit(out):
    pytest.importorskip('kicad2ems')
    import pcbqs
    from pcbqs.circuit.__main__ import main
    model = json.loads((Path(pcbqs.__file__).parent / 'data' / 'wpthb_bridge.json').read_text())
    model['sweep'] = {'parameters.f': [90000], 'parameters.P': [10]}
    path = Path(out) / 'wpthb_bridge.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model))
    main([str(path), '--board', str(WPTHB), '--copper', '--out', str(Path(out) / 'run')])
    c, = json.loads((Path(out) / 'run' / 'circuit.json').read_text())['cases']
    return dict(load_power_W=c['load_power_W'], input_power_W=c['source_power_W'], efficiency=c['efficiency'],
                phase_deg=c['knobs']['drives.legb.phase_deg'], primary_rms_A=c['measures']['primary_rms_A'],
                copper_loss_W=c['copper']['loss_W'])

CASES = {'cm5_near_field': (CM5, cm5_near_field), 'wpthb_pdn': (WPTHB, wpthb_pdn),
         'wpthb_circuit': (WPTHB, wpthb_circuit)}


def _close(got, want, path=''):
    if isinstance(want, dict):
        assert set(got) == set(want), path
        for k in want:
            _close(got[k], want[k], f'{path}.{k}')
    elif isinstance(want, (list, tuple)):
        np.testing.assert_allclose(np.asarray(got, float), np.asarray(want, float), rtol=RTOL, err_msg=path)
    elif isinstance(want, (int, float)):
        assert got == pytest.approx(want, rel=RTOL), path
    else:
        assert got == want, path


@pytest.mark.parametrize('name', list(CASES))
def test_real_board_answers_unchanged(name, tmp_path):
    board, run = CASES[name]
    if not board.is_file():
        pytest.skip(f'{board} not installed')
    want = json.loads(DATA.read_text())[name]
    _close(run(tmp_path), want, name)


if __name__ == '__main__' and '--update' in sys.argv:
    import tempfile
    stored = json.loads(DATA.read_text()) if DATA.is_file() else {}
    for name, (board, run) in CASES.items():
        if board.is_file():
            with tempfile.TemporaryDirectory(dir=Path.home()) as d:
                stored[name] = run(d)
            print(f'{name}: updated')
    DATA.parent.mkdir(exist_ok=True)
    DATA.write_text(json.dumps(stored, indent=1, default=float) + '\n')
