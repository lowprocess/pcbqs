"""Parts at their temperatures (circuit/temperature.py) and the electro-thermal runs (circuit/warm.py, R-E9)."""
import json
import shutil
from pathlib import Path

import pytest

import pcbqs
from pcbqs.circuit import model as M
from pcbqs.circuit import temperature as T
from pcbqs.circuit.engine import fingerprint


def test_each_kind_moves_by_its_own_law_from_25_C():
    hot = 125.0
    L = T.apply(dict(ref='L1', type='inductor', value=1e-6, DCR_ohm=0.1, AC_R_ohm=[[1e5, 0.2]], temperature_C=hot))
    assert L['DCR_ohm'] == pytest.approx(0.1 * (1 + T.COPPER_PER_K * 100))
    assert L['AC_R_ohm'][0][1] == pytest.approx(0.2 * (1 + T.COPPER_PER_K * 100))
    assert 1 + T.COPPER_PER_K * 100 == pytest.approx((1 + 0.00393 * 105) / (1 + 0.00393 * 5))    # copper's, from 20 C
    Q = T.apply(dict(ref='Q1', type='switch', model=dict(type='vdmos', Ron_ohm=0.01), temperature_C=hot))
    assert Q['model']['Ron_ohm'] == pytest.approx(0.016)                                         # 1.6x at 125 C
    own = T.apply(dict(ref='Q2', type='switch', model=dict(Ron_ohm=0.01, Ron_tempco_per_K=0.004), temperature_C=hot))
    assert own['model']['Ron_ohm'] == pytest.approx(0.014)
    D = T.apply(dict(ref='D1', type='diode', model=dict(Vf_V=0.5), temperature_C=hot))
    assert D['model']['Vf_V'] == pytest.approx(0.3)
    R = T.apply(dict(ref='R1', type='resistor', value=10.0, temperature_C=hot))
    assert R['value'] == 10.0 and 'nominal' not in R                                               # ppm/K: left
    vendor = T.apply(dict(ref='Q3', type='switch', model=dict(type='subcircuit', Ron_ohm=0.01), temperature_C=hot))
    assert vendor['model']['Ron_ohm'] == 0.01


def test_applying_twice_is_applying_once_and_the_25_C_values_are_kept():
    p = T.apply(dict(ref='L1', type='inductor', value=1e-6, DCR_ohm=0.1, temperature_C=75.0))
    once = p['DCR_ohm']
    assert T.apply(p)['DCR_ohm'] == once and p['nominal'] == {'DCR_ohm': 0.1}


def test_the_board_copper_and_the_fingerprint_follow_the_boards_temperature():
    model = dict(R_ohm=[[1.0, 0.1], [0.1, 1.0]], L_H=[[1e-9, 0], [0, 1e-9]], foster=[dict(R_ohm=[[2.0]], pole_rad_s=1.0)])
    warm = T.copper_model(model, 75.0)
    k = 1 + T.COPPER_PER_K * 50
    assert warm['R_ohm'][0][1] == pytest.approx(0.1 * k) and warm['foster'][0]['R_ohm'][0][0] == pytest.approx(2 * k)
    assert warm['L_H'] == model['L_H'] and model['R_ohm'][0][0] == 1.0
    m = M.template('04_buck')
    cold = fingerprint(M.resolve(m))
    m['board']['temperature_C'] = 60.0
    assert fingerprint(M.resolve(m)) != cold


def test_resolve_puts_each_part_at_its_temperature():
    m = M.template('04_buck')
    next(p for p in m['parts'] if p['ref'] == 'L1')['temperature_C'] = '25 + 50'
    L1 = next(p for p in M.resolve(m)['parts'] if p['ref'] == 'L1')
    assert L1['DCR_ohm'] == pytest.approx(0.03 * (1 + T.COPPER_PER_K * 50))


@pytest.mark.skipif(not shutil.which('ngspice'), reason='ngspice not installed')
def test_the_warm_runs_set_each_part_to_its_maps_temperature_until_none_moves(tmp_path):
    from pcbqs.circuit import warm
    m = M.template('04_buck')
    m['board']['file'] = str(Path(pcbqs.__file__).parent / 'circuit' / 'examples' / '04_buck.kicad_pcb')
    s = warm.run(m, tmp_path, 4, log=lambda s: None)
    w = s['warm']
    assert w['settled'] and w['passes'][-1]['largest_move_K'] <= warm.TOLERANCE_K < w['passes'][0]['largest_move_K']
    c, = s['cases']
    at = c['temperatures_C']['parts']
    assert set(at) == {'C1', 'C2', 'D1', 'L1', 'Q1'}                    # on the board; the source and load are not
    table = json.loads((tmp_path / 'thermal' / 'thermal-cases.json').read_text())
    for ref, t in at.items():                                          # the last map agrees with what was run
        assert table['parts'][ref][c['name']]['part_C'] == pytest.approx(t, abs=warm.TOLERANCE_K)
    ran = {p['ref']: p for p in json.loads((Path(c['folder']) / 'model.json').read_text())['parts']}
    assert ran['D1']['model']['Vf_V'] < 0.45 and ran['D1']['nominal'] == {'model.Vf_V': 0.45}
    assert ran['L1']['DCR_ohm'] > 0.03
    assert json.loads((tmp_path / 'circuit.json').read_text())['warm'] == w
