"""The example circuits (pcbqs/circuit/examples), run with ngspice and checked against what each must give: its
closed form where there is one, charge and power balance where there is not. One run of each for the module."""
import math
import shutil

import numpy as np
import pytest

from pcbqs.circuit import model as M
from pcbqs.circuit import run

pytestmark = pytest.mark.skipif(not shutil.which('ngspice'), reason='ngspice not installed')
_results = {}


@pytest.fixture
def solved(tmp_path_factory):
    def solve(name):
        if name not in _results:
            model = M.template(name)
            summary = run(model, tmp_path_factory.mktemp(name), jobs=3, log=lambda s: None)
            assert summary['complete'] and not summary['failed'], summary['failed']
            _results[name] = (M.validate(model), summary['cases'])
        return _results[name]
    return solve


def values(model, point=None):
    r = M.resolve(model, point)
    return r['parameters'], {p['ref']: p for p in r['parts']}


@pytest.mark.parametrize('name', list(M.templates()))
def test_every_example_runs_clean_and_its_books_balance(solved, name):
    model, cases = solved(name)
    assert len(cases) == len(M.cases(model))
    for c in cases:
        assert c['settled'] and c['notes'] == [] and c['reltol'] == 1e-4          # the strict run was the one reported
        assert c['periodic_error'] < 0.003 and c['energy_drift_relative'] < 0.01      # and that is counted out
        for t in c['targets']:
            assert t['reached'] == pytest.approx(t['value'], rel=0.005)
        assert c['power_balance_relative'] < 1e-6                                 # every part is accounted for
        parts = c['parts']
        lossy = sum(p['power_W'] for p in parts.values() if p['type'] not in ('source', 'load'))
        assert c['loss_W'] == pytest.approx(lossy + c['gate_drive_power_W'], abs=2e-3 * c['source_power_W'])
        for ref, p in parts.items():                                              # nothing passive makes power
            if p['type'] in ('resistor', 'capacitor', 'diode', 'switch') or (p['type'] == 'inductor' and 'coupled_with' not in p):
                assert p['power_W'] > -2.5e-3 * c['source_power_W'], ref   # within what it may still be storing
        if c['efficiency'] is not None:
            assert 0.5 < c['efficiency'] < 1
        for name_ in ('transient.npz', 'circuit.cir', 'case.json', 'model.json', 'waveforms.png'):
            assert (__import__('pathlib').Path(c['folder']) / name_).is_file()


def test_rc_filter_is_its_closed_form(solved):
    model, (c,) = solved('rc_filter')
    T, tau = 1 / 10e3, 1e3 * 100e-9
    assert c['measures']['vout_avg'] == pytest.approx(2.5, rel=2e-3)
    assert c['measures']['vout_ripple'] == pytest.approx(5 * math.tanh(T / (4 * tau)), rel=3e-3)
    assert c['parts']['R1']['power_W'] == pytest.approx(-c['parts']['V1']['power_W'], rel=1e-3)


def test_rectifiers_balance_charge_and_power(solved):
    model, (c,) = solved('half_wave_rectifier')
    p = c['parts']
    assert p['D1']['avg_A'] == pytest.approx(p['R1']['avg_A'], rel=5e-3)          # all the load's charge came through it
    droop = p['R1']['avg_A'] * 0.02 / 1000e-6                                     # were the capacitor alone all period
    assert 0.75 * droop < c['measures']['vout_ripple'] < droop
    assert 17 - 0.8 - 0.5 * c['measures']['id_peak'] - c['measures']['vout_ripple'] < c['measures']['vout_avg'] < 17 - 0.6
    model, (c,) = solved('bridge_rectifier')
    p = c['parts']
    assert p['P1']['power_W'] == pytest.approx(20, rel=2e-3)                       # a constant-power load
    for d in ('D1', 'D2', 'D3', 'D4'):
        assert p[d]['avg_A'] == pytest.approx(p['P1']['avg_A'] / 2, rel=5e-3)     # each diode: half the load's current
    assert p['D1']['peak_V'] == pytest.approx(c['measures']['vdc_avg'] + c['measures']['vdc_ripple'] / 2, rel=0.05)
    assert c['measures']['vdc_ripple'] < p['P1']['avg_A'] * 0.01 / 2200e-6


def test_buck_and_boost_ripple_and_duty_are_the_textbook_ones(solved):
    model, (c,) = solved('buck')
    par, by = values(model)
    p, duty = c['parts'], c['knobs']['drives.pwm.duty']
    assert c['measures']['vout_avg'] == pytest.approx(5, rel=5e-3) and p['L1']['avg_A'] == pytest.approx(2, rel=5e-3)
    off = 5 + 0.45 + 2 * by['L1']['DCR_ohm']                       # across the inductor while the diode carries it
    assert c['measures']['il_ripple'] == pytest.approx(off * (1 - duty) / (by['L1']['value'] * par['fsw']), rel=0.02)
    on = p['V1']['avg_V'] - 2 * (by['Q1']['model']['Ron_ohm'] + by['L1']['DCR_ohm']) - 5
    assert duty == pytest.approx(off / (off + on), rel=0.02)       # volt-seconds balance
    assert p['L1']['power_W'] == pytest.approx(by['L1']['DCR_ohm'] * p['L1']['rms_A'] ** 2, rel=0.03)
    assert p['Q1']['avg_A'] + p['D1']['avg_A'] == pytest.approx(p['L1']['avg_A'], rel=5e-3)

    model, (c,) = solved('boost')
    par, by = values(model)
    p, duty = c['parts'], c['knobs']['drives.pwm.duty']
    assert c['measures']['vout_avg'] == pytest.approx(24, rel=5e-3)
    i = p['L1']['avg_A']
    assert i == pytest.approx(c['source_power_W'] / p['V1']['avg_V'], rel=5e-3)
    on = p['V1']['avg_V'] - i * (by['L1']['DCR_ohm'] + by['Q1']['model']['Ron_ohm'])
    assert c['measures']['il_ripple'] == pytest.approx(on * duty / (by['L1']['value'] * par['fsw']), rel=0.02)
    assert p['D1']['avg_A'] == pytest.approx(p['R1']['avg_A'], rel=5e-3)
    assert duty == pytest.approx(1 - 12 / 24, abs=0.03) and duty > 0.5           # the losses ask for a little more

    model, (c,) = solved('buck_boost')
    assert c['measures']['vout_avg'] == pytest.approx(-15, rel=5e-3)               # the inverting one
    assert c['parts']['D1']['avg_A'] == pytest.approx(0.5, rel=0.01)
    assert c['knobs']['drives.pwm.duty'] == pytest.approx(15 / 27, abs=0.03)


def test_synchronous_buck_sweep_holds_the_output_and_loses_more_with_load(solved):
    model, cases = solved('buck_synchronous')
    assert [c['sweep'] for c in cases] == [{'I1.current_A': i} for i in (0.5, 2, 4)]
    assert [c['name'] for c in cases] == ['I1-current-A=0.5', 'I1-current-A=2', 'I1-current-A=4']
    for c, load in zip(cases, (0.5, 2, 4)):
        p = c['parts']
        assert c['measures']['vout_avg'] == pytest.approx(3.3, rel=5e-3)
        assert p['L1']['avg_A'] == pytest.approx(load, rel=0.01) and p['I1']['avg_A'] == pytest.approx(load, rel=1e-3)
        assert p['L1']['power_W'] == pytest.approx(0.03 * p['L1']['rms_A'] ** 2, rel=0.05, abs=2e-3)
        assert p['Q2']['avg_A'] < 0                                # the lower switch carries the current backwards
    eff = [c['efficiency'] for c in cases]
    assert eff[0] > eff[1] > eff[2] > 0.9
    assert cases[2]['knobs']['drives.pwm.duty'] > cases[0]['knobs']['drives.pwm.duty']


def test_flyback_and_full_bridge_account_for_their_windings_together(solved):
    model, (c,) = solved('flyback')
    p = c['parts']
    assert c['measures']['vout_avg'] == pytest.approx(5, rel=5e-3) and p['D1']['avg_A'] == pytest.approx(1, rel=0.01)
    (group,) = c['coupled']
    assert group['inductors'] == ['LP', 'LS'] and p['LP']['coupled_with'] == ['LS']
    copper = 0.1 * p['LP']['rms_A'] ** 2 + 0.03 * p['LS']['rms_A'] ** 2
    assert group['power_W'] == pytest.approx(copper, rel=0.1)      # what the pair loses; each alone passes the power on
    assert p['LP']['power_W'] > 5 and p['LS']['power_W'] < -5
    assert 24 + 5.45 / 0.5 < c['measures']['vsw_peak'] < 80        # the reflected output, and the clamp holds the rest
    assert p['R2']['power_W'] > 0.1                                # the leakage's energy, burnt in the clamp

    model, (c,) = solved('full_bridge_phase_shift')
    p = c['parts']
    assert c['load_power_W'] == pytest.approx(10, rel=5e-3)
    assert 5 < c['knobs']['drives.legb.phase_deg'] < 180
    (group,) = c['coupled']
    assert group['power_W'] == pytest.approx(0.11 * (p['L1']['rms_A'] ** 2 + p['L2']['rms_A'] ** 2), rel=0.03)
    assert c['measures']['bus_ripple'] < 0.5                       # volts, on a 24 V bus
    assert max(p[q]['peak_V'] for q in ('Q1', 'Q2', 'Q3', 'Q4')) < 27


def test_resonant_half_bridge_follows_its_tank_across_the_sweep(solved):
    model, cases = solved('half_bridge_resonant')
    assert [c['name'] for c in cases] == ['fsw=115k', 'fsw=130k', 'fsw=160k']
    R = 5 + 0.05 + 0.01 + 0.02                                     # the load, the coil, the capacitor, a switch
    for c, f in zip(cases, (115e3, 130e3, 160e3)):
        rms2 = 0.0
        for n in range(1, 40, 2):                                  # a 0..24 V square wave, harmonic by harmonic
            w = 2 * math.pi * n * f
            rms2 += (2 * 24 / (n * math.pi) / math.sqrt(2)) ** 2 / (R ** 2 + (w * 22e-6 - 1 / (w * 100e-9)) ** 2)
        assert c['measures']['ir_rms'] == pytest.approx(math.sqrt(rms2), rel=0.01)
        assert c['parts']['CR']['avg_V'] == pytest.approx(12, rel=0.01)           # it blocks half the bus
        assert c['parts']['LR']['power_W'] == pytest.approx(0.05 * c['measures']['ir_rms'] ** 2, rel=0.03)
    assert cases[0]['load_power_W'] > cases[1]['load_power_W'] > cases[2]['load_power_W']


def test_an_unchanged_point_is_read_back_and_a_target_out_of_reach_says_so(tmp_path):
    model = M.template('buck')
    said = []
    first = run(model, tmp_path, log=said.append)
    again = run(model, tmp_path, log=said.append)
    assert said[-1] == 'run: unchanged, reusing its results' and again['cases'] == first['cases']
    model['targets'][0]['value'] = 15                              # above the input: no duty gets there
    out = run(model, tmp_path / 'high', log=lambda s: None)
    assert not out['cases'] and 'vout_avg cannot reach 15 with drives.pwm.duty in [0.05, 0.95]' in out['failed'][0]['error']


def test_a_capacitor_s_own_voltage_and_a_sampled_measure(tmp_path):
    """<ref>.cap_V of a capacitor with no ESR or ESL (its voltage), and a measure taken at so many instants a period
    (samples: a scope's settled value): the RC filter's ripple at 64 instants is its peak to peak, to the step."""
    import copy
    m = M.template('rc_filter')
    m['measures'].update(c_pp=dict(pp='C1.cap_V'), sampled=dict(pp='vout', samples=64), sampled_avg=dict(avg='vout', samples=64))
    out = run(m, tmp_path, log=lambda s: None)
    assert not out['failed'], out['failed']
    ms = out['cases'][0]['measures']
    assert ms['c_pp'] == pytest.approx(ms['vout_ripple'], rel=1e-6)
    assert ms['sampled'] == pytest.approx(ms['vout_ripple'], rel=0.02) and ms['sampled'] <= ms['vout_ripple'] * (1 + 1e-9)
    assert ms['sampled_avg'] == pytest.approx(ms['vout_avg'], rel=0.01)
    bad = copy.deepcopy(m)
    bad['measures']['sampled']['samples'] = 2
    with pytest.raises(M.ModelError, match='samples'):
        M.validate(bad)
