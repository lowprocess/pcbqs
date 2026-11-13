"""The piecewise-linear engine (circuit/pwl.py): a switched circuit's periodic steady state from each interval's
exact solution. A buck against its textbook formulas, in continuous and in discontinuous conduction (the diode's
state is the circuit's own); the example converters against ngspice; what it leaves to ngspice."""
import copy
import json
import os
import shutil
import time

import numpy as np
import pytest

from pcbqs.circuit import model as M
from pcbqs.circuit import pwl
from pcbqs.circuit.engine import run

ngspice = pytest.mark.skipif(not shutil.which('ngspice'), reason='ngspice not installed')
QUIET = dict(log=lambda s: None)


def example(name):
    import pcbqs.circuit
    with open(os.path.join(os.path.dirname(pcbqs.circuit.__file__), 'examples', name + '.json'), encoding='utf-8') as fh:
        return json.load(fh)


def buck(load=5.0, duty=0.3, vin=12.0, L=10e-6, f=200e3):
    """A buck of parts as near ideal as the model allows: a 1 uOhm switch with no body diode, a diode of 50 mV."""
    return dict(format=2, name='buck', parameters={}, parts=[
        dict(ref='V1', type='source', nodes=['in', '0'], dc_V=vin),
        dict(ref='Q1', type='switch', nodes=['in', 'sw'], drive='pwm', model=dict(type='ideal', body_diode=False, rise_s=1e-12, fall_s=1e-12)),
        dict(ref='D1', type='diode', nodes=['0', 'sw'], model=dict(Vf_V=0.05, If_A=1.0, R_ohm=1e-4)),
        dict(ref='L1', type='inductor', nodes=['sw', 'out'], value=L),
        dict(ref='C1', type='capacitor', nodes=['out', '0'], value=100e-6, ESR_ohm=1e-3),
        dict(ref='R1', type='load', nodes=['out', '0'], resistance_ohm=load)],
        drives=dict(pwm=dict(frequency_Hz=f, duty=duty)),
        probes=dict(vout=dict(voltage=['out', '0']), il=dict(current='L1')),
        measures=dict(vout_avg=dict(avg='vout'), il_ripple=dict(pp='il'), il_min=dict(min='il')),
        analysis=dict(engine='pwl'))


def test_a_diode_s_line_meets_its_curve_where_it_is_rated():
    vth, ron = pwl.diode_line(0.45, 2.0, 1.0, 0.03)
    assert vth + ron * 2.0 == pytest.approx(0.45) and 0 < vth < 0.45 and ron == pytest.approx(0.03 + pwl.VT / 2.0)


def test_a_buck_in_continuous_and_in_discontinuous_conduction(tmp_path):
    """Continuous: the output is the duty times the input (less the diode's 50 mV for the rest of the period),
    the inductor's ripple (Vin - Vout) D / (L f). A light load: the inductor's current reaches zero, the diode
    turns off there (no gate tells it to), and the output rises to 2 Vin / (1 + sqrt(1 + 8 L f / (R D^2)))."""
    vin, D, L, f = 12.0, 0.3, 10e-6, 200e3
    case = run(buck(load=5.0), tmp_path / 'ccm', **QUIET)['cases'][0]
    vout = case['measures']['vout_avg']
    assert vout == pytest.approx(D * vin - 0.05 * (1 - D), abs=0.01) and case['measures']['il_min'] > 0
    assert case['measures']['il_ripple'] == pytest.approx((vin - vout) * D / (L * f), rel=0.005)
    assert case['engine'] == 'pwl' and case['reltol'] is None and case['runs'] == 1 and case['settled']
    assert case['power_balance_relative'] < 1e-6 and case['efficiency'] == pytest.approx(1.0, abs=0.01)
    light = 100.0
    case = run(buck(load=light), tmp_path / 'dcm', **QUIET)['cases'][0]
    ideal = 2 * vin / (1 + np.sqrt(1 + 8 * L * f / (light * D * D)))
    assert ideal == pytest.approx(9.0) and case['measures']['vout_avg'] == pytest.approx(ideal, rel=0.01)
    assert abs(case['measures']['il_min']) < 1e-3                    # it stops at zero: nothing rings on after it
    data = np.load(os.path.join(case['folder'], 'transient.npz'))
    il = data['L1.I'][:len(data['time_s']) // pwl.PERIODS]
    assert (il < 1e-4).mean() == pytest.approx(1 - D - D * (vin - ideal) / ideal, abs=0.02)   # the idle share


def test_the_steady_state_is_found_in_a_few_periods_not_a_transient():
    """Newton on the period's map: the state of a filter that takes hundreds of periods to settle is found in a
    handful, to a part in a thousand million; a period from it ends where it began."""
    plant = pwl.Plant(M.resolve(buck()))
    x, kept = plant.steady_state(samples=1024)
    assert plant.periods <= 12 and plant.nx == 2
    x1, _, _, _ = plant.one_period(x, 512)
    assert np.abs(x1 - x).max() < 1e-8 * np.abs(x).max()
    far, _, _, _ = plant.one_period(np.zeros(2), 512)               # one period from nothing: nowhere near
    assert np.abs(far - x).max() > 0.5 * np.abs(x).max()
    sig = plant.signals(kept)
    assert {'time_s', 'node.out', 'L1.I', 'C1.cap_V', 'Q1.V', 'Q1.gate_V', 'D1.I', 'V1.I', 'R1.V'} <= set(sig)
    assert np.mean(sig['V1.I']) == pytest.approx(-np.mean(sig['Q1.I']))      # the source's current: into its +


def test_a_constant_power_load_takes_its_power(tmp_path):
    m = buck()
    m['parts'][-1] = dict(ref='R1', type='load', nodes=['out', '0'], power_W=2.0)
    case = run(m, tmp_path / 'p', **QUIET)['cases'][0]
    assert case['parts']['R1']['power_W'] == pytest.approx(2.0, rel=1e-4) and case['load_power_W'] == pytest.approx(2.0, rel=1e-4)


def test_what_it_leaves_to_ngspice(tmp_path):
    wave = buck()
    wave['parts'][0] = dict(ref='V1', type='source', nodes=['in', '0'],
                            wave=dict(type='pwl', points=[[0, 0], [1e-6, 12]], repeat=False))
    out = run(wave, tmp_path / 'w', **QUIET)
    assert out['cases'] == [] and "a source with a pwl wave is ngspice's" in out['failed'][0]['error']
    fet = buck()
    fet['parts'][1]['model'] = dict(type='mosfet')
    assert "a mosfet switch model is ngspice's" in run(fet, tmp_path / 'f', **QUIET)['failed'][0]['error']
    with pytest.raises(pwl.Unsupported, match="copper"):
        pwl.simulate(M.resolve(buck()), None, 1, tmp_path / 'c', {}, None, None, copper=dict(ports=[1]))
    bad = buck()
    bad['analysis']['engine'] = 'spectre'
    assert 'analysis.engine' in run(bad, tmp_path / 'b', **QUIET)['failed'][0]['error']
    esl = buck()
    esl['parts'][4]['ESL_H'] = 1e-9                                  # a capacitor's ESL: left out, and said
    case = run(esl, tmp_path / 'e', **QUIET)['cases'][0]
    assert any('C1: its ESL is left out' in n for n in case['notes'])


def test_the_command_and_the_window_s_option(tmp_path, capsys):
    from pcbqs.circuit.__main__ import main, describe
    from pcbqs.analyses import Circuit
    path = tmp_path / 'boost.json'
    path.write_text(json.dumps(example('06_boost')))
    assert main([str(path), '--engine', 'pwl', '--out', str(tmp_path / 'out')]) == 0
    summary = json.loads((tmp_path / 'out' / 'circuit.json').read_text())
    case = summary['cases'][0]
    assert case['engine'] == 'pwl' and case['measures']['vout_avg'] == pytest.approx(24.0, rel=0.005)
    assert 'efficiency 97.' in capsys.readouterr().out
    opt = next(o for o in describe()['options'] if o['key'] == 'engine')
    assert opt['choices'] == ['ngspice', 'pwl'] and opt['default'] is None
    assert any(o['key'] == 'engine' and o['only'] == 'general' for o in Circuit().options())


def test_a_sine_and_a_pulse_source(tmp_path):
    """A sine source is two more states that turn at its frequency; a pulse a level that changes at its edges.
    A resistor on a sine takes A^2 / 2R; an RC low-pass on a square wave swings as its time constant says."""
    f, R, C = 1e3, 100.0, 1e-6
    rc = lambda wave: dict(format=2, name='rc', parameters={}, parts=[
        dict(ref='V1', type='source', nodes=['in', '0'], wave=wave),
        dict(ref='R1', type='resistor', nodes=['in', 'out'], value=R),
        dict(ref='C1', type='capacitor', nodes=['out', '0'], value=C),
        dict(ref='R2', type='load', nodes=['in', '0'], resistance_ohm=50.0)],
        probes=dict(vout=dict(voltage=['out', '0'])), measures=dict(pp=dict(pp='vout'), avg=dict(avg='vout')),
        analysis=dict(engine='pwl'))
    case = run(rc(dict(type='sine', amplitude_V=10.0, frequency_Hz=f, offset_V=1.0)), tmp_path / 's', **QUIET)['cases'][0]
    gain = 1 / np.sqrt(1 + (2 * np.pi * f * R * C) ** 2)
    assert case['measures']['pp'] == pytest.approx(2 * 10.0 * gain, rel=2e-3) and case['measures']['avg'] == pytest.approx(1.0, abs=1e-3)
    assert case['parts']['R2']['power_W'] == pytest.approx((10.0 ** 2 / 2 + 1.0) / 50.0, rel=2e-3)
    plant = pwl.Plant(M.resolve(rc(dict(type='sine', amplitude_V=10.0, frequency_Hz=f, phase_deg=90.0))))
    assert plant.nx == 3 and list(plant.free) == [0] and plant.x0[1:] == pytest.approx([1.0, 0.0])    # starts at its crest
    case = run(rc(dict(type='pulse', low_V=0.0, high_V=5.0, frequency_Hz=f, duty=0.5)), tmp_path / 'p', **QUIET)['cases'][0]
    swing = 5.0 * np.tanh(1 / (4 * f * R * C))                          # a square wave through R C, settled
    assert case['measures']['pp'] == pytest.approx(swing, rel=2e-3) and case['measures']['avg'] == pytest.approx(2.5, rel=2e-3)


@ngspice
@pytest.mark.parametrize('name, same, loose', [
    ('03_bridge_rectifier', ('vdc_ripple',), ('vdc_avg', 'iac_rms')),   # (a sine, four diodes: the line for the curve)
    ('06_boost', ('vout_avg', 'vout_ripple', 'il_ripple'), ()),
    ('08_flyback', ('vout_avg',), ('vsw_peak', 'ip_peak')),         # (coupled inductors; a snubbed switch's peak)
    ('10_full_bridge_phase_shift', ('i1_rms', 'i2_rms'), ())])       # (two legs, a phase for its knob)
def test_the_examples_as_ngspice_has_them_and_much_sooner(tmp_path, name, same, loose):
    rows, seconds = {}, {}
    for engine in ('ngspice', 'pwl'):
        m = copy.deepcopy(example(name))
        m.setdefault('analysis', {})['engine'] = engine
        t = time.perf_counter()
        out = run(m, tmp_path / engine, **QUIET)
        seconds[engine] = time.perf_counter() - t
        assert not out['failed'], out['failed']
        rows[engine] = out['cases'][0]
    a, b = rows['ngspice'], rows['pwl']
    for k in same:
        assert b['measures'][k] == pytest.approx(a['measures'][k], rel=0.01), k
    for k in loose:
        assert b['measures'][k] == pytest.approx(a['measures'][k], rel=0.03), k
    for k, v in a['knobs'].items():
        assert b['knobs'][k] == pytest.approx(v, rel=0.01), k
    assert b['efficiency'] == pytest.approx(a['efficiency'], abs=0.006)     # no edges, no capacitance: a little kinder
    assert b['load_power_W'] == pytest.approx(a['load_power_W'], rel=0.01)
    if a['runs'] > 1:                                                   # (a rectifier's one run is quick either way)
        assert seconds['pwl'] < 0.4 * seconds['ngspice']
