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
    bad = buck()
    bad['analysis']['engine'] = 'spectre'
    assert 'analysis.engine' in run(bad, tmp_path / 'b', **QUIET)['failed'][0]['error']
    gate = buck()
    gate['parts'][1]['model'] = dict(type='switch', Cgd_F=1e-10, body_diode=False)     # what is left out is said
    case = run(gate, tmp_path / 'e', **QUIET)['cases'][0]
    assert any('Q1: its gate capacitances' in n for n in case['notes'])
    body = buck()
    body['parts'][1]['model'] = dict(type='switch', diode_TT_s=2e-8)                   # its body diode's recovery:
    plant = pwl.Plant(M.resolve(body))                                                  # a state, not left out
    qd = next(q for q in plant.caps if q['ref'] == 'Q1:body:Qd')
    assert qd['C'] == pytest.approx(2e-8 * plant.di[qd['diode']]['g'])
    case = run(body, tmp_path / 'r', **QUIET)['cases'][0]
    assert not any('recovery' in n for n in case['notes'])


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
    assert any(o['key'] == 'engine' for o in Circuit().options())


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


def test_nodes_held_only_by_inductors_are_islands(tmp_path):
    """A node between two windings and a capacitor between two inductances (as a part's pads are on its board's
    copper) have no conductance to ground: their potentials are what keep the windings' currents in Kirchhoff's
    law. The same buck with its inductance split in two, and with its output capacitor behind two small
    inductances, gives the buck's answer."""
    base = run(buck(), tmp_path / 'b', **QUIET)['cases'][0]
    split = buck()
    split['parts'][3] = dict(ref='L1', type='inductor', nodes=['sw', 'mid'], value=4e-6)
    split['parts'].append(dict(ref='L2', type='inductor', nodes=['mid', 'out'], value=6e-6))
    behind = buck()
    behind['parts'][4]['nodes'] = ['c+', 'c-']
    behind['parts'] += [dict(ref='La', type='inductor', nodes=['out', 'c+'], value=1e-9),
                        dict(ref='Lb', type='inductor', nodes=['c-', '0'], value=1e-9)]
    for m in (split, behind):
        case = run(m, tmp_path / m['parts'][-1]['ref'], **QUIET)['cases'][0]
        assert case['measures']['vout_avg'] == pytest.approx(base['measures']['vout_avg'], rel=1e-4)
        assert case['measures']['il_ripple'] == pytest.approx(base['measures']['il_ripple'], rel=1e-3)
        assert case['power_balance_relative'] < 1e-6
    plant = pwl.Plant(M.resolve(behind))
    m = plant.matrices(plant.gates_at(0.0), (False,))
    assert np.isfinite(m['A']).all()


def test_a_switch_s_capacitance_is_a_state_and_its_charge_a_loss(tmp_path):
    """Cds_F across a switch (type switch, as ngspice's deck has it): a capacitor of the plant, discharged through
    the switch when it turns on. A synchronous buck's high side turns on hard: about C V^2 f more loss."""
    m = buck(duty=0.4)
    m['parts'][1]['model'] = dict(type='switch', Ron_ohm=0.02, body_diode=True)
    m['parts'].insert(2, dict(ref='Q2', type='switch', nodes=['sw', '0'], drive='pwm', invert=True,
                              model=dict(type='switch', Ron_ohm=0.02, body_diode=True)))
    m['parts'] = [p for p in m['parts'] if p['ref'] != 'D1']
    m['drives']['pwm']['dead_time_s'] = 50e-9
    rows = {}
    for c in (0.0, 1e-9):
        mm = copy.deepcopy(m)
        for p in mm['parts']:
            if p['type'] == 'switch':
                p['model']['Cds_F'] = c
        rows[c] = run(mm, tmp_path / f"c{c:g}", **QUIET)['cases'][0]
        assert len(pwl.Plant(M.resolve(mm)).caps) == 1 + (2 if c else 0)
    extra = rows[1e-9]['loss_W'] - rows[0.0]['loss_W']
    assert extra == pytest.approx(1e-9 * 12.0 ** 2 * 200e3, rel=0.25)


EX_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'pcbqs', 'circuit', 'examples')


def on_board(name):
    m = example(name)
    m['board']['file'] = os.path.join(EX_DIR, m['board']['file'])
    m.setdefault('analysis', {})['engine'] = 'pwl'
    return m


def test_on_its_board_as_ngspice_has_it_in_a_tenth_of_the_time(tmp_path):
    """The board's copper (circuit/copper.py's R-L model, every part on its own pads) in the piecewise-linear
    engine. The synchronous buck's three loads against ngspice's run of the same model and board (2026-11-13: 22
    to 50 s a load, here a few): its efficiency, its copper's loss, its output, its inductor's ripple."""
    ngspice_rows = {                                    # efficiency, copper loss W, vout_avg, il_ripple
        'I1-current-A=0.5': (0.981754, 0.001912, 3.298765, 0.437710),
        'I1-current-A=2': (0.966313, 0.008455, 3.296591, 0.442873),
        'I1-current-A=4': (0.941158, 0.028776, 3.290672, 0.448115)}
    t = time.perf_counter()
    out = run(on_board('05_buck_synchronous'), tmp_path, copper=True, **QUIET)
    assert not out['failed'], out['failed']
    assert time.perf_counter() - t < 60
    for c in out['cases']:
        eff, cu, vout, ripple = ngspice_rows[c['name']]
        assert c['efficiency'] == pytest.approx(eff, abs=0.001)
        assert c['copper']['loss_W'] == pytest.approx(cu, rel=0.05)
        assert c['measures']['vout_avg'] == pytest.approx(vout, rel=0.002)
        assert c['measures']['il_ripple'] == pytest.approx(ripple, rel=0.005)
        assert c['power_balance_relative'] < 1e-4 and c['engine'] == 'pwl'     # (the copper: its dissipation, not V x I)
        with np.load(os.path.join(c['folder'], 'port-spectrum.npz')) as sp:      # what the fields are mapped from
            assert sp['coefficients_A'].shape[1] == len(c['copper']['ports']) == 10
            dc = sp['coefficients_A'][0].real
            assert dc[[p['name'] for p in c['copper']['ports']].index('T5')] == pytest.approx(-float(c['name'].split('=')[1]), rel=0.01)


@ngspice
def test_a_rectifier_on_its_board_as_ngspice_has_it(tmp_path):
    rows = {}
    for engine in ('ngspice', 'pwl'):
        m = on_board('02_half_wave_rectifier')
        m['analysis']['engine'] = engine
        out = run(m, tmp_path / engine, copper=True, **QUIET)
        assert not out['failed'], out['failed']
        rows[engine] = out['cases'][0]
    a, b = rows['ngspice'], rows['pwl']
    assert b['efficiency'] == pytest.approx(a['efficiency'], abs=0.002)
    assert b['copper']['loss_W'] == pytest.approx(a['copper']['loss_W'], rel=0.03)
    for k in ('vout_avg', 'vout_ripple', 'id_peak'):
        assert b['measures'][k] == pytest.approx(a['measures'][k], rel=0.01), k


def edge_buck(phase=0.0, package=None, duty=0.25, engine='pwl'):
    """A 12 V, 100 kHz buck at a quarter's duty with a 20 ns dead time (its edges 20 ns): at 270 degrees its off
    edge falls on the period's start."""
    sw = dict(type='switch', Ron_ohm=0.02, body_diode=False, **({'package_L_H': package} if package else {}))
    return dict(format=2, name='b', parameters={}, parts=[
        dict(ref='V1', type='source', nodes=['in', '0'], dc_V=12.0),
        dict(ref='Q1', type='switch', nodes=['in', 'sw'], drive='pwm', model=sw),
        dict(ref='D1', type='diode', nodes=['0', 'sw'], model=dict(Vf_V=0.5, If_A=1.0, R_ohm=0.01)),
        dict(ref='L1', type='inductor', nodes=['sw', 'out'], value=22e-6),
        dict(ref='C1', type='capacitor', nodes=['out', '0'], value=100e-6, ESR_ohm=5e-3),
        dict(ref='R1', type='load', nodes=['out', '0'], resistance_ohm=2.0)],
        drives=dict(pwm=dict(frequency_Hz=100e3, duty=duty, phase_deg=phase, dead_time_s=20e-9)),
        probes=dict(vout=dict(voltage=['out', '0'])), measures=dict(vout=dict(avg='vout')), analysis=dict(engine=engine))


def test_an_edge_on_the_period_s_start_is_the_period_s(tmp_path):
    """An edge that falls exactly on the period's start is where the period begins: the switch is off from there,
    as the same drive at 0 degrees has it (it was dropped, and the switch left on all period: 11.6 V for 2.6)."""
    m = edge_buck(270.0)
    assert (0.0, 0, False) in pwl.Plant(M.resolve(m)).edges
    a = run(edge_buck(0.0), tmp_path / 'a', **QUIET)['cases'][0]
    b = run(m, tmp_path / 'b', **QUIET)['cases'][0]
    assert b['measures']['vout'] == pytest.approx(a['measures']['vout'], rel=1e-6)
    assert b['measures']['vout'] == pytest.approx(2.5866, rel=2e-3)     # ngspice's, 2026-11-14


def test_a_current_source_into_open_parts_finds_the_diode_that_takes_it(tmp_path):
    """A current-fed boost: a source of 1 A into a node only an open switch and an off diode are on. What it
    brings the node turns the diode on (its node is an island no inductor reaches: the leakage and G_MIN against
    the current, not the leakage alone); half the period into a 10 ohm load: 5 V, as ngspice has it."""
    m = dict(format=2, name='c', parameters={}, parts=[
        dict(ref='I1', type='source', nodes=['sw', '0'], dc_A=1.0),
        dict(ref='Q1', type='switch', nodes=['sw', '0'], drive='pwm', model=dict(type='switch', Ron_ohm=0.01, body_diode=False)),
        dict(ref='D1', type='diode', nodes=['sw', 'out'], model=dict(Vf_V=0.5, If_A=1.0, R_ohm=0.01)),
        dict(ref='C1', type='capacitor', nodes=['out', '0'], value=10e-6),
        dict(ref='R1', type='load', nodes=['out', '0'], resistance_ohm=10.0)],
        drives=dict(pwm=dict(frequency_Hz=100e3, duty=0.5)),
        probes=dict(vout=dict(voltage=['out', '0'])), measures=dict(vout=dict(avg='vout')), analysis=dict(engine='pwl'))
    case = run(m, tmp_path, **QUIET)['cases'][0]
    assert case['measures']['vout'] == pytest.approx(5.0, rel=1e-3)         # ngspice: 5.00007 V
    assert case['parts']['R1']['power_W'] == pytest.approx(2.5, rel=2e-3)


def test_a_cut_current_s_energy_is_the_switch_s_whatever_the_hold(tmp_path, monkeypatch):
    """A switch with a package inductance and no capacitance cuts that inductance's current as it opens: the
    current jumps to zero at the edge (it went on for a period over HOLD, as much again lost at the input's
    voltage the faster it was held), and the half L I^2 it held is the switch's loss. The same at any HOLD, and the
    power balance closes."""
    rows = {}
    for hold in (2000, 32000):
        monkeypatch.setattr(pwl, 'HOLD', hold)
        rows[hold] = run(edge_buck(package=1e-9, duty=0.4), tmp_path / str(hold), **QUIET)['cases'][0]
    a, b = rows[2000], rows[32000]
    assert b['parts']['Q1']['power_W'] == pytest.approx(a['parts']['Q1']['power_W'], rel=1e-6)
    assert a['power_balance_relative'] < 1e-6
    bare = run(edge_buck(duty=0.4), tmp_path / 'bare', **QUIET)['cases'][0]
    ipk = a['parts']['L1']['peak_A']
    extra = a['parts']['Q1']['power_W'] - bare['parts']['Q1']['power_W']
    assert extra == pytest.approx(0.5 * 1e-9 * ipk ** 2 * 100e3, rel=0.15)


def test_a_board_with_no_ports_leaves_the_parts_at_their_nodes():
    """copper.prepare's answer where no net has two of the model's pads (no ports): no pin is put on a pad of its
    own, as the deck has it (they were, each on a node of its own: the buck's output read 0 V)."""
    m = M.resolve(buck())
    pins = {('Q1', 0): 'Q1.in', ('Q1', 1): 'Q1.sw', ('L1', 1): 'L1.out'}
    plant = pwl.Plant(m, dict(ports=[], model=None, pins=pins))
    assert set(plant.index) == set(pwl.Plant(m).index) == {'in', 'sw', 'out'}


def test_the_port_spectrum_is_the_copper_s_ports_alone(tmp_path):
    """PCB<i>.I in port order (PCB2 before PCB10); a part of the model called PCB_TRACE is not a port."""
    n = 64
    one = {'time_s': np.arange(n) / (n * 1e3), 'PCB0.I': np.ones(n), 'PCB10.I': 2 * np.ones(n),
           'PCB2.I': 3 * np.ones(n), 'PCB_TRACE.I': np.zeros(n), 'PCB_TRACE.V': np.zeros(n)}
    pwl.port_spectrum(tmp_path, one, 1e-3)
    with np.load(tmp_path / 'port-spectrum.npz') as sp:
        assert list(sp['port_labels']) == ['PCB0.I', 'PCB2.I', 'PCB10.I']
        assert sp['coefficients_A'][0].real == pytest.approx([1.0, 3.0, 2.0])


@pytest.mark.parametrize('engine', ['pwl', pytest.param('ngspice', marks=ngspice)])
def test_a_capacitor_s_own_voltage_and_a_sampled_measure(tmp_path, engine):
    """<ref>.cap_V of a capacitor with no ESR or ESL (its voltage: both engines have it), and a measure taken at
    so many instants a period (samples): the RC filter's ripple at 64 instants is its peak to peak, to the
    sampling's step."""
    m = example('01_rc_filter')
    m['measures'].update(c_pp=dict(pp='C1.cap_V'), sampled=dict(pp='vout', samples=64), sampled_avg=dict(avg='vout', samples=64))
    m.setdefault('analysis', {})['engine'] = engine
    out = run(m, tmp_path, **QUIET)
    assert not out['failed'], out['failed']
    ms = out['cases'][0]['measures']
    assert ms['c_pp'] == pytest.approx(ms['vout_ripple'], rel=1e-6)
    assert ms['sampled'] == pytest.approx(ms['vout_ripple'], rel=0.02) and ms['sampled'] <= ms['vout_ripple'] * (1 + 1e-9)
    assert ms['sampled_avg'] == pytest.approx(ms['vout_avg'], rel=0.01)
    bad = copy.deepcopy(m)
    bad['measures']['sampled']['samples'] = 2
    with pytest.raises(M.ModelError, match='samples'):
        M.validate(bad)


def test_a_probe_on_a_node_no_enabled_part_is_on_fails_its_point(tmp_path):
    m = buck()
    m['parts'].append(dict(ref='R9', type='resistor', nodes=['x', '0'], value=1.0, enabled=False))
    m['probes']['vx'] = dict(voltage=['x', '0'])
    out = run(m, tmp_path, **QUIET)
    assert out['cases'] == [] and 'probe vx' in out['failed'][0]['error']


def test_the_reported_run_keeps_samples_per_cycle(tmp_path):
    """The reported run is sampled as finely as the circuit rings; what it keeps is samples_per_cycle a period
    (the bridge: 2^18 and 8192), and it starts from the accepted run's state."""
    import pcbqs
    with open(os.path.join(os.path.dirname(pcbqs.__file__), 'data', 'wpthb_bridge.json'), encoding='utf-8') as fh:
        m = json.load(fh)
    m['sweep'] = {'parameters.f': [90e3], 'parameters.P': [10.0]}
    m['analysis'] = dict(engine='pwl')
    case = run(m, tmp_path, **QUIET)['cases'][0]
    perf = json.loads(open(os.path.join(case['folder'], 'performance.json'), encoding='utf-8').read())
    assert perf['samples_per_cycle'] == perf['ringing_samples'] > 8192 and perf['stored_samples_per_cycle'] == 8192
    with np.load(os.path.join(case['folder'], 'transient.npz')) as d:
        assert len(d['time_s']) == pwl.PERIODS * 8192
    assert os.path.getsize(os.path.join(case['folder'], 'transient.npz')) < 40e6
    assert case['load_power_W'] == pytest.approx(10.0, rel=0.005) and case['efficiency'] == pytest.approx(0.9376, abs=0.002)
    assert case['measures']['bus_spike_pp_V'] >= case['measures']['bus_ripple_V'] > 0


def recovery_buck(tt):
    """A 24 V buck whose high side, a fast ideal switch, turns on through 100 nH against a diode of transit time tt
    (the inductance, not the gate, sets di/dt), an RC snubber across the diode."""
    return dict(format=2, name='rr', parameters={}, parts=[
        dict(ref='V1', type='source', nodes=['in', '0'], dc_V=24.0),
        dict(ref='LS', type='inductor', nodes=['in', 'x'], value=100e-9),
        dict(ref='Q1', type='switch', nodes=['x', 'sw'], drive='pwm',
             model=dict(type='ideal', Ron_ohm=0.01, body_diode=False, rise_s=1e-9, fall_s=1e-9)),
        dict(ref='D1', type='diode', nodes=['0', 'sw'], model=dict(Vf_V=0.8, If_A=5.0, R_ohm=0.01, TT_s=tt)),
        dict(ref='RS', type='resistor', nodes=['sw', 'snub'], value=10.0),
        dict(ref='CS', type='capacitor', nodes=['snub', '0'], value=1e-9),
        dict(ref='L1', type='inductor', nodes=['sw', 'out'], value=47e-6),
        dict(ref='C1', type='capacitor', nodes=['out', '0'], value=47e-6, ESR_ohm=5e-3),
        dict(ref='R1', type='load', nodes=['out', '0'], resistance_ohm=2.4)],
        drives=dict(pwm=dict(frequency_Hz=100e3, duty=0.5)),
        probes=dict(vout=dict(voltage=['out', '0']), id=dict(current='D1'), iq=dict(current='Q1')),
        measures=dict(vout=dict(avg='vout'), irr=dict(min='id'), iq_peak=dict(max='iq')),
        analysis=dict(engine='pwl'))


def test_a_diode_s_stored_charge_is_its_reverse_recovery(tmp_path):
    """A diode's transit time (TT_s; a body diode's diode_TT_s): SPICE's diffusion charge, TT times its junction's
    current, a state while it conducts; it turns off when that charge is out, so a current reversed through it
    flows on until then. Where the circuit sets di/dt (the high side turning on through 100 nH, against 50 ns):
    the recovery current, the switch's peak, the snubber's loss and the efficiency as ngspice has them
    (2026-11-14: -7.597 A, 11.752 A, 366.1 mW, 95.641 %); with no transit time, no reverse current."""
    rows = {tt: run(recovery_buck(tt), tmp_path / f"{tt:g}", **QUIET)['cases'][0] for tt in (0.0, 50e-9)}
    a, b = rows[0.0], rows[50e-9]
    assert a['measures']['irr'] > -1e-3
    assert b['measures']['irr'] == pytest.approx(-7.597, rel=0.03)
    assert b['measures']['iq_peak'] == pytest.approx(11.752, rel=0.02)
    assert b['parts']['RS']['power_W'] == pytest.approx(0.3661, rel=0.08)
    assert b['efficiency'] == pytest.approx(0.95641, abs=5e-4) and b['power_balance_relative'] < 1e-6
    plant = pwl.Plant(M.resolve(recovery_buck(50e-9)))
    qd = next(q for q in plant.caps if q['ref'] == 'D1:Qd')
    assert qd['C'] == pytest.approx(50e-9 * plant.di[0]['g']) and qd['rest'] == plant.di[0]['vth']
