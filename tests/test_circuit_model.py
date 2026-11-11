"""The general circuit model (pcbqs.circuit): its values, paths, checks, sweep, decks and the search's pieces.
None of these needs ngspice; tests/test_circuit_examples.py runs the example circuits with it."""
import copy
import math

import numpy as np
import pytest

from pcbqs.circuit import model as M
from pcbqs.circuit import engine as E
from pcbqs.circuit.deck import build, gate_timing, period_of


def test_values_are_numbers_engineering_notation_or_expressions():
    assert M.number(4.7) == 4.7 and M.number('22u') == pytest.approx(22e-6) and M.number('4.7k') == 4700
    assert M.number('100n') == pytest.approx(1e-7) and M.number('2.2M') == 2.2e6 and M.number('1e-3') == 1e-3
    assert M.number('1/(2*pi*f*sqrt(l*c))', dict(f=1e3, l=1e-3, c=1e-6)) == pytest.approx(1 / (2 * math.pi * 1e3 * math.sqrt(1e-9)))
    assert M.number('max(a, 2)**2 - abs(-1)', dict(a=3)) == 8


@pytest.mark.parametrize('text, says', [
    ('fsw*2', "unknown name 'fsw'"), ('__import__("os")', 'only numbers'), ('a.b', 'only numbers'),
    ('1/0', 'cannot work out'), ('2 +', 'cannot work out'), ('[1, 2]', 'only numbers'), ('10**400', 'cannot work out')])
def test_an_expression_is_arithmetic_and_nothing_else(text, says):
    with pytest.raises(M.ModelError, match=says):
        M.number(text, {}, 'L1.value')


def test_a_path_names_a_number_anywhere_in_the_model():
    m = M.template('buck')
    assert M.get(m, 'drives.pwm.duty') == 0.45 and M.get(m, 'L1.value') == '22u' and M.get(m, 'parameters.vin') == 12
    assert M.get(m, 'parts.Q1.model.Ron_ohm') == M.get(m, 'Q1.model.Ron_ohm') == 0.03
    M.put(m, 'D1.model.Vf_V', 0.5)
    M.put(m, 'L1.DCR_ohm', 0.1)
    assert M.get(m, 'D1.model.Vf_V') == 0.5 and M.get(m, 'L1.DCR_ohm') == 0.1
    with pytest.raises(M.ModelError, match="no part 'L9'"):
        M.get(m, 'L9.value')


def broken(change):
    m = M.template('buck')
    change(m)
    return m


@pytest.mark.parametrize('change, says', [
    (lambda m: m.update(format=1), 'format must be 2'),
    (lambda m: m['parts'].append(dict(m['parts'][0])), 'used twice'),
    (lambda m: m['parts'][0].update(type='transformer'), 'type must be one of'),
    (lambda m: m['parts'][0].update(nodes=['in']), 'two node names'),
    (lambda m: m['parts'][2].update(drive='gate'), 'drive must name one of the drives'),
    (lambda m: m['parts'][0].pop('dc_V'), 'needs one of dc_V, dc_A or wave'),
    (lambda m: m['parts'][-1].update(power_W=5), 'needs one of resistance_ohm, power_W or current_A'),
    (lambda m: m['measures'].update(x={'avg': 'vnothing'}), "'vnothing' is not a probe"),
    (lambda m: m['probes'].update(vx={'voltage': ['nowhere', '0']}), "no part is on node 'nowhere'"),
    (lambda m: m['measures'].update(x={'median': 'vout'}), 'one of avg, rms, pp, max, min'),
    (lambda m: m['targets'][0].update(measure='vx'), "measure 'vx' is not one of"),
    (lambda m: m['targets'][0].update(adjust='drives.pwm.width'), 'not in the model'),
    (lambda m: m.update(couplings=[dict(inductors=['L1', 'C1'], k=0.5)]), "'C1' is not an inductor"),
    (lambda m: m.update(sweep={'R9.value': [1, 2]}), "no part 'R9'"),
])
def test_validate_says_what_is_wrong_and_where(change, says):
    with pytest.raises(M.ModelError, match=says):
        M.validate(broken(change))


def test_the_sweep_is_every_combination_and_each_point_has_a_folder_name():
    m = M.template('buck')
    assert M.cases(m) == [{}] and M.case_name({}) == 'run'
    m['sweep'] = {'parameters.fsw': [100e3, 250e3], 'R1.resistance_ohm': [2.5, 5]}
    points = M.cases(M.validate(m))
    assert len(points) == 4 and points[1] == {'parameters.fsw': 100e3, 'R1.resistance_ohm': 5}
    names = [M.case_name(p) for p in points]
    assert names[1] == 'fsw=100k_R1-resistance-ohm=5' and len(set(names)) == 4


def test_resolve_works_out_parameters_expressions_and_the_point():
    m = M.template('buck')
    m['parameters']['iout'] = 'pout/vout'                         # parameters may refer to each other, in any order
    m['parameters']['pout'] = 20
    r = M.resolve(m, {'parameters.vin': 24})
    assert r['parameters'] == dict(fsw=250e3, vin=24, vout=5, iout=4, pout=20)
    by = {p['ref']: p for p in r['parts']}
    assert by['R1']['resistance_ohm'] == 1.25 and by['L1']['value'] == pytest.approx(22e-6) and by['V1']['dc_V'] == 24
    assert r['targets'][0]['initial'] == pytest.approx(5 / 24) and r['targets'][0]['value'] == 5
    assert m['parameters']['vin'] == 12                           # the model itself is left as written
    m['parameters']['pout'] = 'iout*vout'
    with pytest.raises(M.ModelError, match='unknown name'):
        M.resolve(m)


def test_every_example_validates_resolves_and_builds():
    names = list(M.templates())
    assert names[:4] == ['rc_filter', 'half_wave_rectifier', 'bridge_rectifier', 'buck'] and len(names) == 10
    assert all(M.templates().values())                            # each says what it is
    for name in names:
        m = M.validate(M.template(name))
        for point in M.cases(m):
            deck = build(M.resolve(m, point))
            assert deck.period > 0 and set(deck.kinds) == {p['ref'] for p in m['parts']}
    with pytest.raises(M.ModelError, match='no template'):
        M.template('cuk')


def renamed(model, refs, nodes):
    """The model with its parts and nodes called something else."""
    m = copy.deepcopy(model)
    for p in m['parts']:
        p['ref'] = refs[p['ref']]
        p['nodes'] = [nodes.get(n, n) for n in p['nodes']]
    for c in m.get('couplings', []):
        c['inductors'] = [refs[r] for r in c['inductors']]
    m.pop('probes', None), m.pop('measures', None), m.pop('targets', None), m.pop('sweep', None)
    return m


@pytest.mark.parametrize('name', ['boost', 'flyback', 'full_bridge_phase_shift'])
def test_the_deck_goes_by_the_parts_and_nodes_not_by_what_they_are_called(name):
    """Nothing in the deck depends on a reference or a net name: the same circuit under other names is the same
    deck, name for name."""
    m = M.template(name)
    refs = {p['ref']: f"X{i}" for i, p in enumerate(m['parts'])}
    nodes = {n: f"net{i}" for i, n in enumerate(sorted({n for p in m['parts'] for n in p['nodes']} - {'0'}))}
    plain, other = build(M.resolve(renamed(m, {r: r for r in refs}, {}))), build(M.resolve(renamed(m, refs, nodes)))
    back = {v: k for k, v in refs.items()}
    undo = lambda line: ' '.join('_'.join(back.get(w, w) for w in tok.split('_')) for tok in line.split(' '))
    assert [undo(l) for l in other.lines] == plain.lines


def test_a_switch_and_its_complement_are_never_on_together():
    for duty, phase, dead in [(0.5, 0, 100e-9), (0.2, 90, 50e-9), (0.85, 300, 0.0), (0.03, 0, 20e-9)]:
        drive = dict(frequency_Hz=200e3, duty=duty, phase_deg=phase, dead_time_s=dead)
        (d1, w1, T), (d2, w2, _) = gate_timing(drive, False, {}, 5e-6), gate_timing(drive, True, {}, 5e-6)
        edge = 20e-9                                              # the default rise and fall
        on1 = (d1 + edge / 2, d1 + edge + w1 + edge / 2)          # between the middles of its rise and its fall
        on2 = (d2 + edge / 2, d2 + edge + w2 + edge / 2)
        for off, on in ((on1[1], on2[0]), (on2[1], on1[0])):       # one goes off, the dead time, the other comes on
            gap = (on - off) % T
            assert min(gap, T - gap) == pytest.approx(dead, abs=1e-12)
        assert (on1[1] - on1[0]) + (on2[1] - on2[0]) + 2 * dead == pytest.approx(T)
        assert on1[1] - on1[0] == pytest.approx(duty * T - dead)


def test_the_period_is_the_one_all_the_drives_and_sources_share():
    m = M.resolve(M.template('buck'))
    assert period_of(m) == pytest.approx(4e-6)
    m['drives']['second'] = dict(frequency_Hz=375e3)              # 250 kHz and 375 kHz: 125 kHz
    assert period_of(m) == pytest.approx(8e-6)
    assert period_of(M.resolve(M.template('bridge_rectifier'))) == pytest.approx(0.02)
    m['analysis'] = dict(period_s=1e-3)
    assert period_of(m) == 1e-3
    dc = dict(parts=[dict(ref='V1', type='source', nodes=['a', '0'], dc_V=1)])
    with pytest.raises(ValueError, match='nothing in the model is periodic'):
        period_of(dc)


def test_a_continued_run_starts_every_part_where_the_last_one_stopped():
    m = M.resolve(M.template('buck_synchronous'), M.cases(M.template('buck_synchronous'))[0])
    first = build(m)
    assert not first.uic and not any(' IC=' in l for l in first.lines)
    state = dict(C1=dict(initial_V=11.9, initial_I=0.1), C2=dict(initial_V=3.3, initial_I=0.0), L1=dict(initial_A=2.0),
                 Q1=dict(initial_A=0.0, initial_V=11.9), Q2=dict(initial_A=-2.0, initial_V=0.01))
    again = build(m, state)
    line = lambda start: next(l for l in again.lines if l.startswith(start))
    assert again.uic and line('C_C2 ').endswith('IC=3.2999999999999998') and line('L_L1 ').endswith(' IC=2')
    assert line('C_Q1_Cds_F ').endswith('IC=11.9') and line('D_Q1 ').endswith('IC=-11.9')
    assert 'C2.cap_V' in again.signals and 'C1.cap_V' in again.signals      # the capacitance's own voltage


def test_the_knob_finds_a_target_by_secant_steps_and_says_when_it_cannot():
    knob = E.Knob(dict(adjust='drives.pwm.duty', measure='vout_avg', value=5.0, range=[0.05, 0.95], initial=0.3))
    f = lambda d: 12 * d - 0.4                                    # a buck, near enough
    for n in range(12):
        err = knob.error(f(knob.x))
        if abs(err) < 1e-6:
            break
        knob.record(err)
        knob.step()
    assert n <= 3 and knob.x == pytest.approx(0.45)
    far = E.Knob(dict(adjust='drives.pwm.duty', measure='vout_avg', value=24.0, range=[0.05, 0.95]))
    with pytest.raises(RuntimeError, match=r'vout_avg cannot reach 24 with drives.pwm.duty in \[0.05, 0.95\]: at its high end it is 11'):
        for _ in range(20):
            far.record(far.error(f(far.x)))
            far.step()


def waves(n, envelope, period=1e-5, spc=64):
    """n periods of a triangle current and a capacitor voltage, their size following envelope(period index)."""
    t = np.arange(n * spc + 1) * period / spc
    k = t / period
    deck = build(M.resolve(M.template('buck')))
    tri = 2 + 0.3 * (2 * np.abs((k % 1) - 0.5) - 0.5)
    return dict(time_s=t, **{'L1.I': tri * envelope(k), 'C2.cap_V': 5 * envelope(k), 'C1.cap_V': 12 + 0 * t}), deck, period


def test_settled_means_every_period_of_the_half_run_matches_the_last():
    steady, deck, T = waves(100, lambda k: 1 + 0 * k)
    assert max(E.periodic_error(steady, deck, T).values()) < 1e-9
    drift, deck, T = waves(100, lambda k: 1 + 0.0004 * k)         # 0.04 % a period: 2 % over the half-run,
    assert E.periodic_error(drift, deck, T)['C2.cap_V'] == pytest.approx(0.9 * 0.02 / 1.04, rel=0.03)   # 9 in 10 within 1.8 %
    # a filter ringing once every 25 periods, back where it was every 50: only looking at every period shows it
    ring, deck, T = waves(100, lambda k: 1 + 0.03 * np.sin(2 * np.pi * k / 25))
    assert E.periodic_error(ring, deck, T)['C2.cap_V'] > 0.025
    for period in (70, 99):                                       # one edge of one period stepped over differently,
        blip, deck, T = waves(100, lambda k: 1 + 0 * k)           # in the middle of the run or in its last period
        blip['L1.I'][period * 64 + 3] += 0.2
        assert max(E.periodic_error(blip, deck, T).values()) < 1e-9
    with pytest.raises(ValueError, match='at least three periods'):
        E.periodic_error({k: v[:100] for k, v in steady.items()}, deck, T)


def test_a_ringing_run_is_continued_from_its_middle_and_a_drifting_one_from_its_end():
    m = M.resolve(M.template('buck'))
    ring, deck, T = waves(100, lambda k: 1 + 0.05 * np.cos(2 * np.pi * k / 20))
    extra = {'C1.I': 0 * ring['time_s'], 'C2.I': 0 * ring['time_s'], 'Q1.I': 0 * ring['time_s'], 'Q1.V': 0 * ring['time_s'],
             'D1.V': 0 * ring['time_s']}
    ring.update(extra)
    assert E.end_state(ring, m, deck)['C2']['initial_V'] == pytest.approx(5.25)             # where it stopped
    assert E.end_state(ring, m, deck, T)['C2']['initial_V'] == pytest.approx(5.0, abs=0.02)  # where it is heading
    drift, deck, T = waves(100, lambda k: 1 + 0.001 * k)
    drift.update(extra)
    assert E.end_state(drift, m, deck, T)['C2']['initial_V'] == pytest.approx(5.5)


def test_a_voltage_no_circuit_gives_is_called_a_solver_spike():
    parts = dict(V1=dict(type='source', avg_V=24.0), C1=dict(type='capacitor', avg_V=24.0, ripple_V=0.1),
                 Q1=dict(type='switch', peak_V=26.0), Q2=dict(type='switch', peak_V=5.6e5))
    notes = E.plausibility(dict(parts=parts))
    assert len(notes) == 1 and notes[0].startswith('Q2: a peak of 5.6e+05 V') and 'Cds_F' in notes[0]
    tank = dict(V1=dict(type='source', avg_V=24.0), CR=dict(type='capacitor', avg_V=0.0, ripple_V=900.0),
                Q1=dict(type='switch', peak_V=30.0), D1=dict(type='diode', peak_V=470.0))
    assert E.plausibility(dict(parts=tank)) == []                 # a resonant tank's voltage is the circuit's own


def test_what_the_circuit_stores_must_have_stopped_changing():
    """A 1 % ring on an output capacitor passes for settled period against period, and is tens of per cent of
    the power going in and out of that capacitor: an efficiency taken there would be off by as much."""
    m = M.resolve(M.template('buck'))
    steady, deck, T = waves(100, lambda k: 1 + 0 * k, period=4e-6)
    assert np.abs(E.energy_drift(steady, m, deck, T)).max() < 1e-9
    ring, deck, T = waves(100, lambda k: 1 + 0.01 * np.sin(2 * np.pi * k / 50), period=4e-6)
    drift = E.energy_drift(ring, m, deck, T)
    c, l = 47e-6, 22e-6
    energy = lambda e: 0.5 * c * (5 * e) ** 2 + 0.5 * l * (2.15 * e) ** 2 + 0.5 * 22e-6 * 12 ** 2
    assert drift[0] == pytest.approx((energy(1.0) - energy(1 + 0.01 * np.sin(2 * np.pi * 99 / 50))) / T, rel=1e-6)
    assert np.abs(drift).max() == pytest.approx(0.01 * 2 * np.pi / 50 * 2 * (energy(1) - 0.5 * 22e-6 * 144) / T, rel=0.02)
    stats = dict(source_power_W=10.7, load_power_W=10.0, parts=dict(R1=dict(power_W=10.0)))
    often, last = E.unsteadiness(ring, m, deck, stats)
    assert often > 0.03 and last == pytest.approx(abs(drift[0]) / 10.7)          # against a tolerance of 0.002
