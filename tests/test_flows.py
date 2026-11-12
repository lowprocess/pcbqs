"""From one tool to the next: a converter the near-field tool selects as a circuit model (circuit/converter.py),
what a circuit draws from its supply (circuit/supply.py) as a rail's loads and current spectra (pdn/cli.py's
ripple) and as the near field's input current (mqs_cli.py's --spectrum), and what each result proposes to the
others (reports.py)."""
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest
from shapely.geometry import box

from pcbqs import reports
from pcbqs.circuit import boards as B
from pcbqs.circuit import model as M
from pcbqs.circuit import supply as S
from pcbqs.circuit.converter import model_of
from pcbqs.mqs_cli import parser, run as near_field

ngspice = pytest.mark.skipif(not shutil.which('ngspice'), reason='ngspice not installed')


def regulator(folder, diode=False, inductor=True):
    """A buck regulator's board: islands in, sw and out over a ground strip, U1 (pads VIN, SW, GND) where the three
    meet, C1 on the input, L1 from sw to out, C2 and C3 on the output; with `diode`, D1 from ground to sw."""
    b = B.Board('regulator', 40, 23)
    b.row(['in', 'sw', 'out'], [12, 8, 14], height=14)
    parts = [dict(ref='C1', type='capacitor', nodes=['in', '0'], value=22e-6),
             dict(ref='C2', type='capacitor', nodes=['out', '0'], value=47e-6),
             dict(ref='C3', type='capacitor', nodes=['out', '0'], value=100e-9)]
    if inductor:
        parts.append(dict(ref='L1', type='inductor', nodes=['sw', 'out'], value=10e-6))
    if diode:
        parts.append(dict(ref='D1', type='diode', nodes=['0', 'sw']))
    b.place(dict(parts=parts), packages=dict(C1='C_0805', C2='C_0805'))
    b.footprints.append(dict(ref='U1', value='BUCK', lib='Package_SO:SOIC-8', x=14.2, y=9.0, pads=[
        ('1', 'VIN', 'in', box(12.4, 4.0, 13.6, 6.0)), ('2', 'SW', 'sw', box(14.8, 4.0, 16.0, 6.0)),
        ('3', 'GND', 'GND', box(13.4, 17.0, 15.0, 18.0))]))
    b.check()
    folder.mkdir(parents=True, exist_ok=True)
    return b.write(folder)


def settings(path, *more):
    return parser().parse_args([str(path), '--vin-volts', '12', '--vout', '5', '--fsw', '250e3', '--current', '2',
                                '--cap-radius', '30', *more])


@pytest.fixture(scope='module')
def board(tmp_path_factory):
    pytest.importorskip('kicad2ems')
    return regulator(tmp_path_factory.mktemp('regulator'))


# ---- a converter as a circuit ----
def test_a_converter_is_a_circuit_model(board, tmp_path):
    from pcbqs._cli import load_board
    m = model_of(load_board(str(board)), board, settings(board))
    M.validate(m)
    kinds = {p['ref']: (p['type'], p['nodes']) for p in m['parts']}
    assert kinds == dict(V1=('source', ['in', '0']), C1=('capacitor', ['in', '0']), QH=('switch', ['in', 'sw']),
                         QL=('switch', ['sw', '0']), L1=('inductor', ['sw', 'out']), C2=('capacitor', ['out', '0']),
                         C3=('capacitor', ['out', '0']), RL=('load', ['out', '0']))
    assert m['board'] == dict(file=str(board), parts=dict(QH='U1', QL='U1'), nets={'in': 'in', 'sw': 'sw', 'out': 'out'})
    assert m['parameters'] == dict(fsw=250e3, vin=12.0, vout=5.0, iout=2.0)
    value = {p['ref']: p.get('value') for p in m['parts']}
    assert value['C1'] == pytest.approx(22e-6) and value['L1'] == pytest.approx(10e-6)
    assert next(p for p in m['parts'] if p['ref'] == 'QL')['invert'] is True
    assert 'Assumed: 50 mOhm switches; 30 mOhm DCR for L1.' in m['description']
    # a duty alone: the voltages are assumed, and said
    duty = parser().parse_args([str(board), '--duty', '0.25', '--cap-radius', '30'])
    m2 = model_of(load_board(str(board)), board, duty)
    assert m2['parameters']['vin'] == 12.0 and m2['parameters']['vout'] == 3.0 and '12 V in and 3 V out' in m2['description']
    # the output capacitors out of reach: all of the net's in one, not on the board
    far = parser().parse_args([str(board), '--duty', '0.25', '--cap-radius', '0.5', '--capacitors', 'C1', '--converter',
                               'U1', '--vin', 'in'])
    m3 = model_of(load_board(str(board)), board, far)
    cout = next(p for p in m3['parts'] if p['ref'] == 'COUT')
    assert cout['value'] == pytest.approx(47.1e-6) and 'COUT, the 2 capacitors on out in one' in m3['description']
    # the command line prints it
    p = subprocess.run([sys.executable, '-m', 'pcbqs.mqs', str(board), '--vin-volts', '12', '--vout', '5', '--fsw',
                        '250e3', '--current', '2', '--cap-radius', '30', '--circuit'], capture_output=True, text=True)
    assert p.returncode == 0 and json.loads(p.stdout) == m
    # a board's own diode from ground to the switch node is the low side
    with_diode = regulator(tmp_path / 'd', diode=True)
    m4 = model_of(load_board(str(with_diode)), with_diode, settings(with_diode))
    low = next(p for p in m4['parts'] if p['ref'] == 'D1')
    assert low['type'] == 'diode' and low['nodes'] == ['0', 'sw'] and m4['board']['parts'] == dict(QH='U1')
    assert not any(p['ref'] == 'QL' for p in m4['parts'])
    # no inductor: no converter to make a circuit of
    bare = regulator(tmp_path / 'n', inductor=False)
    with pytest.raises(ValueError, match='no inductor on a net it shares'):
        model_of(load_board(str(bare)), bare, parser().parse_args([str(bare), '--converter', 'U1', '--vin', 'in',
                                                                   '--cap-radius', '30']))


# ---- what a circuit draws from its supply ----
def test_a_spectrum_file_and_its_waveform(tmp_path):
    f = tmp_path / 's.csv'
    f.write_text('# a measured current\nf_Hz,A,deg\n2e6,0.5,90\n1e6,1.0\n\n3e6, 0.25, -90\n')
    hz, amp = S.read_spectrum(f)
    assert hz.tolist() == [1e6, 2e6, 3e6] and amp == pytest.approx([1.0, 0.5j, -0.25j])
    periodic, pp = S.waveform(hz, amp)
    t = np.arange(4096) / 4096
    v = sum((a * np.exp(2j * np.pi * k * t)).real for k, a in zip((1, 2, 3), amp))
    assert periodic and pp == pytest.approx(v.max() - v.min(), rel=2e-3)
    assert S.waveform([1e6, 1.5e6], [1.0, 0.5]) == (False, 3.0)      # no common period: the bound
    for text, why in (('f_Hz,A\n', 'no spectrum'), ('1e6,1\nx,2\n', 'line 2 is not numbers'), ('-1,1\n', 'no spectrum'),
                      ('1e6\n', 'expected f_Hz, A')):
        f.write_text(text)
        with pytest.raises(ValueError, match=why):
            S.read_spectrum(f)


def test_supply_currents_of_a_run():
    """A supply node's parts but its capacitors, at their pins on it: the average, the harmonics, parts on one
    footprint together."""
    period, n = 1e-6, 256
    t = np.arange(3 * n + 1) * period / n
    pulse = np.where((t / period) % 1 < 0.25, 2.0, 0.0)             # a 2 A pulse, a quarter of the period
    model = dict(parts=[dict(ref='V1', type='source', nodes=['in', '0'], dc_V=12.0),
                        dict(ref='C1', type='capacitor', nodes=['in', '0'], value=1e-6),
                        dict(ref='QH', type='switch', nodes=['in', 'sw']),
                        dict(ref='R9', type='resistor', nodes=['x', 'in'], value=10.0),
                        dict(ref='QL', type='switch', nodes=['sw', '0'])],
                 board=dict(parts=dict(QH='U1', QL='U1'), nets={'in': '+12V'}))
    result = {'time_s': t, 'QH.I': pulse, 'C1.I': -pulse, 'R9.I': np.full(len(t), -0.5), 'QL.I': 2.0 - pulse}
    rows = S.currents(model, result, period)
    assert [(r['node'], r['net'], r['source'], r['volts'], r['capacitors']) for r in rows] == [
        ('in', '+12V', 'V1', 12.0, ['C1'])]
    loads = {l['footprint']: l for l in rows[0]['loads']}
    assert set(loads) == {'U1', 'R9'} and loads['U1']['parts'] == ['QH']         # not QL: not on the node
    u1 = loads['U1']
    assert u1['avg_A'] == pytest.approx(0.5) and u1['rms_A'] == pytest.approx(1.0)
    k = np.arange(1, 6)
    assert u1['A'][:5] == pytest.approx(2 * 2 * 0.25 * np.abs(np.sinc(k * 0.25)), abs=0.02)
    assert u1['f_Hz'][:3] == pytest.approx([1e6, 2e6, 3e6]) and len(u1['A']) == 40
    assert loads['R9']['avg_A'] == pytest.approx(0.5) and max(loads['R9']['A']) < 1e-9   # drawn at its second pin


@ngspice
def test_the_regulator_runs_and_says_what_it_draws(board, tmp_path):
    from pcbqs._cli import load_board
    from pcbqs.circuit.engine import run
    m = model_of(load_board(str(board)), board, settings(board))
    out = run(m, tmp_path / 'run', log=lambda s: None)
    case = out['cases'][0]
    assert case['measures']['vout_avg'] == pytest.approx(5.0, rel=0.01)
    row, = case['supply']
    load, = row['loads']
    assert (row['net'], row['volts'], load['footprint'], load['parts']) == ('in', 12.0, 'U1', ['QH'])
    assert load['avg_A'] == pytest.approx(case['parts']['V1']['avg_A'], rel=0.02)       # the capacitors carry no DC
    assert load['rms_A'] == pytest.approx(case['parts']['QH']['rms_A'], rel=0.02)
    hz, amp = S.read_spectrum(tmp_path / 'run' / case['name'] / load['file'])
    duty = case['knobs']['drives.pwm.duty']
    assert hz[0] == pytest.approx(250e3) and len(hz) == 40
    assert abs(amp[0]) == pytest.approx(2 * 2.0 * duty * np.sinc(duty), rel=0.1)       # a 2 A pulse's fundamental
    assert np.sqrt(load['avg_A'] ** 2 + np.sum(np.abs(amp) ** 2) / 2) == pytest.approx(load['rms_A'], rel=0.03)
    # what the run proposes: the rail with its load and spectrum, and the regulator's input current
    pads = {'U1': {'in', 'sw', 'GND'}, 'C1': {'in', 'GND'}}
    pdn, mqs = reports.circuit_suggest(out, str(tmp_path / 'run'), pads)
    file = str(tmp_path / 'run' / case['name'] / load['file'])
    assert pdn['to'] == 'pdn' and pdn['values'] == dict(rail='in', ground='GND', volts=12.0,
                                                        load=f"U1={load['avg_A']:.4g}", load_spectrum=f"U1={file}")
    assert mqs['to'] == 'mqs' and mqs['values'] == dict(converter='U1', vin='in', ground='GND', capacitors='C1',
                                                        spectrum=file)
    assert reports.circuit_suggest(out, str(tmp_path / 'run'), {'J1': {'in'}}) == []   # its parts not on that board


@ngspice
def test_the_regulator_on_its_copper(board, tmp_path):
    """The model's two switches are one footprint's pads (board.parts): the copper between the regulator, its
    capacitors and its inductor is in the circuit, and what it draws is still the regulator's."""
    from pcbqs._cli import load_board
    from pcbqs.circuit.engine import run
    m = model_of(load_board(str(board)), board, settings(board))
    out = run(m, tmp_path / 'run', log=lambda s: None, copper=True)
    case = out['cases'][0]
    assert case['measures']['vout_avg'] == pytest.approx(5.0, rel=0.01)
    assert 0 < case['copper']['loss_W'] < 0.1 and len(case['copper']['ports']) >= 4
    assert any(n.startswith("not on the board, so at their nets' nodes") and 'V1' in n and 'RL' in n for n in case['notes'])
    load, = case['supply'][0]['loads']
    assert load['footprint'] == 'U1' and load['avg_A'] == pytest.approx(case['parts']['V1']['avg_A'], rel=0.02)


# ---- a rail's ripple from its loads' currents ----
def test_a_rails_ripple_from_a_load_spectrum(tmp_path):
    from test_pdn_rail import board as rail_board
    from pcbqs.pdn.cli import main
    path = tmp_path / 'b_board.json'
    path.write_text(json.dumps(rail_board()))
    spectrum = tmp_path / 'u1.csv'
    spectrum.write_text('f_Hz,A,deg\n1e6,0.5,0\n3e6,0.1,180\n')
    common = [str(path), '--rail', 'P', '--volts', '3.3', '--load', 'U1=2', '--load', 'U2=0.1', '--dmax', '1',
              '--points', '7']
    assert main(common + ['--load-spectrum', f'U1={spectrum}', '--out', str(tmp_path / 'a')]) == 0
    r = json.loads((tmp_path / 'a' / 'pdn.json').read_text())
    rp = r['pdn']['ripple']
    assert rp['allowed_V'] == pytest.approx(3.3 * 0.05) and rp['highest_Hz'] == 3e6
    assert rp['sources']['U1'] == dict(file=str(spectrum), harmonics=2, fundamental_Hz=1e6, largest_A=0.5)
    u1, u2 = rp['loads']['U1'], rp['loads']['U2']
    assert u1['f'] == [1e6, 3e6] and not u1['bound']
    assert u1['rms_V'] == pytest.approx(np.sqrt(np.sum(np.square(u1['V'])) / 2))
    assert 0 < u2['pp_V'] < u1['pp_V'] <= 2 * sum(u1['V']) + 1e-12       # the rail further away sees less
    # each harmonic: the current times the impedance at the planes, which is below what the load's pads see
    z = np.interp(1e6, r['pdn']['f'], r['pdn']['z']['U1']['mag'])
    assert 0 < u1['V'][0] / 0.5 <= z * 1.05
    rep = reports.report('pdn', str(tmp_path / 'a'))
    sec = next(s for s in rep['sections'] if s['title'] == 'Ripple from the current spectra')
    assert sec['table']['rows'][0][0] == 'U1' and 'u1.csv: 2 harmonics of 1 MHz' in sec['text']
    assert f"ripple {u1['pp_V'] * 1e3:.3g} mV peak to peak at U1 from the current spectra" in rep['summary']
    sets = reports.curves('pdn', str(tmp_path / 'a'))
    assert [s['title'] for s in sets] == ['Impedance each load sees', 'Ripple on the rail by harmonic']
    assert [c['name'] for c in sets[1]['curves']] == ['U1', 'U2'] and sets[1]['curves'][0]['y'][0] == pytest.approx(u1['V'][0] * 1e3)
    # two loads' spectra add where they meet; one with no common period is a bound
    other = tmp_path / 'u2.csv'
    other.write_text('1.5e6,0.2\n')
    main(common + ['--load-spectrum', f'U1={spectrum}', '--load-spectrum', f'U2={other}', '--out', str(tmp_path / 'b')])
    both = json.loads((tmp_path / 'b' / 'pdn.json').read_text())['pdn']['ripple']
    assert both['loads']['U1']['f'] == [1e6, 1.5e6, 3e6] and both['loads']['U1']['bound']
    assert both['loads']['U1']['pp_V'] == pytest.approx(2 * sum(both['loads']['U1']['V']))
    for bad, why in ((['--load-spectrum', f'C1={spectrum}'], 'not a load'), (['--load-spectrum', 'U1=nothing.csv'], 'U1'),
                     (['--load-spectrum', 'U1'], 'expected REF=FILE')):
        with pytest.raises(SystemExit, match=why):
            main(common + bad + ['--out', str(tmp_path / 'c')])


# ---- the near field with a circuit's current ----
def test_the_near_field_takes_a_circuits_current(tmp_path):
    from test_mqs_cli import board as loop_board
    b = tmp_path / 'board.json'
    b.write_text(json.dumps(loop_board()))
    common = [str(b), '--harmonics', '1,2', '--scan-points', '5', '--heights', '1', '--dmax', '1', '--fine', '1',
              '--fill-backend', 'numpy', '--fsw', '1e6', '--current', '2', '--duty', '0.25']
    assumed = near_field(parser().parse_args(common + ['--out', str(tmp_path / 'a')]))
    assert 'spectrum' not in assumed and assumed['harmonic_peak_A'] == pytest.approx(
        2 * 2 * 0.25 * np.abs(np.sinc(np.array([1, 2]) * 0.25)), rel=1e-3)
    # that board's inductor has no capacitor behind it: no circuit of it, and the result says why
    assert 'circuit_model' not in assumed and any(n.startswith('No circuit model of the converter') for n in assumed['notes'])
    assert [s['to'] for s in reports.mqs_suggest(assumed, str(tmp_path / 'a'))] == ['pdn']
    spectrum = tmp_path / 'supply-U1.csv'
    spectrum.write_text('f_Hz,A,deg\n2e6,0.8,10\n4e6,0.2,20\n6e6,0.1,0\n')            # a circuit at 2 MHz
    given = near_field(parser().parse_args(common + ['--spectrum', str(spectrum), '--out', str(tmp_path / 'b')]))
    assert given['frequencies_Hz'] == [2e6, 4e6] and given['harmonic_peak_A'] == pytest.approx([0.8, 0.2])
    assert given['spectrum']['fsw_Hz'] == 2e6 and len(given['spectrum']['assumed_peak_A']) == 2
    # the field follows the current: the same copper, each harmonic scaled
    assert np.array(given['peak_H_Am'])[0, 0] / 0.8 == pytest.approx(
        np.array(assumed['peak_H_Am'])[1, 0] / assumed['harmonic_peak_A'][1], rel=0.02)   # both at 2 MHz
    rep = reports.report('mqs', str(tmp_path / 'b'))
    assert rep['summary'].endswith("2 harmonics of a circuit's input current")
    assert rep['sections'][1]['table']['columns'][2] == 'assumed pulse (A)' and 'supply-U1.csv' in rep['sections'][0]['text']
    with pytest.raises(ValueError, match='no harmonic 4 of its lowest frequency'):
        near_field(parser().parse_args(common[:1] + ['--harmonics', '1,4', '--spectrum', str(spectrum)] + common[3:]))


def test_a_near_field_run_writes_its_converter_as_a_circuit(board, tmp_path):
    out = tmp_path / 'nf'
    res = near_field(settings(board, '--harmonics', '1', '--scan-points', '9', '--heights', '2', '--field-tol', '0.2',
                              '--dmax', '3', '--fine', '1.5', '--out', str(out)))
    assert res['circuit_model'] == 'circuit-model.json'
    model = json.loads((out / 'circuit-model.json').read_text())
    assert M.validate(model)['board']['parts'] == dict(QH='U1', QL='U1')
    pdn, circuit = reports.mqs_suggest(res, str(out))
    assert (pdn['to'], circuit['to']) == ('pdn', 'circuit') and circuit['label'] == 'Open U1 as a circuit'
    assert circuit['values'] == dict(model=str(out / 'circuit-model.json'), copper=True)


# ---- the full bridge's runs ----
def test_a_bridge_s_high_side_switches_are_its_rail_s_loads():
    """pcbqs.power_circuit's model (format 1): what each high-side switch draws from the bridge's rail, the rail
    named from the supply's side with the filter's inductors on the way."""
    from pcbqs import power_circuit
    w = power_circuit.template('wpthb')
    f, n = 100e3, 256
    t = np.arange(3 * n + 1) / (n * f)
    first = (t * f) % 1 < 0.5
    result = {'time_s': t, 'Q1.I': np.where(first, 2.0, 0.0), 'Q2.I': np.where(first, 0.0, 2.0),
              'Q3.I': np.where(first, 0.0, 2.0), 'Q4.I': np.where(first, 2.0, 0.0)}
    row, = S.bridge_currents(w, result, f)
    assert (row['node'], row['net'], row['source'], row['volts'], row['through']) == ('VHB', 'VIN', 'supply', 24.0, ['L3'])
    assert row['dcr'] == dict(L3=0.03)
    assert row['capacitors'] == ['C19', 'C20', 'C21', 'C22', 'C23', 'C24']
    q1, q2 = row['loads']
    assert (q1['footprint'], q2['footprint']) == ('Q1', 'Q2') and q1['avg_A'] == q2['avg_A'] == pytest.approx(1.0)
    amp = lambda l, k: l['A'][k - 1] * np.exp(1j * np.radians(l['deg'][k - 1]))
    assert abs(amp(q1, 1)) == pytest.approx(4 / np.pi, rel=1e-3)                 # a 2 A square wave's fundamental
    assert abs(amp(q1, 1) + amp(q2, 1)) < 1e-9                                   # the two halves: it cancels in the rail
    assert S.bridge_currents(dict(w, switches=[]), result, f) == []
    bare = power_circuit.template('simple')                                      # its switches on the supply's own net
    row, = S.bridge_currents(bare, result, f)
    assert row['net'] != row['node'] or row['through'] == []


@ngspice
def test_a_bridge_run_proposes_its_rail(tmp_path):
    from pcbqs import power_circuit
    c = power_circuit.template('wpthb')
    c['frequencies_Hz'], c['powers_W'] = [90000.0], [30.0]
    power_circuit.validate(c)
    rows, _ = power_circuit.run_cases(c, tmp_path, log=lambda *a: None)
    row, = rows[0]['supply']
    assert row['net'] == 'VIN' and row['through'] == ['L3'] and [l['file'] for l in row['loads']] == ['supply-Q1.csv', 'supply-Q2.csv']
    assert sum(l['avg_A'] for l in row['loads']) == pytest.approx(rows[0]['input_power_W'] / 24.0, rel=0.05)
    hz, a1 = S.read_spectrum(tmp_path / '90kHz_30W' / 'supply-Q1.csv')
    _, a2 = S.read_spectrum(tmp_path / '90kHz_30W' / 'supply-Q2.csv')
    assert hz[0] == pytest.approx(90e3) and abs(a1[0] + a2[0]) < 0.05 * abs(a1[0])      # the rail sees twice the frequency
    assert abs(a1[1] + a2[1]) > 1.0
    res = dict(format=1, cases=rows)
    pdn, = reports.circuit_suggest(res, str(tmp_path))                           # no near field of a bridge
    assert pdn['to'] == 'pdn' and pdn['values']['rail'] == 'VIN' and pdn['values']['through'] == 'L3'
    assert pdn['values']['dcr'] == 'L3=0.03'
    assert pdn['values']['volts'] == 24.0 and pdn['values']['load'].splitlines()[0].startswith('Q1=')
    assert pdn['values']['load_spectrum'].splitlines() == [f"Q1={tmp_path / '90kHz_30W' / 'supply-Q1.csv'}",
                                                           f"Q2={tmp_path / '90kHz_30W' / 'supply-Q2.csv'}"]
    assert 'through L3' in pdn['note']
