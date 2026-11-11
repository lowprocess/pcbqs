"""The example circuits' boards (circuit/boards.py), reading KiCad into the general model (circuit/kicad.py) and the
board's copper in the circuit (circuit/copper.py)."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from pcbqs.circuit import boards as B
from pcbqs.circuit import copper as C
from pcbqs.circuit import kicad as K
from pcbqs.circuit import model as M

kicad2ems = pytest.importorskip('kicad2ems')
EXAMPLES = Path(B.__file__).parent / 'examples'
ngspice = pytest.mark.skipif(not shutil.which('ngspice'), reason='ngspice not installed')


# ---- the layouts ----
def test_the_board_files_are_what_the_generator_writes(tmp_path):
    made = B.layouts()
    assert list(made) == list(M.templates())
    for name, (board, names) in made.items():
        written = board.write(tmp_path)
        assert written.read_text() == (EXAMPLES / written.name).read_text(), name
        model = json.loads((EXAMPLES / (board.name + '.json')).read_text())
        assert model['board'] == dict(file=written.name, parts=names)


def test_a_layout_that_is_wrong_is_refused():
    resolved = M.resolve(M.template('buck'))
    b = B.Board('x', 30, 20)
    b.row(['in', 'sw'], [10, 12])                                   # no island for the output
    with pytest.raises(ValueError, match='no gap between an island of sw and one of out for L1'):
        b.place(resolved)
    b = B.Board('x', 40, 20)
    b.row(['in', 'sw', 'out'], [10, 4, 10])                         # the switch node too small for what stands on it
    b.place(resolved, names=dict(V1='J1', R1='J2'))
    with pytest.raises(ValueError, match='overlap|is not on an island|apart'):
        b.check()
    b = B.Board('x', 40, 20)
    b.row(['in', 'sw', 'out'], [10, 12, 10])
    b.island('in', 12.2, 2, 14, 6)                                  # the input drawn over the switch node
    with pytest.raises(ValueError, match='in and sw are 0.00 mm apart'):
        b.check()
    b = B.Board('x', 40, 20)
    b.row(['in', 'sw', 'out'], [10, 12, 10])
    b.island('out', 2, 17.5, 6, 18.4)                               # a piece of the output joined to nothing
    with pytest.raises(ValueError, match='the copper of out is in 2 pieces'):
        b.check()


def test_a_gate_pad_gets_a_notch_and_no_stitching_via_beside_it():
    board, _ = B.layouts()['buck_synchronous']
    q2 = next(f for f in board.footprints if f['ref'] == 'Q2')
    gate = next(g for _, fn, net, g in q2['pads'] if fn == 'G')
    assert next(net for _, fn, net, _ in q2['pads'] if fn == 'G') == 'Net-(Q2-G)'
    for net, island in board.islands:
        assert island.distance(gate) >= B.GAP - 1e-9
    assert all(np.hypot(x - gate.centroid.x, y - gate.centroid.y) > 1 for _, x, y in board.vias)
    ground = next(i for n, i in board.islands if n == 'GND')
    assert len(ground.exterior.coords) > 5 and not ground.interiors  # the strip, notched from its edge


# ---- KiCad in ----
@pytest.mark.parametrize('name', list(M.templates()))
def test_every_board_reads_back_as_its_model(name):
    """The board has the model's netlist: each part that KiCad can tell is found with its type, value and nets."""
    model = M.template(name)
    resolved = M.resolve(model, M.cases(model)[0])
    document = K.read(model['board']['file'])
    read = {p['ref']: p for p in K.model_from(document)['parts']}
    on_board = model['board']['parts']
    ground = lambda n: '0' if n in ('GND', '0') else n
    for p in resolved['parts']:
        if p['type'] not in K.IMPORTED or p['ref'] in on_board:
            assert p['ref'] not in read                             # a source, a load, a coil on a connector
            continue
        got = read[p['ref']]
        assert got['type'] == p['type']
        nodes, want = [ground(n) for n in got['nodes']], [ground(n) for n in p['nodes']]
        assert nodes == want if p['type'] in ('diode', 'switch') else sorted(nodes) == sorted(want)
        if 'value' in p:
            assert got['value'] == pytest.approx(p['value'], rel=1e-3)
    where, terminals, absent = C.pins(resolved, __import__('pcbqs._cli', fromlist=['x']).load_board(model['board']['file']))
    assert absent == [] and len(where) == 2 * len(resolved['parts'])    # every pin of the model is a pad of the board


def test_what_kicad_tells_and_what_it_does_not():
    text = '''(kicad_pcb (footprint "Package_TO_SOT_SMD:SOT-23" (layer "F.Cu")
      (property "Reference" "Q7" (at 0 0)) (property "Value" "say \\"(pad\\" here")
      (pad "1" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 1 "g") (pinfunction "G") (pintype "input"))
      (pad "2" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 2 "GND") (pinfunction "S"))
      (pad "3" smd rect (at 0 0) (size 1 1) (layers "F.Cu") (net 3 "/sw") (pinfunction "D")))
     (footprint "x:y" (property "Reference" "R1") (pad "1" smd rect (net 1 "a")) (pad "2" smd rect (net 2 "b"))))'''
    assert K.pad_functions(text) == {'Q7': {'1': 'G', '2': 'S', '3': 'D'}}
    pin = lambda number, name, net: dict(number=number, name=name, type='', net=net)
    comp = lambda ref, kind, value, pins, **more: dict(ref=ref, kind=kind, value=value, value_si=more.pop('si', None), pins=pins,
                                                       lib_id=more.pop('lib', ''), fields=more.pop('fields', {}), dnp=False,
                                                       exclude_from_sim=False, **more)
    fet = comp('Q7', 'mosfet', 'BSC', [pin('1', 'G', 'g'), pin('2', 'S', 'GND'), pin('3', 'D', '/sw')], fields={'Rds': '8m'})
    part, _ = K.part_of(fet, {'hb': {}})
    assert part == dict(ref='Q7', type='switch', nodes=['sw', 'GND'], drive='hb', model=dict(type='switch', Ron_ohm=0.008, Cds_F='300p'))
    assert K.part_of(comp('D2', 'unsupported', 'SS34', [pin('1', '', 'out'), pin('2', '', 'sw')]))[0] == \
        dict(ref='D2', type='diode', nodes=['sw', 'out'], model=dict(Vf_V=0.45, Cj_F='100p'))      # pad 1 is the cathode
    assert K.part_of(comp('D3', 'unsupported', '1N4148', [pin('1', 'A', 'x'), pin('2', 'K', 'y')]))[0]['nodes'] == ['x', 'y']
    cap = comp('C5', 'capacitor', '22u', [pin('1', '', '/vout'), pin('2', '', 'GND')], si=22e-6, fields={'ESR': '5m'})
    assert K.part_of(cap)[0] == dict(ref='C5', type='capacitor', value=22e-6, nodes=['vout', 'GND'], ESR_ohm=0.005)
    for bad, why in ((comp('U1', 'unsupported', 'LM', [pin('1', '', 'a'), pin('2', '', 'b')]), 'not a resistor'),
                     (comp('R9', 'resistor', 'DNF', [pin('1', '', 'a'), pin('2', '', 'b')]), "its value 'DNF' is not a number"),
                     (comp('C9', 'capacitor', '1u', [pin('1', '', 'a'), pin('2', '', 'a')], si=1e-6), 'on 1 net, not two'),
                     (comp('J1', 'unsupported', 'conn', [pin('1', '', 'a'), pin('2', '', 'b')]), 'add it as a source or a load')):
        part, said = K.part_of(bad)
        assert part is None and why in said
        assert K.status(bad, dict(parts=[])) == ('Not taken over: ' + said, False, False)


def test_importing_into_a_model_adds_updates_and_keeps_what_is_the_models():
    model = M.template('buck')
    document = K.read(model['board']['file'])
    c2 = next(p for p in model['parts'] if p['ref'] == 'C2')
    c2.update(value='10u', nodes=['sw', '0'])                       # out of date against the board
    model['parts'] = [p for p in model['parts'] if p['ref'] != 'D1']
    by = {c['ref']: c for c in document['components']}
    assert K.status(by['D1'], model) == ('New diode', True, True)
    assert K.status(by['C2'], model) == ('In the model: its value and nodes brought up to date', True, True)
    assert K.status(by['J1'], model)[1:] == (False, False)
    kept = K.apply(model, document, ['C2', 'D1', 'Q1'], update_connections=False)
    part = lambda m, ref: next(p for p in m['parts'] if p['ref'] == ref)
    assert part(kept, 'C2')['value'] == pytest.approx(47e-6) and part(kept, 'C2')['nodes'] == ['sw', '0']
    assert part(kept, 'C2')['ESR_ohm'] == 0.008                    # the model's own fields stay
    assert part(kept, 'Q1')['model']['Ron_ohm'] == 0.03 and part(kept, 'D1')['nodes'] == ['GND', 'sw']
    moved = K.apply(model, document, ['C2'])
    assert part(moved, 'C2')['nodes'] == ['out', 'GND']
    assert moved['board']['file'] == document['path'] and moved['schematic']['source']['kind'] == 'kicad_pcb'
    assert part(model, 'C2')['value'] == '10u'                     # the model given is left alone
    by['C1']['dnp'] = True
    assert K.status(by['C1'], model)[0].endswith('(not fitted: left out of the simulation)')
    assert part(K.apply(model, document, ['C1']), 'C1')['enabled'] is False
    with pytest.raises(ValueError, match='J1: not a resistor'):
        K.apply(model, document, ['J1'])


def test_a_netlist_is_read_the_same_way(tmp_path):
    xml = '''<?xml version="1.0" encoding="utf-8"?><export version="E"><design><source>buck.kicad_sch</source><tool>Eeschema 8</tool></design>
    <components>
     <comp ref="L1"><value>22u</value><libsource lib="Device" part="L"/><fields><field name="DCR">30m</field></fields></comp>
     <comp ref="C2"><value>47u</value><libsource lib="Device" part="C"/></comp>
     <comp ref="D1"><value>SS34</value><libsource lib="Device" part="D_Schottky"/></comp>
     <comp ref="Q1"><value>BSC</value><libsource lib="Transistor_FET" part="BSC"/></comp>
     <comp ref="J1"><value>Conn</value><libsource lib="Connector" part="Conn_01x02"/></comp></components>
    <libparts/>
    <nets>
     <net code="1" name="/VIN"><node ref="Q1" pin="3" pinfunction="D"/><node ref="J1" pin="1"/></net>
     <net code="2" name="/SW"><node ref="Q1" pin="2" pinfunction="S"/><node ref="L1" pin="1"/><node ref="D1" pin="1" pinfunction="K"/></net>
     <net code="3" name="/VOUT"><node ref="L1" pin="2"/><node ref="C2" pin="1"/></net>
     <net code="4" name="GND"><node ref="C2" pin="2"/><node ref="D1" pin="2" pinfunction="A"/><node ref="J1" pin="2"/></net>
     <net code="5" name="Net-(Q1-G)"><node ref="Q1" pin="1" pinfunction="G"/></net></nets></export>'''
    path = tmp_path / 'buck.xml'
    path.write_text(xml)
    model = K.model_from(K.read(path))
    assert [(p['ref'], p['type'], p['nodes']) for p in model['parts']] == [
        ('L1', 'inductor', ['SW', 'VOUT']), ('C2', 'capacitor', ['VOUT', 'GND']), ('D1', 'diode', ['GND', 'SW']),
        ('Q1', 'switch', ['VIN', 'SW'])]
    assert model['parts'][0]['DCR_ohm'] == 0.03 and model['parts'][2]['model']['Vf_V'] == 0.45
    assert model['drives'] == {'pwm': dict(frequency_Hz=100e3, duty=0.5)} and 'J1 not taken over' in model['description']
    assert 'board' not in model                                    # a netlist is no layout


# ---- the copper ----
def test_a_model_and_a_board_that_do_not_fit_say_where():
    from pcbqs._cli import load_board
    model = M.resolve(M.template('boost'))
    board = load_board(model['board']['file'])
    where, terminals, absent = C.pins(model, board)
    assert where[('V1', 0)] == 'J1.in' and where[('R1', 1)] == 'J2.GND' and where[('Q1', 0)] == 'Q1.sw'
    model['board']['parts']['V1'] = 'J9'
    with pytest.raises(C.BoardError, match="V1 is given footprint 'J9', which the board does not have"):
        C.pins(model, board)
    del model['board']['parts']['V1']                              # no footprint of that name: off the board
    assert C.pins(model, board)[2] == ['V1']
    model['parts'][2]['nodes'] = ['in', 'elsewhere']                # L1, on a net the board does not have it on
    with pytest.raises(C.BoardError, match="L1: footprint L1 has no pad on net 'elsewhere' .*its pads are on in, sw"):
        C.pins(model, board)
    model['board']['file'] = '/nowhere/board.kicad_pcb'
    with pytest.raises(C.BoardError, match='no file'):
        C.board_file(model)
    assert C.net_name(dict(board=dict(nets={'vbus': '+24V'})), 'vbus', set()) == '+24V'
    assert C.net_name({}, '0', {'PGND', 'x'}) == 'PGND' and C.net_name({}, 'sw', set()) == 'sw'


@pytest.fixture(scope='module')
def boost_copper(tmp_path_factory):
    pytest.importorskip('triangle')
    out = tmp_path_factory.mktemp('copper')
    said = []
    model = M.template('boost')
    cu = C.prepare(model, out, said.append)
    return model, cu, out, said


def test_the_boards_copper_is_a_small_passive_network_between_its_pads(boost_copper):
    model, cu, out, said = boost_copper
    assert said[0] == 'Extracting the copper of 06_boost.kicad_pcb: 14 pads on 4 nets'
    assert len(cu['ports']) == 10 and cu['same'] == {} and cu['absent'] == []
    reference = {p['net']: p['reference'] for p in cu['ports']}
    assert reference == {'in': 'J1.in', 'GND': 'J1.GND', 'sw': 'L1.sw', 'out': 'J2.out'}   # the connectors' pads first
    R, L = np.array(cu['model']['R_ohm']), np.array(cu['model']['L_H'])
    for matrix in (R, L):
        assert np.allclose(matrix, matrix.T) and np.linalg.eigvalsh(matrix).min() > -1e-9 * abs(matrix).max()
    assert 5e-5 < np.diag(R).min() and np.diag(R).max() < 5e-3      # tenths of a milliohm to a few
    assert 5e-11 < np.diag(L).min() and np.diag(L).max() < 2e-8     # a fraction of a nanohenry to a few
    assert cu['fit_error'] < 0.05
    resolved = M.resolve(model)
    nodes = C.nodes(resolved, cu)
    assert nodes[('V1', 0)] == 'in' and nodes[('V1', 1)] == '0' and nodes[('C1', 0)] == 'C1.in' and nodes[('R1', 0)] == 'out'
    assert C.port_nodes(resolved, cu)['T0'] == ('C1.in', 'in')
    again = []
    assert C.prepare(model, out, again.append)['model'] == cu['model'] and again == ['The copper of 06_boost.kicad_pcb: as extracted before']
    from pcbqs.circuit.deck import build
    deck = build(resolved, None, cu)
    assert len(deck.ports) == 10 and 'PCB9.I' in deck.signals and any(l.startswith('LMODE') for l in deck.lines)
    start = build(resolved, dict(__currents__=dict(L1=1.0, C2=-0.25)), cu)      # the first run on the board
    port = {p['terminal']: k for k, p in enumerate(cu['ports'])}
    currents = C.port_currents(resolved, cu, dict(L1=1.0, C2=-0.25))
    assert currents[port['L1.in']] == -1.0 and currents[port['C2.out']] == 0.25 and currents[port['C2.GND']] == -0.25
    assert start.uic


@ngspice
def test_boost_on_its_board(boost_copper, tmp_path):
    from pcbqs.circuit import run
    model, cu, out, said = boost_copper
    log = []
    alone = run(model, tmp_path / 'alone', log=lambda s: None)['cases'][0]
    board = run(model, out, log=log.append, copper=True)['cases'][0]
    assert board['notes'] == [] and board['reltol'] == 1e-4 and alone['copper'] is None
    assert [h['copper'] for h in board['history']].count(True) >= 1 and not board['history'][0]['copper']
    assert sum("with the board's copper" in line for line in log) >= 1
    assert board['knobs']['drives.pwm.duty'] == pytest.approx(alone['knobs']['drives.pwm.duty'], abs=0.003)
    assert board['measures']['vout_avg'] == pytest.approx(24, rel=5e-3)
    copper = board['copper']
    assert copper['board'] == '06_boost.kicad_pcb' and len(copper['ports']) == 10
    assert 5e-4 < copper['loss_W'] < 0.02                           # milliwatts, of 12 W
    by = {p['terminal']: p for p in copper['ports']}
    assert by['L1.in']['rms_A'] == pytest.approx(board['parts']['L1']['rms_A'], rel=0.01)   # the pad carries its part's current
    assert by['J2.GND']['rms_A'] == pytest.approx(board['parts']['R1']['avg_A'], rel=0.02)
    assert board['efficiency'] == pytest.approx(alone['efficiency'], abs=0.004)
    assert board['loss_W'] == pytest.approx(sum(p['power_W'] for p in board['parts'].values() if p['type'] not in ('source', 'load'))
                                            + copper['loss_W'], abs=2e-3 * board['source_power_W'])
    # the loop's inductance against the diode's capacitance: its reverse voltage overshoots on the board
    assert board['parts']['D1']['peak_V'] > 1.15 * alone['parts']['D1']['peak_V']
    assert (Path(board['folder']) / 'waveforms.png').is_file() and (out / 'copper-model.json').is_file()


# ---- the fields over the board ----
def test_a_signals_harmonics_put_it_back_together():
    from pcbqs.circuit import fields as F
    T = 5e-6
    t = np.arange(8 * 256 + 1) * T / 256
    x = 1.5 + 0.8 * np.cos(2 * np.pi * t / T + 0.4) - 0.3 * np.sin(2 * np.pi * 3 * t / T)
    c, at = F.harmonics_of(dict(time_s=t, a=x), ['a'], T, 8)
    assert c.shape == (9, 1) and c[0, 0] == pytest.approx(1.5) and abs(c[1, 0]) == pytest.approx(0.4, rel=1e-6)
    assert abs(c[3, 0]) == pytest.approx(0.15, rel=1e-6) and abs(c[2, 0]) < 1e-9
    n = np.arange(9)[:, None]
    back = (np.r_[1.0, np.full(8, 2.0)][:, None] * c * np.exp(2j * np.pi * n * (at - at[0])[None, :] / T)).real.sum(axis=0)
    assert back == pytest.approx(np.interp(at, t, x), abs=1e-9)
    assert F.settings(dict(fields=dict(harmonics=8)))['harmonics'] == 8 and F.settings({})['heights_mm'] == (1.0, 5.0)


def test_a_volt_on_a_net_is_a_field_over_that_net(boost_copper):
    """E for one volt on each net alone: strongest over that net's own island, and the islands' capacitances are
    those of their plates over the ground plane with the fringe on top."""
    from pcbqs._cli import load_board
    from pcbqs.circuit import fields as F
    model, cu, out, said = boost_copper
    board = load_board(model['board']['file'])
    x, y = np.linspace(2, 36, 35), np.linspace(-13, -1, 13)         # over the three islands, 1 mm up
    unit, cap = F.unit_fields(M.resolve(model), board, x, y, np.array([1.5 + 1.0]), lambda *a: None)
    assert set(unit) == {'in', 'sw', 'out', 'GND'}
    plate = lambda w, h: 8.854e-12 * 4.5 * w * h * 1e-6 / 1.5e-3    # the island alone, as a parallel plate
    assert plate(13, 10) < cap['sw'] < 1.6 * plate(13, 10) and plate(10, 10) < cap['in'] < 1.6 * plate(10, 10)
    assert cap['in'] == pytest.approx(cap['out'], rel=0.05) and cap['GND'] > cap['sw']
    mag = {n: np.linalg.norm(E[:, :, 0, :], axis=-1) for n, E in unit.items()}
    cols = dict(sw=(x > 13) & (x < 25), out=x > 26.5)
    cols['in'] = x < 11.5
    for net, other in (('in', 'out'), ('out', 'in'), ('sw', 'in')):
        assert mag[net][cols[net]].mean() > 3 * mag[net][cols[other]].mean(), net
    assert np.abs(unit['sw'][:, :, 0, 2]).max() > 100              # V/m for a volt, a millimetre up: of the order of V / d


@ngspice
def test_boost_fields_come_from_its_own_currents_and_voltages(boost_copper):
    """H over the loop the switch and the diode commutate round, E over the switch node, both from the run."""
    from pcbqs.circuit import run
    model, cu, out, said = boost_copper
    log = []
    summary = run(model, out, log=log.append, copper=True, fields=True)
    case = summary['cases'][0]
    f = case['fields']
    assert f['harmonics'] == 16 and f['highest_Hz'] == pytest.approx(3.2e6) and f['heights_mm'] == [1.0, 5.0]
    assert f['net_voltage_swing_V']['sw'] == pytest.approx(24.4, abs=1.5) and f['net_voltage_swing_V']['in'] < 0.5
    assert 0 <= f['port_current_AC_left_out'] < 0.2 and f['grid']['converged']
    with np.load(f['file']) as d:
        x, y, H, E = d['x_mm'], d['y_mm'], d['H_ac_rms_A_per_m'], d['E_ac_rms_V_per_m']
        Et, t = d['E_V_per_m'], d['time_s']
        assert d['H_A_per_m'].shape == (64, len(x), len(y), 2, 3) and Et.shape == d['H_A_per_m'].shape
        assert (d['H_rms_A_per_m'] >= H - 1e-6).all() and (d['E_rms_V_per_m'] >= E - 1e-3).all()
    at = lambda F, k: (x[np.unravel_index(F[:, :, k].argmax(), F[:, :, k].shape)[0]], y[np.unravel_index(F[:, :, k].argmax(), F[:, :, k].shape)[1]])
    ex, ey = at(E, 1)                                              # 5 mm up: over the switch node's island
    assert 12 < ex < 26 and -12.5 < ey < -1.5
    hx, hy = at(H, 1)                                              # over Q1, D1 and the output capacitor: the hot loop
    assert 17 < hx < 31
    assert H[:, :, 0].max() > 3 * H[:, :, 1].max() and E[:, :, 0].max() > 2 * E[:, :, 1].max()     # falls off with height
    i, j = np.abs(x - 19).argmin(), np.abs(y + 7).argmin()         # over the middle of the switch node, 5 mm up
    ez = Et[:, i, j, 1, 2]
    assert ez.max() - ez.min() > 10 * abs(Et[:, np.abs(x - 7).argmin(), j, 1, 2].max() - Et[:, np.abs(x - 7).argmin(), j, 1, 2].min())
    high = ez > (ez.max() + ez.min()) / 2                          # up while the switch is off: 1 - duty of the period
    assert high.mean() == pytest.approx(1 - case['knobs']['drives.pwm.duty'], abs=0.04)
    assert (out / 'run' / 'fields.png').is_file() and case['notes'] == []
    assert any('fields at 1 mm' in line for line in log)
    again = run(model, out, log=lambda s: None, copper=True, fields=True)['cases'][0]
    assert again['notes'] == [] and again['fields']['E_ac_rms_max_V_per_m'] == f['E_ac_rms_max_V_per_m']
