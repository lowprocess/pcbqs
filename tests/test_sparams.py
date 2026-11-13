"""Low-frequency S-parameters from the quasi-static copper (pcbqs.sparams): a microstrip thru against its series
element and a transmission line, the return's path (a slot under the trace; a ground via beside a layer change, far
from it, or missing), the charge solved with the currents, the command, and the kicad2ems solver."""
import json
import os

import numpy as np
import pytest
from shapely.geometry import Point, box, mapping

from pcbqs import sparams as SP

W, LEN, H, T = 1.0, 30.0, 1.5, 0.035
QUIET = dict(log=lambda *a: None)


def port(name, layer, ref, x, sgn, w=W):
    return dict(name=name, layer=layer, ref_layer=ref, c=[x, 10.0], axis=0, sgn=sgn, w=w, z0=50.0, net='SIG', kind='z')


def thru(slot=False):
    """A 30 mm microstrip, 1 mm wide, 1.5 mm over a 40 x 20 mm plane; slot: a 1 x 14 mm slot in the plane across it."""
    plane = box(0, 0, 40, 20)
    if slot:
        plane = plane.difference(box(19.5, 3, 20.5, 17))
    return dict(stack=dict(copper=[dict(name='B.Cu', z=0.0, t=T), dict(name='F.Cu', z=H, t=T)],
                           layers=[dict(name='core', z0=0.0, z1=H, er=4.3, tand=0.02, kind='diel')], thickness=H),
                copper={'F.Cu': [dict(net='SIG', polygon=mapping(box(5, 10 - W / 2, 5 + LEN, 10 + W / 2)))],
                        'B.Cu': [dict(net='GND', polygon=mapping(plane))]}, vias=[],
                outline=mapping(box(0, 0, 40, 20)), ground_nets=['GND'], prims=dict(footprints=[]),
                ports=[port('P1', 'F.Cu', 'B.Cu', 5.0, 1), port('P2', 'F.Cu', 'B.Cu', 5.0 + LEN, -1)])


def layer_change(ground_vias):
    """A 0.35 mm trace on F.Cu over the In1 ground, through a via to B.Cu over the In2 ground; the two grounds are
    joined by the vias given."""
    w, z = 0.35, {'B.Cu': 0.0, 'In2.Cu': 0.2, 'In1.Cu': 1.3, 'F.Cu': 1.5}
    plane = box(0, 0, 40, 20).difference(Point(20, 10).buffer(0.7, 16))
    cu = {'F.Cu': [dict(net='SIG', polygon=mapping(box(5, 10 - w / 2, 20.4, 10 + w / 2)))],
          'B.Cu': [dict(net='SIG', polygon=mapping(box(19.6, 10 - w / 2, 35, 10 + w / 2)))],
          'In1.Cu': [dict(net='GND', polygon=mapping(plane))], 'In2.Cu': [dict(net='GND', polygon=mapping(plane))]}
    vias = [dict(x=20.0, y=10.0, r=0.15, z0=0.0, z1=1.5, net='SIG')]
    vias += [dict(x=x, y=y, r=0.15, z0=0.2, z1=1.3, net='GND') for x, y in ground_vias]
    diel = [dict(name=n, z0=a, z1=b, er=4.3, tand=0.02, kind='diel') for n, a, b in (('d3', 0, 0.2), ('d2', 0.2, 1.3), ('d1', 1.3, 1.5))]
    return dict(stack=dict(copper=[dict(name=n, z=v, t=T) for n, v in sorted(z.items(), key=lambda kv: kv[1])],
                           layers=diel, thickness=1.5), copper=cu, vias=vias, outline=mapping(box(0, 0, 40, 20)),
                ground_nets=['GND'], prims=dict(footprints=[]),
                ports=[port('P1', 'F.Cu', 'In1.Cu', 5.0, 1, w), port('P2', 'B.Cu', 'In2.Cu', 35.0, -1, w)])


def path(r):
    p, = SP.paths(r['f'], r['S'], r['names'], r['zref'])
    return p


@pytest.fixture(scope='module')
def thru_rl():
    return SP.solve(thru(), np.geomspace(1e3, 1e7, 9), **QUIET)


def test_a_thru_is_its_copper_s_resistance_and_inductance(thru_rl):
    r = thru_rl
    S, f = r['S'], r['f']
    assert not r['info']['capacitance'] and r['info']['backend'] == 'dense' and r['names'] == ['P1', 'P2']
    assert np.allclose(S[:, 0, 0] + S[:, 1, 0], 1, atol=1e-9)                 # a series element: S11 + S21 = 1
    assert np.allclose(S, np.swapaxes(S, 1, 2), atol=1e-9) and np.abs(S).max() <= 1 + 1e-9
    p = path(r)
    R, L = np.array(p['R_ohm']), np.array(p['L_H']) * 1e9
    trace = LEN * 1e-3 / (5.8e7 * T * 1e-3 * W * 1e-3)                        # 14.8 mohm: the plane adds a little
    assert trace < R[0] < 1.12 * trace and R[-1] > R[0]                       # the return crowds under the trace
    assert L[-1] == pytest.approx(487.4e-9 * 29.5e-3 * 1e9, rel=0.06)         # the 2D field solver's L' x its length
    assert L[0] > 1.1 * L[-1]                                                 # at DC the return spreads: a larger loop
    assert p['joined'] and p['C_F'] is None


def test_the_drives_share_each_frequency_s_factors(thru_rl):
    """One solve per port in turn is the same as the ports solved together (MQS.solve with a list of drives)."""
    q, f = thru_rl['q'], thru_rl['f'][[0, 4, 8]]
    drives = [{'P1.I': 1.0, 'P2.I': 0.0}, {'P1.I': 0.0, 'P2.I': 1.0}]
    together = q.solve(f, amps=drives)
    for d, r in zip(drives, together):
        alone = q.solve(f, amps=d)
        assert np.allclose(r.i, alone.i, rtol=1e-9, atol=1e-12) and r.names == alone.names
        assert np.allclose(r.z['P1.I' if d['P1.I'] else 'P2.I'], alone.z['P1.I' if d['P1.I'] else 'P2.I'])


def test_the_compressed_matrices_give_the_same(thru_rl):
    f = thru_rl['f'][[0, 4, 8]]
    r = SP.solve(thru(), f, backend='hierarchical', **QUIET)
    assert r['info']['backend'] == 'hierarchical'
    assert np.allclose(r['S'], thru_rl['S'][[0, 4, 8]], atol=1e-5)


def test_the_copper_s_thickness_brings_the_skin_effect(thru_rl):
    """--thickness: two face sheets. The same at DC; at 100 MHz the current is on the faces (35 um copper, a skin
    depth of 6.6 um), where one sheet keeps its DC resistance."""
    f = np.array([1e3, 1e8])
    sheet = path(SP.solve(thru(), f, capacitance='off', **QUIET))
    r = SP.solve(thru(), f, capacitance='off', thickness=True, **QUIET)
    faces = path(r)
    assert r['info']['thickness'] and r['info']['triangles'] == 2 * thru_rl['info']['triangles']
    assert 'two faces' in r['info']['model'] and 'one sheet' in thru_rl['info']['model']
    assert faces['R_ohm'][0] == pytest.approx(sheet['R_ohm'][0], rel=0.01)
    assert faces['R_ohm'][1] > 2.2 * sheet['R_ohm'][1] and sheet['R_ohm'][1] < 1.1 * sheet['R_ohm'][0]
    assert faces['L_H'][1] == pytest.approx(sheet['L_H'][1], rel=0.03) and faces['L_H'][1] < sheet['L_H'][1]


def test_a_slot_under_the_trace_lengthens_the_return(thru_rl):
    slot = SP.solve(thru(slot=True), thru_rl['f'], **QUIET)
    L0, L1 = np.array(path(thru_rl)['L_H']), np.array(path(slot)['L_H'])
    assert (L1 > L0 + 1.0e-9).all() and (L1 < L0 + 4e-9).all()               # the current goes round the slot


def test_where_the_ground_via_is_and_whether_there_is_one():
    f = np.geomspace(1e3, 1e7, 5)
    kw = dict(dmax=3.0, fine=0.7, **QUIET)
    L = {}
    for name, vias in (('beside', [(21.2, 10.0)]), ('away', [(20.0, 2.0)]), ('corner', [(38.0, 18.0)])):
        r = SP.solve(layer_change(vias), f, **kw)
        L[name] = np.array(path(r)['L_H'])[2] * 1e9
        assert abs(r['S'][0, 1, 0]) > 0.999                                    # joined: a thru at 1 kHz
    assert L['beside'] + 0.7 < L['away'] < L['corner'] - 0.7                  # nH: the return's detour
    none = SP.solve(layer_change([]), f, **kw)
    p = path(none)
    assert not p['joined'] and p['R_ohm'] is None and np.abs(none['S'][:, 1, 0]).max() < 1e-9
    assert 'ground via missing' in SP.apart_note(p) and 'no current passes at any frequency' in SP.apart_note(p)


def test_the_planes_capacitance_closes_a_return_no_via_joins():
    r = SP.solve(layer_change([]), np.geomspace(1e3, 1e9, 7), dmax=3.0, fine=0.7, **QUIET)
    s21 = 20 * np.log10(np.abs(r['S'][:, 1, 0]))
    assert r['info']['capacitance'] and s21[0] < -80 and s21[-1] > -1 and (np.diff(s21) > 0).all()
    p = path(r)
    plates = 8.854e-12 * 4.3 * 40e-3 * 20e-3 / 1.1e-3                          # the two planes, 1.1 mm apart
    assert not p['joined'] and plates < p['C_F'] < 1.3 * plates
    assert f"{p['C_F'] * 1e12:.3g} pF closes the path" in SP.apart_note(p)


def test_with_the_charge_it_is_a_transmission_line():
    """To a tenth of a wavelength and past it: the thru against the line of the 2D field solver's L' and C'."""
    f = np.array([1e8, 3e8])
    b = thru()
    rl = SP.solve(b, f, capacitance='off', **QUIET)
    rlc = SP.solve(b, f, capacitance='auto', **QUIET)
    assert rlc['info']['capacitance'] and not rl['info']['capacitance']
    assert rlc['info']['valid_to_Hz'] == pytest.approx(rlc['info']['tenth_wavelength_Hz'])
    assert rl['info']['valid_to_Hz'] == pytest.approx(rl['info']['capacitance_limit_Hz'])
    assert rl['info']['valid_to_Hz'] < rlc['info']['valid_to_Hz']
    Lp, Cp, length, z0 = 487.4e-9, 68.3e-12, 29.5e-3, 50.0
    zc, g = np.sqrt(Lp / Cp), 2j * np.pi * f * np.sqrt(Lp * Cp)
    A, B, C = np.cosh(g * length), zc * np.sinh(g * length), np.sinh(g * length) / zc
    line = np.abs((B / z0 - C * z0) / (2 * A + B / z0 + C * z0))
    assert np.abs(rlc['S'][:, 0, 0]) == pytest.approx(line, rel=0.15)
    assert (np.abs(rl['S'][:, 0, 0]) > 1.4 * line).all()                       # without its capacitance: far off
    assert SP.solve(b, [1e5], **QUIET)['info']['capacitance'] is False         # a band below the limit: R and L
    q = rlc['q']                                                               # each frequency's factor, both drives
    assert not q._peec['factors'] and q.factor_cache_bytes == 64 * 1024 ** 2   # nothing kept, the budget as it was
    alone = q.solve(f, amps={'P1.I': 0.0, 'P2.I': 1.0})
    assert np.allclose(rlc['drive'][1][0].i, alone.i, rtol=1e-8, atol=1e-12)
    assert np.allclose(rlc['drive'][1][0].charge, alone.charge, rtol=1e-8, atol=1e-24)


def with_parts(series=dict(R=0.5, L=10e-9, C=None), shunt=dict(R=0.2, L=1e-9, C=100e-12)):
    """The thru with a 1 mm gap at 20 mm bridged by a part in series, and a part from the trace at 25 mm down to a
    pad with a via to the plane (a low-pass filter's two parts)."""
    b = thru()
    half = lambda net, x0, x1: dict(net=net, polygon=mapping(box(x0, 10 - W / 2, x1, 10 + W / 2)))
    b['copper']['F.Cu'] = [half('IN', 5, 19.5), half('OUT', 20.5, 5 + LEN), dict(net='GND', polygon=mapping(box(24.5, 7, 25.5, 8)))]
    b['vias'] = [dict(x=25.0, y=7.5, r=0.15, z0=0.0, z1=H, net='GND')]
    b['ports'][0]['net'], b['ports'][1]['net'] = 'IN', 'OUT'
    part = lambda ref, axis, s, e, v: dict(ref=ref, layer='F.Cu', z=H, axis=axis, s=s + [H], e=e + [H], **v)
    b['parts'] = [part('L1', 0, [19.5, 10 - W / 2], [20.5, 10 + W / 2], series),
                  part('C1', 1, [24.5, 8.0], [25.5, 10 - W / 2], shunt)]
    return b


def test_the_board_s_parts_are_elements_between_their_pads(thru_rl):
    """A part of the board (a kicad2ems board's `parts`: a gap between two pads and its R, L, C) is a series
    element between patches of the copper at the gap's two ends. With the trace cut and bridged by a part, and
    a capacitor from it to ground, S is that of the ladder the parts make with the copper's own resistance and
    inductance; without the parts nothing would get through the cut."""
    b = with_parts()
    took, left = SP.part_elements(b)
    assert [(e['ref'], e['nets']) for e in took] == [('L1', ['IN', 'OUT']), ('C1', ['GND', 'OUT'])] and left == []
    assert SP.signal_nets(b, b['ports']) == {'IN', 'OUT'}           # (through the series part; not the ground)
    f = np.array([1e3, 1e5, 1e6, 1e7])
    r = SP.solve(b, f, **QUIET)
    assert r['info']['parts'] == ['L1', 'C1'] and r['info']['parts_left_out'] == []
    p = path(thru_rl)                                               # the uncut trace's own R and L, per mm
    Rm, Lm = p['R_ohm'][0] / LEN, p['L_H'][-1] / LEN
    w = 2 * np.pi * f
    cu = lambda mm: mm * (Rm + 1j * w * Lm)
    z1 = cu(14.5) + 0.5 + 1j * w * 10e-9 + cu(4.5)                  # port 1 to the capacitor's tap
    y = 1 / (0.2 + 1j * w * 1e-9 + 1 / (1j * w * 100e-12))
    z2 = cu(10.0)
    A, B, C, D = 1 + z1 * y, z1 + z2 + z1 * z2 * y, y, 1 + z2 * y
    den = A + B / 50 + C * 50 + D
    assert np.abs(r['S'][:, 1, 0] - 2 / den).max() < 0.01 and np.abs(r['S'][:, 0, 0] - (A + B / 50 - C * 50 - D) / den).max() < 0.01
    assert abs(r['S'][0, 1, 0]) == pytest.approx(100 / (100 + 0.5 + Rm * 29), abs=2e-4)     # 1 kHz: the resistances
    assert abs(r['S'][3, 0, 0]) > 0.1 > abs(r['S'][0, 0, 0])        # 10 MHz: the capacitor counts
    # a part with no value, and one whose pad is on no copper, are left out and named
    b = with_parts(series=dict(R=None, L=None, C=None))
    b['parts'][1]['s'][1] = 5.0                                     # (the capacitor's lower pad: off its pad)
    b['parts'][1]['e'][1] = 6.0
    took, left = SP.part_elements(b)
    assert took == [] and left == [('L1', 'no value'), ('C1', 'a pad on no copper of F.Cu')]
    r = SP.solve(b, f[:1], **QUIET)
    assert abs(r['S'][0, 1, 0]) < 1e-6 and r['info']['parts_left_out'] == [list(x) for x in left]
    # a single value is that element alone: a 0 ohm link passes everything
    link = SP.solve(with_parts(series=dict(R=0.0, L=None, C=None), shunt=dict(R=None, L=None, C=1e-15)), f[:1], **QUIET)
    assert abs(link['S'][0, 1, 0]) == pytest.approx(100 / (100 + Rm * 29), abs=2e-4)


def test_what_it_refuses():
    b = thru()
    for change, msg in ((dict(ports=[]), 'no ports'), (dict(ports=[dict(b['ports'][0], kind='gap')]), 'no gap ports'),
                        (dict(ports=[dict(b['ports'][0], c=[38.0, 18.0])]), 'no copper on F.Cu')):
        with pytest.raises(ValueError, match=msg):
            SP.solve(dict(b, **change), [1e6], **QUIET)
    for kw, msg in ((dict(backend='fast'), 'backend'), (dict(capacitance='maybe'), 'capacitance')):
        with pytest.raises(ValueError, match=msg):
            SP.solve(b, [1e6], **kw, **QUIET)
    with pytest.raises(ValueError, match='positive'):
        SP.solve(b, [0.0, 1e6], **QUIET)


def test_the_command(tmp_path, capsys):
    path_ = tmp_path / 'thru_board.json'
    path_.write_text(json.dumps(thru()))
    out = tmp_path / 'out'
    assert SP.main([str(path_), '--out', str(out), '--points', '7', '--fmin', '1e3', '--fmax', '1e7']) == 0
    text = capsys.readouterr().out
    assert 'P1 to P2:' in text and 'mOhm' in text and '(dense, R, L)' in text and 'thru_pcbqs.s2p' in text
    r = json.loads((out / 'sparams.json').read_text())
    assert r['tool'] == 'pcbqs.sparams' and r['touchstone'] == 'thru_pcbqs.s2p' and len(r['f_Hz']) == 7
    assert r['paths'][0]['joined'] and 'Resistance and inductance of the copper only' in r['notes'][0]
    lines = (out / 'thru_pcbqs.s2p').read_text().splitlines()
    assert lines[0].startswith('! pcbqs') and lines[1] == '! ports: 1=P1, 2=P2' and lines[2] == '# Hz S RI R 50'
    row = np.array(lines[3].split(), float)
    assert row[0] == 1e3 and row[1] + 1j * row[2] == pytest.approx(complex(r['S_re'][0][0][0], r['S_im'][0][0][0]), rel=1e-6)
    assert row[3] == pytest.approx(r['S_re'][0][1][0], rel=1e-6)               # S21 second, as a two-port file has it
    v = r['viewer']
    assert v['port'] == 'P1' and [d['layer'] for d in v['dumps']] == ['F.Cu', 'B.Cu'] and len(v['frequencies_Hz']) == 3
    info = json.loads((out / 'sparams' / 'exc_P1' / 'dump_info.json').read_text())
    assert [d['unit'] for d in info['dumps']] == ['A/mm per A'] * 2 and [p['name'] for p in info['ports']] == ['P1', 'P2']
    with np.load(out / v['dumps'][1]['file']) as d:                            # the plane's return current
        j = np.abs(d['data'][:, :, :, 0, :2])
        assert d['data'].shape[0] == 3 and d['freqs'][0] == 1e3
        x, y = d['x'], d['y']
    under = np.abs(y - 10.0) < 1.0                                             # the strip of the plane under the trace
    share = lambda k: j[k][:, under].sum() / j[k].sum()
    assert share(2) > 1.5 * share(0)                                           # the return gathers under the trace
    capsys.readouterr()
    assert SP.main(['--describe', '--json']) == 0
    keys = [o['key'] for o in json.loads(capsys.readouterr().out)['options']]
    assert keys[:3] == ['fmin', 'fmax', 'points'] and {'capacitance', 'thickness', 'maps', 'z0'} <= set(keys)
    empty = dict(thru(), ports=[])
    path_.write_text(json.dumps(empty))
    assert SP.main([str(path_), '--out', str(out)]) == 2 and 'no ports' in capsys.readouterr().err


def test_it_is_a_kicad2ems_solver(tmp_path):
    """--solvers pcbqs in a kicad2ems run: registered, never picked by auto, its Touchstone file and its maps
    written with the others'; an example's thru is a thru at 10 kHz."""
    pytest.importorskip('kicad2ems')
    import tomllib
    from kicad2ems.api.solver import FEATURES, registry
    from kicad2ems.cli import main
    from kicad2ems.examples import examples
    from kicad2ems.study import engines
    from pcbqs.analyses import QuasiStatic
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pyproject.toml'), 'rb') as fh:
        assert tomllib.load(fh)['project']['entry-points']['kicad2ems.solvers'] == {'pcbqs': 'pcbqs.analyses:QuasiStatic'}
    sv = QuasiStatic()
    assert set(sv.capabilities) == set(FEATURES) and not sv.in_auto and not sv.in_all and sv.consumes == 'board'
    assert sv.check()[0]
    assert [o.key for o in sv.options] == ['qs_dmax', 'qs_fine', 'qs_capacitance', 'qs_thickness']
    if 'pcbqs' not in registry():
        pytest.skip('pcbqs is not installed where kicad2ems looks for its solvers')
    assert 'sparams' in engines()['pcbqs'].provides and 'antenna' not in engines()['pcbqs'].provides
    from kicad2ems.config import _solvers
    assert 'pcbqs' not in _solvers('all') and _solvers('qs') == ['pcbqs'] == _solvers('quasistatic')
    out = tmp_path / 'run'
    assert main([examples()['sma_launch'], '--solvers', 'pcbqs', '--fmin', '0.00001', '--fmax', '0.001', '--points', '5',
                 '--qs-dmax', '4', '--qs-fine', '1.2', '--out', str(out)]) == 0
    lines = (out / 'sma_launch_pcbqs.s2p').read_text().splitlines()
    assert 'pcbqs quasi-static' in lines[0] and lines[1] == '! ports: 1=P1, 2=P2'
    first = np.array(lines[3].split(), float)
    assert first[0] == 1e4 and abs(first[3] + 1j * first[4]) > 0.999           # S21 at 10 kHz
    run = json.loads((out / 'sma_launch_run.json').read_text())
    info = run['solved']['pcbqs']
    assert info['capacitance'] is False and info['triangles'] > 100 and 'quasi-static' in info['about']
    assert (out / 'pcbqs' / 'exc_P1' / 'dump_info.json').exists() and (out / 'pcbqs_board.json').exists()
