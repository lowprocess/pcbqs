"""The rail tree, the DC drop over it with pads as terminals and series parts, decaps through their copper, and the
PDN over several plane pairs, on a small synthetic board."""
import numpy as np
import pytest
from shapely.geometry import box, mapping, Point

from pcbqs._numerics.base import MU0, EPS0
from pcbqs.pdn.rail import Rail, parse_value
from pcbqs.pdn.dc import solve_rail
from pcbqs.pdn.stack import RailPDN

W, H = 40.0, 30.0
# F.Cu 0.6 / 0.1 / In1 GND 0.5 / 0.1 / In2 P plane 0.4 / 0.1 / In3 GND 0.3 / 0.3 / B.Cu 0
STACK = dict(layers=[dict(name='d4', z0=0.0, z1=0.3, er=4.0, tand=0.0, kind='diel'),
                     dict(name='d3', z0=0.3, z1=0.4, er=4.0, tand=0.0, kind='diel'),
                     dict(name='d2', z0=0.4, z1=0.5, er=4.0, tand=0.0, kind='diel'),
                     dict(name='d1', z0=0.5, z1=0.6, er=4.0, tand=0.0, kind='diel')],
             copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='In3.Cu', z=0.3, t=0.035),
                     dict(name='In2.Cu', z=0.4, t=0.035), dict(name='In1.Cu', z=0.5, t=0.035),
                     dict(name='F.Cu', z=0.6, t=0.035)], thickness=0.6)


def pad(name, net, x, y, w=1.0, h=1.0, layer='F.Cu'):
    return dict(name=name, net=net, xy=[x, y], layers=[layer], geom=mapping(box(x - w / 2, y - h / 2, x + w / 2, y + h / 2)))


def fp(ref, value, pads, lib='C_0402_1005Metric'):
    return dict(ref=ref, value=value, props={'Value': value}, lib=lib, side='F', dnp=False, exclude_from_sim=False,
                pads=pads, x=0, y=0, rot=0)


def board(trace_decap=True, via_in_pad=False):
    """A 40 x 30 mm board: a buck (U3 driving L1 from SW) to VOUT, a 10 mOhm sense R1 to the plane net P (In2,
    between the In1 and In3 grounds), a bead FB1 to the filtered PF (F.Cu pour, its own C3 and load U2), the feedback
    R2 (10 k) to FBK, decaps C1 (on vias) and C2 (5 mm of 0.5 mm trace to its via), the load U1."""
    cu = {l['name']: [] for l in STACK['copper']}
    plane = box(1, 1, W - 1, H - 1)
    cu['In1.Cu'].append(dict(net='GND', polygon=mapping(plane)))
    cu['In3.Cu'].append(dict(net='GND', polygon=mapping(plane)))
    cu['In2.Cu'].append(dict(net='P', polygon=mapping(plane)))
    F = []                                              # F.Cu: (net, geometry)
    F += [('SW', box(3, 13, 6, 17)), ('VOUT', box(7, 13, 12, 17)), ('P', box(13, 13, 17, 17)),
          ('GND', box(3, 6, 8, 11)), ('FBK', box(7, 20, 9, 21))]
    F += [('P', box(30, 13, 34, 17)), ('GND', box(30, 8, 34, 12))]          # U1's pads
    F += [('P', box(22, 4, 23, 5)), ('P', box(22.25, 5, 22.75, 6.4)), ('GND', box(22, 2, 23, 3)),
          ('GND', box(22.25, 1.1, 22.75, 2))]                                # C1 and its short necks
    F += [('P', box(25, 24, 26, 25)), ('P', box(25.25, 19.5, 25.75, 24.5)), ('GND', box(25, 26, 26, 27))]   # C2 + trace
    F += [('P', box(17, 22, 18, 23)), ('PF', box(19, 21, 24.8, 23)), ('GND', box(19, 26.5, 21, 28))]
    for n, gm in F:
        cu['F.Cu'].append(dict(net=n, polygon=mapping(gm)))
    vias = []
    def via(x, y, net, z0=0.0, z1=0.6, r=0.15):
        vias.append(dict(x=x, y=y, r=r, z0=z0, z1=z1, net=net))
    for x in (14, 15, 16):
        via(x, 14, 'P')                                 # the sense resistor's plane side down to In2
    for x in (31, 33):
        via(x, 14, 'P'); via(x, 10, 'GND')              # U1
    via(22.5, 6.0, 'P'); via(22.5, 1.5, 'GND')        # C1 (a short neck to each via)
    via(25.5, 19.75, 'P'); via(25.5, 26.5, 'GND')     # C2: its via at the end of the trace (ground: in its pad)
    via(17.5, 22.5, 'P')                              # the bead's plane side
    via(20, 27.2, 'GND')
    for x in (4, 6):
        via(x, 8, 'GND')                                # the regulator's ground
    fps = [fp('U3', 'BUCK', [pad('1', 'SW', 4.5, 15), pad('2', 'GND', 5.5, 8.5, 3, 3)]),
           fp('L1', '2.2uH', [pad('1', 'SW', 5.5, 15), pad('2', 'VOUT', 8.0, 15)], lib='L_1210'),
           fp('R1', '0R01', [pad('1', 'VOUT', 11.0, 15), pad('2', 'P', 14.0, 15)], lib='R_0805_2012Metric'),
           fp('R2', '10k', [pad('1', 'VOUT', 8.0, 16.5, 0.5, 0.5), pad('2', 'FBK', 8.0, 20.5, 0.5, 0.5)]),
           fp('FB1', 'BLM18 600R@100MHz', [pad('1', 'P', 17.5, 22.5), pad('2', 'PF', 19.5, 22)], lib='L_0603_1608Metric'),
           fp('C3', '1u', [pad('1', 'PF', 20.0, 22.5), pad('2', 'GND', 20.0, 27.2)]),
           fp('U2', 'LOAD2', [pad('1', 'PF', 24.0, 22), pad('2', 'GND', 20.5, 27.5)]),
           fp('U1', 'LOAD', [pad('1', 'P', 32, 15, 3, 3), pad('2', 'GND', 32, 10, 3, 3)]),
           fp('C1', '100n', [pad('1', 'P', 22.5, 4.5), pad('2', 'GND', 22.5, 2.5)]),
           fp('C2', '10u', [pad('1', 'P', 25.5, 24.5), pad('2', 'GND', 25.5, 26.5)], lib='C_0805_2012Metric')]
    if via_in_pad:
        via(32, 15, 'P')                                # a via in the middle of U1's supply pad
    return dict(stack=STACK, outline=mapping(box(0, 0, W, H)), copper=cu, vias=vias,
                prims=dict(footprints=fps))


def test_values():
    assert parse_value('0R001') == pytest.approx(1e-3) and parse_value('4k7') == 4700
    assert parse_value('1u8') == pytest.approx(1.8e-6) and parse_value('100n') == pytest.approx(1e-7)


def test_rail_tree_follows_sense_resistors_and_beads_not_dividers():
    r = Rail(board(), 'P', loads={'U1': 2.0, 'U2': 0.5})
    assert r.nets == {'P', 'VOUT', 'PF'}
    kinds = {s.part.ref: s.kind for s in r.series}
    assert kinds == {'R1': 'R', 'FB1': 'bead'}
    assert [p.ref for p in r.source_parts] == ['L1'] and {q.net for q in r.source_pads} == {'VOUT'}
    assert {q.part for q in r.ground_source_pads} == {'U3'}
    assert {p.ref for p in r.decaps} == {'C1', 'C2', 'C3'}
    # a 0.5 ohm sense resistor is in the tree at the default 1 ohm, out of it at 0.1
    b = board()
    next(f for f in b['prims']['footprints'] if f['ref'] == 'R1')['props']['Value'] = '0R5'
    assert 'VOUT' in Rail(b, 'P', source='L1').nets
    with pytest.raises(ValueError):                     # VOUT is then not on the tree: L1 feeds nothing
        Rail(b, 'P', max_ohms=0.1)


def test_dc_over_the_tree_drops_the_sense_resistor():
    r = Rail(board(), 'P', loads={'U1': 2.0, 'U2': 0.5})
    res = solve_rail(r, 3.3, dmax=0.8, log=lambda *a: None)
    s = res['series']
    assert s['R1']['I'] == pytest.approx(2.5, rel=1e-4)              # all of the loads' current
    assert s['R1']['drop'] == pytest.approx(2.5 * 0.01, rel=1e-4)    # through its 10 mOhm
    assert s['FB1']['I'] == pytest.approx(0.5, rel=1e-4)
    u1 = res['loads']['U1']
    assert 0.025 < u1['power_drop'] < 0.05                           # the sense R and some copper
    assert u1['ground_rise'] > 0


def test_via_in_pad_is_part_of_the_pad():
    r = Rail(board(via_in_pad=True), 'P', loads={'U1': 2.0, 'U2': 0.5})
    res = solve_rail(r, 3.3, dmax=0.8, log=lambda *a: None)
    v = [q for q in res['power'].vias if (q['x'], q['y']) == (32, 15)][0]
    assert v['I'] > 0.1                                              # it carries a share of U1's current


def test_decap_path_through_its_trace():
    """C2's supply pad is 4.25 mm of 0.5 mm trace from its via: 8.5 squares less the via's ring, 0.1 mm over In1."""
    r = Rail(board(), 'P', loads={'U1': 2.0, 'U2': 0.5})
    pdn = RailPDN(r, dmax=1.0, log=lambda *a: None)
    c2 = pdn.parts['C2']['rail']
    sq = c2.L_path / (MU0 * c2.h * 1e-3)
    assert 6.5 < sq < 9.5 and c2.h == pytest.approx(0.1)            # In1 is 0.1 mm under F.Cu
    assert pdn.parts['C1']['rail'].L_path < c2.L_path / 2
    assert pdn.parts['C3']['rail'].node == 'PF'                      # the filtered net: a lumped node


def test_two_plane_pairs_in_parallel():
    """In2 between the In1 and In3 grounds: two cavities of 0.1 mm, joined; at low frequency the planes alone are
    their two capacitances in parallel."""
    r = Rail(board(), 'P', loads={'U1': 2.0, 'U2': 0.5})
    pdn = RailPDN(r, dmax=1.0, log=lambda *a: None)
    assert {(c.rail_layer, c.ground_layer) for c in pdn.cavities} == {('In2.Cu', 'In1.Cu'), ('In2.Cu', 'In3.Cu')}
    C = sum(EPS0 * c.er * c.geom.area * 1e-6 / (c.d * 1e-3) for c in pdn.cavities)
    f = 1e6
    z = pdn.z([f], observe=['U1'], bare=True)['U1'][0]
    assert abs(z) == pytest.approx(1 / (2 * np.pi * f * C), rel=0.03)


def test_rail_pdn_with_its_parts():
    r = Rail(board(), 'P', loads={'U1': 2.0, 'U2': 0.5})
    pdn = RailPDN(r, dmax=1.0, vrm=dict(R=2e-3, L=20e-9), log=lambda *a: None)
    f = np.logspace(3, 9, 61)
    z = pdn.z(f)
    # at 1 kHz the regulator, its sense resistor: R_vrm + R1 (+ copper)
    assert abs(z['U1'][0]) == pytest.approx(2e-3 + 0.01, rel=0.15)
    # U2, behind the bead: at 1 kHz the bead's DCR in series too; at 100 MHz its own decap C3 (0402: 0.35 nH ESL and
    # a lumped node's nominal 0.5 nH), the bead's 600 ohm cutting it off from the plane
    assert abs(z['U2'][0]) == pytest.approx(2e-3 + 0.01 + 0.05, rel=0.15)
    k = np.argmin(abs(f - 1e8))
    assert abs(z['U2'][k]) == pytest.approx(2 * np.pi * f[k] * 0.85e-9, rel=0.1)
    z0 = pdn.z(f[k:k + 1], observe=['U1'], bare=True)['U1'][0]
    assert abs(z['U1'][k]) < 0.5 * abs(z0)                           # C1, C2 (~0.7 nH together) on the planes


def test_cli_describes_plane_min():
    # The window's Power Integrity form is generated from describe(): a small
    # rail's plane threshold must be offered there.
    import pcbqs.pdn.cli as cli
    options = {o['key']: o for o in cli.describe()['options']}
    assert options['plane_min']['flag'] == '--plane-min' and options['plane_min']['default'] == 0.0
    assert options['plane_min']['group'] == 'PDN impedance'


def test_followed_inductor_keeps_its_inductance_and_dcr():
    # An inductor named in through= is a series filter inductor: its henries and
    # DCR stay in the model (DC drop and PDN impedance), not a 10 uOhm link.
    import numpy as np
    from pcbqs.pdn.rail import DEFAULT_INDUCTOR_DCR
    from pcbqs.pdn.stack import RailPDN
    r = Rail(board(), 'P', loads={'U1': 2.0, 'U2': 0.5}, source='U3', through=('L1',), dcr={'L1': 0.02, 'R1': 0.012})
    series = {s.part.ref: s for s in r.series}
    assert r.nets == {'P', 'VOUT', 'PF', 'SW'}
    L1 = series['L1']
    assert (L1.kind, L1.L, L1.R, L1.defaulted) == ('inductor', pytest.approx(2.2e-6), 0.02, False)
    assert series['R1'].R == 0.012 and series['R1'].L == 0.0
    assert any('L1 inductor 2.2 uH 0.02 ohm' in line for line in r.describe())
    w = 2 * np.pi * 1e5
    z = RailPDN._series_z(None, dict(s=L1, esl=0.5e-9), w)
    assert z == pytest.approx(0.02 + 1j * w * (2.2e-6 + 0.5e-9))
    assumed = {s.part.ref: s for s in Rail(board(), 'P', loads={'U1': 2.0}, source='U3', through=('L1',)).series}['L1']
    assert assumed.R == DEFAULT_INDUCTOR_DCR and assumed.defaulted


def test_cli_describes_dcr():
    import pcbqs.pdn.cli as cli
    options = {o['key']: o for o in cli.describe()['options']}
    assert options['dcr']['kind'] == 'list' and options['dcr']['flag'] == '--dcr'


def test_prefixed_capacitors_are_decaps_with_their_values():
    # Bulk electrolytics are often CA*/CE*/CP*: they must be decaps, with
    # electrolytic package defaults (flagged) unless datasheet values are given.
    from pcbqs.pdn.stack import RailPDN, package
    assert package('CP_Radial_D8.0mm_P3.50mm') == 'radial' and package('CP_Elec_6.3x5.9') == 'elec_smd'
    b = board()
    c2 = next(f for f in b['prims']['footprints'] if f['ref'] == 'C2')
    c2['ref'], c2['lib'] = 'CA2', 'CP_Radial_D8.0mm_P3.50mm'
    for p in c2['pads']:
        p['ref'] = 'CA2'
    r = Rail(b, 'P', loads={'U1': 2.0, 'U2': 0.5})
    assert 'CA2' in {p.ref for p in r.decaps}
    pdn = RailPDN(r, dmax=1.0, vrm=dict(R=2e-3, L=20e-9), log=lambda *a: None)
    el = pdn.parts['CA2']['el']
    assert (el['C'], el['ESR'], el['ESL'], el['defaulted']) == (pytest.approx(10e-6), 0.05, 3e-9, True)
    given = RailPDN(r, dmax=1.0, vrm=dict(R=2e-3, L=20e-9), log=lambda *a: None,
                    caps={'CA2': dict(C=47e-6, ESR=0.02, ESL=1e-9)}).parts['CA2']['el']
    assert (given['C'], given['ESR'], given['ESL'], given['defaulted']) == (47e-6, 0.02, 1e-9, False)


def test_cli_describes_cap():
    import pcbqs.pdn.cli as cli
    options = {o['key']: o for o in cli.describe()['options']}
    assert options['cap']['kind'] == 'list' and options['cap']['flag'] == '--cap'


def _with_high_side(b):
    """Q1: a load with a rail pad only (a bridge's high-side FET: its current returns through the low side)."""
    b['prims']['footprints'].append(fp('Q1', 'FET', [pad('1', 'P', 15.0, 16.0, 0.8, 0.8)], lib='SON-8'))
    return b


def test_a_load_returns_through_another_part():
    """--load REF=A@RETURN (C9): a load without ground pads returns through the named part's, so its ground rise
    is modelled instead of noted as missing."""
    b = _with_high_side(board())
    alone = solve_rail(Rail(b, 'P', loads={'Q1': 1.0}), 3.3, dmax=0.8, log=lambda *a: None)
    assert alone['loads']['Q1']['ground_rise'] == 0 and any('Q1: no ground pad' in n for n in alone['notes'])
    assert alone['no_return'] == ['Q1']
    back = solve_rail(Rail(b, 'P', loads={'Q1': 1.0}, returns={'Q1': 'U1'}), 3.3, dmax=0.8, log=lambda *a: None)
    assert back['loads']['Q1']['ground_rise'] > 0 and not any('Q1: no ground pad' in n for n in back['notes'])
    with pytest.raises(ValueError):
        Rail(b, 'P', loads={'Q1': 1.0}, returns={'Q1': 'R2'})            # R2 has no ground pad


def test_a_half_bridge_returns_through_its_low_side():
    """A load with no pad on ground returns, unasked, through the one part of its kind on its other net and on
    ground (a half bridge's low-side switch), and the result says so; a return that is named is the one taken."""
    b = board()
    b['prims']['footprints'] += [
        fp('Q1', 'FET', [pad('1', 'P', 15.0, 16.0, 0.8, 0.8), pad('2', 'HB', 15.0, 19.0, 0.8, 0.8)], lib='SON-8'),
        fp('Q2', 'FET', [pad('1', 'HB', 31.0, 19.0, 0.8, 0.8), pad('2', 'GND', 31.0, 11.0, 0.8, 0.8)], lib='SON-8'),
        fp('U9', 'DRV', [pad('1', 'HB', 16.0, 19.0, 0.5, 0.5), pad('2', 'GND', 33.0, 11.0, 0.5, 0.5)])]
    rail = Rail(b, 'P', loads={'Q1': 1.0})
    assert rail.returns == {'Q1': 'Q2'}                 # not U9, the driver: on the node and on ground, not a switch
    assert any('Q1 has no pad on ground: its return is taken through Q2' in n for n in rail.notes)
    res = solve_rail(rail, 3.3, dmax=0.8, log=lambda *a: None)
    assert res['no_return'] == [] and res['loads']['Q1']['ground_rise'] > 0
    named = Rail(b, 'P', loads={'Q1': 1.0}, returns={'Q1': 'U1'})
    assert named.returns == {'Q1': 'U1'} and not any('no pad on ground' in n for n in named.notes)
    b['prims']['footprints'].append(fp('Q3', 'FET', [pad('1', 'HB', 32.0, 19.0, 0.8, 0.8),
                                                    pad('2', 'GND', 32.0, 9.0, 0.8, 0.8)], lib='SON-8'))
    two = Rail(b, 'P', loads={'Q1': 1.0})               # two to choose from: none taken
    assert two.returns == {}
    assert solve_rail(two, 3.3, dmax=0.8, log=lambda *a: None)['no_return'] == ['Q1']


def test_cli_load_returns_and_targets(tmp_path):
    """The PDN CLI: --load REF=A@RETURN, per-load targets (--load-target) and none (--no-target) (C10)."""
    import json
    from pcbqs.pdn.cli import main
    path = tmp_path / 'b_board.json'
    path.write_text(json.dumps(_with_high_side(board())))
    common = [str(path), '--rail', 'P', '--volts', '3.3', '--load', 'U1=2', '--load', 'Q1=1@U1', '--dmax', '1',
              '--points', '7']
    main(common + ['--load-target', 'U1=0.02', '--out', str(tmp_path / 'a')])
    r = json.loads((tmp_path / 'a' / 'pdn.json').read_text())
    assert r['loads']['Q1']['ground_rise'] > 0
    assert r['pdn']['z']['U1']['target'] == 0.02 and r['pdn']['z']['Q1']['target'] == r['pdn']['target'] > 0
    main(common + ['--no-target', '--out', str(tmp_path / 'b')])
    r = json.loads((tmp_path / 'b' / 'pdn.json').read_text())
    assert r['pdn']['target'] is None and all(z['target'] is None and z['over'] == [] for z in r['pdn']['z'].values())


def test_cli_writes_the_viewer_maps(tmp_path):
    """The DC solve where kicad2ems's viewer reads it (pdn/viewer.py): the board, the mesh, per layer the drop (mV,
    as the solve's loads see it) and the sheet current, as 0 Hz plane dumps."""
    import json
    from pcbqs.pdn.cli import main, describe
    assert 'viewer' in describe()['features']
    path = tmp_path / 'b_board.json'
    path.write_text(json.dumps(_with_high_side(board())))
    main([str(path), '--rail', 'P', '--volts', '3.3', '--load', 'U1=2', '--dmax', '1', '--no-pdn', '--out', str(tmp_path / 'o')])
    r = json.loads((tmp_path / 'o' / 'pdn.json').read_text())
    v = r['viewer']
    assert (tmp_path / 'o' / v['board']).is_file() and (tmp_path / 'o' / v['mesh']).is_file()
    assert {(d['side'], d['field']) for d in v['dumps']} == {('power', 'V'), ('power', 'J'), ('ground', 'V'), ('ground', 'J')}
    info = json.loads((tmp_path / 'o' / 'pdn' / 'exc_power' / 'dump_info.json').read_text())
    assert {d['unit'] for d in info['dumps']} == {'mV', 'A/mm'} and info['outline'] and info['cu_z']
    drop = max(float(np.abs(np.load(tmp_path / 'o' / d['file'])['data']).max()) for d in v['dumps']
               if d['side'] == 'power' and d['field'] == 'V')
    assert drop == pytest.approx(r['loads']['U1']['power_drop'] * 1e3, rel=0.15)    # the map's worst: the load's


def test_loads_from_the_parts_fields(tmp_path):
    """A part's pdn.load field is its load: amps for the rail it is on, NET=AMPS for a part on several rails,
    AMPS@RETURN; --load overrides one, --no-part-loads ignores them, a field that cannot be read is noted."""
    import json
    from pcbqs.pdn.cli import main, describe
    from pcbqs.pdn.rail import field_loads
    b = _with_high_side(board())
    fps = {f['ref']: f for f in b['prims']['footprints']}
    fps['U1']['props']['pdn.load'] = '2 A'
    fps['U2']['props']['PDN.Load'] = 'PF=500m, +1V8=3'                  # only its share on this rail's nets
    fps['Q1']['props']['pdn.load'] = '1@U1'
    fps['R2']['props']['pdn.load'] = 'lots'
    fps['U3']['props']['pdn.load'] = '+12V=4'                           # on VOUT, but its field names another rail
    assert field_loads(b, {'P', 'PF'}) == ({'U1': 2.0, 'U2': 0.5, 'Q1': 1.0}, {'Q1': 'U1'}, [])
    assert field_loads(b, {'P'})[0] == {'U1': 2.0, 'Q1': 1.0}             # U2 is on PF: not this net's
    loads, _, notes = field_loads(b, {'VOUT', 'P', 'PF'})               # R2 hangs on VOUT: its field is read, and bad
    assert 'R2' not in loads and notes == ["R2: its pdn.load field 'lots' is not AMPS, NET=AMPS or AMPS@RETURN (ignored)"]
    assert any(o['key'] == 'no_part_loads' and o['group'] == 'Rail' for o in describe()['options'])

    path = tmp_path / 'b_board.json'
    path.write_text(json.dumps(b))
    common = [str(path), '--rail', 'VOUT', '--volts', '3.3', '--dmax', '1', '--no-pdn']
    main(common + ['--out', str(tmp_path / 'a')])                       # no --load at all
    r = json.loads((tmp_path / 'a' / 'pdn.json').read_text())
    assert {k: v['I'] for k, v in r['loads'].items()} == {'U1': 2.0, 'U2': 0.5, 'Q1': 1.0}
    assert r['field_loads'] == ['Q1', 'U1', 'U2'] and r['loads']['Q1']['ground_rise'] > 0 and not r['assumed_loads']
    assert any(n.startswith("R2: its pdn.load field 'lots'") for n in r['notes'])
    main(common + ['--load', 'U1=3', '--out', str(tmp_path / 'b')])     # one typed: it overrides that part's field
    r = json.loads((tmp_path / 'b' / 'pdn.json').read_text())
    assert r['loads']['U1']['I'] == 3.0 and r['loads']['U2']['I'] == 0.5 and r['field_loads'] == ['Q1', 'U2']
    main(common + ['--load', 'U1=3', '--no-part-loads', '--out', str(tmp_path / 'c')])
    r = json.loads((tmp_path / 'c' / 'pdn.json').read_text())
    assert list(r['loads']) == ['U1'] and r['field_loads'] == []
    with pytest.raises(SystemExit, match='a pdn.load field on the parts'):
        main(common + ['--no-part-loads', '--out', str(tmp_path / 'd')])
