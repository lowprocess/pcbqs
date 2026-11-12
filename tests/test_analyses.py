"""pcbqs's tools as kicad2ems analyses (analyses.py), what a result folder holds for a front end (reports.py: the
report to read, the curves to plot), the temperature maps where its viewer reads them (thermal/viewer.py) and the
circuit examples as configs."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box, mapping

from pcbqs import reports as R

EXAMPLES = Path(__file__).resolve().parents[1] / 'src' / 'pcbqs' / 'circuit' / 'examples'
A = pytest.importorskip('pcbqs.analyses', reason='the plugins need kicad2ems (its api.analysis)')


def cells(rep):
    """Every table of a report: {section title: (columns, rows)}; each row as long as its columns and formats."""
    out = {}
    for s in rep['sections']:
        t = s.get('table')
        if t:
            assert all(len(r) == len(t['columns']) == len(t['formats']) for r in t['rows']), s['title']
            for r in t['rows']:
                for v, f in zip(r, t['formats']):
                    if isinstance(v, (int, float)) and f:
                        format(v, f)                                # the format a number is shown with
            out[s['title']] = (t['columns'], t['rows'])
    return out


# ---- the plugins -------------------------------------------------------------------------------------------
def test_the_four_register_with_kicad2ems():
    """Declared as kicad2ems expects (its api, a name, a result file, options of its kinds), in the Study's order."""
    from kicad2ems.api.analysis import Analysis, KINDS, API_VERSION
    made = [cls() for cls in (A.PowerIntegrity, A.NearField, A.Circuit, A.Thermal)]
    assert [a.name for a in made] == ['pdn', 'mqs', 'circuit', 'thermal'] == [a.name for a in sorted(made, key=lambda a: a.order)]
    for a in made:
        assert isinstance(a, Analysis) and a.api[0] == API_VERSION[0] and a.api <= API_VERSION
        assert a.label and a.tool.startswith('pcbqs.') and a.result.endswith('.json') and a.about
        assert a.check() == (True, a.check()[1]) and 'pcbqs' in a.check()[1]
        opts = a.options()
        assert opts and all(o['kind'] in KINDS and o['flag'].startswith('--') for o in opts)
        assert set(a.paths) <= {o['key'] for o in opts}
    assert made[1].previewed and made[1].preview_result == 'mqs-preview.json' and made[2].page
    import tomllib
    eps = tomllib.load(open(Path(__file__).resolve().parents[1] / 'pyproject.toml', 'rb'))['project']['entry-points']
    assert eps['kicad2ems.analyses'] == {a.name: f"pcbqs.analyses:{type(a).__name__}" for a in made}


def test_power_integrity_and_near_field_commands(tmp_path):
    b = tmp_path / 'b.kicad_pcb'
    b.write_text('')
    pdn, mqs = A.PowerIntegrity(), A.NearField()
    with pytest.raises(ValueError, match='rail'):
        pdn.command(dict(volts='3.3'), str(b), 'out')
    with pytest.raises(ValueError, match='board'):
        pdn.command(dict(rail='P'), None, 'out')
    cmd = pdn.command(dict(rail='+3V3', volts='3.3', load='U1=2\nU2=1@U1', cap='CA1=47e-6,0.02,1e-9, C2=1e-7',
                           no_pdn=True, dmax='', fmin='10000.0'), str(b), 'out')
    assert cmd[:6] == [sys.executable, '-m', 'pcbqs.pdn', str(b), '--out', 'out']
    assert cmd[6:] == ['--rail', '+3V3', '--load', 'U1=2', '--load', 'U2=1@U1', '--cap', 'CA1=47e-6,0.02,1e-9',
                       '--cap', 'C2=1e-7', '--no-pdn']                  # volts and fmin at their defaults: not passed
    assert pdn.command(dict(rail='P', load=['U1=2', 'U2=1']), str(b), 'o')[-4:] == ['--load', 'U1=2', '--load', 'U2=1']
    assert pdn.tag(dict(rail=' +3V3 ')) == '+3V3' and pdn.status('pdn: DC: meshing') == 'DC: meshing'
    assert pdn.status('RuntimeWarning: x') is None
    cmd = mqs.command(dict(converter='U7', fsw='1e6'), str(b), 'out', preview=True)
    assert cmd[2] == 'pcbqs.mqs' and cmd[-1] == '--preview' and ['--converter', 'U7'] == cmd[6:8]
    assert mqs.tag({}) == 'auto' and mqs.status('mqs: filling') == 'filling'
    assert mqs.discover(dict(ground='AGND'), str(b)) == [sys.executable, '-m', 'pcbqs.mqs', str(b), '--list-converters',
                                                         '--ground', 'AGND']
    found = [dict(converter='U7', inductor='L3', switch='SW', inputs=[
        dict(vin='VIN', capacitors=['C1', 'C2']), dict(vin='VDRV', capacitors=[])])]
    picks = mqs.choices('reading the board\n' + json.dumps(found))
    assert [v for _l, v in picks] == [dict(converter='U7', vin='VIN', capacitors='C1,C2'),
                                      dict(converter='U7', vin='VDRV', capacitors='')]
    assert 'inductor L3' in picks[0][0] and '2 capacitors: C1, C2' in picks[0][0] and mqs.choices('no json') == []


def circuit_folder(tmp_path, board=None, name='circuit-run'):
    run = tmp_path / name
    run.mkdir()
    (run / 'model.json').write_text(json.dumps(dict(format=2, parts=[], **(dict(board=dict(file=str(board))) if board else {}))))
    (run / 'circuit.json').write_text(json.dumps(dict(format=2, cases=[])))
    return run


def test_where_the_temperature_takes_its_losses_and_its_board(tmp_path):
    """Typed losses on the board given; else a circuit run's (the one named, else the latest), on that run's
    board when its model names one."""
    th = A.Thermal()
    study, theirs = tmp_path / 'study.kicad_pcb', tmp_path / 'theirs.kicad_pcb'
    study.write_text('')
    theirs.write_text('')
    with pytest.raises(ValueError, match='losses'):
        th.command({}, str(study), 'out')
    with pytest.raises(ValueError, match='board'):
        th.command(dict(loss='Q1=0.2'), None, 'out')
    cmd = th.command(dict(loss='Q1=0.2\nL1=0.5', film='12', method='grid'), str(study), 'out')
    assert cmd[2:] == ['pcbqs.thermal', str(study), '--out', 'out', '--loss', 'Q1=0.2', '--loss', 'L1=0.5', '--film', '12']
    run = circuit_folder(tmp_path, theirs)
    cmd = th.command({}, str(study), 'out', results=dict(circuit=str(run)))      # the latest run, on its board
    assert cmd[2:] == ['pcbqs.thermal', str(theirs), '--out', 'out', '--circuit', str(run)]
    bare = circuit_folder(tmp_path, None, 'bare')
    cmd = th.command(dict(circuit=str(bare), case='full'), str(study), 'out')    # a run that names no board
    assert cmd[2:] == ['pcbqs.thermal', str(study), '--out', 'out', '--circuit', str(bare), '--case', 'full']
    with pytest.raises(ValueError, match='not a circuit run'):
        th.command(dict(circuit=str(tmp_path)), str(study), 'out')
    assert th.tag(dict(loss='Q1=1')) == 'losses' and th.tag(dict(circuit=str(run) + '/')) == 'circuit'
    assert th.tag({}) == 'circuit' and th.tag(dict(circuit=str(run), case='full')) == 'full'
    assert th.status('0.800 W into the board, 0.800 W out through the films') and th.status('note: x') is None


def test_the_circuit_command(tmp_path):
    c = A.Circuit()
    with pytest.raises(ValueError, match='model'):
        c.command({}, None, 'out')
    model = tmp_path / 'm.json'
    model.write_text(json.dumps(dict(format=2)))
    assert c.command(dict(model=str(model), jobs=2), None, 'out') == [
        sys.executable, '-u', '-m', 'pcbqs.circuit', str(model), '--out', 'out', '--jobs', '2']
    with pytest.raises(ValueError, match='board'):
        c.command(dict(model=str(model), copper=True), None, 'out')
    b = tmp_path / 'b.kicad_pcb'
    b.write_text('')
    line = c.command(dict(model=str(model), fields='true', set='drives.pwm.duty=0.4\nparameters.vin=9', powers='5'), str(b), 'out')
    assert line[7:] == ['--fields', '--set', 'drives.pwm.duty=0.4', '--set', 'parameters.vin=9', '--copper', '--jobs', '4',
                        '--board', str(b)]                          # the bridge's powers: not this tool's
    model.write_text(json.dumps(dict(format=1)))                    # the full bridge on a board's terminals
    assert c.command(dict(model=str(model)), None, 'out')[3:] == ['pcbqs.power_circuit', str(model), '--out', 'out', '--jobs', '4']
    line = c.command(dict(model=str(model), copper=True, frequencies='90000,110000', powers='5', coupling=0.3, receiver_c='',
                          load=2.5, switch_type='vdmos', override=['tank.ESR_ohm=0.004', 'filter.DCR_ohm=0.01'], jobs=2,
                          set='x=1', field_harmonics=16), str(b), 'out')
    assert line[7:] == ['--copper', '--frequencies', '90000,110000', '--powers', '5', '--coupling', '0.3', '--load', '2.5',
                        '--switch-type', 'vdmos', '--override', 'tank.ESR_ohm=0.004', '--override', 'filter.DCR_ohm=0.01',
                        '--jobs', '2', '--board', str(b)]           # its default harmonics and the other tool's --set: not passed
    assert c.tag(dict(model=str(model))) == 'm' and c.env
    opts = {o['key']: o for o in c.options()}                        # the two tools' own tables, by the kind of model
    assert [k for k, o in opts.items() if o['only'] == ''] == ['model', 'copper', 'fields', 'jobs']
    assert [k for k, o in opts.items() if o['only'] == 'general'] == ['set'] and opts['jobs']['default'] == 4
    assert [k for k, o in opts.items() if o['only'] == 'bridge'] == ['field_harmonics', 'frequencies', 'powers', 'coupling',
                                                                     'receiver_c', 'load', 'switch_type', 'override']
    assert opts['switch_type']['choices'][0] == 'vdmos' and 'board' not in opts


# ---- the circuit examples as configs -----------------------------------------------------------------------
def test_the_examples_are_configs_beside_their_models(tmp_path):
    """Each a config kicad2ems reads: its board and its model beside it, the circuit ticked; prepared in a folder
    of the user's, a run of it writes there and not into the package."""
    import tomllib
    ex = A.Circuit().examples()
    assert [e['name'] for e in ex][:4] == ['circuit_rc_filter', 'circuit_half_wave_rectifier',
                                           'circuit_bridge_rectifier', 'circuit_buck']
    assert len(ex) == 11 and {e['group'] for e in ex} == {'Circuits/Passive and rectifiers',
                                                           'Circuits/DC-DC converters', 'Circuits/Bridges'}
    for e in ex:
        cfg = tomllib.load(open(e['config'], 'rb'))
        here = Path(e['config']).parent
        assert (here / cfg['analysis']['circuit']['model']).is_file() and 'circuit' in cfg['analysis']['run']
        assert all((here / f).is_file() for f in e['files']) and cfg['analysis']['circuit']['model'] in e['files']
        if 'board' in cfg:
            assert cfg['board'] in e['files'] and e['title']
    assert len({m.name for m in EXAMPLES.glob('*.json')}) == len(list(EXAMPLES.glob('*.toml'))) == 10
    from kicad2ems.api import frontend
    from kicad2ems import analyses
    if 'circuit' not in analyses.registry():
        pytest.skip('pcbqs is not installed where kicad2ems looks for plugins (its entry points)')
    names = [e['name'] for e in frontend.example_index() if e['source'] == 'circuit']
    assert names == [e['name'] for e in ex]
    cfg = frontend.prepare_example('circuit_buck', str(tmp_path))
    assert cfg == str(tmp_path / 'circuit_buck.toml') and (tmp_path / '04_buck.kicad_pcb').is_file()
    run, tables = frontend.read_analyses(cfg)
    assert run == ['circuit', 'thermal'] and tables['circuit']['model'] == str(tmp_path / '04_buck.json')
    s = frontend.build_settings({}, cfg)
    assert s.board == str(tmp_path / '04_buck.kicad_pcb') and s.notes == []
    assert frontend.analysis_of(str(tmp_path)) is None


# ---- what a result holds -------------------------------------------------------------------------------------
def test_power_integrity_report_and_curves(tmp_path):
    from pcbqs.pdn.cli import main
    from test_pdn_rail import board, _with_high_side
    path = tmp_path / 'b_board.json'
    path.write_text(json.dumps(_with_high_side(board())))
    out = tmp_path / 'o'
    main([str(path), '--rail', 'P', '--volts', '3.3', '--load', 'U1=2', '--load', 'Q1=1@U1', '--dmax', '1',
          '--points', '7', '--out', str(out)])
    res = json.loads((out / 'pdn.json').read_text())
    rep = R.report('pdn', str(out))
    assert rep['title'] == 'P: 3.3 V' and rep['summary'].startswith('Power integrity, rail P: worst DC drop')
    t = cells(rep)
    cols, rows = t['Loads at DC']
    assert cols[0] == 'load' and {r[0] for r in rows} == {'U1', 'Q1'}
    worst = max(res['loads'].values(), key=lambda l: l['drop'])
    assert rows[0][3] == pytest.approx(worst['drop'] * 1e3)        # the worst first, in mV
    assert {'Impedance each load sees', 'Parts'} <= set(t) and any(k.startswith('Plane pairs') for k in t)
    (cs,) = R.curves('pdn', str(out))
    assert cs['logx'] and cs['logy'] and cs['info'] == rep['summary']
    roles = {c['role'] for c in cs['curves']}
    assert roles == {'data', 'reference', 'limit'} and len(cs['curves'][0]['x']) == 7
    a = A.PowerIntegrity()
    assert a.report(str(out)) == rep and a.curves(str(out)) == [cs]
    assert a.report(str(tmp_path)) is None and a.curves(str(tmp_path)) == [] and a.suggest(str(out)) == []
    main([str(path), '--rail', 'P', '--volts', '3.3', '--load', 'U1=2', '--dmax', '1', '--no-pdn', '--out', str(out)])
    assert R.curves('pdn', str(out)) == [] and 'Parts' not in cells(R.report('pdn', str(out)))     # the DC drop alone


def test_near_field_report_curves_and_its_rail_for_power_integrity(tmp_path):
    from pcbqs.mqs_cli import parser, run
    from test_mqs_cli import board
    b = tmp_path / 'board.json'
    b.write_text(json.dumps(board()))
    out = tmp_path / 'result'
    args = [str(b), '--out', str(out), '--harmonics', '1,2', '--scan-points', '5', '--heights', '1', '--dmax', '1',
            '--fine', '1', '--fill-backend', 'numpy', '--vin-volts', '12', '--vout', '5']
    run(parser().parse_args(args + ['--preview']))
    a = A.NearField()
    rep = a.report(str(out))                                        # a preview alone: its mesh
    assert rep['title'].endswith('mesh preview') and 'Mesh' in cells(rep) and a.curves(str(out)) == []
    res = run(parser().parse_args(args))
    rep = a.report(str(out))
    cols, rows = cells(rep)['Harmonics']
    assert len(rows) == 2 and rows[0][0] == res['frequencies_Hz'][0] and rows[0][3] == pytest.approx(res['copper_L_H'][0] * 1e9)
    assert rep['title'] == 'U1: VIN / GND' and rep['sections'][-1]['notes']
    (cs,) = a.curves(str(out))
    assert cs['curves'][0]['unit'] == 'A/m' and cs['curves'][0]['y'] == [max(r) for r in res['peak_H_Am']]
    (s,) = a.suggest(str(out))
    assert s['to'] == 'pdn' and s['values'] == dict(rail='VIN', ground='GND', ground_source='U1', volts=12.0, load='')


def plate():
    sq = box(0, 0, 30, 20)
    return dict(stack=dict(layers=[], copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='F.Cu', z=1.5, t=0.035)],
                           thickness=1.5),
                copper={n: [dict(net='GND', polygon=mapping(sq))] for n in ('B.Cu', 'F.Cu')}, vias=[],
                outline=mapping(sq.difference(box(22, 12, 31, 21))),          # a corner cut away
                prims=dict(footprints=[dict(ref='Q1', value='', props={}, pads=[dict(
                    name='1', net='GND', layers=['F.Cu'], geom=mapping(box(8, 8, 12, 12)), xy=[10, 10])])]))


@pytest.mark.parametrize('method', ['grid', 'both'])
def test_temperature_report_and_its_maps_for_the_viewer(tmp_path, method):
    """The parts' and layers' tables; each layer's map a 0 Hz plane dump in C with NaN off the board, its colours
    spanning its own values; with both solvers, each one's and their difference."""
    from pcbqs.thermal.__main__ import main
    b = tmp_path / 'b_board.json'
    b.write_text(json.dumps(plate()))
    out = tmp_path / 'th'
    assert main([str(b), '--loss', 'Q1=0.5', '--theta', 'Q1=10', '--method', method, '--out', str(out)]) == 0
    res = json.loads((out / 'thermal.json').read_text())
    one = res['grid'] if method == 'both' else res
    rep = A.Thermal().report(str(out))
    t = cells(rep)
    cols, rows = t['Parts']
    assert cols == ['part', 'loss (W)', *(['board °C (grid)', 'board °C (mesh)'] if method == 'both' else ['board °C (grid)']),
                    'part °C', 'rise (K)']
    assert rows[0][0] == 'Q1' and rows[0][-2] == pytest.approx(one['parts']['Q1']['board_C'] + 5.0)
    assert [r[0] for r in t['Layers'][1]] == ['B.Cu', 'F.Cu'] and rep['title'].startswith('Hottest copper')
    assert ('Mesh against grid' in {s['title'] for s in rep['sections']}) == (method == 'both') and rep['images'] == []
    v = one['viewer']
    assert (out / v['board']).is_file() and json.loads((out / v['board']).read_text())['active_parts'] == ['Q1']
    info = json.loads((out / 'thermal' / 'exc_board' / 'dump_info.json').read_text())
    temps = [d for d in info['dumps'] if d['unit'] == '°C']
    assert all(d['field'] == 'T' and d['range'] == 'data' and d['freqs'] == [0.0] and d['normal'] == 2 for d in temps)
    assert len(temps) == (4 if method == 'both' else 2) and len(info['dumps']) == (6 if method == 'both' else 2)
    assert info['components'][0]['ref'] == 'Q1' and info['cu_z'] == {'B.Cu': 0.0, 'F.Cu': 1.5}
    top = next(d for d in v['dumps'] if d['layer'] == 'F.Cu' and d['method'] == 'grid')
    z = np.load(out / top['file'])
    T = z['data'][0, :, :, 0, 0].real                               # (nx, ny)
    assert np.nanmax(T) == pytest.approx(one['layers']['F.Cu']['hottest_C'], abs=1e-3)
    assert np.isnan(T[-1, -1]) and not np.isnan(T[0, 0]) and np.nanmin(T) > 25.0     # the cut corner: off the board
    frames = pytest.importorskip('kicad2ems.fields.frames')
    srcs = frames.find_dumps(str(out))
    assert len(srcs) == len(info['dumps']) and all(s.static and s.unit in ('°C', 'K') for s in srcs)
    s = next(s for s in srcs if s.label.startswith('thermal board: T F.Cu'))
    mesh, data = s.load(0)
    pl = frames.plane(mesh, data, 2, 0)
    lo, hi = frames.levels([pl.img], pl.signed, False, span=s.spans)
    assert s.spans and lo == pytest.approx(np.nanmin(T), abs=1e-3) and hi == pytest.approx(np.nanmax(T), abs=1e-3)


def test_circuit_report_and_waveforms(tmp_path):
    """A general run: a row a point, each point's losses and stress, its waveforms as curve sets."""
    from pcbqs.circuit.__main__ import main
    out = tmp_path / 'run'
    if main([str(EXAMPLES / '01_rc_filter.json'), '--out', str(out)]) != 0:
        pytest.skip('ngspice is not installed')
    res = json.loads((out / 'circuit.json').read_text())
    a = A.Circuit()
    rep = a.report(str(out))
    t = cells(rep)
    assert rep['title'] == res['name'] and len(t['Points'][1]) == len(res['cases']) == 1
    case = res['cases'][0]
    loss = next(v for k, v in t.items() if 'where the' in k)
    assert sum(r[1] for r in loss[1]) == pytest.approx(case['loss_W'], rel=1e-3)
    assert any(k.endswith(': stress') for k in t) and rep['footer'] == 'Completed.'
    sets = a.curves(str(out))
    assert sets and not sets[0]['logx'] and sets[0]['title'].endswith(('voltages', 'currents'))
    c = sets[0]['curves'][0]
    assert len(c['x']) == len(c['y']) > 10 and c['x'][0] == pytest.approx(0.0, abs=1e-6)
    old = dict(format=1, complete=True, cases=[dict(frequency_Hz=9e4, load_power_W=10.0, input_power_W=10.6,
                                                    efficiency=0.94, phase_deg=120.0, primary_rms_A=1.2)])
    assert len(cells(R.circuit_report(old))['Operating points'][1]) == 1 and R.circuit_curves(old) == []   # no waveforms saved


def test_the_full_bridge_s_run_is_a_circuit_result_too(tmp_path):
    """pcbqs.power_circuit's power-circuit.json: the circuit analysis's result as circuit.json is: its operating
    points and what to read them with, and each point's waveforms as curve sets (the windings, the bridge, the
    series parts between the supply's rails found by what they join, the bus; again at the solver's own steps
    where the run saved them)."""
    import numpy as np
    from pcbqs import power_circuit
    run = tmp_path / 'run'
    case = run / 'case-90k-10W'
    case.mkdir(parents=True)
    model = power_circuit.template('simple')                        # its filter: R1 and L1, not wpthb's L3 and R9
    (run / 'model.json').write_text(json.dumps(model))
    f = 9e4
    t = np.linspace(0, 4 / f, 4 * 256 + 1)
    ts = np.sort(np.concatenate([t, 3.5 / f + np.linspace(0, 1e-7, 40)]))       # the solver's own: denser at an edge
    wave = lambda x, a=1.0: a * np.sin(2 * np.pi * f * x)
    np.savez(case / 'transient.npz', time_s=t, **{'PRIMARY.winding_I': wave(t, 2.0), 'SECONDARY.winding_I': wave(t), 'bridge.V': wave(t, 24),
             'L1.I': 0.4 + 0 * t, 'R1.I': 0.01 * wave(t), 'bus.V': 24 + 0.1 * wave(t), 'other.I': t,
             'steps.time_s': ts, 'steps.bus.V': 24 + 0.1 * wave(ts) + 3.0 * (ts > 3.5 / f) * (ts < 3.5 / f + 1e-7)})
    res = dict(format=1, complete=True, name='A bridge', matching_definition='Receiver matched per frequency.',
               assumptions=['The receiver coil is not on the board.'],
               cases=[dict(frequency_Hz=f, target_W=10, load_power_W=9.98, input_power_W=10.6, efficiency=0.941, phase_deg=120.0,
                           primary_rms_A=1.41, bus_ripple_V=0.2, bus_spike_pp_V=3.2, runtime_s=50.0, folder=str(case),
                           field_inference=dict(harmonics=16), averaged_cycles=9, slow_mode_Hz=10378.0)])
    (run / 'power-circuit.json').write_text(json.dumps(res))
    a = A.Circuit()
    assert a.owns(str(run)) and not a.owns(str(run), preview=True) and not a.owns(str(tmp_path))
    rep = a.report(str(run))
    assert rep['title'] == 'A bridge' and rep['summary'] == 'Circuit, A bridge: 1 operating point, efficiency up to 94.1 %'
    assert cells(rep)['Operating points'][1] == [[90.0, 9.98, 10.6, 94.1, 120.0, 1.41, 0.2, 3.2, 50.0]] and rep['footer'] == 'Completed.'
    notes = ' '.join(rep['notes'])
    assert 'solver timestep' in notes and 'Receiver matched per frequency.' in notes and 'not on the board' in notes
    assert 'Fields:' in notes and 'which the viewer opens' in notes
    assert "slow mode (10.4 kHz: 9 switching cycles at 90 kHz)" in notes
    sets = a.curves(str(run))
    assert [s['title'] for s in sets] == [
        '90 kHz, 10 W: winding currents', '90 kHz, 10 W: bridge and coil voltages', '90 kHz, 10 W: supply filter currents',
        '90 kHz, 10 W: bus voltages', '90 kHz, 10 W: bus voltages, solver steps']
    by = {s['title'].split(': ')[1]: s for s in sets}
    assert [c['name'] for c in by['supply filter currents']['curves']] == ['R1.I', 'L1.I']       # by what they join
    assert [c['name'] for c in by['winding currents']['curves']] == ['PRIMARY.winding_I', 'SECONDARY.winding_I']
    x = by['bus voltages']['curves'][0]['x']
    assert x[0] == pytest.approx(0.0, abs=1e-9) and x[-1] == pytest.approx(2e6 / f) and len(x) == 513   # two periods, µs
    stepped = by['bus voltages, solver steps']['curves'][0]
    assert max(stepped['y']) > 26.5 > max(by['bus voltages']['curves'][0]['y'])          # the edge's spike: only there
    (run / 'model.json').unlink()                                   # a run without its model: every other current
    assert [c['name'] for c in {s['title']: s for s in a.curves(str(run))}['90 kHz, 10 W: supply filter currents']['curves']] == \
        ['L1.I', 'R1.I', 'other.I']


def test_the_tools_are_engines_of_the_questions_they_answer(tmp_path):
    """kicad2ems.study: the rail tool answers the drop and the impedance from one run (the drop alone when only it
    is asked), the others one question each; the circuit can add fields; a thermal.json of another tool's is not
    pcbqs's."""
    import json
    from kicad2ems import study
    from pcbqs import analyses as A
    pdn, mqs, circuit, thermal = A.PowerIntegrity(), A.NearField(), A.Circuit(), A.Thermal()
    assert [a.provides for a in (pdn, mqs, circuit, thermal)] == [('dc', 'pdn'), ('nearfield',), ('circuit',), ('thermal',)]
    assert all(q in study.QUESTIONS for a in (pdn, mqs, circuit, thermal) for q in a.provides)
    assert {a.engine for a in (pdn, mqs, circuit, thermal)} == {'pcbqs'} and pdn.api == (2, 6)
    assert circuit.fields == 'fields' and any(o['key'] == 'fields' and o['kind'] == 'bool' for o in circuit.options())
    assert pdn.for_questions(dict(rail='P'), ('dc',)) == dict(rail='P', no_pdn=True)
    assert pdn.for_questions(dict(rail='P'), ('dc', 'pdn')) == dict(rail='P')
    argv = pdn.command(pdn.for_questions(dict(rail='P', load='U1=1'), ('dc',)), __file__, str(tmp_path / 'o'))
    assert '--no-pdn' in argv
    (tmp_path / 'thermal.json').write_text(json.dumps(dict(method='grid', hottest_C=40.0)))
    assert thermal.owns(str(tmp_path)) and not thermal.owns(str(tmp_path), preview=True)
    # the figures engines of one question name alike (kicad2ems.study.compare): the rail's own copper, the
    # hottest point and the board under each part; this tool's own after them
    assert thermal.figures(str(tmp_path)) == [] and pdn.figures(str(tmp_path)) == []
    (tmp_path / 'thermal.json').write_text(json.dumps(dict(
        grid=dict(hottest_C=77.4, parts=dict(D1=dict(power_W=0.5, board_C=76.2, part_C=80.0))),
        mesh=dict(hottest_C=78.0, parts=dict(D1=dict(power_W=0.5, board_C=76.9, part_C=80.5))))))
    assert thermal.figures(str(tmp_path)) == [dict(name='hottest', value=77.4, unit='C'), dict(name='D1', value=76.2, unit='C')]
    (tmp_path / 'pdn.json').write_text(json.dumps(dict(loss_W=0.002159, loads=dict(
        C2=dict(I=2.0, drop=0.0010807, power_drop=5.79e-05, ground_rise=0.0010228)))))
    assert [(f['name'], round(f['value'], 4), f['unit']) for f in pdn.figures(str(tmp_path))] == \
        [('drop C2', 0.0579, 'mV'), ('drop with return C2', 1.0807, 'mV'), ('loss with return', 2.159, 'mW')]
    (tmp_path / 'thermal.json').write_text(json.dumps(dict(tool='kicad2ems_fem.elmer.thermal', T_max=40.0)))
    assert not thermal.owns(str(tmp_path))
    (tmp_path / 'thermal.json').write_text('{')
    assert not thermal.owns(str(tmp_path))


def test_a_load_with_no_pad_on_ground_is_still_needed(monkeypatch, tmp_path):
    """Analysis.missing (kicad2ems 2.9) of the rail tool: besides what its command raises, a load typed without a
    return that has no pad on ground (a bridge's high-side switch: wpthb's Q1), which the run would only note,
    its return left out of the drop. It names the part to return through where the board has one such: of the
    load's kind, on one of its nets that is no rail, and on ground."""
    from pcbqs import analyses as A
    board = tmp_path / 'b.kicad_pcb'
    board.write_text('')
    nets = dict(Q1={'', '/GH', '/SW', 'VHB'}, Q3={'/SW', '/GL', 'GND'}, U6={'/SW', 'GND', 'VCC'}, C32={'/SW', '/BOOT'},
                Q2={'/SW2', 'VHB'}, Q4={'/SW2', 'GND'}, Q5={'/SW2', 'GND'}, U1={'VIN', 'GND'},
                **{f'C{k}': {'VHB', 'GND'} for k in range(1, 10)})
    monkeypatch.setattr(A, '_pad_nets', lambda path: nets)
    pi = A.PowerIntegrity()
    need = lambda **v: [(m['key'], m['text']) for m in pi.missing(v, str(board))]
    assert need(rail='', load='U1=0.1') == [(None, 'name the rail (its net)')]
    assert need(rail='VIN', load='U1=0.1') == []                                 # on ground: its return is there
    got = need(rail='VIN', load='Q1=0.67\nQ2=0.67\nU1=0.1')                     # a form's list: one a line
    assert [k for k, _ in got] == ['load', 'load']
    assert got[0][1] == ('Q1 has no pad on ground: name the part its current returns through, Q1=AMPS@Q3 '
                         '(Q3 is on its other net and on ground)')               # not U6, the driver on that node
    assert got[1][1] == 'Q2 has no pad on ground: name the part its current returns through, Q2=AMPS@PART'   # Q4 or Q5
    assert need(rail='VIN', load=['Q1=0.67@Q3', 'Q2=0.67@Q4']) == []             # named: nothing more
    assert need(rail='VIN', load='Q9=1') == [] and need(rail='VIN', load='Q1=1', ground='GND,/SW') == []
    assert A.QuasiStatic.basic == ('qs_dmax',) and A.Thermal.uses == ('circuit',)
    wpthb = os.path.expanduser('~/Desktop/wpthb/wpthb.kicad_pcb')
    if os.path.isfile(wpthb):                                                    # the board it was met on
        monkeypatch.undo()
        text = [m['text'] for m in A.PowerIntegrity().missing(dict(rail='VIN', load='Q1=0.67'), wpthb)]
        assert text == ['Q1 has no pad on ground: name the part its current returns through, Q1=AMPS@Q3 '
                        '(Q3 is on its other net and on ground)']
