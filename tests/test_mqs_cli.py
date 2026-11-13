"""CLI contracts, converter ambiguity, and real solver/dump round trip."""
import json
import subprocess
import sys

import numpy as np
import pytest
from shapely.geometry import box, mapping
from pcbqs.mqs_cli import parser, validate, select, discover, run


def board():
    def fp(ref,value,pads):
        return dict(ref=ref,value=value,props={},pads=[dict(name=str(i),net=net,layers=['F.Cu'],
                     geom=mapping(poly),xy=list(poly.centroid.coords)[0]) for i,(net,poly) in enumerate(pads)])
    vin,ground=box(0,0,4,1),box(0,2,4,3)
    return dict(outline=mapping(box(-1,-1,5,4)),stack=dict(copper=[dict(name='F.Cu',z=0,t=.035)],layers=[],thickness=0),
        copper={'F.Cu':[dict(net='VIN',polygon=mapping(vin)),dict(net='GND',polygon=mapping(ground))]},vias=[],
        prims=dict(footprints=[fp('U1','buck',[('VIN',box(0,0,1,1)),('GND',box(0,2,1,3)),('SW',box(0,4,1,5))]),
                              fp('C1','10u',[('VIN',box(3,0,4,1)),('GND',box(3,2,4,3))]),
                              fp('L1','1uH',[('SW',box(1,4,2,5)),('OUT',box(3,4,4,5))])]))


def test_describe_is_clean_json():
    p=subprocess.run([sys.executable,'-m','pcbqs.mqs','--describe','--json'],capture_output=True,text=True,check=True)
    d=json.loads(p.stdout)
    assert d['format']==1 and any(o['key']=='fill_backend' for o in d['options'])
    assert not p.stderr


def test_discovery_and_explicit_override():
    b=board();a=parser().parse_args([])
    assert discover(b)[0]['switch']=='SW'
    selected,terms=select(b,a)
    assert selected['converter']=='U1' and selected['vin']=='VIN' and selected['capacitors']==['C1']
    assert len(terms)==4
    # Ambiguous extra regulator requires selection, never chooses the first.
    other=dict(b['prims']['footprints'][0],ref='U2')
    b['prims']['footprints'].append(other)
    with pytest.raises(ValueError,match='select --converter'):
        select(b,a)
    a.converter='U1';a.vin='VIN'
    assert select(b,a)[0]['converter']=='U1'


@pytest.mark.parametrize('args',[['--fsw','0'],['--harmonics','1.5'],['--heights','nan'],['--heights','1,1'],
                                 ['--duty','1'],['--vin-volts','5'],['--scan-points','1'],['--edge-time','-1']])
def test_invalid_settings(args):
    with pytest.raises(ValueError):validate(parser().parse_args(args))


def test_solver_outputs_are_standard_dumps(tmp_path):
    b=tmp_path/'board.json';b.write_text(json.dumps(board()))
    a=parser().parse_args([str(b),'--out',str(tmp_path/'result'),'--harmonics','1,2','--scan-points','5',
                          '--heights','1,2','--dmax','1','--fine','1','--fill-backend','numpy'])
    result=run(a)
    assert result['triangles']>0 and np.array(result['peak_H_Am']).shape==(2,2)
    assert np.isfinite(result['copper_L_H']).all()
    pytest.importorskip('kicad2ems.fields.frames',reason='optional frontend interoperability check')
    from kicad2ems.fields.frames import find_dumps
    dumps=find_dumps(str(tmp_path/'result'))
    assert len(dumps)==3
    # Writer and frontend reader agree on units, axes and frequency/component ordering.
    files=list((tmp_path/'result'/'mqs'/'exc_converter').glob('*.npz'))
    for path in files:
        d=np.load(path)
        assert d['data'].shape==(2,len(d['x']),len(d['y']),len(d['z']),3)
        assert np.isfinite(d['data']).all()
    assert (tmp_path/'result'/'mqs.json').exists()

    # Reusing an output folder cannot expose stale planes/volume to the GUI.
    a.heights='1'
    run(a)
    assert len(find_dumps(str(tmp_path/'result')))==1


def test_preview_never_fills_or_solves_and_exports_real_layer_heights(tmp_path, monkeypatch):
    from pcbqs.mqs import MQS
    def forbidden(*args, **kwargs):
        raise AssertionError('preview must not fill or solve')
    monkeypatch.setattr(MQS, 'solve', forbidden)
    monkeypatch.setattr(MQS, '_edge_matrices', forbidden)
    b=board()
    b['prims']['footprints'][1]['value']='unknown'  # geometry preview needs no capacitance value
    # A second physical layer verifies mesh z coordinates, independently of plan data.
    b['stack']['copper'].append(dict(name='B.Cu',z=-.7,t=.035))
    b['copper']['B.Cu']=[dict(net='GND',polygon=mapping(box(0,0,4,3)))]
    b['copper']['F.Cu'].append(dict(net='OTHER',polygon=mapping(box(1,1.2,2,1.8))))
    path=tmp_path/'board.json';path.write_text(json.dumps(b))
    a=parser().parse_args([str(path),'--preview','--out',str(tmp_path/'preview'),'--fsw','0','--fill-backend','cupy'])
    result=run(a)
    assert result['preview'] and 'fill' not in result
    layout=json.loads((tmp_path/'preview/mqs_board.json').read_text())
    assert any(not r['active'] for r in layout['copper']['F.Cu'])
    assert layout['active_parts']==['C1','U1'] and layout['components']
    pytest.importorskip('kicad2ems.meshes',reason='optional frontend interoperability check')
    from kicad2ems.meshes import find,segments
    mesh=np.load(find(str(tmp_path/'preview'))['mqs'])
    assert set(np.round(mesh['nodes'][:,2],4))=={0.,-.7}
    assert segments(find(str(tmp_path/'preview'))['mqs']).shape[1]==3
    assert not (tmp_path/'preview/mqs.json').exists()
    assert not list((tmp_path/'preview').glob('**/dump*'))


def test_nominal_component_body_has_rotation_and_is_not_an_electrical_part():
    from pcbqs.mqs_layout import crop_layout
    b=board();fp=b['prims']['footprints'][0]
    fp.update(lib='Example:IC_4x2x0.75mm',x=1,y=2,rot=90)
    layout=crop_layout(b,box(-5,-5,10,10),{'VIN','GND'},{'U1'})
    c=next(c for c in layout['components'] if c['ref']=='U1')
    from shapely.geometry import shape
    xmin,ymin,xmax,ymax=shape(c['polygon']).bounds
    assert xmax-xmin==pytest.approx(2) and ymax-ymin==pytest.approx(4)
    assert c['height']==.75 and c['active']
    assert layout['parts']==[]


EXAMPLES = __import__('pathlib').Path(__file__).resolve().parents[1] / 'src' / 'pcbqs' / 'circuit' / 'examples'


def test_discrete_switching_cells_on_the_example_boards():
    """A converter of discrete parts: a transistor or diode up from the switch node to a decoupled net, one down
    to ground. Every converter example has one (a full bridge two); a rectifier's diodes, a flyback's transformer
    and an inverting buck-boost (its loop through both capacitors) none."""
    pytest.importorskip('kicad2ems')
    from pcbqs._cli import load_board
    found = {b: [(c['converter'], c.get('low'), c['switch'], c['inductor'], [i['vin'] for i in c['inputs']])
                 for c in discover(load_board(str(EXAMPLES / f'{b}.kicad_pcb')))]
             for b in ('03_bridge_rectifier', '04_buck', '05_buck_synchronous', '06_boost', '07_buck_boost', '08_flyback',
                       '09_half_bridge_resonant', '10_full_bridge_phase_shift')}
    assert found == {'03_bridge_rectifier': [], '04_buck': [('Q1', 'D1', 'sw', 'L1', ['in'])],
                     '05_buck_synchronous': [('Q1', 'Q2', 'sw', 'L1', ['in'])],
                     '06_boost': [('D1', 'Q1', 'sw', 'L1', ['out'])],             # its hot loop: the output's
                     '07_buck_boost': [], '08_flyback': [],
                     '09_half_bridge_resonant': [('Q1', 'Q2', 'sw', '', ['in'])],
                     '10_full_bridge_phase_shift': [('Q1', 'Q2', 'a', '', ['in']), ('Q3', 'Q4', 'b', '', ['in'])]}


def test_a_discrete_cell_is_selected_and_solved(tmp_path):
    """Its loop: through the upper part to the switch node, through the lower one to ground; the switch node's
    copper in the mesh; a bridge's legs one at a time; its capacitors by connection when none is within reach."""
    pytest.importorskip('kicad2ems')
    from pcbqs._cli import load_board
    buck = load_board(str(EXAMPLES / '04_buck.kicad_pcb'))
    a = parser().parse_args([])
    selected, terms = select(buck, a)
    assert selected == dict(converter='Q1', vin='in', ground='GND', capacitors=['C1'], values=selected['values'],
                            switch='sw', low='D1')
    assert set(terms) == {'C1:in', 'C1:GND', 'Q1:in', 'Q1:sw', 'D1:sw', 'D1:GND'}
    bridge = load_board(str(EXAMPLES / '10_full_bridge_phase_shift.kicad_pcb'))
    with pytest.raises(ValueError, match='select --converter and --vin; candidates: Q1, Q3'):
        select(bridge, parser().parse_args([]))
    leg, _ = select(bridge, parser().parse_args(['--converter', 'Q3']))
    assert (leg['low'], leg['switch'], leg['capacitors']) == ('Q4', 'b', ['C1'])     # C1 beyond the 8 mm: by connection
    res = run(parser().parse_args([str(EXAMPLES / '04_buck.kicad_pcb'), '--harmonics', '1', '--scan-points', '5',
                                   '--heights', '1', '--fsw', '250e3', '--out', str(tmp_path / 'nf')]))
    assert res['selection']['low'] == 'D1' and 0.5e-9 < res['copper_L_H'][0] < 5e-9
    assert any(n.startswith('A discrete switching cell: the current through Q1 to the switch node sw') for n in res['notes'])
    model = json.loads((tmp_path / 'nf' / 'circuit-model.json').read_text())     # a discrete buck: its own parts
    assert model['board']['parts'] == dict(QH='Q1', V1='J1', RL='J2')
    assert [p['ref'] for p in model['parts'] if p['type'] == 'diode'] == ['D1']
    from pcbqs.reports import mqs_suggest, mqs_report
    rep = mqs_report(res, str(tmp_path / 'nf'))
    assert rep['summary'] == 'Near field, Q1 and D1 on in: 1 harmonic' and 'the switch node sw and D1' in rep['sections'][0]['text']
    layout = json.loads((tmp_path / 'nf' / 'layout.json').read_text()) if (tmp_path / 'nf' / 'layout.json').exists() else None
    assert mqs_suggest(res, str(tmp_path / 'nf'))[0]['values']['ground_source'] == 'D1'   # the cell's ground pad
