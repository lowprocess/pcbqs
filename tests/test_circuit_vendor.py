"""Vendor models (R-E12): a subcircuit's pins beyond D, G and S, a library's .model for a VDMOS and a diode, and
KiCad's Sim.* fields taking them over."""
import shutil

import pytest

from pcbqs.circuit import model as M
from pcbqs.circuit import run
from pcbqs.circuit.deck import build
from pcbqs.circuit.kicad import part_of
from pcbqs.power_spice import library_entry

LIBRARY = """* a vendor's library, as they come
.model DSCHOTTKY D(IS=1e-6 N=1.1 RS=0.02 CJO=150p)
.model QVD VDMOS(VTO=3 KP=20 RDS=1e10 CGDMIN=50p CGDMAX=50p CGS=1n CJO=500p RB=0.02 IS=1e-12)
.SUBCKT QSUB drain gate source Tj
+ Tcase PARAMS: dVth=0
M1 drain gate source QVD
R_th Tj Tcase 1
.ENDS
.subckt DSUB anode cathode
D1 anode cathode DSCHOTTKY
.ends
"""


@pytest.fixture
def library(tmp_path):
    path = tmp_path / 'vendor.lib'
    path.write_text(LIBRARY)
    return path


def test_a_library_says_what_each_name_is(library):
    assert library_entry(library, 'qsub') == ('subckt', ['drain', 'gate', 'source', 'Tj', 'Tcase'])
    assert library_entry(library, 'DSUB') == ('subckt', ['anode', 'cathode'])
    assert library_entry(library, 'QVD') == ('model', 'vdmos') and library_entry(library, 'nothing') is None


def _buck(q1, d1):
    m = M.template('04_buck')
    for p in m['parts']:
        if p['ref'] == 'Q1':
            p['model'] = q1
        elif p['ref'] == 'D1':
            p['model'] = d1
    return M.validate(m)


def test_a_subcircuits_other_pins_are_nodes_held_or_its_own(library):
    m = _buck(dict(type='subcircuit', library=str(library), subcircuit='QSUB', pin_order=['D', 'G', 'S', 'Tj', 'Tcase'],
                   pins={'Tcase': 'temperature'}), dict(model_name='DSCHOTTKY', library=str(library)))
    q1 = next(p for p in m['parts'] if p['ref'] == 'Q1')
    q1['temperature_C'] = 60
    deck = build(M.resolve(m))
    text = '\n'.join(deck.lines)
    x = next(l for l in deck.lines if l.startswith('X_Q1 '))
    assert x.split()[-1] == 'QSUB' and len(x.split()) == 7
    tj, tcase = x.split()[4:6]
    assert f'R_Q1_Tj {tj} 0 1e9' in text and f'V_Q1_Tcase {tcase} 0 60' in text      # floating; at its temperature
    assert next(l for l in deck.lines if l.startswith('D_D1 ')).split()[3] == 'DSCHOTTKY'
    assert text.count('.include') == 2
    q1['model']['pin_order'] = ['D', 'G', 'Tj', 'Tcase']
    with pytest.raises(ValueError, match='pin_order must name D, G, S'):
        build(M.resolve(m))


@pytest.mark.skipif(not shutil.which('ngspice'), reason='ngspice not installed')
@pytest.mark.parametrize('q1, d1', [
    (dict(type='subcircuit', subcircuit='QSUB', pin_order=['D', 'G', 'S', 'Tj', 'Tcase'], pins={'Tcase': 25}),
     dict(model_name='DSCHOTTKY')),
    (dict(type='vdmos', model_name='QVD'), dict(subcircuit='DSUB', pin_order=['A', 'K'])),
])
def test_a_buck_runs_on_its_vendors_models(library, tmp_path, q1, d1):
    m = _buck(dict(q1, library=str(library)), dict(d1, library=str(library)))
    s = run(m, tmp_path, log=lambda s: None)
    assert s['complete'] and not s['failed'], s['failed']
    c, = s['cases']
    assert c['targets'][0]['reached'] == pytest.approx(5.0, rel=0.005) and 0.8 < c['efficiency'] < 1
    assert c['parts']['D1']['power_W'] > 0.05 and c['power_balance_relative'] < 1e-6
    assert not any('generic MOSFET' in n for n in c['notes'])         # the vendor's own capacitances


def _component(ref, pins, fields, value='X'):
    return dict(ref=ref, value=value, kind='mosfet' if ref.startswith('Q') else 'unsupported', lib_id='', description='',
                fields=fields, pins=[dict(number=n, name=name, net=net) for n, name, net in pins])


def test_kicads_sim_fields_make_the_vendors_model(library):
    q = _component('Q1', [('1', 'G', 'gate'), ('2', 'D', 'sw'), ('3', 'S', 'GND'), ('4', 'TJ', 'unconnected-(Q1-TJ)'),
                          ('5', 'TC', 'case')],
                   {'Sim.Library': '${KIPRJMOD}/vendor.lib', 'Sim.Name': 'QSUB',
                    'Sim.Pins': '1=gate 2=drain 3=source 4=Tj 5=Tcase'})
    part, why = part_of(q, folder=library.parent)
    assert why == '' and part['model']['type'] == 'subcircuit' and part['model']['library'] == str(library)
    assert part['model']['pin_order'] == ['D', 'G', 'S', 'Tj', 'Tcase'] and part['model']['pins'] == {'Tcase': 'case'}
    d = _component('D1', [('1', 'K', 'sw'), ('2', 'A', 'GND')], {'Sim.Library': 'vendor.lib', 'Sim.Name': 'DSCHOTTKY'})
    part, why = part_of(d, folder=library.parent)
    assert part['model'] == dict(library=str(library), model_name='DSCHOTTKY') and part['nodes'] == ['GND', 'sw']
    d2 = _component('D2', [('1', 'K', 'sw'), ('2', 'A', 'GND')], {'Sim.Device': 'D', 'Sim.Params': 'is=2n n=1.2 rs=10m'})
    part, _ = part_of(d2)
    assert part['model']['spice_parameters'] == pytest.approx(dict(IS=2e-9, N=1.2, RS=0.01))
    q2 = _component('Q2', [('1', 'G', 'g'), ('2', 'D', 'sw'), ('3', 'S', 'GND')],
                    {'Sim.Device': 'NMOS', 'Sim.Type': 'VDMOS', 'Sim.Params': 'type=VDMOS vto=2.5 kp=30'})
    part, _ = part_of(q2)
    assert part['model']['vdmos_parameters'] == dict(VTO=2.5, KP=30.0)
    bad = _component('Q3', [('1', 'G', 'g'), ('2', 'D', 'sw'), ('3', 'S', 'GND')],
                     {'Sim.Library': 'vendor.lib', 'Sim.Name': 'QSUB', 'Sim.Pins': '1=gate 2=drain'})
    part, why = part_of(bad, folder=library.parent)
    assert part is None and 'source' not in why and 'S' in why
