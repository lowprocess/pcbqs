"""Connectivity import, selective model updates and input preservation."""
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest
from pcbqs.power_circuit import template
from pcbqs.schematic import (engineering_value,read_netlist,apply_import,import_status,
    validate_edit,sync_primary_bank,read_schematic)


def netlist(tmp_path):
    root=ET.Element('export',version='E');design=ET.SubElement(root,'design');ET.SubElement(design,'source').text='original.kicad_sch';ET.SubElement(design,'sheet',name='/receiver/',number='2');comps=ET.SubElement(root,'components');nets=ET.SubElement(root,'nets')
    def add(ref,value,symbol,pins,fields=None,properties=None):
        c=ET.SubElement(comps,'comp',ref=ref);ET.SubElement(c,'value').text=value;ET.SubElement(c,'footprint').text='Capacitor_SMD:C_0603';ET.SubElement(c,'libsource',lib='Device',part=symbol);ET.SubElement(c,'sheetpath',names='/receiver/')
        fs=ET.SubElement(c,'fields')
        for k,v in (fields or {}).items():ET.SubElement(fs,'field',name=k).text=v
        for k,v in (properties or {}).items():ET.SubElement(c,'property',name=k,value=v)
        units=ET.SubElement(ET.SubElement(c,'units'),'unit',name='A');ps=ET.SubElement(units,'pins')
        for number,name,net in pins:
            ET.SubElement(ps,'pin',num=number)
            if net is None:continue
            n=next((n for n in nets if n.get('name')==net),None)
            if n is None:n=ET.SubElement(nets,'net',code=str(len(nets)+1),name=net)
            ET.SubElement(n,'node',ref=ref,pin=number,pinfunction=name,pintype='passive')
    add('C25','47n','C_Small',[('1','A','/HL_SW'),('2','B','/HL_AC')],{'sim.esr':'12mOhm','Manufacturer':'Test'})
    add('RNEW','4k7','R_Small',[('A','','/receiver/VOUT'),('B','','GND')])
    add('C17','33n','C_Small',[('1','','/HL_SW'),('2','','/HL_AC')],properties={'dnp':''})
    add('R2','2k','R',[('1','','VHB'),('2','','GND')],properties={'dnp':'false','exclude_from_sim':''})
    add('U1','Controller','Controller',[('1','IN','VHB'),('2','OUT','/HL_SW'),('3','GND','GND')])
    add('RNC','1k','R',[('1','','VHB'),('2','',None)])
    add('RZERO','0R','R',[('1','','VHB'),('2','','GND')])
    add('Q1','MOSFET','Q_NMOS',[('1','G','/GATE'),('2','S','/HR_SW'),('3','D','VHB')])
    path=tmp_path/'sample.xml';ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True);return path


@pytest.mark.parametrize('text,expected',[('33n',33e-9),('24 µH',24e-6),('110 mΩ',.11),('4k7',4700),('1e-6 F',1e-6),('5mOhm',.005),('1R0',1),('0R',0)])
def test_engineering_values(text,expected):assert engineering_value(text)==pytest.approx(expected)


@pytest.mark.parametrize('text',['nan','inf','33n 10%','BLM18 600R@100MHz','bad'])
def test_ambiguous_or_nonfinite_values_are_not_guessed(text):
    with pytest.raises(ValueError):engineering_value(text)


def test_netlist_preserves_pins_hierarchy_fields_and_exclusions(tmp_path):
    path=netlist(tmp_path);before=path.read_bytes();doc=read_netlist(path);c={p['ref']:p for p in doc['components']}
    assert path.read_bytes()==before and doc['sheets'][0]['name']=='/receiver/'
    assert c['RNEW']['pins'][0]['number']=='A' and c['RNEW']['pins'][0]['net']=='/receiver/VOUT'
    assert c['C25']['fields']['Manufacturer']=='Test'
    assert c['C17']['dnp'] and c['R2']['exclude_from_sim'] and not c['R2']['dnp']
    assert c['Q1']['kind']=='mosfet' and c['U1']['kind']=='unsupported'
    assert c['RNEW']['supported_passive'] and not c['RNC']['supported_passive'] and not c['RZERO']['supported_passive']


def test_selective_update_preserves_parasitics_and_does_not_mutate_inputs(tmp_path):
    doc=read_netlist(netlist(tmp_path));original=template();snapshot=copy.deepcopy(original);old=next(p for p in original['components'] if p['ref']=='C25')
    updated=apply_import(original,doc,['C25']);part=next(p for p in updated['components'] if p['ref']=='C25')
    assert original==snapshot and part['a']==old['a'] and part['b']==old['b']
    assert part['value']==pytest.approx(47e-9) and part['ESR_ohm']==.012 and part['ESL_H']==old['ESL_H']
    assert updated['primary_C_F']==pytest.approx(113e-9)
    assert 'RNEW' not in {p['ref'] for p in updated['components']}
    assert updated['switch_model']==original['switch_model']
    assert import_status(next(p for p in doc['components'] if p['ref']=='C17'),original)[1] is False


def test_add_passive_and_explicit_connection_update(tmp_path):
    doc=read_netlist(netlist(tmp_path));cfg=template();updated=apply_import(cfg,doc,['RNEW','C17','Q1'],update_connections=True)
    parts={p['ref']:p for p in updated['components']}
    assert parts['RNEW']['value']==4700 and updated['terminals'][parts['RNEW']['a']]=='/receiver/VOUT'
    assert parts['C17']['enabled'] is False
    sw=next(p for p in updated['switches'] if p['ref']=='Q1')
    assert updated['terminals'][sw['a']]=='VHB' and updated['terminals'][sw['b']]=='/HR_SW'
    assert updated['switch_model']==cfg['switch_model']
    with pytest.raises(ValueError,match='manually'):apply_import(cfg,doc,['U1'])


def test_reference_override_and_optional_metadata_are_respected(tmp_path):
    cfg=template();cfg['primary_C_F']=100e-9;after=apply_import(cfg,read_netlist(netlist(tmp_path)),['C25']);assert after['primary_C_F']==100e-9
    before=template();after=copy.deepcopy(before);after['primary_C_F']=120e-9;sync_primary_bank(before,after);assert after['primary_C_F']==120e-9
    bad=copy.deepcopy(cfg);bad['components'].append(dict(ref='LOAD',type='resistor',value=1,a='0',b='NET:VIN'))
    with pytest.raises(ValueError,match='reserved'):validate_edit(bad)
    bad=copy.deepcopy(cfg);bad['components'][0]['a']='missing'
    with pytest.raises(ValueError,match='Unknown terminal'):validate_edit(bad)


def test_reject_entities_and_wrong_xml(tmp_path):
    path=tmp_path/'bad.xml';path.write_text('<!DOCTYPE export [<!ENTITY x "a">]><export/>')
    with pytest.raises(ValueError,match='DTD'):read_netlist(path)
    path.write_bytes('<!DOCTYPE export [<!ENTITY x "a">]><export/>'.encode('utf-16'))
    with pytest.raises(ValueError,match='DTD'):read_netlist(path)
    path.write_text('<foo/>')
    with pytest.raises(ValueError,match='Expected'):read_netlist(path)


def test_schematic_export_uses_temporary_output_and_preserves_original(tmp_path,monkeypatch):
    source=tmp_path/'example.kicad_sch';source.write_text('(kicad_sch original)');xml=netlist(tmp_path).read_bytes();calls=[]
    from pcbqs import schematic
    monkeypatch.setattr(schematic.shutil,'which',lambda _: '/bin/kicad-cli')
    def export(args,**kwargs):
        calls.append(args);Path(args[args.index('--output')+1]).write_bytes(xml)
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(schematic.subprocess,'run',export);doc=read_schematic(source)
    assert source.read_text()=='(kicad_sch original)' and doc['kind']=='kicad_sch'
    assert doc['path']==str(source.resolve()) and calls[0][-1]==str(source.resolve())


def test_unchanged_value_import_preserves_physics_exactly(tmp_path):
    cfg=template();doc=read_netlist(netlist(tmp_path));c=next(p for p in doc['components'] if p['ref']=='C25');part=next(p for p in cfg['components'] if p['ref']=='C25');c['value_si']=part['value']*(1+2e-16);c['fields']={}
    imported=apply_import(cfg,doc,['C25']);imported.pop('schematic');assert imported==cfg


@pytest.mark.parametrize('change',[
    lambda c:c['components'][0].update(bias_factor=0),
    lambda c:c['switches'][0].setdefault('model',{}).update(Ron_ohm=0),
    lambda c:c.update(secondary_C_F=0),
    lambda c:c.update(schematic={'positions':{'part:C1':[float('nan'),1]}}),
    lambda c:c.update(schematic=None),
])
def test_invalid_editor_parameters_are_rejected(change):
    cfg=template();change(cfg)
    with pytest.raises(ValueError):validate_edit(cfg)
