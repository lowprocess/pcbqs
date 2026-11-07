"""KiCad connectivity and simulation-model editing, without Qt or RF dependencies.

KiCad sources are read-only. XML netlists can be read without KiCad installed;
.kicad_sch files are exported by the installed kicad-cli to a temporary file.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from .pdn.rail import parse_value

PASSIVES=('resistor','capacitor','inductor')
RESERVED={'PRIMARY','SECONDARY','CSECONDARY','LOAD','RSOURCE','LSOURCE'}


def engineering_value(value):
    """Numeric SI or RKM notation, optionally followed by electrical units."""
    if isinstance(value,(int,float)) and not isinstance(value,bool):number=float(value)
    else:
        text=str(value).strip().replace('μ','u').replace('µ','u')
        text=re.sub(r'(?<=\d)\s+(?=[fpnumkKMGRr])','',text)
        text=re.sub(r'\s*(ohms?|Ω|[FHAaVv])\s*$', '',text,flags=re.I).strip()
        try:number=float(text)
        except ValueError:number=parse_value(text)
    if number is None or not math.isfinite(number):raise ValueError('Expected a finite number or SI value, such as 33n or 5m')
    return number


def terminal_net(config,terminal):
    if terminal=='0':return 'GND'
    if terminal.startswith('NET:'):
        if not terminal[4:]:raise ValueError('NET: needs a net name')
        return terminal[4:]
    if terminal in config.get('terminals',{}):return config['terminals'][terminal]
    if terminal.startswith('SEC:'):return terminal
    raise ValueError('Unknown terminal '+terminal+'; choose a terminal or use NET:name')


def validate_edit(config):
    """Catch invalid references/connections before an edited model reaches ngspice."""
    seen=set(RESERVED)
    for part in [*config.get('components',[]),*config.get('switches',[])]:
        ref=part.get('ref','')
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',ref):raise ValueError('Invalid component reference: '+ref)
        if ref.upper() in seen:raise ValueError('Duplicate or reserved component reference: '+ref)
        seen.add(ref.upper())
        for pin in ['a','b']:
            terminal=part.get(pin,'');terminal_net(config,terminal)
            if terminal.startswith('SEC:'):raise ValueError('Internal receiver nodes use the receiver controls; choose a PCB terminal or NET:name')
        if part in config.get('components',[]):
            if part.get('type') not in PASSIVES:raise ValueError(ref+': unsupported passive type')
            if part.get('enabled',True) and engineering_value(part.get('value',''))<=0:raise ValueError(ref+': value must be positive; specify a finite resistance for a zero-ohm link')
            for key in ['ESR_ohm','ESL_H','DCR_ohm']:
                if key in part and engineering_value(part[key])<0:raise ValueError(ref+': '+key+' must be nonnegative')
            if 'bias_factor' in part and engineering_value(part['bias_factor'])<=0:raise ValueError(ref+': bias factor must be positive')
            if part.get('leakage_ohm',0)<0:raise ValueError(ref+': leakage resistance must be nonnegative')
        else:
            model={**config.get('switch_model',{}),**part.get('model',{})}
            if model.get('type','switch') not in ('ideal','switch','mosfet','vdmos','subcircuit'):raise ValueError(ref+': unknown switch model')
            for key in ['Ron_ohm','Roff_ohm','gate_V','rise_s','fall_s']:
                if key in model and engineering_value(model[key])<=0:raise ValueError(ref+': '+key+' must be positive')
            for key in ['gate_R_ohm','dead_time_s','Cds_F','Cgd_F','Cgs_F','diode_TT_s','package_L_H','source_L_H']:
                if key in model and engineering_value(model[key])<0:raise ValueError(ref+': '+key+' must be nonnegative')
    for key in ['secondary_C_F','load_ohm']:
        if key in config and engineering_value(config[key])<=0:raise ValueError(key+' must be positive, or blank for automatic tuning')
    metadata=config.get('schematic',{})
    if not isinstance(metadata,dict):raise ValueError('schematic metadata must be an object')
    source=metadata.get('source',{})
    if not isinstance(source,dict) or not isinstance(source.get('components',[]),list):raise ValueError('Invalid schematic source metadata')
    if any(not isinstance(c,dict) or not {'ref','value','kind','pins'}<=set(c) or not isinstance(c['pins'],list) or any(not isinstance(p,dict) or not {'number','net'}<=set(p) for p in c['pins']) for c in source.get('components',[])):raise ValueError('Invalid schematic component metadata')
    positions=metadata.get('positions',{})
    if not isinstance(positions,dict) or any(not isinstance(p,(list,tuple)) or len(p)!=2 or any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in p) for p in positions.values()):raise ValueError('Schematic positions must be finite [x, y] coordinates')
    from .power_bridge import validate
    validate(config)
    return config


def _pin_sort(pin):
    n=pin['number'];return (0,int(n)) if n.isdigit() else (1,n)


def read_netlist(path):
    """Read KiCad XML, retaining all components, pins, source fields and exclusions."""
    path=Path(path)
    with path.open('rb') as stream:data=stream.read(64*1024**2+1)
    if len(data)>64*1024**2:raise ValueError('Netlist exceeds the 64 MiB import limit')
    declarations=data.replace(b'\x00',b'').upper()
    if b'<!DOCTYPE' in declarations or b'<!ENTITY' in declarations:raise ValueError('DTD/entity declarations are not supported')
    try:root=ET.fromstring(data)
    except ET.ParseError as exc:raise ValueError('Invalid KiCad XML netlist: '+str(exc)) from exc
    if root.tag!='export' or root.find('components') is None or root.find('nets') is None:raise ValueError('Expected a KiCad XML netlist (export/components/nets)')
    libraries={}
    for lib in root.findall('./libparts/libpart'):
        libraries[(lib.get('lib',''),lib.get('part',''))]={p.get('num',''):dict(number=p.get('num',''),name=p.get('name',''),type=p.get('type',''),net='') for p in lib.findall('./pins/pin')}
    comps=[];byref={}
    for node in root.findall('./components/comp'):
        ref=node.get('ref','')
        if not ref or ref in byref:raise ValueError('Duplicate or missing reference in netlist: '+ref)
        fields={f.get('name',''):f.text or '' for f in node.findall('./fields/field')}
        properties={f.get('name',''):f.get('value','') for f in node.findall('property')}
        lib=node.find('libsource');libname=lib.get('lib','') if lib is not None else '';symbol=lib.get('part','') if lib is not None else ''
        pins=copy.deepcopy(libraries.get((libname,symbol),{}))
        for p in node.findall('./units/unit/pins/pin'):
            n=p.get('num','');pins.setdefault(n,dict(number=n,name='',type='',net=''))
        flag=lambda key:key in properties and properties[key].lower() not in ('0','false','no') or key in {c.tag for c in node}
        dnp=flag('dnp')
        excluded=flag('exclude_from_sim') or fields.get('Sim.Enable','1').lower() in ('0','false','no')
        sheet=node.find('sheetpath')
        c=dict(ref=ref,value=node.findtext('value',''),footprint=node.findtext('footprint',''),description=node.findtext('description',''),lib_id=libname+':'+symbol,fields=fields,properties=properties,pins=pins,dnp=dnp,exclude_from_sim=excluded,sheet=sheet.get('names','/') if sheet is not None else '/')
        comps.append(c);byref[ref]=c
    nets=[]
    for node in root.findall('./nets/net'):
        name=node.get('name','');members=[]
        for pin in node.findall('node'):
            ref,n=pin.get('ref',''),pin.get('pin','')
            if ref not in byref:raise ValueError('Net references unknown component '+ref)
            info=byref[ref]['pins'].setdefault(n,dict(number=n,name='',type='',net=''))
            if info['net'] and info['net']!=name:raise ValueError(ref+'.'+n+' occurs in different nets')
            info.update(net=name,name=pin.get('pinfunction',info['name']),type=pin.get('pintype',info['type']));members.append(dict(ref=ref,pin=n))
        nets.append(dict(name=name,nodes=members))
    for c in comps:
        c['pins']=sorted(c['pins'].values(),key=_pin_sort);part=c['lib_id'].split(':')[-1].lower();device=c['fields'].get('Sim.Device','').lower()
        kind=next((kind for prefix,kind in [('r','resistor'),('c','capacitor'),('l','inductor')] if (part.startswith(prefix) and (len(part)==1 or part[1]=='_')) or device==prefix),None)
        if kind is None and re.match(r'^[RCL]\d',c['ref']):kind={'R':'resistor','C':'capacitor','L':'inductor'}[c['ref'][0]]
        names={p['name'].upper() for p in c['pins']}
        if {'D','G','S'}<=names:kind='mosfet'
        c['kind']=kind or 'unsupported';c['value_si']=None
        if kind in PASSIVES:
            try:c['value_si']=engineering_value(c['value'])
            except ValueError:pass
        connected=[p for p in c['pins'] if p['net'] and not p['net'].startswith('unconnected-')]
        c['supported_passive']=kind in PASSIVES and len(c['pins'])==2 and len(connected)==2 and len({p['net'] for p in connected})==2 and c['value_si'] is not None and c['value_si']>0 and bool(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',c['ref']))
    return dict(format=1,path=str(path.resolve()),kind='kicadxml',sha256=hashlib.sha256(data).hexdigest(),design_source=root.findtext('./design/source',''),tool=root.findtext('./design/tool',''),components=comps,nets=nets,sheets=[dict(name=s.get('name','/'),number=s.get('number','')) for s in root.findall('./design/sheet')])


def read_schematic(path,timeout=120):
    path=Path(path).expanduser().resolve()
    if path.suffix.lower()!='.kicad_sch':return read_netlist(path)
    exe=shutil.which('kicad-cli')
    if not exe:raise RuntimeError('kicad-cli is required for .kicad_sch; import a KiCad XML netlist instead')
    with tempfile.TemporaryDirectory(prefix='pcbqs-schematic-') as tmp:
        target=Path(tmp)/'netlist.xml'
        result=subprocess.run([exe,'sch','export','netlist','--format','kicadxml','--output',str(target),str(path)],capture_output=True,text=True,timeout=timeout)
        if result.returncode or not target.is_file():raise RuntimeError('KiCad netlist export failed: '+(result.stderr or result.stdout).strip())
        doc=read_netlist(target)
    doc.update(path=str(path),kind='kicad_sch',sha256=hashlib.sha256(path.read_bytes()).hexdigest());return doc


def import_status(component,config):
    ref=component['ref'];existing=next((p for p in config.get('components',[]) if p['ref']==ref),None)
    switch=next((p for p in config.get('switches',[]) if p['ref']==ref),None)
    if component['dnp']:return 'DNP — imported disabled',False
    if component['exclude_from_sim']:return 'Excluded from simulation',False
    if existing and component['supported_passive']:
        if existing['type']!=component['kind']:return 'Type differs — review before applying',False
        return 'Update configured passive',True
    if switch and component['kind']=='mosfet':return 'Configured switch — optional pin mapping; gate drive stays configured',False
    if component['supported_passive']:return 'Available passive — select to add',False
    return 'Not modeled automatically — inspect pins/fields',False


def _parasitics(component):
    out={};fields={k.lower():v for k,v in component['fields'].items()}
    for key,names in [('ESR_ohm',['esr','esr_ohm','sim.esr']),('ESL_H',['esl','esl_h','sim.esl']),('DCR_ohm',['dcr','dcr_ohm','sim.dcr'])]:
        for name in names:
            if name not in fields:continue
            value=engineering_value(fields[name])
            if value<0:raise ValueError(component['ref']+': negative '+key)
            out[key]=value;break
    return out


def primary_bank(config):
    try:
        a=terminal_net(config,config['bridge_terminals'][0]);b=terminal_net(config,config['primary_terminals'][0])
        return [p for p in config['components'] if p.get('enabled',True) and p.get('type')=='capacitor' and {terminal_net(config,p['a']),terminal_net(config,p['b'])}=={a,b}]
    except (KeyError,ValueError):return []


def sync_primary_bank(before,after):
    old=sum(p['value']*p.get('bias_factor',1) for p in primary_bank(before));new=sum(p['value']*p.get('bias_factor',1) for p in primary_bank(after))
    if old>0 and new>0 and not math.isclose(old,new,rel_tol=1e-12) and math.isclose(before.get('primary_C_F',0),old,rel_tol=1e-8):after['primary_C_F']=new


def apply_import(config,document,references,update_connections=False):
    result=copy.deepcopy(config);byref={p['ref']:p for p in result['components']};switches={p['ref']:p for p in result.get('switches',[])};source={p['ref']:p for p in document['components']}
    for ref in references:
        if ref not in source:raise ValueError('Unknown imported reference '+ref)
        part=source[ref]
        if part['supported_passive']:
            existing=byref.get(ref)
            if ref in switches:raise ValueError(ref+' is already configured as a switch')
            if existing is None:
                existing=dict(ref=ref,type=part['kind'],value=part['value_si']);result['components'].append(existing);byref[ref]=existing
                reconnect=True
            else:reconnect=update_connections
            value=part['value_si']
            if isinstance(existing.get('value'),(int,float)) and math.isclose(existing['value'],value,rel_tol=1e-14):value=existing['value']
            existing.update(type=part['kind'],value=value)
            if part['dnp'] or part['exclude_from_sim'] or 'enabled' in existing:existing['enabled']=not(part['dnp'] or part['exclude_from_sim'])
            existing.update(_parasitics(part))
            if reconnect:
                pins=part['pins'];old_a=terminal_net(config,existing['a']) if 'a' in existing else None
                if old_a==pins[1]['net']:pins=list(reversed(pins))
                for key,pin in zip(['a','b'],pins):
                    terminal=ref+':'+pin['net'];result.setdefault('terminals',{})[terminal]=pin['net'];existing[key]=terminal
        elif ref in switches and part['kind']=='mosfet':
            if update_connections:
                for role in ['D','S']:
                    if len({p['net'] for p in part['pins'] if p['name'].upper()==role})!=1:raise ValueError(ref+': ambiguous '+role+' pin nets; map connections manually')
                pins={p['name'].upper():p for p in part['pins']}
                for key,name in [('a','D'),('b','S')]:
                    if not pins[name]['net']:raise ValueError(ref+': unconnected '+name)
                    terminal=ref+':'+pins[name]['net'];result.setdefault('terminals',{})[terminal]=pins[name]['net'];switches[ref][key]=terminal
        else:raise ValueError(ref+': assign a supported simulation model manually')
    result.setdefault('schematic',dict(format=1))['source']=copy.deepcopy(document)
    sync_primary_bank(config,result);validate_edit(result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source');args=parser.parse_args();print(json.dumps(read_schematic(args.source),indent=2))
