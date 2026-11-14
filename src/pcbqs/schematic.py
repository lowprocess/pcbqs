"""KiCad connectivity, without Qt or RF dependencies: what a schematic or a netlist has (circuit/kicad.py takes it
into a circuit model).

KiCad sources are read-only. XML netlists can be read without KiCad installed;
.kicad_sch files are exported by the installed kicad-cli to a temporary file.
"""
import copy
import hashlib
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from .pdn.rail import parse_value

PASSIVES=('resistor','capacitor','inductor')


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


def _parasitics(component):
    out={};fields={k.lower():v for k,v in component['fields'].items()}
    for key,names in [('ESR_ohm',['esr','esr_ohm','sim.esr']),('ESL_H',['esl','esl_h','sim.esl']),('DCR_ohm',['dcr','dcr_ohm','sim.dcr'])]:
        for name in names:
            if name not in fields:continue
            value=engineering_value(fields[name])
            if value<0:raise ValueError(component['ref']+': negative '+key)
            out[key]=value;break
    return out
