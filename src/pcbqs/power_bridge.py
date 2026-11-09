"""Configurable full-bridge example: electrical circuit, matching and PCB terminals."""
import copy,json
from pathlib import Path
import numpy as np
from .power_spice import Circuit,periodic_samples
from .wpt import match_receiver,link,impedance


def validate(config):
    if config.get('format')!=1:raise ValueError('unsupported power-circuit config format')
    for key in ['primary','secondary']:
        c=config[key]
        for k in ['L_H','R_ohm']:
            if not np.isfinite(c[k]) or c[k]<=0:raise ValueError(key+'.'+k+' must be positive')
    if not 0<config['coupling']<1:raise ValueError('coupling must be between 0 and 1')
    for k in ['voltage_V','primary_C_F']:
        if not np.isfinite(config[k]) or config[k]<=0:raise ValueError(k+' must be positive')
    for k in ['frequencies_Hz','powers_W']:
        if not config[k] or not all(np.isfinite(v) and v>0 for v in config[k]):raise ValueError(k+' must be positive')
    return config


def load_config(path):
    path=Path(path).resolve();c=validate(json.loads(path.read_text()))
    for model in [c.get('switch_model',{}),*[s.get('model',{}) for s in c['switches']]]:
        if model.get('library'):model['library']=str((path.parent/model['library']).resolve())
    return c


def matching(config,f):
    return match_receiver(f,config['primary'],config['secondary'],config['coupling'],config['primary_C_F'],
                          config.get('primary_cap_ESR_ohm',0),config.get('secondary_cap_ESR_ohm',0),
                          C2=config.get('secondary_C_F'),load=config.get('load_ohm'))


def predicted_phase(config,f,power,match):
    r=link(f,config['primary'],config['secondary'],config['coupling'],config['primary_C_F'],match['C2_F'],match['load_ohm'],
           config.get('primary_cap_ESR_ohm',0),config.get('secondary_cap_ESR_ohm',0))
    # Fundamental sinusoidal estimate only; ngspice adjusts against actual load power.
    V=np.sqrt(power/match['load_ohm'])*abs(r['input_Z_ohm'])/abs(r['secondary_per_primary'])
    vfull=2*np.sqrt(2)*config['voltage_V']/np.pi
    return float(np.degrees(2*np.arcsin(min(V/vfull,.999))))


def end_state(result,config,ports=None):
    """The circuit's state at the end of a run (a whole number of switching periods: the start's phase).

    Capacitors: the voltage on the capacitance itself (terminal voltage less the ESR drop); inductors,
    the source inductance, windings: their currents; coil self-capacitance: the terminal voltage; the
    copper model: its port currents and modal currents. build(..., state=) starts the next run from it (a warm start).
    """
    last=lambda key:float(result[key][-1])
    stores=[p for p in config['components'] if p.get('enabled',True)]
    stores+=[dict(ref='LSOURCE',type='inductor'),dict(ref='CSECONDARY',type='capacitor',ESR_ohm=config.get('secondary_cap_ESR_ohm',0))]
    parts={}
    for p in stores:
        ref=p['ref']
        if ref+'.I' not in result:continue
        if p['type']=='capacitor':parts[ref]=dict(initial_V=last(ref+'.V')-p.get('ESR_ohm',0)*last(ref+'.I'),initial_I=last(ref+'.I'))
        elif p['type']=='inductor':parts[ref]=dict(initial_A=last(ref+'.I'))
    switches={sw['ref']:last(sw['ref']+'.I') for sw in config.get('switches',[]) if sw['ref']+'.I' in result}
    modes=None                                              # the copper model's modal inductor currents
    if 'MODE0.I' in result:
        modes=[];k=0
        while f'MODE{k}.I' in result:modes.append(last(f'MODE{k}.I'));k+=1
    coils={name:dict(initial_A=last(name+'.winding_I'),initial_V=last(name+'.V')) for name in ('PRIMARY','SECONDARY')
           if name+'.winding_I' in result}
    port_currents=[last(f'PCB{i}.I') for i in range(len(ports or []))] if ports and all(f'PCB{i}.I' in result for i in range(len(ports))) else None
    return dict(parts=parts,coils=coils,ports=port_currents,switches=switches,modes=modes)


def build(config,f,phase,match,copper=None,ports=None,state=None):
    """The ngspice circuit at `phase`. state (end_state of a previous run) warm-starts it: every capacitor,
    inductor, winding and copper mode begins where that run ended, instead of the precharge estimate.
    The config is validated by load_config/run_case, not here (build runs once per phase trial)."""
    if not np.isfinite(phase) or not 0<=phase<=180:raise ValueError('phase must be in [0,180]')
    config=copy.deepcopy(config);c=Circuit();ports=ports or []
    precharged=config.get('transient',{}).get('initialization','ramp')=='precharged' or state is not None
    initial_input=0.
    if precharged:
        primary_initial=secondary_initial=primary_v=secondary_v=cap1_v=cap2_v=0.
        for n in range(1,32,2):
            vc=-1j*config['voltage_V']/(np.pi*n)*(1-np.exp(-1j*n*np.deg2rad(phase)))
            r=link(f*n,config['primary'],config['secondary'],config['coupling'],config['primary_C_F'],match['C2_F'],match['load_ohm'],config.get('primary_cap_ESR_ohm',0),config.get('secondary_cap_ESR_ohm',0))
            ip=vc/r['input_Z_ohm'];isec=r['secondary_per_primary']*ip;z=impedance(f*n,config['primary'],config['secondary'],config['coupling'])
            primary_initial+=2*ip.real;secondary_initial+=2*isec.real
            primary_v+=2*(z[0,0]*ip+z[0,1]*isec).real;secondary_v+=2*(z[1,0]*ip+z[1,1]*isec).real
            cap1_v+=2*(-ip/(2j*np.pi*f*n*config['primary_C_F'])).real;cap2_v+=2*(isec/(2j*np.pi*f*n*match['C2_F'])).real
            initial_input+=2*(abs(ip)**2*r['input_Z_ohm'].real)/config['voltage_V']
        initial_input=config.get('transient',{}).get('initial_input_A',initial_input)
        config['primary'].update(initial_A=primary_initial,initial_V=primary_v)
        config['secondary'].update(initial_A=secondary_initial,initial_V=secondary_v)
        for p in config['components']:
            if p['ref'] in ('C25','C27','C30') and p['type']=='capacitor':p['initial_V']=cap1_v
            elif p['ref'] in ('C19','C20','C21','C22','C23','C24') and p['type']=='capacitor':p['initial_V']=-(config['voltage_V']-.085*initial_input)
            elif p['ref'].startswith('CA') and p['type']=='capacitor':p['initial_V']=config['voltage_V']-.055*initial_input
            elif p['ref']=='L3' and p['type']=='inductor':p['initial_A']=initial_input
        if state is not None:                               # a warm start overrides the estimates
            for p in config['components']:
                p.update(state['parts'].get(p['ref'],{}))
            for name,key in (('PRIMARY','primary'),('SECONDARY','secondary')):
                config[key].update(state['coils'].get(name,{}))
            cap2_v=state['parts'].get('CSECONDARY',{}).get('initial_V',cap2_v)
        c.uic=True
    # Every selected pad is explicit. Unselected upstream parts connect at net references.
    nodes={t:'NET:'+net for t,net in config.get('terminals',{}).items()}
    for p in ports:nodes[p['terminal']]='PAD:'+p['terminal']
    def node(t):
        if t.startswith('NET:') or t=='0':return t
        if t not in nodes:raise ValueError('unknown component terminal '+t)
        return nodes[t]
    if copper is not None:
        initial_ports=None
        if precharged:
            currents={}
            for sw in config['switches']:
                currents[sw['a']]=currents.get(sw['a'],0)-initial_input/2
                currents[sw['b']]=currents.get(sw['b'],0)+initial_input/2
            initial_ports=[currents.get(p['terminal'],0) for p in ports]
            if state is not None and state.get('ports') is not None and len(state['ports'])==len(ports):
                initial_ports=list(state['ports'])
        c.copper(copper,{p['name']:(nodes[p['terminal']],'NET:'+p['net']) for p in ports},initial_ports,
                 (state or {}).get('modes'))
    supply=config.get('supply',{});ramp=supply.get('ramp_s',100e-6)
    c.lines.append(f'VSUP {c.node("NET:SOURCE")} 0 '+(f'{config["voltage_V"]:.17g}' if precharged else f'PWL(0 0 {ramp:.17g} {config["voltage_V"]:.17g})'))
    c.signals['supply.I']='i(VSUP)';c.voltage('supply.V','NET:SOURCE')
    c.passive('RSOURCE','NET:SOURCE','NET:SRC_R',dict(type='resistor',value=max(supply.get('R_ohm',.05),1e-9)))
    source_A=state['parts'].get('LSOURCE',{}).get('initial_A',initial_input) if state is not None else initial_input
    c.passive('LSOURCE','NET:SRC_R','NET:VIN',dict(type='inductor',value=max(supply.get('L_H',100e-9),1e-15),**({'initial_A':source_A} if precharged else {})))
    for p in config['components']:
        c.passive(p['ref'],node(p['a']),node(p['b']),p)
    for sw in config['switches']:
        m={**config.get('switch_model',{}),**sw.get('model',{})}
        if state is not None and sw['ref'] in state.get('switches',{}):m['initial_A']=state['switches'][sw['ref']]
        c.switch(sw['ref'],node(sw['a']),node(sw['b']),m,f,phase if sw['leg']=='B' else 0,sw['high'])
    a,b=config['primary_terminals'];c.coil('PRIMARY',node(a),node(b),config['primary'])
    c.nodes['SEC:A']='0'  # receiver voltage gauge; no environmental coupling is modeled
    c.coil('SECONDARY','SEC:A','SEC:B',config['secondary'])
    c.lines.append(f'KLINK L_PRIMARY L_SECONDARY {config["coupling"]:.17g}')
    secondary_cap=dict(type='capacitor',value=match['C2_F'],ESR_ohm=config.get('secondary_cap_ESR_ohm',0),ESL_H=config.get('secondary_cap_ESL_H',0))
    if precharged:secondary_cap['initial_V']=cap2_v
    if state is not None:secondary_cap.update({k:v for k,v in state['parts'].get('CSECONDARY',{}).items() if k=='initial_I'})
    c.passive('CSECONDARY','SEC:B','SEC:LOAD',secondary_cap)
    c.passive('LOAD','SEC:LOAD','SEC:A',dict(type='resistor',value=match['load_ohm']))
    c.voltage('load.V','SEC:LOAD','SEC:A');c.voltage('bus.V','NET:VHB');c.voltage('input.V','NET:VBUS')
    c.voltage('bridge.V',node(config['bridge_terminals'][0]),node(config['bridge_terminals'][1]))
    if precharged and state is None:                         # a warm start's capacitors already set these nodes
        for net,drop in [('VBUS',.055),('VHB',.085)]:
            c.lines.append(f'.ic V({c.node("NET:"+net)})={config["voltage_V"]-drop*initial_input:.17g}')
    c.measures=dict(input_power_W=('AVG','-@supply.V@*@supply.I@'),load_power_W=('AVG',f'@load.V@^2/{match["load_ohm"]:.17g}'),primary_rms_A=('RMS','@PRIMARY.winding_I@'),secondary_rms_A=('RMS','@SECONDARY.winding_I@'),primary_peak_A=('MAX','abs(@PRIMARY.winding_I@)'),bus_ripple_V=('PP','@bus.V@'))
    gate_expr='+'.join(f'-@{s["ref"]}.gate_V@*@{s["ref"]}.gate_I@' for s in config['switches'])
    c.measures['gate_drive_power_W']=('AVG',gate_expr)
    for ref in c.power_refs:c.measures['part_power.'+ref]=('AVG',f'@{ref}.V@*@{ref}.I@')
    return c


def metrics(result,f,match,samples=512):
    """Results of a run: ngspice's measurements over the final cycle (on its adaptive grid, so narrow
    input/diode pulses count) and the cycle-to-cycle change of the main signals."""
    _,_,errors=periodic_samples(result,f,samples=samples)
    measured=result['_measures'];power=measured['load_power_W'];pin=measured['input_power_W'];gate=measured['gate_drive_power_W']
    part_power={k.split('.',1)[1]:v for k,v in measured.items() if k.startswith('part_power.')}
    return dict(component_power_W=part_power,terminal_power_balance_relative=abs(pin-sum(part_power.values()))/max(abs(pin),1e-12) if part_power else None,load_power_W=power,input_power_W=pin,gate_drive_power_W=gate,
                efficiency=power/(pin+gate) if pin+gate>0 else None,
                **{k:measured[k] for k in ('primary_rms_A','secondary_rms_A','primary_peak_A','bus_ripple_V')},
                periodic_error={k:errors[k] for k in ['PRIMARY.winding_I','SECONDARY.winding_I','bus.V','supply.I','load.V']})


def make_board_ports(board,config,log=print):
    from shapely.ops import unary_union
    from .mqs import MQS
    from .pdn.rail import parts_of
    selected=config['mesh'];nets=set(selected['nets']);parts={p.ref:p for p in parts_of(board)};terms={}
    for terminal,net in config['terminals'].items():
        if net not in nets:continue
        ref=terminal.split(':',1)[0]
        if ref not in parts:raise ValueError('missing PCB component '+ref)
        pads=parts[ref].on({net})
        terms[terminal]=[(l,p.geom) for p in pads for l in p.layers]
        if not terms[terminal]:raise ValueError('missing PCB terminal '+terminal)
    extent=unary_union([p for vv in terms.values() for _,p in vv]);crop=extent.buffer(selected.get('margin_mm',2)).envelope
    q=MQS(board,nets,terminals=terms,region=crop,dmax=selected.get('coarse_mm',3),fine=extent.buffer(.6).envelope,
          dmax_fine=selected.get('fine_mm',1.6),conductor_loss='slab',conductor_faces=2,
          fill_backend=selected.get('backend','auto'),fill_workers=selected.get('workers',8),log=log)
    refs={net:config.get('net_references',{}).get(net,next(t for t in terms if config['terminals'][t]==net)) for net in nets}
    ports=[]
    for term in sorted(terms):
        net=config['terminals'][term];ref=refs[net]
        if q.term_node[term]==q.term_node[ref]:continue
        name='T'+str(len(ports));q.source(name,term,ref,part_h=0)
        ports.append(dict(name=name,terminal=term,reference=ref,net=net))
    return q,ports,crop
