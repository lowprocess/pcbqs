"""Switching circuit simulation, coupled receiver sweeps and MQS time-domain inference.

python -m pcbqs.power_circuit --template > model.json
python -m pcbqs.power_circuit model.json --board board.kicad_pcb --out result --copper --fields
"""
import argparse,hashlib,json,shutil,copy,time
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from .power_bridge import load_config,validate,matching,predicted_phase,build,metrics,make_board_ports,end_state
from .power_spice import periodic_samples,cancel_active_runs,reset_runs
from .wpt import link
from .mqs_ports import CopperPorts,PortResponse,rl_error,foster_equivalent

FIT_TOLERANCE=.03          # default mesh.circuit_fit_tolerance: copper-model reduction error


def template():
    from importlib.resources import files
    return json.loads(files('pcbqs').joinpath('data/wpthb-power.json').read_text())


def fingerprint(board,cfg):
    from . import mqs,mqs_ports
    from . import __version__
    h=hashlib.sha256(json.dumps(dict(board=board,mesh={k:v for k,v in cfg['mesh'].items() if k!='circuit_fit_tolerance'},terminals=cfg['terminals'],references=cfg.get('net_references')),sort_keys=True).encode())
    h.update(('pcbqs:'+__version__).encode())
    source=Path(__file__).parent
    paths=[Path(mqs.__file__),Path(mqs_ports.__file__),*sorted((source/'_numerics').glob('*.py')),
           *[source/(name+'.py') for name in ['static','static_backends','static_cupy','static_cupy_kernels','conductor','electrostatic']],
           *sorted((source/'core').glob('*.hpp')),*sorted((source/'core').glob('*.cpp'))]
    for p in paths:h.update(p.read_bytes())
    return h.hexdigest()


def reduce_copper(model,response,tolerance=FIT_TOLERANCE):
    from .mqs_ports import model_impedance
    result=copy.deepcopy(model);s=2j*np.pi*response.frequencies;target=response.impedance
    norm=np.maximum(np.linalg.norm(target,axis=(1,2)),1e-30);current=model_impedance(result,response.frequencies)
    matrix=lambda kind:np.asarray(result[kind] if isinstance(kind,str) else result['foster'][kind]['R_ohm'])
    candidates=[]
    for kind in ['L_H',*range(len(result.get('foster',[])))]:
        A=matrix(kind);e,V=np.linalg.eigh(A)
        factor=s if isinstance(kind,str) else s/(s+result['foster'][kind]['pole_rad_s'])
        for j,val in enumerate(e):
            if val<=max(e.max()*1e-8,1e-20):continue
            part=val*np.outer(V[:,j],V[:,j]);delta=factor[:,None,None]*part
            candidates.append((float(np.max(np.linalg.norm(delta,axis=(1,2))/norm)),kind,part,delta))
    removed=0
    for _,kind,part,delta in sorted(candidates,key=lambda t:t[0]):
        if np.max(np.linalg.norm(current-delta-target,axis=(1,2))/norm)<=tolerance:
            current-=delta;A=matrix(kind)-part
            if isinstance(kind,str):result[kind]=A.tolist()
            else:result['foster'][kind]['R_ohm']=A.tolist()
            removed+=1
    for part in [result,*result.get('foster',[])]:
        key='L_H' if part is result else 'R_ohm';A=np.asarray(part[key]);e,V=np.linalg.eigh((A+A.T)/2)
        part[key]=(np.zeros_like(A) if abs(e).max()<1e-15 else (V*np.maximum(e,0))@V.T).tolist()
    result.update(reduction_tolerance=tolerance,removed_modes=removed,retained_dynamic_modes=len(candidates)-removed,
                  relative_impedance_error=rl_error(result,response))
    return result


def prepare(board,cfg,out,need_geometry=False,log=print):
    key=fingerprint(board,cfg);cache=out/'copper-model.json';tolerance=cfg['mesh'].get('circuit_fit_tolerance',FIT_TOLERANCE)
    saved=json.loads(cache.read_text()) if cache.exists() else {}
    if saved.get('fingerprint')==key and saved['model'].get('reduction_tolerance')!=tolerance:
        frequencies=saved['model']['check_frequencies_Hz'];zz=[]
        for f in frequencies:
            with np.load(response_path(out,key,f)) as data:zz.append(data['impedance'])
        rr=PortResponse(np.asarray(frequencies),saved['model']['names'],np.asarray(zz),None)
        saved['model']=reduce_copper(saved['model'],rr,tolerance);cache.write_text(json.dumps(saved,indent=2))
    if saved.get('fingerprint')==key and not need_geometry:return None,saved['ports'],saved['model'],None,key
    q,ports,crop=make_board_ports(board,cfg,log)
    if saved.get('fingerprint')==key:return q,ports,saved['model'],crop,key
    fs=sorted(set([0.,1e3,1e4,3e4,1e5,3e5,3e6,7e6,2e7,*cfg['frequencies_Hz'],*[f*10 for f in cfg['frequencies_Hz']]]))
    response=CopperPorts(q).extract(fs,log=log);model=foster_equivalent(response)
    model=reduce_copper(model,response,tolerance);model['check_frequencies_Hz']=fs   # reduce_copper sets relative_impedance_error
    saved=dict(fingerprint=key,ports=ports,model=model)
    cache.write_text(json.dumps(saved,indent=2))
    for i,f in enumerate(fs):save_response(out,key,f,response.impedance[i],response.currents[i])
    return q,ports,model,crop,key


def response_path(out,key,f):return out/'copper-response'/f'{key[:16]}_{f:.9g}.npz'


def save_response(out,key,f,z,currents):
    path=response_path(out,key,f);path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,frequency_Hz=f,impedance=z,currents=currents)


def extract_cached(q,freqs,out,key,log=print):
    impedance=[];currents=[]
    for f in freqs:
        path=response_path(out,key,f)
        if path.exists():
            with np.load(path) as z:impedance.append(z['impedance']);currents.append(z['currents'])
        else:
            r=CopperPorts(q).extract([f],log=log);impedance.append(r.impedance[0]);currents.append(r.currents[0]);save_response(out,key,f,r.impedance[0],r.currents[0])
    return PortResponse(np.asarray(freqs),[p['name'] for p in q.parts],np.asarray(impedance),np.asarray(currents))


def run_case(cfg,f,power,out,copper=None,ports=None,log=print):
    started=time.perf_counter();validate(cfg)          # once per case: build() does not
    from . import power_bridge,power_spice,wpt
    physics={k:v for k,v in cfg.items() if k not in ('frequencies_Hz','powers_W','assumptions','name','schematic')}
    h=hashlib.sha256(json.dumps(dict(config=physics,copper=copper,f=f,power=power),sort_keys=True).encode())
    for module in [power_bridge,power_spice,wpt]:h.update(Path(module.__file__).read_bytes())
    for switch in [cfg.get('switch_model',{}),*[s.get('model',{}) for s in cfg.get('switches',[])]]:
        if switch.get('library'):
            path=Path(switch['library']);h.update(path.read_bytes())
    case_key=h.hexdigest();previous=out/'case.json'
    if previous.exists() and (out/'transient.npz').exists():
        saved=json.loads(previous.read_text())
        if saved.get('input_fingerprint')==case_key:
            with np.load(out/'transient.npz') as data:result={k:data[k] for k in data.files}
            result['_measures']=json.loads((out/'measurements.json').read_text());log(f'Reusing {f/1000:g} kHz / {power:g} W');return saved,result
    cfg=copy.deepcopy(cfg);match=matching(cfg,f);phase=predicted_phase(cfg,f,power,match);settings=cfg.setdefault('transient',{})
    solver=dict(settings.get('solver',{}))
    if settings.get('method'):solver.setdefault('method',settings['method'])   # transient.method: shorthand for solver.method
    strict=solver.get('reltol',1e-4)
    periodic_tolerance=settings.get('periodic_tolerance',.01);power_tolerance=settings.get('power_tolerance',.005)
    # Phase-search trials at relaxed tolerances (4x faster on wpthb, <=0.04% power/current change); the
    # reported result comes from a short strict run continuing from the accepted state.
    search=dict(solver,reltol=max(settings.get('search_reltol',1e-3),strict),
                abstol=max(settings.get('search_abstol',1e-6),solver.get('abstol',1e-6)))
    warm=settings.get('warm_start',True);state=None
    if copper is not None and settings.get('phase_pilot',True):
        pilot_folder=out/'phase-pilot';pilot_folder.mkdir(exist_ok=True)
        pilot_cfg=copy.deepcopy(cfg);pilot_cfg['transient']['skip_final']=True   # it only seeds phase and state
        pilot,pilot_result=run_case(pilot_cfg,f,power,pilot_folder,None,None,log)
        phase=pilot['phase_deg'];settings['initial_input_A']=pilot['input_power_W']/cfg['voltage_V']
        if warm:state=end_state(pilot_result,cfg)          # its parts' states; copper ports keep the estimate
    cycles=settings.get('cycles',250);history=[];accepted=None

    def settled(m):                                     # supply current: slow filter modes are not judged here (C8)
        return max(v for k,v in m['periodic_error'].items() if k!='supply.I')<=periodic_tolerance

    def record(folder,m,performance,cycles,reltol,**extra):
        history.append(dict(phase_deg=phase,cycles=cycles,reltol=reltol,**extra,relative_power_error=(m['load_power_W']-power)/power,performance=performance,**m))
        (folder/'metrics.json').write_text(json.dumps(history[-1],indent=2))

    def simulate(folder,phase,cycles,options,state):
        # Hard-switched edges occasionally defeat ngspice's step control ("timestep too small"):
        # retry with trapezoidal integration (as accurate at these tolerances), then from a cold start.
        attempts=[(options,state,cycles),(dict(options,method='trap'),state,cycles)]
        if state is not None:attempts.append((options,None,max(cycles,settings.get('cycles',250))))
        for k,(opts,start,n) in enumerate(attempts):
            circ=build(cfg,f,phase,match,copper,ports,state=start)
            try:
                result=circ.run(folder,f,cycles=n,samples_per_cycle=settings.get('samples_per_cycle',256),ramp_s=cfg.get('supply',{}).get('ramp_s',100e-6),linear_solver=settings.get('linear_solver','klu'),threads=settings.get('ngspice_threads',1),options=opts)
                break
            except RuntimeError as exc:
                if 'ngspice transient failed' not in str(exc) or k==len(attempts)-1:raise
                if (folder/'ngspice.log').exists():shutil.copy2(folder/'ngspice.log',folder/f'ngspice-failed-{k}.log')
                log(f'{f/1000:g} kHz / {power:g} W: ngspice did not converge; retrying '+('with trap' if k==0 else 'from a cold start'))
        m=metrics(result,f,match)
        performance=json.loads((folder/'performance.json').read_text());performance['retries']=k
        (folder/'transient.dat').unlink(missing_ok=True)     # reproducible from circuit.cir; transient.npz has the signals
        return result,m,performance

    for trial in range(8):
        folder=out/f'trial-{trial:02d}'
        log(f'{f/1000:g} kHz / {power:g} W: phase {phase:.3f}°, {cycles} cycles'+(' (warm start)' if state else ''))
        result,m,performance=simulate(folder,phase,cycles,search,state)
        periodic=max(m['periodic_error'].values());error=(m['load_power_W']-power)/power
        record(folder,m,performance,cycles,search['reltol'],warm_start=state is not None)
        if warm:state=end_state(result,cfg,ports)          # the next run continues from here
        if not settled(m) or (abs(error)<power_tolerance and periodic>periodic_tolerance):
            cycles=int(cycles*1.6)+1 if not warm else max(int(cycles*.8),8)   # warm: the next run already continues
            if cycles>2000:raise RuntimeError('input filter/circuit did not settle; see '+str(folder))
            continue
        if m['input_power_W']<=0 or m['terminal_power_balance_relative']>.01:raise RuntimeError('failed circuit power balance; see '+str(folder))
        if abs(error)<power_tolerance:accepted=(folder,result,m);break
        sine=np.sin(np.deg2rad(phase)/2)*np.sqrt(power/max(m['load_power_W'],1e-12))
        newphase=float(np.degrees(2*np.arcsin(min(sine,.99999))))
        if phase>179 and error<0:raise RuntimeError(f'{power:g} W is unreachable at {f:g} Hz with this model')
        phase=np.clip(newphase, .01, 179.9)
        if m['input_power_W']>0:settings['initial_input_A']=power*m['input_power_W']/max(m['load_power_W'],1e-12)/cfg['voltage_V']
    if accepted is None:raise RuntimeError('power targeting did not converge; see '+str(out))
    if search['reltol']>strict and not settings.get('skip_final'):
        # The reported run: the strict tolerance, continuing from the accepted (settled) state.
        final_cycles=settings.get('final_cycles',8)
        for attempt in range(4):
            folder=out/f'final-{attempt:02d}'
            log(f'{f/1000:g} kHz / {power:g} W: final, reltol {strict:g}, {final_cycles} cycles')
            result,m,performance=simulate(folder,phase,final_cycles,dict(solver,reltol=strict),state if warm else None)
            record(folder,m,performance,final_cycles,strict,warm_start=warm,final=True)
            if settled(m):
                accepted=(folder,result,m);break
            if warm:state=end_state(result,cfg,ports)
            final_cycles*=2
        else:
            raise RuntimeError('final strict-tolerance run did not settle; see '+str(out))
    folder,result,m=accepted
    for name in ['transient.npz','circuit.cir','ngspice.log','signals.json','measurements.json','port-spectrum.npz','adaptive-ports.raw','performance.json']:
        if (folder/name).exists():shutil.copy2(folder/name,out/name)
    for trial in [*out.glob('trial-*'),*out.glob('final-*')]:(trial/'transient.npz').unlink(missing_ok=True)
    row=dict(input_fingerprint=case_key,frequency_Hz=f,target_W=power,phase_deg=phase,matching=match,**m,history=history,runtime_s=time.perf_counter()-started,folder=str(out))
    (out/'case.json').write_text(json.dumps(row,indent=2))
    return row,result


def sweep_link(cfg,f,match):
    frequencies=np.linspace(50e3,180e3,1001);transfer=[];reactance=[];eff=[]
    for ff in frequencies:
        z=link(ff,cfg['primary'],cfg['secondary'],cfg['coupling'],cfg['primary_C_F'],match['C2_F'],match['load_ohm'],cfg.get('primary_cap_ESR_ohm',0),cfg.get('secondary_cap_ESR_ohm',0))
        transfer.append(abs(z['secondary_per_primary']/z['input_Z_ohm'])**2*match['load_ohm']);reactance.append(z['input_Z_ohm'].imag);eff.append(z['efficiency'])
    peaks=find_peaks(transfer)[0];cross=np.flatnonzero(np.diff(np.sign(reactance)))
    return dict(operating_Hz=f,frequencies_Hz=frequencies.tolist(),load_power_per_Vrms2=transfer,efficiency=eff,input_X_ohm=reactance,
                loaded_power_peak_Hz=frequencies[peaks].tolist(),input_reactance_zero_Hz=[float(frequencies[k]-reactance[k]*(frequencies[k+1]-frequencies[k])/(reactance[k+1]-reactance[k])) for k in cross])


def infer_fields(q,ports,model,cfg,rows,results,crop,out,key,harmonics=16,log=print):
    from .mqs_layout import save_layout,geometry
    references=list({t.split(':',1)[0] for t in cfg['terminals']})
    layout=save_layout(out,q.board,q,crop,dict(converter=references[0],capacitors=references[1:]))
    geo=geometry(layout);x=np.linspace(crop.bounds[0],crop.bounds[2],25);y=np.linspace(crop.bounds[1],crop.bounds[3],25);top=q.z_of.max()*1000;z=top+np.array([1.,3.])
    field_rows=[]
    for f in cfg['frequencies_Hz']:
        freqs=np.arange(harmonics+1)*f;response=extract_cached(q,freqs,out,key,log);errors=rl_error(model,response)
        for row,result in [(r,v) for r,v in zip(rows,results) if r['frequency_Hz']==f]:
            _,values,per=periodic_samples(result,f,samples=1024)
            traces=np.stack([values['PCB'+str(i)+'.I'][-1] for i in range(len(ports))],axis=1)
            spectrum=Path(row['folder'])/'port-spectrum.npz'
            if not spectrum.exists():raise ValueError('adaptive port spectrum missing; rerun the circuit case before field inference')
            with np.load(spectrum) as sampled:
                full=sampled['coefficients_A'];coef=full[:harmonics+1];total_AC=float(np.sum(sampled['AC_mean_square_A2']));integration_points=int(sampled['adaptive_points_last_period'])
                if sampled['port_labels'].tolist()!=['PCB'+str(i)+'.I' for i in range(len(ports))]:raise ValueError('adaptive port order differs from copper model')
            driven=response.drive(coef)
            H,info=q.h_grid(driven,list(range(len(freqs))),x*.001,y*.001,z*.001,rtol=.02,return_info=True)
            # Reconstruct the retained spectral band. These are genuine time samples, not phasor phase frames.
            nt=64;theta=np.arange(nt)/nt
            weights=np.full(harmonics+1,2.);weights[0]=1
            timeH=np.einsum('xyvcn,nt,n->txyvc',H,np.exp(2j*np.pi*np.arange(harmonics+1)[:,None]*theta),weights).real
            folder=Path(row['folder']);save_layout(folder,q.board,q,crop,dict(converter=references[0],capacitors=references[1:]))
            exc=folder/'mqs'/'exc_bridge';exc.mkdir(parents=True,exist_ok=True)
            for old in exc.glob('dump_H_*'):old.unlink()
            for k in range(nt):write_vtr(exc/f'dump_H_operational_td_{k:04d}.vtr',x,y,z[0],timeH[k,:,:,0,:])
            meta=dict(geo,dt_ps=1e12/(nt*f),dumps=[dict(name='dump_H_operational_td',field='H',label='operational PCB, top+1mm',freqs=None,normal=2,start=[x[0],y[0],z[0]],stop=[x[-1],y[-1],z[0]],unit='A/m')],field_scope='Copper and vias only; fictitious terminal ports, parts, external coil, upstream filter geometry and cables excluded',inference='One-way spectral inference from nonlinear transient through exact MQS copper; circuit feedback uses a passive broadband Foster R/L approximation')
            (exc/'dump_info.json').write_text(json.dumps(meta))
            retained_AC=float(2*np.sum(abs(coef[1:])**2));discarded=float(max(0,total_AC-retained_AC)/max(total_AC,1e-30))
            check=dict(current_sampling='Four-point integration of adaptive ngspice piecewise-linear currents',adaptive_points_last_period=integration_points,harmonics=harmonics,maximum_Hz=float(freqs[-1]),discarded_port_current_AC_energy_fraction=discarded,copper_model_error_per_harmonic=errors,FFT_grid=info,peak_H_Am=float(np.linalg.norm(timeH,axis=-1).max()),scope=meta['field_scope'],periodic_error=max(per['PCB'+str(i)+'.I'] for i in range(len(ports))))
            np.savez_compressed(folder/'time_fields.npz',H=timeH,H_coefficients=H,time_s=theta/f,x=x,y=y,z=z,frequencies_Hz=freqs,port_current_coefficients_A=coef)
            (folder/'field-inference.json').write_text(json.dumps(check,indent=2));row['field_inference']=check;field_rows.append(check)
    return field_rows


def write_vtr(path,x,y,z,H):
    def ascii(a):return ' '.join(f'{v:.9g}' for v in np.asarray(a).ravel())
    nx,ny=len(x),len(y);extent=f'0 {nx-1} 0 {ny-1} 0 0'
    path.write_text(f'<VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian"><RectilinearGrid WholeExtent="{extent}"><Piece Extent="{extent}"><PointData Vectors="H"><DataArray type="Float32" Name="H" NumberOfComponents="3" format="ascii">{ascii(H.transpose(1,0,2))}</DataArray></PointData><Coordinates><DataArray type="Float64" format="ascii">{ascii(x)}</DataArray><DataArray type="Float64" format="ascii">{ascii(y)}</DataArray><DataArray type="Float64" format="ascii">{z}</DataArray></Coordinates></Piece></RectilinearGrid></VTKFile>')


def run_cases(cfg,out,copper=None,ports=None,jobs=1,log=print):
    """Parallelize independent circuit children; retain shared extraction and case order."""
    from concurrent.futures import ThreadPoolExecutor,as_completed
    if not isinstance(jobs,int) or jobs<1:raise ValueError('jobs must be a positive integer')
    cases=[(f,p,out/f'{f/1000:g}kHz_{p:g}W') for f in cfg['frequencies_Hz'] for p in cfg['powers_W']]
    if len({str(folder) for _,_,folder in cases})!=len(cases):raise ValueError('duplicate circuit output folders; use distinct frequencies and powers')
    reset_runs();completed={}
    def run(case):
        f,p,folder=case;folder.mkdir(parents=True,exist_ok=True)
        return run_case(cfg,f,p,folder,copper,ports,log)
    try:
        with ThreadPoolExecutor(max_workers=min(jobs,len(cases))) as pool:
            pending={pool.submit(run,case):i for i,case in enumerate(cases)}
            try:
                for future in as_completed(pending):
                    completed[pending[future]]=future.result()
                    rows=[completed[i][0] for i in sorted(completed)]
                    (out/'power-circuit.json').write_text(json.dumps(dict(format=1,complete=False,cases=rows),indent=2))
            except BaseException:
                cancel_active_runs()
                for future in pending:future.cancel()
                raise
    finally:
        reset_runs()
    ordered=[completed[i] for i in range(len(cases))]
    return [r for r,_ in ordered],[v for _,v in ordered]


def prefetch_fields(q,cfg,out,key,harmonics,log=print):
    """Cache field transfers before circuit workers, then release dense MQS matrices."""
    for f in cfg['frequencies_Hz']:
        extract_cached(q,np.arange(harmonics+1)*f,out,key,log)
    # Field evaluation needs mesh/current responses, not the dense extraction
    # matrices. Retaining both beside several ngspice children wastes GBs.
    q._mats=None;q._copper=None


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config',nargs='?');parser.add_argument('--template',action='store_true')
    parser.add_argument('--board');parser.add_argument('--out',default='power-circuit');parser.add_argument('--copper',action='store_true');parser.add_argument('--fields',action='store_true')
    parser.add_argument('--prepare',action='store_true');parser.add_argument('--field-harmonics',type=int,default=16)
    parser.add_argument('--switch-type',choices=['ideal','switch','mosfet','vdmos','subcircuit']);parser.add_argument('--frequencies');parser.add_argument('--powers')
    parser.add_argument('--jobs',type=int,default=1,help='independent ngspice cases in parallel (PCB extraction/field export remain shared)')
    args=parser.parse_args(argv)
    import signal
    def stop(*_):
        cancel_active_runs()
        raise SystemExit('Circuit run stopped')
    signal.signal(signal.SIGTERM,stop)
    if args.template:print(json.dumps(template(),indent=2));return
    if not args.config:parser.error('a circuit configuration JSON is required')
    if args.jobs<1:parser.error('--jobs must be positive')
    if not 1<=args.field_harmonics<32:parser.error('field harmonics must be in [1,31] for the 64-frame time export')
    cfg=load_config(args.config)
    if args.switch_type:cfg['switch_model']['type']=args.switch_type
    for opt,key in [(args.frequencies,'frequencies_Hz'),(args.powers,'powers_W')]:
        if opt:cfg[key]=[float(v) for v in opt.split(',')]
    validate(cfg);out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)   # the overrides, before any extraction
    (out/'model.json').write_text(json.dumps(cfg,indent=2));q=None;ports=None;copper=None;crop=None;key=None
    if args.copper or args.fields or args.prepare:
        if not args.board:parser.error('--board is required for copper/fields')
        from ._cli import load_board
        board=load_board(args.board);q,ports,copper,crop,key=prepare(board,cfg,out,args.fields)
    if args.prepare:return
    if args.fields:prefetch_fields(q,cfg,out,key,args.field_harmonics)
    else:q=None # Circuit-only runs need the reduced model, not the extraction geometry/matrices.
    rows,results=run_cases(cfg,out,copper,ports,args.jobs)
    if args.fields:infer_fields(q,ports,copper,cfg,rows,results,crop,out,key,args.field_harmonics)
    summary=dict(format=1,complete=True,name=cfg.get('name','Full bridge'),cases=rows,copper=copper,ports=ports,
                 frequency_sweeps=[sweep_link(cfg,f,matching(cfg,f)) for f in cfg['frequencies_Hz']],assumptions=cfg.get('assumptions',[]),
                 matching_definition='Receiver capacitor and load are held at supplied secondary_C_F/load_ohm, otherwise tuned/efficiency-matched per frequency. Primary C fixed. Phase regulates load power.',
                 circuit_backend='ngspice; finite transition bidirectional conductance / NMOS / external D,G,S subcircuit',
                 field_inference='Band-limited periodic copper fields inferred from transient port currents; passive broadband Foster feedback is approximate; no full-wave/common-mode compliance prediction.')
    (out/'power-circuit.json').write_text(json.dumps(summary,indent=2))
    from .power_report import render
    render(out)
    print('Completed circuit cases:',len(rows),flush=True)
    return summary


if __name__=='__main__':main()
