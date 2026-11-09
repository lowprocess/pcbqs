"""ngspice transient adapter with explicit RLC, switch and terminal-model choices.

No device parameters are inferred from a footprint. Model libraries and selected
parasitics are inputs. All currents, including package capacitance/diode currents,
are measured at the component terminals for subsequent MQS field inference.
"""
import copy,json,re,shutil,subprocess,threading,time
from pathlib import Path
import numpy as np
from .wpt import coil_capacitance,series_resistance,core_resistance


# ngspice transient options. reltol dominates run time on hard-switched circuits: a 1e-3 search
# tolerance was 3.5x faster than 1e-4 on the wpthb netlist with <=0.03% power/current change.
# abstol 1e-6 A: 1e-9 A made strict runs fail ("timestep too small") at hard-switched edges on wpthb, and
# changed no power/current/ripple result by more than 0.03% on the same netlist.
SOLVER_DEFAULTS=dict(method='gear',maxord=2,reltol=1e-4,abstol=1e-6,vntol=1e-6,chgtol=1e-12,gmin=1e-10,rshunt=1e12)


def solver_options(options=None,method=None):
    """Validated ngspice .options, defaults overridden by `options` (and a legacy `method`)."""
    merged={**SOLVER_DEFAULTS,**(options or {})}
    if method is not None:merged['method']=method
    unknown=set(merged)-set(SOLVER_DEFAULTS)
    if unknown:raise ValueError('unknown solver options: '+', '.join(sorted(unknown)))
    if merged['method'] not in ('gear','trap'):raise ValueError('method must be gear or trap')
    if merged['maxord'] not in (1,2,3,4,5,6):raise ValueError('maxord must be an integer from 1 to 6')
    for key in ('reltol','abstol','vntol','chgtol','gmin','rshunt'):
        value=merged[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
            raise ValueError(key+' must be a positive number')
    if merged['reltol']>=.1:raise ValueError('reltol must be below 0.1')
    return merged


_children=set()
_children_lock=threading.RLock()
_cancelled=threading.Event()


def cancel_active_runs():
    """Stop all circuit children, including runs started by sweep worker threads."""
    _cancelled.set()
    with _children_lock:
        for child in tuple(_children):
            if child.poll() is None:child.terminate()


def reset_runs():
    with _children_lock:
        if _children:raise RuntimeError('circuit simulations are already running')
        _cancelled.clear()


def _positive(value,name,zero=False):
    if not np.isfinite(value) or value<0 or (value==0 and not zero):raise ValueError(f'{name} must be finite and '+('nonnegative' if zero else 'positive'))


class Circuit:
    def __init__(self):
        self.lines=[];self.nodes={'0':'0','NET:GND':'0'};self.signals={};self.signal_signs={};self.measures={};self.power_refs=[];self.uic=False;self.refs=set()

    def node(self,name):
        if name not in self.nodes:self.nodes[name]=f'n{len(self.nodes)}'
        return self.nodes[name]

    def voltage(self,label,a,b='0'):
        na,nb=self.node(a),self.node(b)
        if na=='0':
            self.signals[label]=f'v({nb})';self.signal_signs[label]=-1
        else:self.signals[label]=f'v({na},{nb})' if nb!='0' else f'v({na})'

    def sense(self,ref,a):
        if not re.fullmatch('[A-Za-z][A-Za-z0-9_]*',ref) or ref.lower() in self.refs:raise ValueError('duplicate or invalid component reference: '+ref)
        self.refs.add(ref.lower());n=self.node('IN:'+ref);self.lines.append(f'V_{ref} {self.node(a)} {n} 0')
        self.signals[ref+'.I']=f'i(V_{ref})'
        return n

    def passive(self,ref,a,b,part):
        if not part.get('enabled',True):return
        kind=part['type'];value=part['value'];_positive(value,ref+'.value')
        if kind not in ('resistor','capacitor','inductor'):raise ValueError('unsupported passive type: '+kind)
        n=self.sense(ref,a);input_node=n;end=self.node(b);self.voltage(ref+'.V',a,b);self.power_refs.append(ref)
        R=part.get('ESR_ohm',0) if kind=='capacitor' else part.get('DCR_ohm',0)
        L=part.get('ESL_H',0)
        for key,val in [('R',R),('L',L)]:_positive(val,ref+'.'+key,True)
        if R:
            nxt=self.node('R:'+ref);self.lines.append(f'R_{ref}_series {n} {nxt} {R:.17g}');n=nxt
        if L:                                                  # series ESL carries the branch current
            branch=part.get('initial_I',part.get('initial_A') if kind=='inductor' else None)
            nxt=self.node('L:'+ref);self.lines.append(f'L_{ref}_series {n} {nxt} {L:.17g}'+(f' IC={branch:.17g}' if branch is not None else ''));n=nxt
        code={'resistor':'R','capacitor':'C','inductor':'L'}[kind]
        if kind=='capacitor':
            factor=part.get('bias_factor',1.);_positive(factor,ref+'.bias_factor');value*=factor
        ic=part.get('initial_V' if kind=='capacitor' else 'initial_A') if kind!='resistor' else None
        self.lines.append(f'{code}_{ref} {n} {end} {value:.17g}'+(f' IC={ic:.17g}' if ic is not None else ''))
        if part.get('leakage_ohm'):
            _positive(part['leakage_ohm'],ref+'.leakage_ohm');self.lines.append(f'R_{ref}_leak {input_node} {end} {part["leakage_ohm"]:.17g}')

    def coil(self,ref,a,b,coil):
        L=coil['L_H'];R=series_resistance(coil);C=coil_capacitance(coil)
        for k,v in [('L',L),('R',R),('Cp',C)]:_positive(v,ref+'.'+k,k!='L')
        n=self.sense(ref,a);end=self.node(b);mid=self.node('RL:'+ref)
        self.lines.extend([f'R_{ref} {n} {mid} {max(R,1e-12):.17g}',f'L_{ref} {mid} {end} {L:.17g}'+(f' IC={coil["initial_A"]:.17g}' if 'initial_A' in coil else '')])
        if C:self.lines.append(f'C_{ref}_self {n} {end} {C:.17g}'+(f' IC={coil["initial_V"]:.17g}' if 'initial_V' in coil else ''))
        rc=core_resistance(coil)
        if rc:self.lines.append(f'R_{ref}_core {mid} {end} {rc:.17g}')
        self.power_refs.append(ref)
        self.signals[ref+'.winding_I']=f'i(L_{ref})';self.voltage(ref+'.V',a,b)

    def switch(self,ref,a,b,model,f,phase,high):
        kind=model.get('type','switch')
        if kind not in ('ideal','switch','mosfet','vdmos','subcircuit'):raise ValueError('unknown switch type: '+kind)
        T=1/f;dead=model.get('dead_time_s',0);tr=model.get('rise_s',20e-9);tf=model.get('fall_s',20e-9)
        for k,v in [('dead_time_s',dead),('rise_s',tr),('fall_s',tf)]:_positive(v,ref+'.'+k,k=='dead_time_s')
        width=T/2-dead-(tr+tf)/2
        if width<=0:raise ValueError('dead time/edges exceed half a switching period')
        vgate=model.get('gate_V',10.);_positive(vgate,ref+'.gate_V')
        d=self.sense(ref,a);s=self.node(b);g=self.node('G:'+ref);drv=self.node('DRIVE:'+ref)
        delay=((phase/360+(0 if high else .5))*T+dead/2)%T
        if delay+tr+width+tf>T:delay-=T                        # an on-interval across t=0: on from the start, as in steady state
        self.lines.append(f'VG_{ref} {drv} {s} PULSE(0 {vgate:.17g} {delay:.17g} {tr:.17g} {tf:.17g} {width:.17g} {T:.17g})')
        rg=model.get('gate_R_ohm',2.);_positive(rg,ref+'.gate_R',True)
        self.lines.append(f'RG_{ref} {drv} {g} {max(rg,1e-9):.17g}')
        current=model.get('initial_A')                          # warm start: the switch's drain current
        for pin,key in [('d','package_L_H'),('s','source_L_H')]:
            val=model.get(key,0);_positive(val,ref+'.'+key,True)
            if val:
                ic='' if current is None else f' IC={(current if pin=="d" else -current):.17g}'
                nxt=self.node(pin+':'+ref);self.lines.append(f'L_{ref}_{pin} {d if pin=="d" else s} {nxt} {val:.17g}'+ic)
                if pin=='d':d=nxt
                else:s=nxt
        ron=model.get('Ron_ohm',.008 if kind!='ideal' else 1e-6);roff=model.get('Roff_ohm',1e10)
        _positive(ron,ref+'.Ron');_positive(roff,ref+'.Roff')
        if kind in ('ideal','switch'):
            logoff=np.log(1/roff);logspan=np.log(roff/ron)
            self.lines.append(f'B_{ref} {d} {s} I=V({d},{s})*exp({logoff:.17g}+({logspan:.17g})*(0.5+0.5*tanh((V({g},{s})-{vgate/2:.17g})/{model.get("transition_V",.5):.17g})))')
        elif kind=='mosfet':
            vt=model.get('Vth_V',3.);kp=model.get('Kp_A_V2',1/(ron*max(vgate-vt,.1)))
            if not 0<vt<vgate:raise ValueError('MOSFET threshold must be below gate voltage')
            self.lines.extend([f'.model NM_{ref} NMOS(level=1 VTO={vt:.17g} KP={kp:.17g} LAMBDA={model.get("lambda_per_V",0):.17g} IS=1e-12)',f'M_{ref} {d} {g} {s} {s} NM_{ref} W=100u L=100u'])
        elif kind=='vdmos':
            params=dict(VTO=model.get('Vth_V',3.),KP=model.get('Kp_A_V2',1/(ron*max(vgate-model.get('Vth_V',3.),.1))),
                        RDS=roff,CGDMIN=model.get('Cgd_F',0),CGDMAX=model.get('Cgd_F',0),CGS=model.get('Cgs_F',0),
                        CJO=model.get('Cds_F',0),TT=model.get('diode_TT_s',0),RB=model.get('diode_R_ohm',.02),IS=1e-12)
            params.update(model.get('vdmos_parameters',{}))
            if any(not re.fullmatch('[A-Za-z][A-Za-z0-9_]*',key) or not np.isfinite(value) for key,value in params.items()):raise ValueError('VDMOS parameters must be named finite numbers')
            self.lines.extend([f'.model NM_{ref} VDMOS('+ ' '.join(f'{key}={value:.17g}' for key,value in params.items())+')',f'M_{ref} {d} {g} {s} NM_{ref}'])
        else:
            path=Path(model['library']).expanduser().resolve()
            if not path.is_file():raise ValueError('missing switch model library: '+str(path))
            name=model['subcircuit']
            if not re.fullmatch('[A-Za-z_][A-Za-z0-9_]*',name):raise ValueError('invalid subcircuit name')
            # Explicit D,G,S contract, with optional order for vendor libraries.
            pins={'D':d,'G':g,'S':s};order=model.get('pin_order',['D','G','S'])
            if sorted(order)!=['D','G','S']:raise ValueError('subcircuit pin_order must contain D,G,S once')
            self.lines.extend([f'.include "{path}"',f'X_{ref} '+ ' '.join(pins[k] for k in order)+' '+name])
        if kind not in ('ideal','vdmos') and (kind!='subcircuit' or model.get('external_capacitance',False)):
            for key,pins in [('Cds_F',(d,s)),('Cgd_F',(d,g)),('Cgs_F',(g,s))]:
                val=model.get(key,0);_positive(val,ref+'.'+key,True)
                if val:self.lines.append(f'C_{ref}_{key} {pins[0]} {pins[1]} {val:.17g}')
        if kind in ('ideal','switch') and model.get('body_diode',True):
            tt=model.get('diode_TT_s',20e-9);_positive(tt,ref+'.diode_TT_s',True)
            self.lines.extend([f'.model DD_{ref} D(IS=1e-12 N=1 RS={model.get("diode_R_ohm",.02):.17g} TT={tt:.17g})',f'D_{ref} {s} {d} DD_{ref}'])
        self.power_refs.append(ref)
        self.voltage(ref+'.V',a,b);self.signals[ref+'.gate_I']=f'i(VG_{ref})';self.voltage(ref+'.gate_V','DRIVE:'+ref,b)

    def copper(self,model,term_nodes,initial_currents=None,initial_modes=None):
        """initial_modes (a previous run's MODE<k>.I, in mode order) restores every modal inductor exactly;
        otherwise modes start at the port currents' projection (exact for pure inductance modes only)."""
        names=model['names'];N=len(names);initial=None if initial_currents is None else np.asarray(initial_currents)
        inductive=0
        modes=[]
        matrices=[('R',np.asarray(model['R_ohm']),None),('L',np.asarray(model['L_H']),None)]
        matrices += [('RL',np.asarray(m['R_ohm']),m['pole_rad_s']) for m in model.get('foster',[])]
        # Reciprocal modal transformations use native LINEAR F/G/E sources.
        # Behavioral expression trees make a large otherwise-linear model slow
        # and poorly conditioned inside nonlinear transient Newton iterations.
        for kind,A,pole in matrices:
            if A.shape!=(N,N) or not np.isfinite(A).all() or not np.allclose(A,A.T,rtol=1e-8,atol=1e-18):raise ValueError('invalid copper matrix')
            e,V=np.linalg.eigh((A+A.T)/2)
            if e.min()<-1e-8*max(abs(e).max(),1e-30):raise ValueError('non-passive copper matrix')
            if pole is not None:_positive(pole,'Foster pole')
            for j,val in enumerate(e):
                if val<=max(e.max()*1e-8,1e-20):continue
                tag=str(len(modes));n=self.node('MODE:'+tag)
                reference=1e-6 if kind=='L' else 1.
                gain=V[:,j]*np.sqrt(val/reference)
                for i,g in enumerate(gain):
                    if abs(g)>1e-14:self.lines.append(f'FMODE{tag}_{i} 0 {n} V_PCB{i} {g:.17g}')
                if kind in ('R','RL'):self.lines.append(f'RMODE{tag} {n} 0 1')
                if kind in ('L','RL'):
                    L=reference if kind=='L' else 1/pole
                    if initial_modes is not None and inductive<len(initial_modes):ic=f' IC={float(initial_modes[inductive]):.17g}'
                    elif initial is not None:ic=f' IC={float(gain@initial):.17g}'
                    else:ic=''
                    self.lines.append(f'LMODE{tag} {n} 0 {L:.17g}'+ic)
                    self.signals[f'MODE{inductive}.I']=f'i(LMODE{tag})';inductive+=1
                modes.append((n,gain))
        for i,name in enumerate(names):
            a,b=term_nodes[name];ref='PCB'+str(i);self.voltage(ref+'.V',a,b);self.power_refs.append(ref);n=self.sense(ref,a);summed=self.node('SUM:'+name)
            self.lines.append(f'RSUM{i} {summed} 0 1')
            for j,(mode,gain) in enumerate(modes):
                if abs(gain[i])>1e-14:self.lines.append(f'GSUM{i}_{j} 0 {summed} {mode} 0 {gain[i]:.17g}')
            self.lines.append(f'EPCB{i} {n} {self.node(b)} {summed} 0 1')

    def run(self,out,f,cycles=250,samples_per_cycle=128,ramp_s=100e-6,timeout=600,linear_solver='klu',method=None,threads=1,options=None):
        solver=solver_options(options,method)
        if linear_solver not in ('klu','sparse'):raise ValueError('linear_solver must be klu or sparse')
        if not isinstance(threads,int) or threads<1:raise ValueError('threads must be a positive integer')
        executable=shutil.which('ngspice')
        if not executable:raise RuntimeError('ngspice is required for transient circuit simulation')
        out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
        stop=cycles/f if self.uic else max(cycles/f, ramp_s+20/f)
        labels=list(self.signals);vectors=list(self.signals.values())
        output_names=[f'out{i}' for i in range(len(vectors))]
        lets=[f'let {name}={vector}' for name,vector in zip(output_names,vectors)]
        save_vectors=set()
        for vector in vectors:
            if vector.startswith('i('):save_vectors.add(vector)
            else:
                for node in re.search(r'v\(([^)]+)\)',vector)[1].split(','):
                    if node!='0':save_vectors.add('v('+node+')')
        measures=[]
        for i,(label,(kind,expr)) in enumerate(self.measures.items()):
            for key,name in zip(labels,output_names):expr=expr.replace('@'+key+'@',f'({self.signal_signs.get(key,1)}*{name})')
            measures.extend([f'let measure{i}={expr}',f'meas tran m{i} {kind} measure{i} from={stop-1/f:.17g} to={stop:.17g}'])
        port_vectors=[name for name,label in zip(output_names,labels) if re.fullmatch(r'PCB\d+\.I',label)]
        adaptive_export=['set filetype=binary','write adaptive-ports.raw '+' '.join(port_vectors),'set filetype=ascii'] if port_vectors else []
        lines=['Transient power circuit',*self.lines,'.options '+' '.join(f'{k}={v:.6g}' if isinstance(v,float) else f'{k}={v}' for k,v in solver.items()),
               '.control',*(['set klu'] if linear_solver=='klu' else []),f'set num_threads={threads}','set wr_singlescale','set wr_vecnames','save '+' '.join(sorted(save_vectors)),f'tran {1/f/samples_per_cycle:.17g} {stop:.17g} 0 {1/f/samples_per_cycle:.17g}'+(' uic' if self.uic else ''),
               *lets,*measures,*adaptive_export,'linearize '+' '.join(output_names),'wrdata transient.dat '+' '.join(output_names),'rusage all','quit','.endc','.end']
        (out/'circuit.cir').write_text('\n'.join(lines)+'\n')
        started=time.perf_counter()
        with _children_lock:
            if _cancelled.is_set():raise RuntimeError('circuit run cancelled')
            child=subprocess.Popen([executable,'-b','circuit.cir'],cwd=out,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            _children.add(child)
        try:
            try:stdout,stderr=child.communicate(timeout=timeout)
            except BaseException:
                child.terminate()
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise
            from types import SimpleNamespace
            proc=SimpleNamespace(stdout=stdout,stderr=stderr,returncode=child.returncode)
        finally:
            with _children_lock:_children.discard(child)
            child.stdout.close();child.stderr.close()
        solved=time.perf_counter()
        (out/'ngspice.log').write_text(proc.stdout+'\n'+proc.stderr)
        data=out/'transient.dat'
        if proc.returncode or not data.is_file() or 'simulation(s) aborted' in proc.stderr:raise RuntimeError('ngspice transient failed; see '+str(out/'ngspice.log'))
        table=np.loadtxt(data,skiprows=1)
        if table.ndim!=2 or table.shape[1]!=len(labels)+1 or not np.isfinite(table).all() or table[-1,0]<stop*(1-1e-6):raise RuntimeError('invalid transient data')
        res={'time_s':table[:,0],**{k:table[:,i+1]*self.signal_signs.get(k,1) for i,k in enumerate(labels)}}
        np.savez_compressed(out/'transient.npz',**res)
        (out/'signals.json').write_text(json.dumps(dict(zip(labels,vectors)),indent=2))
        measured={}
        for i,label in enumerate(self.measures):
            match=re.search(rf'^m{i}\s*=\s*([+\-0-9.eE]+)',proc.stdout,re.M)
            if not match:raise RuntimeError('ngspice measure failed: '+label)
            measured[label]=float(match[1])
        (out/'measurements.json').write_text(json.dumps(measured,indent=2))
        res['_measures']=measured
        if port_vectors:
            from .power_waveforms import save_port_spectrum
            save_port_spectrum(out,labels,f)
        stats=dict(ngspice_wall_s=solved-started,postprocess_wall_s=time.perf_counter()-solved,linear_solver=linear_solver,method=solver['method'],reltol=solver['reltol'],ngspice_threads=threads)
        for key,label in dict(analysis_s='Total analysis time (seconds)',load_s='Matrix load time',factor_s='Matrix factor time',solve_s='Matrix solve time',equations='Circuit Equations',iterations='Total iterations',accepted_steps='Accepted timepoints',rejected_steps='Rejected timepoints').items():
            match=re.search(r'^'+re.escape(label)+r'\s*=\s*([+\-0-9.eE]+)',stdout,re.M)
            if match:stats[key]=float(match[1])
        (out/'performance.json').write_text(json.dumps(stats,indent=2))
        return res


def periodic_samples(result,f,cycles=4,samples=512):
    """Uniform samples from the final complete cycles; preserve an absolute time origin."""
    time=np.asarray(result['time_s']);end=np.floor((time[-1]*f)+1e-8)/f;start=end-cycles/f
    if start<time[0] or cycles<2 or samples<16:raise ValueError('insufficient cycles/samples')
    t=start+np.arange(cycles*samples)/(f*samples)
    values={k:np.interp(t,time,v).reshape(cycles,samples) for k,v in result.items() if k!='time_s' and not k.startswith('_')}
    errors={k:float(np.max(abs(v[-1]-v[-2]))/max(np.max(abs(v[-1])),1e-9)) for k,v in values.items()}
    return t.reshape(cycles,samples),values,errors
