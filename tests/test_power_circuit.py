"""Independent circuit physics, terminal signs and actual ngspice transient checks."""
import copy,shutil
from pathlib import Path
import numpy as np
import pytest
from pcbqs.wpt import match_receiver,link,impedance
from pcbqs.power_spice import Circuit,periodic_samples
from pcbqs.mqs_ports import CopperPorts,rl_equivalent,rl_error


def test_receiver_matching_against_closed_form_and_power_balance():
    coil=dict(L_H=24e-6,R_ohm=.11);f=110e3;k=.4;w=2*np.pi*f;M=k*coil['L_H'];C=1/(w*w*coil['L_H'])
    m=match_receiver(f,coil,coil,k,C)
    assert m['C2_F']==pytest.approx(C)
    assert m['load_ohm']==pytest.approx(np.sqrt(.11**2+(w*M)**2),rel=1e-6)
    for R in [.5*m['load_ohm'],2*m['load_ohm']]:assert link(f,coil,coil,k,C,C,R)['efficiency']<m['efficiency']
    z=impedance(f,coil,coil,k);R=m['load_ohm'];j2=-z[1,0]/(z[1,1]+R+1/(1j*w*C))
    pin=(z[0,0]+1/(1j*w*C)+z[0,1]*j2).real
    assert pin==pytest.approx(.11+.11*abs(j2)**2+R*abs(j2)**2)


def test_srf_and_ac_total_resistance_not_double_counted():
    from pcbqs.wpt import coil_capacitance,winding_resistance
    c=dict(L_H=24e-6,R_ohm=.11,SRF_Hz=7.7e6,Rac_ref_ohm=.1216)
    assert coil_capacitance(c)==pytest.approx(17.80113e-12)
    assert winding_resistance(c)==.1216
    assert winding_resistance(dict(c,Rac_ref_ohm=.09))==.11
    for k in [-.2,1]:
        with pytest.raises(ValueError):impedance(1e5,c,c,k)


def test_copper_port_signs_reciprocity_and_passivity():
    from test_mqs import trace_board
    from shapely.geometry import box
    from pcbqs.mqs import MQS
    q=MQS(trace_board(),'N',terminals={'s':[('F.Cu',box(2,9.6,2.6,10.4))],'g':[('F.Cu',box(2,7.7,2.6,8.5))]},dmax=2,log=lambda *_:None)
    q.source('p','s','g',part_h=0)
    extracted=CopperPorts(q).extract([0,1e5,2e5])
    ref=q.solve([1e5,2e5]);assert extracted.impedance[1:,0,0]==pytest.approx(ref.z['p'])
    assert extracted.currents[1:,:len(q.br),0]==pytest.approx(-ref.i[:,:len(q.br)])
    assert np.all(extracted.currents[:,len(q.br):]==0)
    model=rl_equivalent(extracted,1)
    assert np.linalg.eigvalsh(model['R_ohm']).min()>0 and np.linalg.eigvalsh(model['L_H']).min()>0
    assert rl_error(model,extracted)[1]<1e-12


@pytest.mark.skipif(not shutil.which('ngspice'),reason='ngspice not installed')
@pytest.mark.parametrize('method',['gear','trap'])
def test_transient_rc_against_closed_form(tmp_path,method):
    c=Circuit();c.lines.append(f'VIN {c.node("VIN")} 0 PULSE(0 1 0 1n 1n 1 2)')
    c.passive('R','VIN','OUT',dict(type='resistor',value=1000))
    c.passive('C','OUT','0',dict(type='capacitor',value=1e-6,ESR_ohm=0))
    c.voltage('Vout','OUT');r=c.run(tmp_path,1000,cycles=5,samples_per_cycle=100,method=method)
    expected=1-np.exp(-r['time_s']/.001)
    np.testing.assert_allclose(r['Vout'],expected,atol=1e-4)


@pytest.mark.skipif(not shutil.which('ngspice'),reason='ngspice not installed')
def test_copper_spice_mutual_polarity_and_resistance(tmp_path):
    c=Circuit();c.lines.append(f'VS {c.node("A")} 0 PULSE(0 1 0 1n 1n 1 2)')
    model=dict(names=['p','q'],R_ohm=[[1,.1],[.1,1]],L_H=[[1e-3,-.2e-3],[-.2e-3,1e-3]])
    c.copper(model,dict(p=('A','0'),q=('B','0')));c.passive('LOAD','B','0',dict(type='resistor',value=1))
    r=c.run(tmp_path,1000,cycles=10)
    # DC: open-circuit source reaches p=1/(1-.1^2/2), q=-.1*p/2.
    assert r['PCB0.I'][-1]==pytest.approx(1/(1-.1**2/2),rel=.001)
    assert r['PCB1.I'][-1]==pytest.approx(-.1*r['PCB0.I'][-1]/2,rel=.001)


def test_periodic_sampling_detects_unsettled_waveform():
    t=np.linspace(0,.01,10001);r=dict(time_s=t,I=np.sin(2*np.pi*1000*t)*(1-np.exp(-t/.1)))
    _,v,e=periodic_samples(r,1000)
    assert e['I']>.05
    with pytest.raises(ValueError):periodic_samples(r,10)


def test_foster_fit_passivity_and_independent_frequency_interpolation():
    from pcbqs.mqs_ports import PortResponse,foster_equivalent,model_impedance
    f=np.r_[0,np.geomspace(1e2,1e7,24)]
    true=dict(names=['a','b'],R_ohm=[[.1,.02],[.02,.2]],L_H=[[2e-8,-.3e-8],[-.3e-8,3e-8]],foster=[dict(pole_rad_s=2*np.pi*1e4,R_ohm=[[.2,.1],[.1,.2]]),dict(pole_rad_s=2*np.pi*1e6,R_ohm=[[.7,-.2],[-.2,.4]])])
    Z=model_impedance(true,f);r=PortResponse(f,true['names'],Z,None)
    fitted=foster_equivalent(r,poles_Hz=[1e4,1e6],iterations=2500)
    ff=np.geomspace(150,7e6,33)
    err=np.linalg.norm(model_impedance(fitted,ff)-model_impedance(true,ff))/np.linalg.norm(model_impedance(true,ff))
    assert err<.001
    for residue in [fitted['R_ohm'],fitted['L_H'],*[v['R_ohm'] for v in fitted['foster']]]:assert np.linalg.eigvalsh(residue).min()>-1e-12


def test_parallel_core_loss_matches_q_and_damps_srf():
    from pcbqs.wpt import core_resistance,coil_capacitance
    c=dict(L_H=24e-6,R_ohm=.11,SRF_Hz=7.7e6,Rac_ref_ohm=.1216100382,loss_model='parallel_core')
    rc=core_resistance(c);w=2*np.pi*125e3;zz=.11+1/(1/(1j*w*c['L_H'])+1/rc)
    z=zz/(1+1j*w*coil_capacitance(c)*zz)
    assert z.real==pytest.approx(c['Rac_ref_ohm'],rel=1e-9)
    assert 3e4<rc<3.2e4


@pytest.mark.skipif(not shutil.which('ngspice'),reason='ngspice not installed')
def test_foster_spice_step_against_analytic(tmp_path):
    c=Circuit();c.lines.append(f'VS {c.node("A")} 0 PULSE(0 1 0 1n 1n 1 2)')
    model=dict(names=['p'],R_ohm=[[1]],L_H=[[1e-9]],foster=[dict(pole_rad_s=1000,R_ohm=[[1]])])
    c.copper(model,dict(p=('A','0')))
    r=c.run(tmp_path,1000,cycles=10)
    # (1+s/(s+1000))^-1 step -> current 1 - .5 exp(-500t).
    take=r['time_s']>1e-5
    np.testing.assert_allclose(r['PCB0.I'][take],1-.5*np.exp(-500*r['time_s'][take]),atol=1e-4)


@pytest.mark.skipif(not shutil.which('ngspice'),reason='ngspice not installed')
@pytest.mark.parametrize('kind',['ideal','switch','mosfet','vdmos','subcircuit'])
def test_switch_types_actual_transient_on_and_off(tmp_path,kind):
    c=Circuit();c.lines.append(f'VS {c.node("VIN")} 0 5')
    c.passive('LOAD','VIN','D',dict(type='resistor',value=100))
    model=dict(type=kind,Ron_ohm=.01,rise_s=1e-6,fall_s=1e-6,dead_time_s=0,Cds_F=100e-12,Cgs_F=100e-12,Cgd_F=10e-12)
    if kind=='subcircuit':
        lib=tmp_path/'vendor.lib';lib.write_text('.subckt TESTMOS D G S\nM1 D G S S NMODEL W=100u L=100u\n.model NMODEL NMOS(level=1 VTO=3 KP=14.2857142857)\n.ends\n')
        model.update(library=str(lib),subcircuit='TESTMOS',pin_order=['D','G','S'])
    c.switch('Q','D','0',model,1000,0,True)
    r=c.run(tmp_path,1000,cycles=3)
    assert np.interp(.00225,r['time_s'],r['Q.I'])==pytest.approx(.05,rel=.003)
    assert abs(np.interp(.00275,r['time_s'],r['Q.I']))<1e-5


def test_adaptive_fourier_does_not_alias_a_narrow_pulse():
    from pcbqs.power_waveforms import periodic_coefficients
    f=1000.;width=2e-6;start=.000193
    # A narrow triangular pulse missed entirely by a coarse 128-point display grid.
    t=np.unique(np.r_[np.linspace(0,.002,257),.001+start,.001+start+width/2,.001+start+width])
    y=np.maximum(0,1-abs(t-(.001+start+width/2))/(width/2))
    coef,info=periodic_coefficients(t,y,f,16)
    n=np.arange(17);expected=width*f/2*np.sinc(n*f*width/2)**2*np.exp(-2j*np.pi*n*f*(start+width/2))
    np.testing.assert_allclose(coef[:,0],expected,atol=1e-12)
    assert info['mean_square_A2'][0]==pytest.approx(width*f/3)


def test_fixed_receiver_capacitance_and_load_do_not_retune_with_frequency():
    from pcbqs.power_bridge import matching
    c=dict(primary=dict(L_H=24e-6,R_ohm=.11),secondary=dict(L_H=24e-6,R_ohm=.11),coupling=.4,
           primary_C_F=100e-9,secondary_C_F=100e-9,load_ohm=6.4)
    for f in [90e3,110e3]:
        m=matching(c,f)
        assert m['C2_F']==100e-9 and m['load_ohm']==6.4
        assert m['receiver_capacitor_mode']=='fixed' and m['load_mode']=='fixed'
    assert matching(c,90e3)['input_X_ohm']<0<matching(c,110e3)['input_X_ohm']
    del c['load_ohm']
    m=matching(c,90e3)
    assert m['C2_F']==100e-9 and m['load_mode']=='efficiency_matched'
    baseline=link(90e3,c['primary'],c['secondary'],.4,100e-9,100e-9,m['load_ohm'])
    assert baseline['efficiency']>link(90e3,c['primary'],c['secondary'],.4,100e-9,100e-9,m['load_ohm']*2)['efficiency']


@pytest.mark.skipif(not shutil.which('ngspice'),reason='ngspice not installed')
def test_parallel_cases_keep_order_and_independent_waveforms(tmp_path,monkeypatch):
    import json,threading
    from pcbqs import power_circuit
    gate=threading.Barrier(4,timeout=10)
    def run(cfg,f,power,out,*_):
        gate.wait() # All four jobs must actually be able to run concurrently.
        c=Circuit();c.lines.append(f'VIN {c.node("VIN")} 0 PULSE(0 1 0 1n 1n 1 2)')
        c.passive('R','VIN','OUT',dict(type='resistor',value=1000*power))
        c.passive('C','OUT','0',dict(type='capacitor',value=1e-6));c.voltage('Vout','OUT')
        result=c.run(out,f,cycles=5,samples_per_cycle=100)
        np.testing.assert_allclose(result['Vout'],1-np.exp(-result['time_s']/(.001*power)),atol=1e-4)
        stats=json.loads((out/'performance.json').read_text())
        assert stats['ngspice_wall_s']>0 and stats['postprocess_wall_s']>=0 and stats['iterations']>0
        return dict(frequency_Hz=f,target_W=power,folder=str(out)),result
    monkeypatch.setattr(power_circuit,'run_case',run)
    cfg=dict(frequencies_Hz=[1000,2000],powers_W=[1,2])
    rows,results=power_circuit.run_cases(cfg,tmp_path,jobs=4,log=lambda *_:None)
    assert [(r['frequency_Hz'],r['target_W']) for r in rows]==[(1000,1),(1000,2),(2000,1),(2000,2)]
    assert len(results)==4
    assert json.loads((tmp_path/'power-circuit.json').read_text())['cases']==rows
    with pytest.raises(ValueError,match='duplicate'):power_circuit.run_cases(dict(frequencies_Hz=[1000,1000],powers_W=[1]),tmp_path,jobs=2)


@pytest.mark.skipif(__import__('os').name=='nt',reason='POSIX test child')
def test_cancelling_sweep_stops_all_children(tmp_path,monkeypatch):
    import os,time
    from concurrent.futures import ThreadPoolExecutor
    from pcbqs import power_spice
    child=tmp_path/'slow-ngspice';child.write_text('#!/bin/sh\nexec sleep 60\n');child.chmod(0o755)
    monkeypatch.setattr(power_spice.shutil,'which',lambda _:str(child))
    power_spice.reset_runs()
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            runs=[pool.submit(Circuit().run,tmp_path/str(i),1000,timeout=10) for i in range(2)]
            deadline=time.monotonic()+5
            while len(power_spice._children)<2 and time.monotonic()<deadline:time.sleep(.01)
            pids=[p.pid for p in power_spice._children];assert len(pids)==2
            power_spice.cancel_active_runs()
            for future in runs:
                with pytest.raises(RuntimeError,match='transient failed'):future.result(timeout=3)
            for pid in pids:
                with pytest.raises(ProcessLookupError):os.kill(pid,0)
            with pytest.raises(RuntimeError,match='cancelled'):Circuit().run(tmp_path/'pending',1000)
    finally:
        power_spice.cancel_active_runs();power_spice.reset_runs()


def test_field_prefetch_releases_dense_matrices_and_preserves_responses(tmp_path,monkeypatch):
    from test_mqs import trace_board
    from shapely.geometry import box
    from pcbqs.mqs import MQS
    from pcbqs.power_circuit import prefetch_fields,extract_cached
    q=MQS(trace_board(),'N',terminals={'s':[('F.Cu',box(2,9.6,2.6,10.4))],'g':[('F.Cu',box(2,7.7,2.6,8.5))]},dmax=2,log=lambda *_:None)
    q.source('p','s','g',part_h=0)
    expected=CopperPorts(q).extract([0,1e5,2e5]);points=np.array([[.015,.010,.002]])
    before=q.h_field(expected.drive(np.ones((3,1))),1,points)
    prefetch_fields(q,dict(frequencies_Hz=[1e5]),tmp_path,'cache-test',2,log=lambda *_:None)
    assert q._mats is None and q._copper is None
    def forbidden():raise AssertionError('cached fields must not reassemble dense matrices')
    monkeypatch.setattr(q,'_assemble',forbidden)
    response=extract_cached(q,[0,1e5,2e5],tmp_path,'cache-test',log=lambda *_:None)
    np.testing.assert_allclose(response.impedance,expected.impedance)
    np.testing.assert_allclose(q.h_field(response.drive(np.ones((3,1))),1,points),before)
