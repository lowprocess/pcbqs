"""Compare an experimental PEEC PDN adapter, existing PDN and analytical limits.

python tools/trial_pdn_peec.py --out examples/pdn-peec-trial/results
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.constants import epsilon_0, mu_0
from scipy.optimize import brentq
from scipy.special import j1
from shapely.geometry import Point, box, mapping

from pcbqs.pdn.rail import Rail
from pcbqs.pdn.stack import RailPDN
from pcbqs.pdn.peec_trial import PEECPDN
from pcbqs.pdn.dc import solve_rail


SIGMA=5.8e7
PORTS=[('U0',2.,6.),('U1',18.,6.),('C1',16.,6.)]
RADIUS=.1
VRM=dict(R=.002,L=20e-9)
CAP=dict(C=100e-9,ESR=.005,ESL=.5e-9)


def board(slot=False, gap=.3):
    p=box(0,0,20,12);g=p.difference(box(7,5.5,20,6.5)) if slot else p
    fps=[]
    for ref,x,y in PORTS:
        pads=[dict(name=str(i),net=net,layers=[layer],xy=[x,y],
                   geom=mapping(Point(x,y).buffer(RADIUS,quad_segs=4)))
              for i,(net,layer) in enumerate([('P','F.Cu'),('GND','B.Cu')],1)]
        fps.append(dict(ref=ref,value='100n' if ref=='C1' else 'contact',
            props={'ESR':'5m','ESL':'0.5n'} if ref=='C1' else {},lib='C_0402',side='F',pads=pads))
    # Keep the contacts on copper below the edge-reaching slot (y=4).
    if slot:
        for fp in fps:
            for pad in fp['pads']:
                x=pad['xy'][0];pad['xy']=[x,4.]
                pad['geom']=mapping(Point(x,4.).buffer(RADIUS,quad_segs=4))
    return dict(outline=mapping(box(0,0,20,12)),
        stack=dict(copper=[dict(name='B.Cu',z=0.,t=.07),dict(name='F.Cu',z=gap,t=.07)],
                   layers=[dict(name='core',z0=0.,z1=gap,er=4.4,tand=.002,kind='diel')],thickness=gap),
        copper={'F.Cu':[dict(net='P',polygon=mapping(p))], 'B.Cu':[dict(net='GND',polygon=mapping(g))]},
        vias=[],prims=dict(footprints=fps))


def modal_reference(frequencies, gap=.3, modes=160, faces=1, legacy_loss=False):
    """Independent rectangular-cavity modal expansion, disk-averaged ports.

    Implements the PDE's full complex series sheet impedance, including its
    numerator. Uniform-current disk self terms are corrected to equipotential
    disks by subtracting Zsheet_pair/(8 pi). This local port approximation and
    neglect of open-edge fringe fields limit its status as physical ground truth.
    No pcbqs numerical or conductor kernel is called here.
    """
    a,b,h,t=20e-3,12e-3,gap*.001,70e-6
    m=np.arange(modes);kx=m*np.pi/a;ky=m*np.pi/b
    chi=np.where(m==0,1.,np.sqrt(2.))
    k2=kx[:,None]**2+ky[None,:]**2
    kr=np.sqrt(k2)*RADIUS*.001
    disc=np.ones_like(kr);mask=kr!=0;disc[mask]=2*j1(kr[mask])/kr[mask]
    profiles=[]
    for _,x,y in PORTS:
        profiles.append((chi*np.cos(kx*x*.001))[:,None]*(chi*np.cos(ky*y*.001))[None,:]*disc)
    F=np.array(profiles);Z=[]
    for f in frequencies:
        w=2*np.pi*f;delta=np.sqrt(2/(w*mu_0*SIGMA))
        if legacy_loss:
            sheet=1/(SIGMA*delta*(1-np.exp(-t/delta)))
        else:
            u=(1+1j)*t/(faces*delta)
            sheet=u/np.tanh(u)/(SIGMA*t)
        series=2*sheet+1j*w*mu_0*h
        shunt=1j*w*epsilon_0*4.4*(1-.002j)/h
        green=series/(a*b)/(k2+series*shunt)
        z=np.einsum('imn,mn,jmn->ij',F,green,F)
        z-=np.eye(len(PORTS))*series/(8*np.pi)
        Z.append(z)
    return np.array(Z)


def terminate(z,f):
    """Independent port-network termination; order U0, U1, C1."""
    out=[]
    for matrix,frequency in zip(z,f):
        w=2*np.pi*frequency
        y=np.linalg.inv(matrix)
        y[0,0]+=1/(VRM['R']+1j*w*VRM['L'])
        y[2,2]+=1/(CAP['ESR']+1j*w*CAP['ESL']+1/(1j*w*CAP['C']))
        out.append(np.linalg.inv(y)[1,1])
    return np.array(out)


def encode(z):
    return dict(real_ohm=np.real(z).tolist(),imag_ohm=np.imag(z).tolist())


def decode(z):
    return np.array(z['real_ohm'])+1j*np.array(z['imag_ohm'])


def deviation(z,ref):
    scale=np.maximum(abs(ref),1e-12)
    return dict(max_complex_relative=float(np.max(abs(z-ref)/scale)),
                median_complex_relative=float(np.median(abs(z-ref)/scale)),
                max_magnitude_relative=float(np.max(abs(abs(z)/scale-1))))


def resonance(solve,f,z):
    crossings=np.flatnonzero((z.imag[:-1]<0)&(z.imag[1:]>=0))
    if not len(crossings):return None
    i=crossings[-1]
    root=brentq(lambda ff:solve(np.array([ff]))[0].imag,float(f[i]),float(f[i+1]),rtol=1e-6)
    at=solve(np.array([root]))[0]
    return dict(frequency_Hz=float(root),resistance_ohm=float(at.real))


def evaluate(b,dmax,f,workers,landmarks=False):
    rail=Rail(b,'P',source='U0',ground_source='U0',loads={'U1':1.})
    started=time.perf_counter()
    old=RailPDN(rail,dmax=dmax,vrm=VRM,log=lambda *_:None)
    stock=old.z(f)['U1']
    resonances={}
    if landmarks:resonances['cavity_stock']=resonance(lambda ff:old.z(ff)['U1'],f,stock)
    # Plane contacts in this fixture have zero vertical mounting height.
    # Record stock answers, then remove the heuristic mount term for an
    # apples-to-apples comparison with the analytical plane-contact reference.
    mounts={ref:info['mount'] for ref,info in old.parts.items()}
    for info in old.parts.values():info['mount']=0.
    cavity=old.z(f)['U1'];bare=old.z([1e5],bare=True)['U1']
    if landmarks:resonances['cavity_fixture']=resonance(lambda ff:old.z(ff)['U1'],f,cavity)
    timing=dict(cavity_seconds=time.perf_counter()-started)
    result=dict(cavity_stock=encode(stock),cavity_fixture=encode(cavity),
                cavity_bare_100kHz=encode(bare),stock_mount_H=mounts,
                cavity_vertices=old.nv)
    shared=None
    for name,loss,faces in [('peec_skin1','slab',1),('peec_skin2','slab',2),('peec_dc','dc',1)]:
        started=time.perf_counter()
        new=PEECPDN(old,dmax=dmax,conductor_loss=loss,conductor_faces=faces,
                    contact_spacing=dmax/12,fill_workers=workers,log=lambda *_:None)
        if shared is not None:
            for attr in ('nodes','tris','node_of','z_of','t_of'):
                if not np.array_equal(getattr(new.q,attr),getattr(shared.q,attr)):
                    raise RuntimeError('Trial kernel reuse requires identical meshes')
            new.q._copper,new.q._potential=shared.q._copper,shared.q._potential
            new.q._sheet_terms=shared.q._sheet_terms
        z=new.z(f)['U1']
        if landmarks:resonances[name]=resonance(lambda ff:new.z(ff)['U1'],f,z)
        if not np.isfinite(z).all() or z.real.min()<-1e-9:
            raise RuntimeError('Nonfinite or nonpassive PDN impedance')
        result[name]=encode(z)
        timing[name+'_seconds']=time.perf_counter()-started
        if shared is None:
            shared=new
            bare_new=new.z([1e5],bare=True)['U1']
            result['peec_bare_100kHz']=encode(bare_new)
            result['peec_triangles']=len(new.q.tris)
        print(f'  {name}: {len(new.q.tris)} triangles, {timing[name+"_seconds"]:.2f} s',flush=True)
    result['timing']=timing;result['resonances']=resonances
    return result


def plot(out,frequencies,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,len(cases),figsize=(5*len(cases),8),squeeze=False)
    labels=[('cavity_stock','Current PDN (stock)','--'),('cavity_fixture','Current PDN (mount=0)',':'),
            ('peec_skin1','PEEC + 1-face skin','-'),('peec_skin2','PEEC + 2-face skin','-.'),('peec_dc','PEEC + DC copper','--')]
    for j,(name,case) in enumerate(cases.items()):
        finest=case['resolutions'][-1]['result']
        for key,label,style in labels:
            z=decode(finest[key])
            axes[0,j].loglog(frequencies,abs(z)*1e3,style,label=label)
            axes[1,j].loglog(frequencies,z.real*1e3,style,label=label)
        if 'reference_skin1' in case:
            ref=decode(case['reference_skin1'])
            axes[0,j].loglog(frequencies,abs(ref)*1e3,color='black',lw=.8,label='Analytical cavity + exact slab')
            axes[1,j].loglog(frequencies,ref.real*1e3,color='black',lw=.8)
        for i in range(2):axes[i,j].grid(True,which='both',alpha=.25);axes[i,j].set_xlabel('Frequency (Hz)')
        axes[0,j].set(title=name,ylabel='Load |Z| (mΩ)');axes[0,j].legend(fontsize=7)
        axes[1,j].set_ylabel('Load Re(Z) (mΩ)')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('examples/pdn-peec-trial/results'))
    parser.add_argument('--mesh',default='2.5,1.75,1.25')
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    meshes=[float(v) for v in args.mesh.split(',')]
    if len(meshes)<2 or any(not np.isfinite(v) or v<=0 for v in meshes) or any(a<=b for a,b in zip(meshes,meshes[1:])):
        parser.error('--mesh requires decreasing positive spacings, at least two')
    f=np.unique(np.r_[np.geomspace(1e3,30e6,35),np.linspace(5e6,9e6,13)])
    cases={}
    for name,gap,slot in [('tight_planes',.3,False),('wide_gap',1.,False),('slotted_return',.3,True)]:
        b=board(slot,gap);(out/(name+'_board.json')).write_text(json.dumps(b,indent=2)+'\n')
        case=dict(gap_mm=gap,slot=slot,resolutions=[])
        for mesh in meshes:
            print(name+f', mesh {mesh:g} mm',flush=True)
            result=evaluate(b,mesh,f,args.workers,landmarks=mesh==meshes[-1])
            case['resolutions'].append(dict(dmax_mm=mesh,result=result))
        fine=case['resolutions'][-1]['result'];coarse=case['resolutions'][-2]['result']
        case['mesh_change']={key:deviation(decode(fine[key]),decode(coarse[key]))
                             for key in ('cavity_stock','peec_skin1','peec_skin2','peec_dc')}
        case['solver_difference']=deviation(decode(fine['peec_skin1']),decode(fine['cavity_stock']))
        rail=Rail(b,'P',source='U0',ground_source='U0',loads={'U1':1.})
        dc=[]
        for spacing in (.25,.125):
            answer=solve_rail(rail,1.,dmax=spacing,log=lambda *_:None)
            dc.append(dict(dmax_mm=spacing,resistance_ohm=answer['loads']['U1']['drop']+VRM['R']))
        case['independent_dc_reference']=dc
        case['dc_reference_mesh_change']=abs(dc[-1]['resistance_ohm']/dc[-2]['resistance_ohm']-1)
        case['dc_error_at_1kHz']={key:abs(decode(fine[key])[0].real/dc[-1]['resistance_ohm']-1)
            for key in ('cavity_stock','peec_skin1','peec_skin2','peec_dc')}
        if not slot:
            ref=terminate(modal_reference(f,gap,modes=480),f)
            ref_previous=terminate(modal_reference(f,gap,modes=320),f)
            oldref=terminate(modal_reference(f,gap,modes=480,legacy_loss=True),f)
            case['reference_skin1']=encode(ref);case['reference_legacy_loss']=encode(oldref)
            case['reference_mode_change']=deviation(ref,ref_previous)
            case['agreement_with_reference']={key:deviation(decode(fine[key]),ref)
                for key in ('cavity_stock','cavity_fixture','peec_skin1','peec_skin2','peec_dc')}
            case['cavity_fixture_agreement_with_legacy_reference']=deviation(decode(fine['cavity_fixture']),oldref)
            reference_resonance=resonance(lambda ff:terminate(modal_reference(ff,gap,modes=480),ff),f,ref)
            case['reference_resonance']=reference_resonance
            case['resonance_error']={key:dict(frequency_relative=abs(v['frequency_Hz']/reference_resonance['frequency_Hz']-1),
                resistance_relative=abs(v['resistance_ohm']/reference_resonance['resistance_ohm']-1))
                for key,v in fine['resonances'].items() if v is not None}
        cases[name]=case
        (out/'summary.json').write_text(json.dumps(dict(frequencies_Hz=f.tolist(),cases=cases),indent=2,allow_nan=False)+'\n')
    from pcbqs.pdn import peec_trial
    summary=dict(format=1,frequencies_Hz=f.tolist(),meshes_mm=meshes,workers=args.workers,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(Path(peec_trial.__file__).read_bytes()).hexdigest(),cases=cases,
        interpretation=['Reference agreement is not hardware accuracy: the modal reference neglects open-edge fringing and approximates the port self term.',
            'Slotted return has no analytical AC reference here; AC mesh sensitivity and solver disagreement, plus DC conduction-reference error, are reported.',
            'One-face skin is the primary plane-pair assumption; two-face is a sensitivity, not a rigorous bound.',
            'Current PDN stock includes heuristic mounts; mount=0 is a separate diagnostic for direct plane contacts.',
            'All sources are small-signal impedances; load DC current is not converted to a shunt resistance.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    with (out/'impedance.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['case','frequency_Hz','model','R_ohm','X_ohm'])
        for name,case in cases.items():
            for key in ('cavity_stock','cavity_fixture','peec_skin1','peec_skin2','peec_dc'):
                for freq,z in zip(f,decode(case['resolutions'][-1]['result'][key])):
                    writer.writerow([name,freq,key,z.real,z.imag])
    plot(out,f,cases)
    for name,case in cases.items():
        print(name,json.dumps(dict(mesh_change=case['mesh_change'],reference=case.get('agreement_with_reference'),
                                  cavity_reference=case.get('cavity_fixture_agreement_with_legacy_reference')),indent=2))


if __name__=='__main__':main()
