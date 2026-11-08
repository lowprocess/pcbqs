"""
A buck converter's hot loop, three ways, on pcbqs.mqs: its input capacitor tight against the regulator's VIN and
PGND pins (good), 10 mm away on a trace (far), and that with a slot in the ground plane under its return (slot).
The loop impedance, the plane's return current, and the magnetic near field 1 and 3 mm above the board at the
switching harmonics (fsw 500 kHz: 0.5-3 MHz, and the trend to 30 MHz) for a 3 A trapezoidal input current (duty
0.33, 5 ns edges). Compare the original MQS/DC-sheet model with layered PEEC,
DC-sheet loss and one/two-face skin slabs. Lateral crowding is resolved;
through-thickness proximity redistribution is assumed, not resolved. H planes
at 1, 3 and 10 mm also form a coarse volume for the GUI's field viewer.

    OPENBLAS_NUM_THREADS=4 python tools/nearfield_demo.py out/
"""
import argparse
import copy
import json
from pathlib import Path
import os
import time
from types import SimpleNamespace

import numpy as np
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from pcbqs.mqs import MQS

# Layout and waveform retained from lowprocess/mom25d tools/nearfield_demo.py,
# revision 77509a598561e79c292e65369cec10c1816fc25a (GPL-3.0-or-later).
# No runtime dependency on mom25d or kicad2ems.
W, H = 36.0, 26.0
Z_TOP, Z_GND = 1.6, 1.4                           # F.Cu, In1.Cu (the ground plane, 0.2 mm below)
FSW, DUTY, TR, IPK = 500e3, 0.33, 5e-9, 3.0
HARM = [1, 2, 3, 4, 5, 6, 10, 20, 60]             # 0.5 .. 3 MHz, then 5, 10, 30 MHz
VIA_R = 0.15


def harmonic_amps(n):
    """The input current's harmonics (A peak): a trapezoid 0 to IPK, duty DUTY, edges TR, at n x FSW."""
    n = np.asarray(n, float)
    return 2 * IPK * DUTY * np.abs(np.sinc(n * DUTY)) * np.abs(np.sinc(n * FSW * TR))


def layout(case):
    """board.json's dict of one case: the copper of VIN and GND, the vias, the terminals."""
    vin_pad, pgnd_pad = box(15.6, 12.0, 16.4, 14.0), box(19.6, 12.0, 20.4, 14.0)
    if case == 'good':
        c_p, c_n = box(16.0, 10.0, 17.0, 11.0), box(19.0, 10.0, 20.0, 11.0)
        vin = unary_union([vin_pad, box(15.6, 10.0, 17.0, 14.0)])
        gnd_top = unary_union([pgnd_pad, box(19.0, 10.0, 20.4, 14.0)])
        gvias = [(19.4, 12.6), (20.0, 12.6), (19.4, 11.6), (20.0, 11.6)]
        fine = box(13, 7, 23, 17)
    else:
        c_p, c_n = box(15.5, 2.5, 16.5, 3.5), box(19.5, 2.5, 20.5, 3.5)
        vin = unary_union([vin_pad, box(15.75, 3.0, 16.25, 12.0), c_p])
        gnd_top = unary_union([pgnd_pad, c_n])
        gvias = [(20.0, 12.6), (20.0, 13.4), (19.8, 3.0), (20.2, 3.0)]     # the cap's two in its ground pad
        fine = unary_union([box(13, 0.6, 24, 16)] + ([box(6, 5.5, 14, 9.5)] if case == 'slot' else []))
    plane = box(0.5, 0.5, W - 0.5, H - 0.5)
    if case == 'slot':
        plane = plane.difference(box(8.0, 6.8, W - 0.5, 7.6))        # a slot under the return, open to the right
    # the vias' antipads in the plane are not cut: the ground vias join it, the VIN copper has none
    cu = {'F.Cu': [dict(net='VIN', polygon=mapping(vin)), dict(net='GND', polygon=mapping(gnd_top))],
          'In1.Cu': [dict(net='GND', polygon=mapping(plane))], 'B.Cu': []}
    stack = dict(layers=[dict(name='core', z0=0, z1=Z_GND, er=4.5, tand=0.02, kind='diel'),
                         dict(name='prepreg', z0=Z_GND, z1=Z_TOP, er=4.2, tand=0.02, kind='diel')],
                 copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='In1.Cu', z=Z_GND, t=0.035),
                         dict(name='F.Cu', z=Z_TOP, t=0.035)], thickness=Z_TOP)
    vias = [dict(x=x, y=y, r=VIA_R, z0=Z_GND, z1=Z_TOP, net='GND') for x, y in gvias]
    terms = {'U1.VIN': [('F.Cu', vin_pad)], 'U1.PGND': [('F.Cu', pgnd_pad)],
             'C1.1': [('F.Cu', c_p)], 'C1.2': [('F.Cu', c_n)]}
    b = dict(stack=stack, copper=cu, vias=vias, outline=mapping(box(0, 0, W, H)))
    return b, terms, fine


def build_model(case, args, mode, shared=None):
    """Independent topology; reuse only identical immutable copper kernels."""
    b, terms, fine = layout(case)
    q = MQS(b, {'VIN', 'GND'}, terminals=terms, dmax=args.dmax, fine=fine,
            dmax_fine=args.fine, capacitance=mode != 'mqs_dc', dielectric='stack',
            conductor_loss='slab' if 'slab' in mode else 'dc',
            conductor_faces=1 if mode.endswith('1') else 2,
            fill_backend=args.backend, fill_workers=args.workers, factor_cache_bytes=0)
    if shared is not None:
        for name in ['nodes','tris','z_of','t_of','node_of']:
            if not np.array_equal(getattr(q,name),getattr(shared,name)):
                raise ValueError('Cannot reuse kernels for a different mesh')
        q._copper = shared._copper
        q._potential = shared._potential
        q._sheet_terms = shared._sheet_terms
        q.fill_info = copy.deepcopy(shared.fill_info)
    q.source('U1', 'U1.VIN', 'U1.PGND', part_h=.4)
    q.element('C1', 'C1.1', 'C1.2', C=10e-6, ESR=3e-3, ESL=.4e-9, part_h=.6)
    return q


def electrical_metrics(q, r):
    """Peak-phasor powers, normalized to 1 A source; internal and external L."""
    metrics = {k:[] for k in ['port_R_ohm','port_X_ohm','copper_R_ohm',
               'copper_magnetic_L_H','copper_internal_L_H','capacitor_ESR_R_ohm',
               'dielectric_R_ohm','power_balance_relative','charge_continuity_relative']}
    for fi, f in enumerate(r.f):
        w=2*np.pi*f; i=r.i[fi]; dr=q._resistance_delta(f)
        copper=np.vdot(i,(q._mats['R']+dr)@i)
        loss_part=sum(abs(i[q.ne+q.nv+k])**2*p['R'] for k,p in enumerate(q.parts) if p['kind']=='Z')
        loss_diel=w*np.vdot(r.charge[fi],r.potential[fi]).imag if r.charge is not None else 0.
        z=r.z['U1'][fi]; values=dict(port_R_ohm=z.real,port_X_ohm=z.imag,
            copper_R_ohm=copper.real,copper_magnetic_L_H=np.vdot(i,q._inductive_voltage(i)).real,
            copper_internal_L_H=np.vdot(i,dr@i).imag/w,
            capacitor_ESR_R_ohm=loss_part,dielectric_R_ohm=loss_diel,
            power_balance_relative=abs(z.real-copper.real-loss_part-loss_diel)/max(abs(z.real),1e-20))
        if r.charge is not None:
            residue=q._peec['A']@i+1j*w*r.charge[fi]
            values['charge_continuity_relative']=float(np.linalg.norm(residue)/max(np.linalg.norm(q._peec['A']@i),1e-20))
        else:values['charge_continuity_relative']=0.
        for k,v in values.items():metrics[k].append(float(v))
    return metrics


def run_case(case, out, args):
    folder=Path(out)/case;folder.mkdir(parents=True,exist_ok=True)
    b, terms, fine=layout(case); (folder/'board.json').write_text(json.dumps(b))
    freqs=FSW*np.asarray(args.harmonics,float);amplitudes=harmonic_amps(args.harmonics)
    modes=['peec_slab1','peec_slab2','peec_dc','mqs_dc']
    summaries={}; shared=None; plotted=None
    for mode in modes:
        print(f'{case}: {mode} starting',flush=True);started=time.perf_counter()
        q=build_model(case,args,mode,shared);r=q.solve(freqs);elapsed=time.perf_counter()-started
        metrics=electrical_metrics(q,r)
        metrics.update(frequencies_Hz=freqs.tolist(),seconds=elapsed,triangles=len(q.tris),kernels_reused=shared is not None,
            edges=q.ne,vias=q.nv,loops=q._mats['C'].shape[1],fill=copy.deepcopy(q.fill_info))
        metrics['harmonic_copper_loss_W']=(.5*amplitudes**2*np.array(metrics['copper_R_ohm'])).tolist()
        metrics['selected_harmonic_copper_loss_W']=sum(metrics['harmonic_copper_loss_W'])
        if max(metrics['power_balance_relative'])>1e-6:raise RuntimeError('Power balance failed')
        if max(metrics['charge_continuity_relative'])>1e-5:raise RuntimeError('Charge continuity failed')
        if min(metrics['copper_R_ohm'])<0 or min(metrics['port_R_ohm'])<0:raise RuntimeError('Passivity failed')
        if mode=='peec_slab1':
            shared=q
            C,labels,terminal_labels=q.capacitance_matrix()
            if C.shape!=(2,2):raise RuntimeError('Expected exactly VIN and GND conductors')
            e=np.zeros(2);e[terminal_labels['U1.VIN']]=1;e[terminal_labels['U1.PGND']]=-1
            differential=1/(e@np.linalg.solve(C,e))
            metrics['floating_capacitance_F']=dict(real=float(differential.real),imag=float(differential.imag))
            metrics['maxwell_capacitance_F']=dict(real=C.real.tolist(),imag=C.imag.tolist(),terminals=terminal_labels)
            if not args.no_fields:
                field_start=time.perf_counter()
                gx,gy=np.arange(0,W+1e-9,.5),np.arange(0,H+1e-9,.5)
                scan_heights=np.array([1.,3.,10.])
                Hk,info=q.h_grid(r,list(range(len(freqs))),gx*1e-3,gy*1e-3,
                    (Z_TOP+scan_heights)*1e-3,rtol=.02,return_info=True)
                planes={height:np.moveaxis(Hk[:,:,k]*amplitudes,-1,0) for k,height in enumerate(scan_heights)}
                metrics['field_seconds']=time.perf_counter()-field_start
                metrics['field_convergence']=info
                metrics['peak_H_Am']={str(h):[float(np.linalg.norm(planes[h][fi],axis=-1).max()) for fi in range(len(freqs))] for h in scan_heights}
                write_dumps(str(folder),b,freqs,gx,gy,planes,gx,gy,scan_heights,np.moveaxis(Hk*amplitudes,-1,0))
                np.savez_compressed(folder/'solution.npz',frequencies=freqs,currents=r.i,
                    charge=r.charge,potential=r.potential,nodes=q.nodes,tris=q.tris,z_of=q.z_of,
                    node_of=q.node_of,layer_of=q.layer_of)
                np.savez_compressed(folder/'field-planes.npz',x=gx,y=gy,heights=scan_heights,H=Hk*amplitudes)
                plotted=dict(case=case,q=q,r=r,planes=planes,gx=gx,gy=gy,J=q.current_density(r,0),f=freqs)
        summaries[mode]=metrics
        (folder/(mode+'.json')).write_text(json.dumps(metrics,indent=2)+'\n')
        print(f'{case}: {mode} completed in {elapsed:.2f} s; Rcu at first/last = '
              f'{metrics["copper_R_ohm"][0]*1e3:.4g}/{metrics["copper_R_ohm"][-1]*1e3:.4g} mOhm',flush=True)
        if mode!='peec_slab1':
            del q,r
            import gc
            gc.collect()
    (folder/'comparison.json').write_text(json.dumps(summaries,indent=2)+'\n')
    return summaries,plotted


def write_dumps(d, b, f, gx, gy, planes, vx, vy, vz, vol):
    """All computed harmonics on the three planes and their coarse volume."""
    from pcbqs._artifacts import Dump, write_info, write_npz
    from shapely.geometry import shape
    exc = os.path.join(d, 'mqs', 'exc_U1')
    os.makedirs(exc, exist_ok=True)
    dumps = []
    for hgt, data in planes.items():
        z = Z_TOP + hgt
        dm = Dump(f"H@top+{hgt:g}mm", 'H', (gx[0], gy[0], z), (gx[-1], gy[-1], z), list(f), 2, f"top+{hgt:g}mm")
        write_npz(os.path.join(exc, dm.name + '.npz'), dict(x=gx, y=gy, z=np.array([z])), f, data[:, :, :, None, :])
        dumps.append(dm)
    dm = Dump("H@volume", 'H', (vx[0], vy[0], Z_TOP + vz[0]), (vx[-1], vy[-1], Z_TOP + vz[-1]), list(f), None, 'volume')
    write_npz(os.path.join(exc, dm.name + '.npz'), dict(x=vx, y=vy, z=Z_TOP + vz), f, vol)
    dumps.append(dm)
    m = SimpleNamespace(outline=shape(b['outline']), cu_z={c['name']: c['z'] for c in b['stack']['copper']},
                        copper={l: [q for c in cs for q in getattr(shape(c['polygon']), 'geoms', [shape(c['polygon'])])]
                                for l, cs in b['copper'].items() if cs},
                        ports=[], vias=[(v['x'], v['y'], v['r'], v['z0'], v['z1']) for v in b['vias']], thickness=Z_TOP)
    write_info(exc, dumps, m)
    info = json.load(open(os.path.join(exc, 'dump_info.json')))
    for x in info['dumps']:
        x['unit'] = 'A/m, the 3 A trapezoid\'s harmonic'
    json.dump(info, open(os.path.join(exc, 'dump_info.json'), 'w'))


def draw_layouts(cases, out):
    """nearfield_layouts.png: each case's top copper, its ground plane (and slot), vias, pads, parts and hot loop."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import shape
    # the layouts: the top copper, the ground plane (its slot), the vias, the parts, the hot loop's path
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig, ax = plt.subplots(1, len(cases), figsize=(5.4 * len(cases), 4.6))
    ax = np.atleast_1d(ax)
    for a, case in zip(ax, cases):
        b, terms, fine = layout(case)
        for c in b['copper']['In1.Cu']:
            g = shape(c['polygon'])
            for p in getattr(g, 'geoms', [g]):
                a.fill(*p.exterior.xy, color='#cfd8dc', lw=0)
                for h in p.interiors:
                    a.fill(*h.xy, color='white', lw=0)
        for c in b['copper']['F.Cu']:
            g = shape(c['polygon'])
            col = '#e5a33b' if c['net'] == 'VIN' else '#5b8c5a'
            for p in getattr(g, 'geoms', [g]):
                a.fill(*p.exterior.xy, color=col, lw=0.5, ec='k')
        for v in b['vias']:
            a.add_patch(plt.Circle((v['x'], v['y']), v['r'], color='k'))
        for name, pads in terms.items():
            (_, pg), = pads
            x0, y0, x1, y1 = pg.bounds
            a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec='r', lw=1.0))
        u_c = (18.0, 13.0)
        a.add_patch(Rectangle((15.4, 11.6), 5.2, 2.8, fill=False, ec='k', lw=1.2, ls='--'))
        a.text(18.0, 14.8, 'U1 (regulator)', ha='center', fontsize=7)
        cx = 18.0
        cy = 10.5 if case == 'good' else 3.0
        a.text(cx, cy - 1.4, 'C1 10 µF (input)', ha='center', fontsize=7)
        a.add_patch(Rectangle((15.9 if case == 'good' else 15.4, cy - 0.6), 4.2 if case == 'good' else 5.2, 1.2,
                              fill=False, ec='k', lw=1.2, ls='--'))
        # the hot loop: C1+ -> VIN pin -> (switches) -> PGND pin -> C1- (top or the plane) -> back through C1
        path = [(16.3, cy), (16.0, 12.8), (20.0, 12.8), (19.9 if case == 'good' else 20.0, cy)]
        for p0, p1 in zip(path, path[1:] + path[:1]):
            a.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=10, color='crimson', lw=1.2, alpha=0.8))
        if case == 'slot':
            a.text(24, 8.0, 'slot in the ground plane', fontsize=7, color='0.3')
            a.annotate('', xy=(7.6, 7.2), xytext=(19.6, 5.4), arrowprops=dict(arrowstyle='->', color='crimson', ls=':',
                       connectionstyle='arc3,rad=-0.3'))
            a.text(9.5, 4.3, 'return detours round the slot', fontsize=6.5, color='crimson')
        a.set_xlim(4, 32); a.set_ylim(0.0, 17.5); a.set_aspect('equal')
        a.set_xlabel('mm', fontsize=7); a.tick_params(labelsize=6)
        a.set_title({'good': 'good: C1 tight against VIN and PGND, solid plane',
                     'far': 'far: C1 10 mm away, VIN on a 0.5 mm trace',
                     'slot': 'slot: as far, a slot in the plane under the return'}[case], fontsize=9)
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color='#e5a33b', label='VIN (F.Cu)'), Patch(color='#5b8c5a', label='GND (F.Cu)'),
                          Patch(color='#cfd8dc', label='GND plane (In1, 0.2 mm below)'),
                          Patch(fill=False, ec='r', label='pads'), Patch(color='crimson', label='hot loop')],
                 fontsize=6.5, loc='upper left')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'nearfield_layouts.png'), dpi=120); plt.close(fig)


def figures(results, fields, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.tri import Triangulation
    from shapely.geometry import shape

    cases=list(results);draw_layouts(cases,out)
    fig,axes=plt.subplots(1,len(cases),figsize=(5.2*len(cases),4),squeeze=False)
    styles={'mqs_dc':(':','MQS / DC sheet'),'peec_dc':('--','PEEC / DC sheet'),
            'peec_slab2':('-.','PEEC / 2-face slab'),'peec_slab1':('-','PEEC / 1-face slab')}
    for ax,case in zip(axes[0],cases):
        for mode,(ls,label) in styles.items():
            m=results[case][mode];ax.loglog(np.array(m['frequencies_Hz'])/1e6,
                np.array(m['copper_R_ohm'])*1e3,ls,marker='.',label=label)
        ax.set(xlabel='Frequency (MHz)',ylabel='Copper loss resistance (mΩ)',title=case)
        ax.grid(True,which='both',alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(Path(out)/'copper-loss.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(cases,[results[c]['peec_slab1']['floating_capacitance_F']['real']*1e12 for c in cases])
    axes[0].set(ylabel='Floating VIN–GND capacitance (pF)',title='Extracted board capacitance; excludes C1')
    x=np.arange(len(cases));width=.18
    for j,(mode,(_,label)) in enumerate(styles.items()):
        axes[1].bar(x+(j-1.5)*width,[results[c][mode]['selected_harmonic_copper_loss_W']*1e3 for c in cases],width,label=label)
    axes[1].set(xticks=x,xticklabels=cases,ylabel='Copper dissipation (mW)',title='Selected AC harmonics only; excludes DC')
    axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(Path(out)/'capacitance-and-loss.png',dpi=150);plt.close(fig)
    if not fields:return
    for kind in ['H','H10','current','current30','charge','mesh']:
        fig,axes=plt.subplots(1,len(fields),figsize=(5.3*len(fields),4.5),squeeze=False)
        data=[]
        for rec in fields:
            q,r=rec['q'],rec['r']
            if kind in ('H','H10'):values=np.linalg.norm(rec['planes'][10. if kind=='H10' else 3.][0],axis=-1)
            elif kind in ('current','current30'):
                fi=-1 if kind=='current30' else 0
                values=np.linalg.norm(q.current_density(r,fi)['In1.Cu'][1],axis=-1)*harmonic_amps(args_harmonics(r)[fi])
            elif kind=='charge':
                area=np.bincount(q.node_of,weights=q.rwg.area,minlength=q.n_nodes)
                values=abs(r.charge[0][q.node_of]/area[q.node_of])*harmonic_amps(args_harmonics(r)[0])*1e9
            else:values=None
            data.append(values)
        if kind!='mesh':
            vmax=max(float(v.max()) for v in data);norm=LogNorm(vmin=max(vmax/1000,1e-15),vmax=vmax)
        for ax,rec,values in zip(axes[0],fields,data):
            q=rec['q'];board,_,_=layout(rec['case'])
            if kind in ('H','H10'):
                im=ax.pcolormesh(rec['gx'],rec['gy'],values.T,norm=norm,cmap='inferno',shading='auto')
            else:
                idx=q.layer_tris['In1.Cu'] if kind in ('current','current30','mesh') else q.layer_tris['F.Cu']
                tri=Triangulation(q.nodes[:,0]*1e3,q.nodes[:,1]*1e3,q.tris[idx])
                if kind=='mesh':ax.triplot(tri,lw=.25,color='#527d96')
                else:im=ax.tripcolor(tri,facecolors=np.maximum(values if kind in ('current','current30') else values[idx],1e-30),norm=norm,cmap='viridis')
            for layer,color in [('F.Cu','#ee7b30'),('In1.Cu','#777777')]:
                for item in board['copper'][layer]:
                    g=shape(item['polygon'])
                    for p in getattr(g,'geoms',[g]):
                        ax.plot(*p.exterior.xy,color=color,lw=.65)
                        for hole in p.interiors:ax.plot(*hole.xy,color=color,lw=.65)
            ax.set(xlim=(0,W),ylim=(0,H),aspect='equal',title=rec['case'],xlabel='mm',ylabel='mm')
        label={'H':'|H| (A/m peak), 3 mm above copper','H10':'|H| (A/m peak), 10 mm above copper',
               'current':'Ground sheet |K| (A/m peak)','current30':'Ground sheet |K| (A/m peak)',
               'charge':'Top-sheet |charge density| (nC/m² peak)','mesh':'Ground-plane mesh'}[kind]
        fi=-1 if kind=='current30' else 0
        fig.suptitle(label+' · '+str(fields[0]['f'][fi]/1e6)+' MHz' if kind!='mesh' else label)
        if kind!='mesh':fig.colorbar(im,ax=axes[0].tolist(),label=label,shrink=.8)
        fig.savefig(Path(out)/{'H':'magnetic-fields.png','H10':'magnetic-fields-10mm.png','current':'ground-return.png',
            'current30':'ground-return-30MHz.png','charge':'surface-charge.png','mesh':'meshes.png'}[kind],dpi=150,bbox_inches='tight');plt.close(fig)


def args_harmonics(result):
    return result.f/FSW


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',nargs='?',default='nearfield_peec')
    parser.add_argument('--cases',default='good,far,slot')
    parser.add_argument('--harmonics',default=','.join(map(str,HARM)))
    parser.add_argument('--dmax',type=float,default=1.2)
    parser.add_argument('--fine',type=float,default=.4)
    parser.add_argument('--backend',default='openmp')
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--no-fields',action='store_true')
    parser.add_argument('--plot-only',action='store_true',help='Rebuild figures from saved results without a solve')
    args=parser.parse_args();args.harmonics=[int(n) for n in args.harmonics.split(',')]
    cases=args.cases.split(',')
    if not args.harmonics or min(args.harmonics)<=0 or not cases or any(c not in ('good','far','slot') for c in cases):parser.error('Invalid cases/harmonics')
    if not all(np.isfinite(x) and x>0 for x in [args.dmax,args.fine,args.workers]):parser.error('Mesh sizes and workers must be positive')
    Path(args.out).mkdir(parents=True,exist_ok=True);results={};fields=[];started=time.perf_counter()
    if args.plot_only:
        for case in cases:
            folder=Path(args.out)/case;results[case]=json.loads((folder/'comparison.json').read_text())
            q=build_model(case,args,'peec_slab1')
            with np.load(folder/'solution.npz') as data:
                if not np.array_equal(q.tris,data['tris']) or not np.array_equal(q.nodes,data['nodes']):raise ValueError('Plot mesh differs from saved solution')
                r=SimpleNamespace(f=data['frequencies'],i=data['currents'],charge=data['charge'])
            with np.load(folder/'field-planes.npz') as data:
                gx,gy=data['x'],data['y'];planes={height:np.moveaxis(data['H'][:,:,k],-1,0) for k,height in enumerate(data['heights'])}
            fields.append(dict(case=case,q=q,r=r,planes=planes,gx=gx,gy=gy,J=q.current_density(r,0),f=r.f))
        figures(results,fields,args.out);return
    for case in cases:
        result,rec=run_case(case,args.out,args);results[case]=result
        if rec:fields.append(rec)
        (Path(args.out)/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
    (Path(args.out)/'settings.json').write_text(json.dumps(dict(
        fsw_Hz=FSW,duty=DUTY,edge_s=TR,peak_current_A=IPK,harmonics=args.harmonics,
        dmax_mm=args.dmax,fine_mm=args.fine,dielectric='stack',copper_thickness_m=35e-6,
        conductor_primary='one-sided slab',proximity='lateral redistribution solved; thickness proximity assumed, not resolved',
        sigma_Sm=5.8e7,workers=args.workers,backend=args.backend,total_seconds=time.perf_counter()-started,
        note='Selected-harmonic AC copper loss excludes DC and uncomputed harmonics. One/two face comparison is a model sensitivity, not a rigorous bound.'),indent=2)+'\n')
    figures(results,fields,args.out)


if __name__=='__main__':
    main()
