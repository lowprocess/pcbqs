"""Scientific plots of completed power-circuit comparisons and inferred fields."""
import json
from pathlib import Path
import numpy as np


def render(out,animation=True):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,PillowWriter
    from .power_spice import periodic_samples
    out=Path(out);summary=json.loads((out/'power-circuit.json').read_text());cfg=json.loads((out/'model.json').read_text());rows=summary['cases'];frequencies=sorted({r['frequency_Hz'] for r in rows})
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for f in frequencies:
        rr=sorted([r for r in rows if r['frequency_Hz']==f],key=lambda r:r['target_W']);p=[r['load_power_W'] for r in rr];label=f'{f/1000:g} kHz'
        for ax,key,scale in [(axs[0,0],'primary_rms_A',1),(axs[0,1],'phase_deg',1),(axs[1,0],'efficiency',100),(axs[1,1],'secondary_rms_A',1)]:ax.plot(p,[r[key]*scale for r in rr],'o-',label=label)
    for ax,title,label in [(axs[0,0],'Primary winding current','RMS A'),(axs[0,1],f'Phase required at {cfg["voltage_V"]:g} V','Degrees'),(axs[1,0],'Electrical source + ideal gate-source efficiency','Percent'),(axs[1,1],'Secondary winding current','RMS A')]:ax.set(title=title,xlabel='Delivered AC load power (W)',ylabel=label);ax.grid(alpha=.25);ax.legend()
    rx=f'fixed {cfg["secondary_C_F"]*1e9:g} nF' if cfg.get('secondary_C_F') is not None else 'tuned per frequency'
    fig.suptitle(f'Coupled receiver, k={cfg["coupling"]:g} · Primary {cfg["primary"]["L_H"]*1e6:g} µH / {cfg["primary_C_F"]*1e9:g} nF · Secondary {rx}')
    fig.savefig(out/'coupled-comparison.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(11,7),layout='constrained',sharex=True)
    for sweep in summary['frequency_sweeps']:
        r=next(r for r in rows if r['frequency_Hz']==sweep['operating_Hz']);m=r['matching'];f=np.array(sweep['frequencies_Hz'])/1000;label=f'{r["frequency_Hz"]/1000:g} kHz operating point: C₂={m["C2_F"]*1e9:.1f} nF, Rload={m["load_ohm"]:.2f} Ω'
        axs[0].plot(f,sweep['load_power_per_Vrms2'],label=label);axs[1].plot(f,sweep['input_X_ohm'],label=label)
        for ax in axs:ax.axvline(r['frequency_Hz']/1000,lw=.8,ls='--',alpha=.4)
    axs[0].set(title='Loaded link response: each receiver capacitor/load held fixed during its sweep',ylabel='Load W / applied fundamental Vrms²');axs[1].set(xlabel='Frequency (kHz)',ylabel='Input reactance (Ω)',ylim=(-25,25));axs[1].axhline(0,color='grey',lw=.8)
    for ax in axs:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(out/'frequency-response.png',dpi=150);plt.close(fig)
    selected=[max([r for r in rows if r['frequency_Hz']==f],key=lambda r:r['target_W']) for f in frequencies]
    fig,axs=plt.subplots(3,len(selected),figsize=(12,8),layout='constrained',squeeze=False)
    series=[]
    for j,r in enumerate(selected):
        with np.load(Path(r['folder'])/'transient.npz') as data:trans={k:data[k] for k in data.files}
        _,v,_=periodic_samples(trans,r['frequency_Hz'],samples=1024);phase=np.arange(1024)/1024
        for key in v:v[key]=v[key][-1]
        series.append((r,v));axs[0,j].plot(phase,v['bridge.V'],label='Bridge differential');axs[0,j].set(title=f'{r["frequency_Hz"]/1000:g} kHz / {r["load_power_W"]:.2f} W / {r["phase_deg"]:.2f}°',ylabel='Bridge V')
        for key,label in [('PRIMARY.winding_I','Primary'),('SECONDARY.winding_I','Secondary')]:axs[1,j].plot(phase,v[key],label=label)
        axs[1,j].set(ylabel='Winding A');axs[1,j].legend()
        axs[2,j].plot(phase,v['bus.V'],label='VHB');axs[2,j].set(xlabel='Switching phase t/T',ylabel='Local VHB (V)')
        for ax in axs[:,j]:ax.grid(alpha=.2)
    fig.suptitle('Settled electrical waveforms; selected transistor/parasitic parameters are illustrative')
    fig.savefig(out/'electrical-waveforms.png',dpi=150);plt.close(fig)
    field_rows=[r for r in selected if (Path(r['folder'])/'time_fields.npz').exists()]
    if not field_rows:return
    from .mqs_cli import draw_layout
    data=[np.load(Path(r['folder'])/'time_fields.npz') for r in field_rows];layout=json.loads((Path(field_rows[0]['folder'])/'mqs_board.json').read_text())
    mag=[np.linalg.norm(d['H'][:,:,:,0,:],axis=-1) for d in data];limit=max(a.max() for a in mag)
    fig=plt.figure(figsize=(12,8),layout='constrained');fig.set_layout_engine('constrained',rect=(0,.03,1,.97));gs=fig.add_gridspec(2,len(data),height_ratios=[4,1.3]);images=[];cursors=[]
    for j,(r,d,vv) in enumerate(zip(field_rows,data,mag)):
        ax=fig.add_subplot(gs[0,j]);draw_layout(ax,layout);im=ax.imshow(vv[0].T,origin='lower',extent=[d['x'][0],d['x'][-1],d['y'][0],d['y'][-1]],vmin=0,vmax=limit,cmap='inferno',alpha=.8,zorder=2);images.append(im)
        ax.set(xlim=(d['x'][0],d['x'][-1]),ylim=(d['y'][0],d['y'][-1]),aspect='equal',xlabel='x (mm)',ylabel='y (mm)',title=f'{r["frequency_Hz"]/1000:g} kHz / {r["load_power_W"]:.2f} W')
        fig.colorbar(im,ax=ax,label='PCB |H(t)|, top + 1 mm (A/m)',shrink=.85)
        ax=fig.add_subplot(gs[1,j]);_,v=next(t for t in series if t[0]['frequency_Hz']==r['frequency_Hz']);ax.plot(np.arange(1024)/1024,v['PRIMARY.winding_I'],label='Primary winding');ax.set(xlabel='Switching phase t/T',ylabel='A',xlim=(0,1));ax.grid(alpha=.2);cursors.append(ax.axvline(0,color='black',lw=1))
    header=fig.suptitle('')
    fig.text(.5,.003,'Copper/via fields only · Coil/receiver/lead/package field geometry omitted · Bandwidth limited to retained harmonics',ha='center',fontsize=9)
    def update(k):
        for im,v,line in zip(images,mag,cursors):im.set_data(v[k].T);line.set_xdata([k/64,k/64])
        header.set_text(f'Coupled-load driver fields · {cfg["voltage_V"]:g} V, k={cfg["coupling"]:g} · t/T = {k/64:.3f}')
        return [*images,*cursors,header]
    update(20);fig.savefig(out/'coupled-fields.png',dpi=150)
    if animation:FuncAnimation(fig,update,frames=64,interval=80).save(out/'coupled-fields.gif',writer=PillowWriter(fps=12),dpi=85)
    plt.close(fig)
    print('Wrote circuit/field comparison figures',flush=True)


if __name__=='__main__':
    import sys
    render(sys.argv[1])
