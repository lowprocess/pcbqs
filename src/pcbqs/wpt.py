"""Two winding, series-compensated link: SRF capacitance and explicit load matching."""
import numpy as np
from scipy.optimize import minimize_scalar,brentq


def winding_resistance(coil):
    # Q-derived AC loss is a TOTAL resistance, never another full resistance on DCR.
    return max(coil['R_ohm'],coil.get('Rac_ref_ohm',coil['R_ohm']))


def coil_capacitance(coil):
    return coil.get('Cp_F',1/((2*np.pi*coil['SRF_Hz'])**2*coil['L_H']) if coil.get('SRF_Hz') else 0.)


def core_resistance(coil):
    """Parallel core-loss resistor fitted to isolated-coil Q reference, when requested.

    Its constant conductance is an assumption, not measured broadband ferrite data.
    It damps the otherwise unrealistically high-Q DCR-only SRF pole passively.
    """
    if coil.get('core_R_ohm'):return coil['core_R_ohm']
    if coil.get('loss_model')!='parallel_core':return None
    target=coil.get('Rac_ref_ohm',coil['R_ohm']);w=2*np.pi*coil.get('loss_reference_Hz',125e3)
    R=coil['R_ohm'];x=w*coil['L_H'];cp=coil_capacitance(coil)
    def real_z(logr):
        rc=np.exp(logr);z=R+1/(1/(1j*x)+1/rc)
        return (z/(1+1j*w*cp*z)).real
    lo=np.log(max(x,1e-9));hi=np.log(1e15)
    if target<=real_z(hi):return None
    if target>=real_z(lo):raise ValueError('Q loss reference incompatible with parallel-core model')
    return float(np.exp(brentq(lambda t:real_z(t)-target,lo,hi)))


def series_resistance(coil):
    return coil['R_ohm'] if coil.get('loss_model') in ('parallel_core','dc_only') else winding_resistance(coil)


def impedance(f, primary, secondary, k):
    if not 0<=k<1:raise ValueError('coupling k must be in [0, 1)')
    w=2*np.pi*f
    if f<=0:raise ValueError('frequency must be positive')
    M=k*np.sqrt(primary['L_H']*secondary['L_H'])
    ZL=1j*w*np.array([[primary['L_H'],M],[M,secondary['L_H']]])
    core=[core_resistance(c) for c in (primary,secondary)]
    Z=np.linalg.inv(np.linalg.inv(ZL)+np.diag([1/r if r else 0 for r in core]))
    Z+=np.diag([series_resistance(primary),series_resistance(secondary)])
    return np.linalg.inv(np.linalg.inv(Z)+1j*w*np.diag([coil_capacitance(primary),coil_capacitance(secondary)]))


def link(f, primary, secondary, k, C1, C2, load, cap_R1=0., cap_R2=0.):
    Z=impedance(f,primary,secondary,k);w=2*np.pi*f
    sec=Z[1,1]+1/(1j*w*C2)+load+cap_R2
    ratio=-Z[1,0]/sec
    zin=Z[0,0]+1/(1j*w*C1)+cap_R1+Z[0,1]*ratio
    eff=abs(ratio)**2*load/zin.real
    return dict(input_Z_ohm=zin,secondary_per_primary=ratio,efficiency=float(eff))


def match_receiver(f, primary, secondary, k, C1, cap_R1=0., cap_R2=0., C2=None, load=None):
    """Tune/match only the receiver parameters not explicitly held fixed.

    Excludes driver/filter loss; primary C1 remains fixed. Load is a fundamental
    AC-equivalent resistance, not a rectified voltage, DC resistance or fixed wattage.
    """
    Z=impedance(f,primary,secondary,k)
    tuned=C2 is None;matched=load is None
    if tuned:C2=1/(2*np.pi*f*Z[1,1].imag)
    if not np.isfinite(C2) or C2<=0:raise ValueError('receiver cannot be series tuned at this frequency')
    if k==0:raise ValueError('matching requires nonzero coupling')
    def objective(logr):return -link(f,primary,secondary,k,C1,C2,np.exp(logr),cap_R1,cap_R2)['efficiency']
    if matched:
        opt=minimize_scalar(objective,bounds=(-14,14),method='bounded',options={'xatol':1e-10})
        load=float(np.exp(opt.x))
    if not np.isfinite(load) or load<=0:raise ValueError('receiver load must be finite and positive')
    r=link(f,primary,secondary,k,C1,C2,load,cap_R1,cap_R2)
    return dict(receiver_capacitor_mode='tuned' if tuned else 'fixed',load_mode='efficiency_matched' if matched else 'fixed',C2_F=float(C2),load_ohm=float(load),efficiency=r['efficiency'],input_R_ohm=float(r['input_Z_ohm'].real),input_X_ohm=float(r['input_Z_ohm'].imag))
