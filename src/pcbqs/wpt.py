"""A winding's parasitics as the circuit's decks have them: its AC resistance, its self-capacitance (from its SRF)
and a parallel core-loss resistance fitted to its Q."""
import numpy as np
from scipy.optimize import brentq


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
