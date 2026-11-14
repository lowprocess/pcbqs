"""A winding's parasitics as the circuit's decks have them: its self-capacitance (from its SRF) and a parallel
core-loss resistance (core_R_ohm), which fit_core_resistance works out from a measured AC resistance."""
import numpy as np
from scipy.optimize import brentq


def coil_capacitance(coil):
    return coil.get('Cp_F',1/((2*np.pi*coil['SRF_Hz'])**2*coil['L_H']) if coil.get('SRF_Hz') else 0.)


def core_resistance(coil):
    """The winding's parallel core-loss resistor (core_R_ohm), or None."""
    return coil.get('core_R_ohm') or None


def series_resistance(coil):
    return coil['R_ohm']


def fit_core_resistance(L_H, R_ohm, ac_ohm, f_Hz, SRF_Hz=None, Cp_F=None):
    """The parallel core-loss resistance (ohm) that gives a winding of inductance L_H and DC resistance R_ohm the AC
    resistance ac_ohm at f_Hz (its Q there), with its self-capacitance (Cp_F, or from SRF_Hz): a model's
    core_R_ohm. None when the DC resistance alone is already as lossy. Its constant conductance is an assumption,
    not broadband ferrite data; it damps the DCR-only SRF pole passively. (The fit had been done at every deck
    build from Rac_ref_ohm, loss_model and loss_reference_Hz, for wpthb's coils alone: R-F1.)"""
    w = 2 * np.pi * f_Hz
    x = w * L_H
    cp = coil_capacitance(dict(L_H=L_H, SRF_Hz=SRF_Hz, **({'Cp_F': Cp_F} if Cp_F else {})))

    def real_z(logr):
        rc = np.exp(logr); z = R_ohm + 1 / (1 / (1j * x) + 1 / rc)
        return (z / (1 + 1j * w * cp * z)).real
    lo, hi = np.log(max(x, 1e-9)), np.log(1e15)
    if ac_ohm <= real_z(hi):
        return None
    if ac_ohm >= real_z(lo):
        raise ValueError('the AC resistance is past what a parallel core resistance can give')
    return float(np.exp(brentq(lambda t: real_z(t) - ac_ohm, lo, hi)))
