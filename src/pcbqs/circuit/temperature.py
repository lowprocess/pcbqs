"""Parts at a temperature (R-E9). A part's temperature_C (25 C unless given) moves what it loses by, from its
values at 25 C (the datasheet's):

    inductor    DCR_ohm (or R_ohm) and AC_R_ohm's resistances: copper's, COPPER_PER_K, unless tempco_per_K
    resistor    value x (1 + tempco_per_K dT); tempco_per_K 0 unless given (a resistor's is ppm/K)
    capacitor   ESR_ohm x (1 + ESR_tempco_per_K dT); 0 unless given (an electrolytic's ESR falls as it warms)
    switch      model.Ron_ohm x (1 + model.Ron_tempco_per_K dT); 0.006/K unless given, a silicon MOSFET's
                (R_DS(on) 1.6x at 125 C; the datasheet's curve gives the part's own). An ideal switch and a
                subcircuit (a vendor model, with its own temperature) are not moved
    diode       model.Vf_V + model.Vf_tempco_V_per_K dT; -2 mV/K unless given (a junction's)

and the board's copper, board.temperature_C, its resistances by copper's. pcbqs.circuit --warm sets these from the
thermal model of the run's own losses, case by case (circuit/warm.py)."""
import copy

import numpy as np

REFERENCE_C = 25.0
COPPER_PER_K = 0.00393 / (1 + 0.00393 * (REFERENCE_C - 20))    # annealed copper's 0.00393/K from 20 C, from 25 C
SWITCH_PER_K = 0.006
DIODE_V_PER_K = -0.002


def _scale(value, per_k, dt):
    return value * max(1 + per_k * dt, 0.05)


def apply(part):
    """The part's values at its temperature_C, in place (resolved: numbers). Its values at 25 C are kept in
    part['nominal'], and a part that has them is left: applying twice is applying once."""
    t = part.get('temperature_C')
    if t is None or 'nominal' in part:
        return part
    dt = float(t) - REFERENCE_C
    kind = part.get('type')
    nominal = {}

    def move(holder, key, new, where=None):
        nominal[where or key] = holder[key]
        holder[key] = new
    if kind == 'inductor':
        per_k = part.get('tempco_per_K', COPPER_PER_K)
        for key in ('DCR_ohm', 'R_ohm'):
            if part.get(key):
                move(part, key, _scale(part[key], per_k, dt))
        if part.get('AC_R_ohm'):
            move(part, 'AC_R_ohm', [[f, _scale(r, per_k, dt)] for f, r in part['AC_R_ohm']])
    elif kind == 'resistor' and part.get('tempco_per_K'):
        move(part, 'value', _scale(part['value'], part['tempco_per_K'], dt))
    elif kind == 'capacitor' and part.get('ESR_ohm') and part.get('ESR_tempco_per_K'):
        move(part, 'ESR_ohm', _scale(part['ESR_ohm'], part['ESR_tempco_per_K'], dt))
    elif kind == 'switch':
        from ..power_spice import RON_OHM
        m = part.setdefault('model', {})
        if m.get('type', 'vdmos') in ('switch', 'mosfet', 'vdmos'):
            ron = m.get('Ron_ohm', RON_OHM)
            nominal['model.Ron_ohm'] = ron
            m['Ron_ohm'] = _scale(ron, m.get('Ron_tempco_per_K', SWITCH_PER_K), dt)
    elif kind == 'diode':
        m = part.setdefault('model', {})
        vf = m.get('Vf_V', 0.7)
        nominal['model.Vf_V'] = vf
        m['Vf_V'] = max(vf + m.get('Vf_tempco_V_per_K', DIODE_V_PER_K) * dt, 0.05)
    if nominal:
        part['nominal'] = nominal
    return part


def copper_model(model, temperature_C):
    """The copper's R-L model (copper.prepare's 'model') with its resistances at temperature_C."""
    k = max(1 + COPPER_PER_K * (float(temperature_C) - REFERENCE_C), 0.05)
    out = copy.deepcopy(model)
    out['R_ohm'] = (np.asarray(model['R_ohm']) * k).tolist()
    for m in out.get('foster', []):
        m['R_ohm'] = (np.asarray(m['R_ohm']) * k).tolist()
    return out


def at(model, temperatures):
    """A copy of the model with temperatures {'parts': {ref: C}, 'board': C} set (None: the model itself)."""
    if not temperatures:
        return model
    m = copy.deepcopy(model)
    given = temperatures.get('parts', {})
    for p in m.get('parts', []):
        if p['ref'] in given:
            p['temperature_C'] = given[p['ref']]
    if temperatures.get('board') is not None and m.get('board'):
        m['board']['temperature_C'] = temperatures['board']
    return m
