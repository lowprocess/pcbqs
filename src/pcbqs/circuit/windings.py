"""An inductor's losses worked out after the run (R-E10): what its winding loses to its AC resistance and its core to
the flux's swing, from the reported run's last period. The circuit itself has the winding's DC resistance (and a
constant core resistor, core_R_ohm, where one is given); these come on top, from two fields of an inductor:

    AC_R_ohm    [[f_Hz, R_ohm], ...]: the winding's resistance against frequency (a datasheet's ACR curve, or a
                measurement). Each harmonic n of the winding's current I_n (RMS) loses (R(n f) - DCR) I_n^2 more
                than the DCR the circuit has. Between the points R is interpolated on log-log axes; below the first
                the excess over DCR falls as f^2 (eddy and proximity losses at low frequency); above the last R
                grows as sqrt(f) (skin effect). The current's AC above the harmonics taken (HARMONICS) is put at
                the last one's resistance.
    core_loss   {k, alpha, beta, turns, Ae_m2, Ve_m3}: the core material's Steinmetz fit, P = k f^alpha B^beta in
                W/m^3 with f in Hz and B the peak flux density in T (core_k_si converts a datasheet's k for
                mW/cm^3, kHz and mT), and the core's turns, effective area and volume. The loss is the improved generalised Steinmetz equation's (iGSE)
                for the flux's own waveform, B from the winding's voltage less its resistance's drop: one major
                loop a period, and no DC-bias or relaxation terms (a ferrite under DC bias loses more). A
                transformer's core is one winding's: give core_loss on one of its windings.

Both go to the part (core_loss_W, AC_winding_loss_W, and their sum added_loss_W, which the thermal model counts on
its footprint) and to the case's loss; the efficiency counts them as drawn from the source on top of what the run
drew (first order: the run's currents are as they would be with them)."""
import math

import numpy as np

from . import model as M

HARMONICS = 200


def core_k_si(k, alpha, beta, f_unit=1e3, B_unit=1e-3, P_unit=1e3):
    """A Steinmetz k given for f in f_unit Hz, B in B_unit T and P in P_unit W/m^3 (the defaults: kHz, mT and
    mW/cm^3, which is 1e3 W/m^3) as core_loss's k (Hz, T, W/m^3)."""
    return k * P_unit / (f_unit ** alpha * B_unit ** beta)


def check(part):
    """A part's AC_R_ohm and core_loss, resolved to numbers: a ModelError for what cannot be used."""
    ref = part['ref']
    table = part.get('AC_R_ohm')
    if table is not None:
        M._need(isinstance(table, list) and table and all(isinstance(p, list) and len(p) == 2 for p in table),
                f"{ref}.AC_R_ohm: a list of [f_Hz, R_ohm] pairs")
        f = [p[0] for p in table]
        M._need(all(x > 0 for x in f) and all(b > a for a, b in zip(f, f[1:])),
                f"{ref}.AC_R_ohm: the frequencies must be positive and rising")
        dcr = float(part.get('DCR_ohm', part.get('R_ohm', 0.0)) or 0.0)
        low = next((p for p in table if p[1] < dcr * (1 - 1e-9)), None)
        if low is not None:
            raise M.ModelError(f"{ref}.AC_R_ohm: {low[1]:g} ohm at {low[0]:g} Hz is below its DCR_ohm ({dcr:g} ohm)")
    core = part.get('core_loss')
    if core is not None:
        need = ('k', 'alpha', 'beta', 'turns', 'Ae_m2', 'Ve_m3')
        M._need(isinstance(core, dict) and all(k in core for k in need),
                f"{ref}.core_loss: needs {', '.join(need)}")
        M._need(all(isinstance(core[k], (int, float)) and core[k] > 0 for k in need),
                f"{ref}.core_loss: {', '.join(need)} must be positive numbers")
        M._need(not part.get('core_R_ohm'), f"{ref}: core_R_ohm and core_loss both give its core's loss: keep one")


def ac_resistance(table, dcr, f):
    """R(f) from AC_R_ohm's points (see the module's notes), for frequencies f (Hz, an array)."""
    pts = np.asarray(table, float)
    fp, rp = pts[:, 0], np.maximum(pts[:, 1], 1e-30)
    f = np.asarray(f, float)
    r = np.exp(np.interp(np.log(np.maximum(f, 1e-30)), np.log(fp), np.log(rp)))
    below, above = f < fp[0], f > fp[-1]
    r[below] = dcr + (rp[0] - dcr) * (f[below] / fp[0]) ** 2
    r[above] = rp[-1] * np.sqrt(f[above] / fp[-1])
    return r


def last_period(t, y, period):
    """The samples of the run's last period, with its ends interpolated: (t, y)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    end = t[-1]
    start = end - period
    if start < t[0] - 1e-12 * period:
        raise ValueError('the run is shorter than a period')
    inner = (t > start) & (t < end)
    return np.r_[start, t[inner], end], np.r_[np.interp(start, t, y), y[inner], np.interp(end, t, y)]


def igse_coefficient(k, alpha, beta):
    """iGSE's k_i for Steinmetz's k, alpha, beta: a sine of peak B at f gives k f^alpha B^beta."""
    cos_alpha = 2 * math.sqrt(math.pi) * math.gamma((alpha + 1) / 2) / math.gamma(alpha / 2 + 1)   # int |cos|^a, 0..2 pi
    return k / ((2 * math.pi) ** (alpha - 1) * 2 ** (beta - alpha) * cos_alpha)


def igse_loss(t, v, core):
    """The core's loss (W) for its winding's voltage v(t) less the resistive drop (one period, piecewise linear):
    B = int v dt / (turns Ae); P = Ve k_i dB^(beta - alpha) / T int |dB/dt|^alpha dt."""
    k, a, b = core['k'], core['alpha'], core['beta']
    area = core['turns'] * core['Ae_m2']
    dt = np.diff(t)
    flux = np.r_[0.0, np.cumsum(dt * (v[:-1] + v[1:]) / 2)]          # (exact for a piecewise-linear v)
    swing = (flux.max() - flux.min()) / area
    if swing <= 0:
        return 0.0
    u, w = np.polynomial.legendre.leggauss(8)                       # |v|^alpha over each segment
    u, w = (u + 1) / 2, w / 2
    s = v[:-1, None] + (v[1:] - v[:-1])[:, None] * u[None, :]
    mean = np.sum(dt[:, None] * w[None, :] * (np.abs(s) / area) ** a) / (t[-1] - t[0])
    return float(core['Ve_m3'] * igse_coefficient(k, a, b) * swing ** (b - a) * mean)


def ac_winding_loss(t, i, period, table, dcr):
    """What the winding's AC resistance loses above its DCR (W), from its current i(t) over the last period."""
    from ..power_waveforms import periodic_coefficients
    coef, info = periodic_coefficients(t, i, 1 / period, HARMONICS)
    c = coef[1:, 0]
    rms2 = 2 * np.abs(c) ** 2                                       # each harmonic's mean square (A^2)
    f = np.arange(1, HARMONICS + 1) / period
    excess = np.maximum(ac_resistance(table, dcr, f) - dcr, 0.0)
    rest = max(float(info['AC_mean_square_A2'][0]) - float(rms2.sum()), 0.0)
    return float(rms2 @ excess + rest * excess[-1])


def losses(model, result, deck, period):
    """{ref: dict(core_loss_W, AC_winding_loss_W, added_loss_W)} for the model's inductors that give AC_R_ohm or
    core_loss; {} when none does."""
    out = {}
    for part in model.get('parts', []):
        if part.get('type') != 'inductor' or not part.get('enabled', True) \
                or (part.get('AC_R_ohm') is None and part.get('core_loss') is None):
            continue
        ref = part['ref']
        check(part)
        dcr = float(part.get('DCR_ohm', part.get('R_ohm', 0.0)) or 0.0)
        current = result[ref + '.winding_I'] if ref + '.winding_I' in result else result[ref + '.I']
        t, i = last_period(result['time_s'], current, period)
        _, v = last_period(result['time_s'], result[ref + '.V'], period)
        row = dict(core_loss_W=0.0, AC_winding_loss_W=0.0)
        if part.get('AC_R_ohm') is not None:
            row['AC_winding_loss_W'] = ac_winding_loss(result['time_s'], current, period, part['AC_R_ohm'], dcr)
        if part.get('core_loss') is not None:
            row['core_loss_W'] = igse_loss(t, v - dcr * i, part['core_loss'])
        row['added_loss_W'] = row['core_loss_W'] + row['AC_winding_loss_W']
        out[ref] = row
    return out


def apply(stats, added, notes):
    """The losses worked out after the run into the case's results (see the module's notes)."""
    total = 0.0
    for ref, row in added.items():
        stats['parts'][ref].update(row)
        total += row['added_loss_W']
    stats['added_loss_W'] = total
    stats['loss_W'] += total
    supplied = stats['source_power_W'] + stats['gate_drive_power_W'] - stats.get('stored_energy_change_W', 0.0) + total
    stats['efficiency'] = stats['load_power_W'] / supplied if supplied > 0 and stats['load_power_W'] > 0 else None
    for group in stats.get('coupled', []):
        group['added_loss_W'] = sum(added.get(r, {}).get('added_loss_W', 0.0) for r in group['inductors'])
    parts = ', '.join(f"{r} {row['added_loss_W']:.3g} W" for r, row in added.items())
    notes.append(f"inductor losses worked out after the run (AC resistance, iGSE core loss: {parts}) are in the "
                 f"loss and the efficiency as drawn on top of the run's; the run's currents are as without them")
