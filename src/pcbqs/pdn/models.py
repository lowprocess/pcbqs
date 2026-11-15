"""
A part's impedance from a file (R-E14): a vendor's capacitor model (a .s2p measured through it, or a .s1p / .z1p),
or a regulator's output impedance, as the PDN's elements take it (cavity.element_z: el['Zf'] = (f, Z)).

    f, Z = read_impedance('grm155.s2p')            # a 2-port: shunt- or series-through, told by its |S21|
    f, Z = read_impedance('vrm.csv')               # f, Re, Im (or f, |Z|, deg with a header saying deg)
    Z_at = table_z(f, Z, f_sweep)                  # interpolated in log f; beyond the table, its ends held

    S, z0 = to_s(Z, z0)                            # (the port matrix's subcircuit: spice_subcircuit)
"""
import math
import os

import numpy as np

UNITS = {'hz': 1.0, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9}
FIT_MAX = 0.05              # a subcircuit's fit missing by more (RMS in S) is not written (9 points over 5 decades: 6e7)


def read_touchstone(path):
    """(f Hz, M (nf, n, n), kind 'S' | 'Z' | 'Y', z0) of a Touchstone v1 file of n ports (n from its extension)."""
    ext = os.path.splitext(path)[1].lower()
    try:
        n = int(ext[2:-1])
    except ValueError:
        raise ValueError(f"{path}: not a Touchstone file (.s1p, .s2p, .z1p ...)") from None
    unit, kind, fmt, z0 = 1e9, 'S', 'MA', 50.0
    nums = []
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.split('!')[0].strip()
            if not line:
                continue
            if line.startswith('#'):
                t = line[1:].lower().split()
                for i, w in enumerate(t):
                    if w in UNITS:
                        unit = UNITS[w]
                    elif w in ('s', 'z', 'y'):
                        kind = w.upper()
                    elif w in ('ri', 'ma', 'db'):
                        fmt = w.upper()
                    elif w == 'r' and i + 1 < len(t):
                        z0 = float(t[i + 1])
                continue
            nums += [float(x) for x in line.split()]
    per = 1 + 2 * n * n
    if len(nums) % per:
        raise ValueError(f"{path}: {len(nums)} numbers are not whole frequency points of {n} ports")
    a = np.asarray(nums).reshape(-1, per)
    f = a[:, 0] * unit
    p, q = a[:, 1::2], a[:, 2::2]
    if fmt == 'RI':
        v = p + 1j * q
    elif fmt == 'MA':
        v = p * np.exp(1j * np.radians(q))
    else:
        v = 10 ** (p / 20) * np.exp(1j * np.radians(q))
    M = v.reshape(-1, n, n)
    if n == 2:                                                      # v1 two-ports: 11 21 12 22
        M = M.transpose(0, 2, 1)
    return f, M, kind, z0


def two_port_z(S, z0, mode='auto'):
    """A part's impedance from a 2-port measured through it: 'shunt' (the part from the line to ground: Z = z0 S21
    / (2 (1 - S21))), 'series' (in the line: Z = 2 z0 (1 - S21) / S21), 'auto' (shunt where |S21| is mostly under
    0.5, as a decap's is shunt-through and near 1 in series). Returns (Z, the mode taken)."""
    s21 = S[:, 1, 0]
    if mode == 'auto':
        mode = 'shunt' if np.median(np.abs(s21)) < 0.5 else 'series'
    if mode == 'shunt':
        return z0 * s21 / (2 * (1 - s21)), mode
    if mode == 'series':
        return 2 * z0 * (1 - s21) / s21, mode
    raise ValueError(f"a 2-port's mode: shunt, series or auto, not {mode!r}")


def read_impedance(path, mode='auto'):
    """(f Hz, Z) of a part from a file: a Touchstone 1-port (S, Z or Y), a 2-port measured through it (two_port_z),
    or a CSV of f, Re, Im (or f, |Z|, deg when its header says deg or phase). Also the 2-port's mode ('' else)."""
    ext = os.path.splitext(path)[1].lower()
    if ext.endswith('p') and ext[1] in 'szy' and ext[2:-1].isdigit():
        f, M, kind, z0 = read_touchstone(path)
        n = M.shape[1]
        if n == 1:
            m = M[:, 0, 0]
            Z = z0 * (1 + m) / (1 - m) if kind == 'S' else (m if kind == 'Z' else 1 / m)
            return f, Z, ''
        if n == 2:
            S = M if kind == 'S' else _to_s(M, kind, z0)
            Z, taken = two_port_z(S, z0, mode)
            return f, Z, taken
        raise ValueError(f"{path}: {n} ports - a part's model is a 1- or 2-port")
    rows, polar = [], False
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in '#!':
                continue
            parts = [x for x in line.replace(';', ',').replace('\t', ',').split(',') if x.strip()] \
                if (',' in line or ';' in line or '\t' in line) else line.split()
            try:
                rows.append([float(x) for x in parts[:3]])
            except ValueError:
                polar = polar or any(k in line.lower() for k in ('deg', 'phase', 'mag'))
    a = np.asarray(rows, float)
    if a.ndim != 2 or a.shape[1] < 3 or len(a) < 2:
        raise ValueError(f"{path}: expected lines of f, Re, Im (or f, |Z|, deg)")
    Z = a[:, 1] * np.exp(1j * np.radians(a[:, 2])) if polar else a[:, 1] + 1j * a[:, 2]
    return a[:, 0], Z, ''


def _to_s(M, kind, z0):
    I = np.eye(M.shape[1])
    if kind == 'Z':
        return np.linalg.solve((M + z0 * I).transpose(0, 2, 1), (M - z0 * I).transpose(0, 2, 1)).transpose(0, 2, 1)
    Y = M
    return np.linalg.solve((I + z0 * Y).transpose(0, 2, 1), (I - z0 * Y).transpose(0, 2, 1)).transpose(0, 2, 1)


def table_z(ft, Zt, f):
    """Zt (at ft Hz) at f: real and imaginary parts interpolated in log f, the table's ends held beyond it."""
    ft = np.asarray(ft, float)
    order = np.argsort(ft)
    ft, Zt = ft[order], np.asarray(Zt)[order]
    lf, x = np.log(ft), np.log(np.clip(np.atleast_1d(np.asarray(f, float)), 1e-300, None))
    out = np.interp(x, lf, Zt.real) + 1j * np.interp(x, lf, Zt.imag)
    return out if np.ndim(f) else complex(out[0])


def coverage_note(name, ft, f):
    """A note where the sweep reaches past a file's band (its ends are held there), else None."""
    lo, hi = float(np.min(ft)), float(np.max(ft))
    if f[0] >= lo * (1 - 1e-9) and f[-1] <= hi * (1 + 1e-9):
        return None
    return (f"{name}: its file covers {lo:.3g}-{hi:.3g} Hz; the sweep's {f[0]:.3g}-{f[-1]:.3g} Hz beyond it takes "
            f"the file's end values")


def spice_subcircuit(path, f, Z, names, z0=1.0, name='pdn_ports', log=print):
    """A SPICE subcircuit of the impedance matrices Z (nf, n, n): vector fitted (scikit-rf), made passive where it
    was not, written to path (its pins: the ports in order, each against the reference). Returns dict(poles,
    rms error, passive). z0: the fit's reference (an ohm: PDN impedances are milliohms to ohms)."""
    try:
        import skrf
        from skrf.vectorFitting import VectorFitting
    except ImportError:
        raise RuntimeError("a SPICE subcircuit needs scikit-rf (python -m pip install scikit-rf)") from None
    import warnings
    fr = skrf.Frequency.from_f(np.asarray(f, float), unit='hz')
    nw = skrf.Network(frequency=fr, s=skrf.network.z2s(np.asarray(Z, complex), z0), z0=z0, name=name)
    vf = VectorFitting(nw)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        vf.auto_fit()
        passive = bool(vf.is_passive())
        if not passive:
            vf.passivity_enforce()
        vf.write_spice_subcircuit_s(path, fitted_model_name=name)
    err = float(vf.get_rms_error())
    if not np.isfinite(err) or err > FIT_MAX:
        os.remove(path)
        raise ValueError(f"the vector fit missed by {err:.2g} (RMS in S): more frequencies (--points) or a narrower "
                         f"band")
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(f"* pcbqs.pdn: the bare PDN between its parts' ports, vector fitted ({len(vf.poles)} poles, RMS error "
                 f"{err:.2g} in S at {z0:g} ohm{'' if passive else ', made passive'}); pins in order: "
                 f"{', '.join(names)}, each against the reference\n" + text)
    log(f"PDN: SPICE subcircuit of {len(names)} ports, {len(vf.poles)} poles, RMS error {err:.2g}"
        + ('' if passive else ' (made passive)'))
    return dict(poles=int(len(vf.poles)), rms=err, passive_as_fitted=passive)
