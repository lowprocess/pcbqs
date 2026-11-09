"""Adaptive ngspice waveform export and non-aliased periodic Fourier inference."""

from pathlib import Path
import re
import numpy as np


def read_raw(path):
    """Read ngspice's real binary rawfile (native float64), selected vectors only."""
    data = Path(path).read_bytes()
    split = data.find(b'Binary:\n')
    if split < 0:
        raise ValueError('expected a binary ngspice rawfile')
    header = data[:split].decode('utf-8', errors='replace')
    if not re.search(r'^Flags:\s+real', header, re.M):
        raise ValueError('only real transient rawfiles are supported')
    nv = int(re.search(r'No\. Variables:\s+(\d+)', header)[1])
    nt = int(re.search(r'No\. Points:\s+(\d+)', header)[1])
    variables = re.findall(r'^\s*\d+\s+(\S+)\s+\S+', header.rsplit('Variables:', 1)[1], re.M)
    if len(variables) != nv:
        raise ValueError('invalid ngspice variable header')
    values = np.frombuffer(data[split + len(b'Binary:\n') :], dtype=np.float64)
    if values.size != nv * nt:
        raise ValueError('invalid ngspice binary dimensions')
    table = values.reshape(nt, nv)
    return dict(zip(variables, table.T))


def periodic_coefficients(time, currents, frequency, harmonics=31):
    """Integrate piecewise-linear adaptive currents with 4-point Gauss quadrature.

    No uniform resampling precedes the Fourier integral. The coefficients use
    absolute switching phase; mean-square current is integrated analytically.
    """
    time = np.asarray(time)
    values = np.asarray(currents)
    if time.ndim != 1 or values.shape[0] != len(time) or not np.all(np.diff(time) > 0):
        raise ValueError('adaptive time must be increasing')
    end = np.floor(time[-1] * frequency + 1e-8) / frequency
    start = end - 1 / frequency
    if start < time[0]:
        raise ValueError('need one complete switching period')
    inner = (time > start) & (time < end)
    t = np.r_[start, time[inner], end]
    if values.ndim == 1:
        values = values[:, None]
    y = np.vstack(
        [
            np.array([np.interp(start, time, col) for col in values.T]),
            values[inner],
            np.array([np.interp(end, time, col) for col in values.T]),
        ]
    )
    dt = np.diff(t)
    u, w = np.polynomial.legendre.leggauss(4)
    u = (u + 1) / 2
    w = w / 2
    samples = y[:-1, None, :] + (y[1:] - y[:-1])[:, None, :] * u[None, :, None]
    at = t[:-1, None] + dt[:, None] * u[None, :]
    coefficients = np.array(
        [
            frequency * np.einsum('sgp,sg,s,g->p', samples, np.exp(-2j * np.pi * n * frequency * at), dt, w)
            for n in range(harmonics + 1)
        ]
    )
    square = frequency * np.sum(dt[:, None] * (y[:-1] ** 2 + y[:-1] * y[1:] + y[1:] ** 2) / 3, axis=0)
    return coefficients, dict(
        mean_square_A2=square,
        AC_mean_square_A2=np.maximum(0, square - coefficients[0].real ** 2),
        adaptive_points_last_period=len(t),
    )


def save_port_spectrum(folder, signal_labels, frequency, harmonics=31):
    folder = Path(folder)
    ports = [(i, k) for i, k in enumerate(signal_labels) if re.fullmatch(r'PCB\d+\.I', k)]
    if not ports:
        return None
    raw = read_raw(folder / 'adaptive-ports.raw')
    time = raw['time']
    currents = np.stack(
        [raw[next(name for name in ['out' + str(i), 'i(out' + str(i) + ')'] if name in raw)] for i, k in ports], axis=1
    )
    coefficients, info = periodic_coefficients(time, currents, frequency, harmonics)
    np.savez_compressed(
        folder / 'port-spectrum.npz',
        coefficients_A=coefficients,
        port_labels=np.array([k for _, k in ports]),
        frequencies_Hz=np.arange(harmonics + 1) * frequency,
        **info,
    )
    return info
