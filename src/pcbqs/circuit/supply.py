"""
What a circuit draws from its supply, for the other tools: a rail's loads for the power-integrity tool
(pcbqs.pdn --load REF=AMPS, --load-spectrum REF=FILE) and a converter's input current for the near-field tool
(pcbqs.mqs --spectrum FILE).

    rows = currents(model, result, period)          # engine.run_case calls it; a case's 'supply'
    write(folder, rows)                             # <case>/supply-<footprint>.csv, their names into the rows

A supply node is one a DC voltage source is on (its + side; the other on ground). Every other part with a pin on
it draws a current from it, but the capacitors: they are the rail's own decoupling, which a model of the rail has
as parts. A part's current is taken at its pin on the node, over the last period: its average (the rail's DC
load) and its harmonics, i(t) = avg + sum A_n cos(2 pi n t / period + phase_n). Parts on one footprint (a
regulator's switches) are one load, their currents added.

    [dict(node='in', net='+5V', source='V1', volts=5.0, capacitors=['C704', 'C705', 'C706'], loads=[
        dict(footprint='U702', parts=['QH'], avg_A=1.40, rms_A=1.67, f_Hz=[...], A=[...], deg=[...],
             file='supply-U702.csv')])]

(capacitors: the node's capacitors to ground, by footprint: a near-field run's input capacitors.)

The CSV: f_Hz, A (peak), deg, a line a harmonic. Harmonics above HARMONICS are left out; the samples of a period
(engine: 1024, or analysis.final_samples_per_cycle) carry them without aliasing.
"""
import csv
from pathlib import Path

import numpy as np

HARMONICS = 40
GROUND = ('0', 'GND', 'gnd')


def currents(model, result, period, harmonics=HARMONICS):
    """The supply rows of a resolved model's reported run (result: {signal: samples}, uniformly sampled)."""
    from .copper import net_name
    t = np.asarray(result['time_s'])
    n = int(round(period / (t[1] - t[0])))
    if n < 4 or len(t) < n + 1:
        return []
    last = slice(len(t) - 1 - n, len(t) - 1)                        # one period, its end left to the next
    count = min(int(harmonics), n // 2 - 1)
    names = (model.get('board') or {}).get('parts', {})
    rows = []
    for src in model['parts']:
        if src['type'] != 'source' or 'dc_V' not in src or not src.get('enabled', True) or src['nodes'][1] not in GROUND:
            continue
        node = src['nodes'][0]
        drawn = {}
        for part in model['parts']:
            if part is src or part['type'] in ('capacitor', 'source') or not part.get('enabled', True):
                continue
            key = part['ref'] + '.I'
            if node not in part['nodes'] or key not in result or part['nodes'][0] == part['nodes'][1]:
                continue
            i = np.asarray(result[key])[last] * (1.0 if part['nodes'][0] == node else -1.0)
            fp = names.get(part['ref'], part['ref'])
            at = drawn.setdefault(fp, dict(footprint=fp, parts=[], i=0.0))
            at['parts'].append(part['ref'])
            at['i'] = at['i'] + i
        loads = []
        for at in drawn.values():
            i = at.pop('i')
            c = np.fft.rfft(i) / n
            a = 2 * np.abs(c[1:count + 1])
            loads.append(dict(at, avg_A=float(c[0].real), rms_A=float(np.sqrt(np.mean(i ** 2))),
                              f_Hz=[float(k / period) for k in range(1, count + 1)], A=[float(x) for x in a],
                              deg=[float(x) for x in np.degrees(np.angle(c[1:count + 1]))]))
        if loads:
            caps = [names.get(p['ref'], p['ref']) for p in model['parts'] if p['type'] == 'capacitor'
                    and p.get('enabled', True) and node in p['nodes'] and set(p['nodes']) & set(GROUND)]
            rows.append(dict(node=node, net=net_name(model, node, ()), source=src['ref'], volts=float(src['dc_V']),
                             capacitors=caps, loads=loads))
    return rows


def write(folder, rows):
    """Each load's harmonics as <folder>/supply-<footprint>.csv; the rows with the files' names (their f_Hz, A
    and deg stay in the CSV alone)."""
    folder, out = Path(folder), []
    for row in rows:
        loads = []
        for load in row['loads']:
            name = 'supply-' + ''.join(ch if ch.isalnum() or ch in '_-' else '_' for ch in load['footprint']) + '.csv'
            with open(folder / name, 'w', newline='', encoding='utf-8') as fh:
                w = csv.writer(fh)
                w.writerow(['f_Hz', 'A', 'deg'])
                w.writerows([f"{f:.9g}", f"{a:.6g}", f"{d:.3f}"] for f, a, d in zip(load['f_Hz'], load['A'], load['deg']))
            loads.append({**{k: v for k, v in load.items() if k not in ('f_Hz', 'A', 'deg')}, 'file': name,
                          'harmonics': len(load['f_Hz']), 'largest_A': max(load['A'], default=0.0)})
        out.append(dict(row, loads=loads))
    return out


def read_spectrum(path):
    """A current spectrum file: (f Hz, complex peak amplitudes), i(t) = Re sum A_n exp(2 pi i f_n t). Columns f_Hz,
    A (peak) and, optionally, deg; a header line and lines that start with # are skipped. ValueError says what is
    wrong with it."""
    f, a = [], []
    with open(path, newline='', encoding='utf-8') as fh:
        for k, line in enumerate(csv.reader(fh)):
            cells = [c.strip() for c in line if c.strip()]
            if not cells or cells[0].startswith('#'):
                continue
            try:
                v = [float(c) for c in cells[:3]]
            except ValueError:
                if not f and k < 3:                                 # its header
                    continue
                raise ValueError(f"{path}: line {k + 1} is not numbers: {', '.join(cells)}") from None
            if len(v) < 2:
                raise ValueError(f"{path}: line {k + 1}: expected f_Hz, A[, deg]")
            f.append(v[0])
            a.append(v[1] * np.exp(1j * np.radians(v[2] if len(v) > 2 else 0.0)))
    f, a = np.asarray(f, float), np.asarray(a, complex)
    if not len(f) or not np.isfinite(f).all() or (f <= 0).any() or not np.isfinite(a).all():
        raise ValueError(f"{path}: no spectrum (lines of f_Hz, A[, deg], the frequencies positive)")
    order = np.argsort(f)
    return f[order], a[order]


def waveform(f, a, samples=512):
    """(whether the frequencies are harmonics of the lowest, peak-to-peak) of v(t) = Re sum a_n exp(2 pi i f_n t):
    over the lowest's period when they are its harmonics; otherwise the bound 2 sum |a_n|, which no alignment of
    the phases exceeds."""
    f, a = np.asarray(f, float), np.asarray(a, complex)
    if not len(f):
        return True, 0.0
    k = f / f[0]
    if np.allclose(k, np.round(k), atol=1e-6) and np.round(k).max() < samples / 2:
        t = np.arange(samples) / samples
        v = (a[None, :] * np.exp(2j * np.pi * np.round(k)[None, :] * t[:, None])).sum(axis=1).real
        return True, float(v.max() - v.min())
    return False, float(2 * np.abs(a).sum())
