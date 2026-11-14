"""
What a circuit draws from its supply, for the other tools: a rail's loads for the power-integrity tool
(pcbqs.pdn --load REF=AMPS, --load-spectrum REF=FILE) and a converter's input current for the near-field tool
(pcbqs.mqs --spectrum FILE).

    rows = currents(model, result, period)          # engine.run_case calls it; a case's 'supply'
    write(folder, rows)                             # <case>/supply-<footprint>.csv, their names into the rows

A supply node is one a DC voltage source is on (its + side; the other on ground). Every other part with a pin on
it draws a current from it, but the capacitors: they are the rail's own decoupling, which a model of the rail has
as parts. Where all the source feeds is one inductor or resistor in series to a node its capacitors hold (an
input filter; a boost's inductor feeds its switch, which no capacitor holds, and is a load), the rail is the node
behind it, as far as such parts lead: its loads are drawn from there, and the row says which footprints the way
passes on the board (through) and their DC resistance (dcr), so a model of the rail has them (pcbqs.pdn
--through, --dcr). A part's current is taken at its pin on the node, over the last period: its average (the
rail's DC load) and its harmonics, i(t) = avg + sum A_n cos(2 pi n t / period + phase_n). Parts on one footprint
(a regulator's switches) are one load, their currents added.

    [dict(node='in', net='+5V', rail_net='+5V', source='V1', source_footprint=None, volts=5.0,
          capacitors=['C704', 'C705', 'C706'], through=[], dcr={}, loads=[
        dict(footprint='U702', parts=['QH'], avg_A=1.40, rms_A=1.67, f_Hz=[...], A=[...], deg=[...],
             file='supply-U702.csv')])]

(net: the board's net of the source's node, where a model of the rail starts; rail_net: the board's net of the
node the loads are on (behind a filter: another). capacitors: the node's capacitors to ground, by footprint: a
near-field run's input capacitors. source_footprint: the supply's own footprint, when the model has a board and
does not put the supply off it: the rail's source.)

The CSV: f_Hz, A (peak), deg, a line a harmonic. Harmonics above HARMONICS are left out; the samples of a period
(engine: 1024, or analysis.final_samples_per_cycle) carry them without aliasing.
"""
import csv
from pathlib import Path

import numpy as np

HARMONICS = 40
GROUND = ('0', 'GND', 'gnd')


def _period(result, period, harmonics):
    """(the samples of the last period, how many, the harmonics that fit), or None: less than a period sampled."""
    t = np.asarray(result['time_s'])
    n = int(round(period / (t[1] - t[0])))
    if n < 4 or len(t) < n + 1:
        return None
    return slice(len(t) - 1 - n, len(t) - 1), n, min(int(harmonics), n // 2 - 1)   # its end left to the next period


def _load(at, i, n, count, period):
    """A load's row from its current over the period: its average, its RMS and its harmonics."""
    c = np.fft.rfft(i) / n
    return dict(at, avg_A=float(c[0].real), rms_A=float(np.sqrt(np.mean(i ** 2))),
                f_Hz=[float(k / period) for k in range(1, count + 1)], A=[float(x) for x in 2 * np.abs(c[1:count + 1])],
                deg=[float(x) for x in np.degrees(np.angle(c[1:count + 1]))])


def currents(model, result, period, harmonics=HARMONICS):
    """The supply rows of a resolved model's reported run (result: {signal: samples}, uniformly sampled)."""
    from .copper import net_name, footprint_of
    got = _period(result, period, harmonics)
    if got is None:
        return []
    last, n, count = got
    live = [p for p in model['parts'] if p.get('enabled', True) and p['nodes'][0] != p['nodes'][1]]
    where = lambda part, node: footprint_of(model, part['ref'], part['nodes'].index(node)) or part['ref']
    held = lambda node: any(p['type'] == 'capacitor' and node in p['nodes'] and set(p['nodes']) & set(GROUND)
                            for p in live)
    rows = []
    for src in live:
        if src['type'] != 'source' or 'dc_V' not in src or src['nodes'][1] not in GROUND:
            continue
        node, through, dcr, passed = src['nodes'][0], [], {}, {src['ref']}
        while True:                                                 # behind a lone series part: an input filter
            drawing = [p for p in live if node in p['nodes'] and p['ref'] not in passed
                       and p['type'] not in ('capacitor', 'source')]
            if len(drawing) != 1 or drawing[0]['type'] not in ('inductor', 'resistor'):
                break
            step = drawing[0]
            other = step['nodes'][1] if step['nodes'][0] == node else step['nodes'][0]
            if other in GROUND or not held(other):                  # (a boost's inductor: its switch's node)
                break
            passed.add(step['ref'])
            fps = [footprint_of(model, step['ref'], k) for k in (0, 1)]
            fp = next((f for f in fps if f), None)                  # (on the board: its footprint; off it, nothing)
            if fp and fp not in through:
                through.append(fp)
                ohms = step.get('DCR_ohm') if step['type'] == 'inductor' else step.get('value')
                if ohms:
                    dcr[fp] = float(ohms)
            node = other
        drawn = {}
        for part in live:
            if part['ref'] in passed or part['type'] in ('capacitor', 'source') or node not in part['nodes']:
                continue
            key = part['ref'] + '.I'
            if key not in result:
                continue
            i = np.asarray(result[key])[last] * (1.0 if part['nodes'][0] == node else -1.0)
            fp = where(part, node)
            at = drawn.setdefault(fp, dict(footprint=fp, parts=[], i=0.0))
            at['parts'].append(part['ref'])
            at['i'] = at['i'] + i
        loads = [_load(at, at.pop('i'), n, count, period) for at in drawn.values()]
        if loads:
            caps = [where(p, node) for p in live if p['type'] == 'capacitor' and node in p['nodes']
                    and set(p['nodes']) & set(GROUND)]
            rows.append(dict(node=node, net=net_name(model, src['nodes'][0], ()), rail_net=net_name(model, node, ()),
                             source=src['ref'], source_footprint=footprint_of(model, src['ref'], 0) if model.get('board') else None,
                             volts=float(src['dc_V']), capacitors=caps, through=through, dcr=dcr, loads=loads))
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
