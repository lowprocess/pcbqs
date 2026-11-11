"""
Running a model: the starting state, the periods until it repeats, the targets' knobs, what is measured.

    summary = run(model, out, jobs=1)           # every point of the sweep, each in its own folder under out
    row = run_case(model, point, folder)        # one point

One point goes like this (analysis: the model's settings for it, all optional):
    1. The first run starts from ngspice's DC solution at time zero, where every gate is still low: the rails are
       charged through the diodes and the inductors carry their DC current, whatever the circuit.
    2. Each run is `cycles` periods (200) and starts where the last one ended (end_state: every capacitor's
       voltage, every inductor's and switch's current).
    3. It is settled when every capacitor voltage and inductor current repeats: the last period against every
       period of the half-run before it, within periodic_tolerance (0.003) of that signal's size (a slow drift or
       a ringing output filter shows over half a run when it does not from one period to the next), and what the
       circuit stores (the energy in its capacitors and inductors) changes over a period by no more than
       energy_tolerance (0.002) of the power it handles: that change is the error of an efficiency taken over
       the period. A run that has not settled is followed by one twice as long, up to eight times `cycles`,
       started from the middle of whatever was still ringing.
    4. Settled, each target is compared with its measure; a knob that is off moves by a secant step, kept inside
       its range, and the runs go on. A target whose measure cannot reach its value inside the range ends the
       point with an error saying which end it stopped at.
    5. The reported run repeats a few periods at the strict tolerance (analysis.solver.reltol, 1e-4; the search
       runs at 1e-3) and with steps four times finer (final_samples_per_cycle). If ngspice cannot hold that
       tolerance the search's result is reported and a note says so.

Reported for every point, with no request: each part's power; a switch's and a diode's average, RMS and peak
current and peak voltage; a capacitor's average voltage, ripple and RMS current; an inductor's average, ripple
and peak current; what the sources deliver, what the loads take, the loss, the efficiency and the power balance
(what the sources deliver against what the parts absorb), and the gate drives' power.
"""
import copy
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from ..power_spice import cancel_active_runs, reset_runs
from . import model as M
from .deck import build, period_of

DEFAULTS = dict(cycles=200, final_cycles=8, max_cycles=20000, max_trials=40, periodic_tolerance=0.003,
                energy_tolerance=0.002,
                target_tolerance=0.005, search_reltol=1e-3, strict=True, samples_per_cycle=None,
                final_samples_per_cycle=None, linear_solver='klu', threads=1, timeout_s=600)
PART_STATS = dict(
    resistor=dict(rms_A=('RMS', '@{r}.I@'), rms_V=('RMS', '@{r}.V@')),
    capacitor=dict(avg_V=('AVG', '@{r}.V@'), ripple_V=('PP', '@{r}.V@'), rms_A=('RMS', '@{r}.I@')),
    inductor=dict(avg_A=('AVG', '@{r}.I@'), ripple_A=('PP', '@{r}.I@'), peak_A=('MAX', 'abs(@{r}.I@)'),
                  rms_A=('RMS', '@{r}.I@')),
    diode=dict(avg_A=('AVG', '@{r}.I@'), rms_A=('RMS', '@{r}.I@'), peak_A=('MAX', 'abs(@{r}.I@)'),
               peak_V=('MAX', 'abs(@{r}.V@)')),
    switch=dict(avg_A=('AVG', '@{r}.I@'), rms_A=('RMS', '@{r}.I@'), peak_A=('MAX', 'abs(@{r}.I@)'),
                peak_V=('MAX', 'abs(@{r}.V@)'), gate_power_W=('AVG', '-@{r}.gate_V@*@{r}.gate_I@')),
    source=dict(avg_V=('AVG', '@{r}.V@'), avg_A=('AVG', '-@{r}.I@'), rms_A=('RMS', '@{r}.I@')),
    load=dict(avg_V=('AVG', '@{r}.V@'), ripple_V=('PP', '@{r}.V@'), avg_A=('AVG', '@{r}.I@')),
)


def settings(model):
    a = {**DEFAULTS, **{k: v for k, v in model.get('analysis', {}).items() if k != 'solver'}}
    a['solver'] = dict(model.get('analysis', {}).get('solver', {}))
    return a


def add_measures(deck, model):
    """The deck's measures: each part's power and the statistics of its type, then the model's own."""
    for ref, kind in deck.kinds.items():
        deck.measures[f"part.{ref}.power_W"] = ('AVG', f"@{ref}.V@*@{ref}.I@")
        for name, (stat, expr) in PART_STATS[kind].items():
            deck.measures[f"part.{ref}.{name}"] = (stat, expr.replace('{r}', ref))
    labels = sorted(deck.signals, key=len, reverse=True)
    token = re.compile('|'.join(re.escape(l) for l in labels) or r'$^')
    for name, ms in model.get('measures', {}).items():
        stat, expr = next(iter(ms.items()))
        deck.measures['measure.' + name] = (stat.upper(), token.sub(lambda m: '@' + m[0] + '@', str(expr)))


def collect(measured, deck):
    """ngspice's measurements as the results: the model's measures, each part's statistics, the totals."""
    parts = {ref: dict(type=kind) for ref, kind in deck.kinds.items()}
    measures = {}
    for key, v in measured.items():
        if key.startswith('measure.'):
            measures[key[8:]] = v
        else:
            _, ref, name = key.split('.', 2)
            parts[ref][name] = v
    delivered = {r: -p['power_W'] for r, p in parts.items() if p['type'] == 'source'}
    source = sum(v for v in delivered.values() if v > 0)
    load = sum(p['power_W'] for p in parts.values() if p['type'] == 'load') + sum(-v for v in delivered.values() if v < 0)
    absorbed = sum(p['power_W'] for p in parts.values() if p['type'] != 'source')
    gate = sum(p.get('gate_power_W', 0.0) for p in parts.values())
    loss = source + gate - load
    for group in deck.coupled:                                      # windings pass power to each other: what a
        for ref in group:                                           # transformer loses is its windings' sum
            parts[ref]['coupled_with'] = [r for r in group if r != ref]
    return dict(
        measures=measures, parts=parts,
        coupled=[dict(inductors=g, power_W=sum(parts[r]['power_W'] for r in g)) for g in deck.coupled], source_power_W=source, load_power_W=load, loss_W=loss,
        gate_drive_power_W=gate, efficiency=load / (source + gate) if source + gate > 0 and load > 0 else None,
        # what the sources deliver against what every other part absorbs: the gate resistors' share is not a part's
        power_balance_relative=abs(sum(delivered.values()) - absorbed) / max(abs(source), 1e-12) if source > 0 else None)


def state_signals(deck):
    """The signals that hold the circuit's state: {label: 'V' | 'A'} (capacitor voltages, inductor currents)."""
    out = {}
    for ref, kind in deck.kinds.items():
        if kind == 'capacitor':
            out[(ref + '.cap_V') if ref + '.cap_V' in deck.signals else ref + '.V'] = 'V'
        elif kind == 'inductor':
            out[(ref + '.winding_I') if ref + '.winding_I' in deck.signals else ref + '.I'] = 'A'
    return out


def periodic_error(result, deck, period, ignore=()):
    """How far the state is from repeating: {label: error}, the RMS difference of each state signal between the
    last period and the periods of the half-run before it (the difference that nine in ten of them are within),
    over that signal's size (at least a hundredth
    of the largest signal of its unit, so a quiet node is not judged against its own noise). Every period of the
    half-run, not one or two of them: an output filter ringing at any rate shows, as does a slow drift. The RMS
    over the period and nine periods in ten, not the largest difference anywhere: a switching edge that the
    solver stepped over differently in one period is not a circuit that has not settled."""
    t = np.asarray(result['time_s'])
    n = int(np.floor((t[-1] - t[0]) / period + 1e-6))
    if n < 3:
        raise ValueError('a run needs at least three periods to judge whether it repeats')
    back = np.arange(max(n // 2, 1) + 1)
    times = t[-1] - (back[:, None] + 1) * period + (np.arange(64) / 64 * period)[None, :]
    signals = {k: u for k, u in state_signals(deck).items() if k in result and k not in ignore
               and k.split('.')[0] not in ignore}
    cycles = {k: np.interp(times, t, result[k]) for k in signals}  # [periods back, phase]
    scale = {u: max([np.max(np.abs(cycles[k][0])) for k in signals if signals[k] == u] or [0.0]) for u in ('V', 'A')}
    out = {}
    for k, u in signals.items():
        size = max(np.max(np.abs(cycles[k][0])), 0.01 * scale[u], 1e-9)
        c = cycles[k]
        # against the last period, and against the one before it in case the odd one out is the last
        worst = [np.quantile(np.sqrt(np.mean((np.delete(c, ref, axis=0) - c[ref]) ** 2, axis=1)), 0.9) for ref in (0, 1)]
        out[k] = float(min(worst) / size)
    return out


def energy_drift(result, model, deck, period):
    """The power going into (+) or coming out of what the circuit stores, period by period back over the
    half-run: [W], the last period first. The energy in every capacitor, inductor (with its couplings) and
    source's series inductance at the end of a period, less that a period earlier, over the period. In a
    settled circuit it is nothing; in one that is not, it is by how much the sources' power and the loads' and
    losses' fail to add up over that period, so it is the error of an efficiency taken there."""
    t = np.asarray(result['time_s'])
    n = int(np.floor((t[-1] - t[0]) / period + 1e-6))
    ends = t[-1] - np.arange(max(n // 2, 1) + 1) * period
    at = lambda key: np.interp(ends, t, result[key])
    energy, current = 0.0, {}
    for p in model['parts']:
        ref, kind = p['ref'], p['type']
        if ref not in deck.kinds:
            continue
        if kind == 'capacitor':
            v = at(ref + '.cap_V' if ref + '.cap_V' in result else ref + '.V')
            energy = energy + 0.5 * p['value'] * p.get('bias_factor', 1.0) * v ** 2
        elif kind == 'inductor':
            current[ref] = at(ref + '.winding_I' if ref + '.winding_I' in result else ref + '.I')
            energy = energy + 0.5 * p['value'] * current[ref] ** 2
        elif kind == 'source' and p.get('L_H'):
            energy = energy + 0.5 * p['L_H'] * at(ref + '.I') ** 2
    value = {p['ref']: p['value'] for p in model['parts'] if p['type'] == 'inductor'}
    for c in model.get('couplings', []):
        a, b = c['inductors']
        if a in current and b in current:
            energy = energy + c['k'] * np.sqrt(value[a] * value[b]) * current[a] * current[b]
    return (energy[:-1] - energy[1:]) / period if np.ndim(energy) else np.zeros(1)


def unsteadiness(result, model, deck, stats):
    """energy_drift over the power the circuit handles: the size nine in ten of the half-run's periods are
    within, and the last period's own (the error of what that period's measures add up to)."""
    drift = np.abs(energy_drift(result, model, deck, deck.period))
    scale = max(stats['source_power_W'], stats['load_power_W'], max(abs(p['power_W']) for p in stats['parts'].values()), 1e-12)
    return float(np.quantile(drift, 0.9) / scale), float(drift[0] / scale)


def end_state(result, model, deck, period=None):
    """Where a run stopped, for the next to start from: {ref: {initial_V | initial_A, initial_I}}. A capacitor:
    the voltage on the capacitance itself (<ref>.cap_V, without its ESR's and ESL's drops) and its branch current
    (for its ESL); an inductor and a source's series inductance: the current; a winding: its current and voltage; a
    switch: its drain current (its package inductance) and its voltage (its capacitance); a diode: its voltage."""
    t = np.asarray(result['time_s'])
    n = int(np.floor((t[-1] - t[0]) / period + 1e-6)) if period else 0
    ends = t[-1] - np.arange(max(n // 2, 1) + 1) * period if n >= 8 else None

    def last(key):
        """The signal where the run stopped; or, of a run that has not settled (period given), its mean over the
        ends of the half-run's periods when it swings about that mean (a filter ringing about where it will
        settle: the mean is nearer to it than wherever the ring happens to be at the end)."""
        v = np.asarray(result[key])
        if ends is None:
            return float(v[-1])
        x = np.interp(ends, t, v)
        swing = x - x.mean()
        crossings = int(np.sum(swing[1:] * swing[:-1] < 0))
        return float(x.mean()) if crossings >= 2 else float(v[-1])
    out = {}
    for p in model['parts']:
        ref, kind = p['ref'], p['type']
        if ref not in deck.kinds:
            continue
        if kind == 'capacitor':
            out[ref] = dict(initial_V=last((ref + '.cap_V') if ref + '.cap_V' in result else ref + '.V'),
                            initial_I=last(ref + '.I'))
        elif kind == 'inductor':
            if ref + '.winding_I' in result:
                out[ref] = dict(initial_A=last(ref + '.winding_I'), initial_V=last(ref + '.V'))
            else:
                out[ref] = dict(initial_A=last(ref + '.I'))
        elif kind == 'switch':
            out[ref] = dict(initial_A=last(ref + '.I'), initial_V=last(ref + '.V'))
        elif kind == 'diode':
            out[ref] = dict(initial_V=last(ref + '.V'))
        elif kind == 'source' and p.get('L_H'):
            out[ref] = dict(initial_A=last(ref + '.I'))
    return out


class Knob:
    """A target's knob: where it is, what it gave, where to try next (a secant step between the last two points,
    kept between the best points on either side of the target once there are some, and inside the range)."""

    def __init__(self, target):
        self.path, self.measure, self.value = target['adjust'], target['measure'], target['value']
        self.lo, self.hi = target['range']
        self.tolerance = target.get('tolerance')
        self.x = min(max(target.get('initial', (self.lo + self.hi) / 2), self.lo), self.hi)
        self.points = []                                            # (x, error), as tried
        self.below = self.above = None                              # the nearest points with error < 0 and > 0

    def error(self, measured):
        return (measured - self.value) / (abs(self.value) if self.value else 1.0)

    def record(self, err):
        self.points.append((self.x, err))
        if err < 0 and (self.below is None or abs(err) < abs(self.below[1])):
            self.below = (self.x, err)
        if err > 0 and (self.above is None or abs(err) < abs(self.above[1])):
            self.above = (self.x, err)

    def step(self):
        """The next value to try; raises when the range's end has been tried and the target is still off."""
        (x1, e1) = self.points[-1]
        if len(self.points) == 1:
            span = 0.03 * (self.hi - self.lo)                        # a small first move: its slope is all that is wanted
            new = x1 + span if x1 + span <= self.hi else x1 - span
        else:
            (x0, e0) = self.points[-2]
            new = x1 - e1 * (x1 - x0) / (e1 - e0) if e1 != e0 and x1 != x0 else x1 + 0.1 * (self.hi - self.lo)
            if self.below and self.above:                           # bracketed: stay inside it
                a, b = sorted((self.below[0], self.above[0]))
                if not a < new < b:
                    new = (self.below[0] * abs(self.above[1]) + self.above[0] * abs(self.below[1])) / \
                          (abs(self.above[1]) + abs(self.below[1]))
                    if abs(new - x1) < 1e-3 * (b - a):
                        new = (a + b) / 2
            else:                                                   # not yet: no further than a quarter of the range
                limit = 0.25 * (self.hi - self.lo)
                new = x1 + max(-limit, min(limit, new - x1))
        new = min(max(new, self.lo), self.hi)
        tried_end = [x for x, _ in self.points if x in (self.lo, self.hi)]
        if new in (self.lo, self.hi) and new in tried_end:
            end = 'low' if new == self.lo else 'high'
            raise RuntimeError(f"{self.measure} cannot reach {self.value:g} with {self.path} in "
                               f"[{self.lo:g}, {self.hi:g}]: at its {end} end it is "
                               f"{self.value * (1 + [e for x, e in self.points if x == new][-1]) if self.value else [e for x, e in self.points if x == new][-1]:g}")
        self.x = new
        return new


def plausibility(stats):
    """Notes on results no circuit gives: a part whose peak voltage is far above every source and capacitor
    voltage of the circuit. It is the solver's answer to a current cut off with nowhere to go (a switch or a
    diode with no capacitance in a loop with inductance), not a measurement."""
    parts = stats['parts']
    level = max([abs(p.get('avg_V', 0.0)) for p in parts.values() if p['type'] in ('capacitor', 'source')]
                + [p.get('ripple_V', 0.0) / 2 for p in parts.values() if p['type'] == 'capacitor'] + [1e-3])
    out = []
    for ref, p in parts.items():
        if p.get('peak_V', 0.0) > 20 * level:
            out.append(f"{ref}: a peak of {p['peak_V']:.3g} V in a circuit whose voltages are about {level:.3g} V is a "
                       f"solver spike, not a result: give the switches and diodes of that loop their capacitance "
                       f"(a switch's Cds_F, a diode's Cj_F) or the loop a snubber")
    return out


def measured_value(name, stats):
    if name in stats['measures']:
        return stats['measures'][name]
    if stats.get(name) is None:
        raise RuntimeError(f"{name} has no value in this circuit (no load, or no source)")
    return stats[name]


def simulate(model, state, cycles, folder, a, reltol, log, label='', final=False):
    """One ngspice run of a resolved model, with what it measured. A run that ngspice abandons ('timestep too
    small' at a hard-switched edge) is tried again with trapezoidal integration, then from the DC start.

    The measures are taken over the run's last period, and ngspice takes them over the solver's own time points
    inside the window it is given: a window that opens between two points loses the part of the period before
    the first of them (as much as a whole step of a signal that may be at its largest there). So a source that
    does nothing puts a solver point where the last period begins, and the window is opened a millionth of a
    period early and closed as much late, so that point and the run's last are always inside it."""
    solver = dict(a['solver'], reltol=reltol)
    attempts = [(solver, state), (dict(solver, method='trap' if solver.get('method', 'gear') == 'gear' else 'gear'), state)]
    if state is not None:
        attempts.append((solver, None))
    for k, (opts, start) in enumerate(attempts):
        deck = build(model, start)
        add_measures(deck, model)
        period = deck.period
        spc = a['samples_per_cycle']
        if not spc:                                                 # 256 a switching period, within reason
            fastest = max([d['frequency_Hz'] for d in model.get('drives', {}).values()] or [1 / period])
            spc = int(min(max(256 * round(fastest * period), 256), 16384))
        if final:                                                   # the reported run: four times finer steps
            spc = int(a['final_samples_per_cycle'] or min(4 * spc, 65536))
        n = max(cycles, 3 if start is not None else 20)
        f = 1 / period
        stop = n / f                                                # as Circuit.run works it out
        mark = deck.node('WINDOW:')
        deck.lines += [f"VWINDOW_MARK {mark} 0 PWL(0 0 {stop - 1 / f:.17g} 0 {stop:.17g} 0)", f"RWINDOW_MARK {mark} 0 1k"]
        deck.measure_lead = 1e-6 * period
        try:
            result = deck.run(folder, f, cycles=n, samples_per_cycle=spc,
                              ramp_s=0.0, timeout=a['timeout_s'], linear_solver=a['linear_solver'], threads=int(a['threads']),
                              options=opts)
            break
        except RuntimeError as exc:
            if 'ngspice transient failed' not in str(exc) or k == len(attempts) - 1:
                raise
            if (folder / 'ngspice.log').exists():
                shutil.copy2(folder / 'ngspice.log', folder / f"ngspice-failed-{k}.log")
            log(f"{label}ngspice did not converge; retrying " + ('with the other integration method' if k == 0 else 'from the DC start'))
    (folder / 'transient.dat').unlink(missing_ok=True)              # reproducible from circuit.cir
    return result, deck, collect(result['_measures'], deck), k


def fingerprint(model):
    h = hashlib.sha256(json.dumps({k: v for k, v in model.items() if k not in ('name', 'description', 'sweep', 'schematic')},
                                  sort_keys=True).encode())
    from .. import power_spice
    from . import deck as D
    for module in (power_spice, D, M, __import__(__name__, fromlist=['x'])):
        h.update(Path(module.__file__).read_bytes())
    return h.hexdigest()


def run_case(model, point, out, log=print):
    """One sweep point: {sweep, knobs, targets, measures, parts, totals, settled, history, ...}, also written to
    <out>/case.json beside the final run's circuit.cir, transient.npz (every signal, uniformly sampled),
    measurements.json and waveforms.png. An unchanged model's point is read back, not run again."""
    started = time.perf_counter()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = M.resolve(model, point)
    key = fingerprint(base)
    saved = out / 'case.json'
    if saved.exists() and (out / 'transient.npz').exists():
        row = json.loads(saved.read_text())
        if row.get('fingerprint') == key:
            log(f"{M.case_name(point)}: unchanged, reusing its results")
            return row
    a = settings(base)
    knobs = [Knob(t) for t in base.get('targets', [])]
    label = (M.case_name(point) + ': ') if point else ''
    state, history, notes, total = None, [], [], 0
    cycles = int(a['cycles'])
    accepted = None
    trials = out / 'trials'
    if trials.exists():
        shutil.rmtree(trials)
    for trial in range(int(a['max_trials'])):
        current = copy.deepcopy(model)
        for path, value in (point or {}).items():
            M.put(current, path, value)
        for kn in knobs:
            M.put(current, kn.path, kn.x)
        resolved = M.resolve(current)
        folder = trials / f"{trial:02d}"
        log(label + f"run {trial + 1}: {cycles} periods" + (''.join(f", {kn.path} = {kn.x:.6g}" for kn in knobs))
            + ('' if state is None else ' (continuing)'))
        result, deck, stats, retries = simulate(resolved, state, cycles, folder, a, a['search_reltol'], log, label)
        total += cycles
        state = end_state(result, resolved, deck)
        errors = periodic_error(result, deck, deck.period, a.get('ignore', ()))
        worst = max(errors.values(), default=0.0)
        drifting, _ = unsteadiness(result, resolved, deck, stats)
        entry = dict(trial=trial, cycles=cycles, knobs={kn.path: kn.x for kn in knobs}, periodic_error=worst,
                     energy_drift=drifting,
                     retries=retries, **{k: stats[k] for k in ('source_power_W', 'load_power_W', 'efficiency')},
                     measures=stats['measures'])
        history.append(entry)
        if worst > a['periodic_tolerance'] or drifting > a['energy_tolerance']:
            state = end_state(result, resolved, deck, deck.period)  # ringing: start the next run from its middle
            cycles = min(cycles * 2, 8 * int(a['cycles']))          # slow to settle: longer runs, fewer of them
            if total >= a['max_cycles']:
                slow = max(errors, key=errors.get)
                raise RuntimeError(
                    f"{label}not settled after {total} periods: " +
                    (f"{slow} still differs by {100 * errors[slow]:.2g} % from period to period (raise "
                     f"analysis.max_cycles, or list it under analysis.ignore)" if worst > a['periodic_tolerance'] else
                     f"what the circuit stores still changes by {100 * drifting:.2g} % of its power (raise "
                     f"analysis.max_cycles or analysis.energy_tolerance)"))
            continue
        errs = [kn.error(measured_value(kn.measure, stats)) for kn in knobs]
        entry['target_error'] = dict(zip([kn.measure for kn in knobs], errs))
        if all(abs(e) <= (kn.tolerance or a['target_tolerance']) for kn, e in zip(knobs, errs)):
            accepted = (folder, result, deck, stats, resolved)
            break
        cycles = int(a['cycles'])
        for kn, e in zip(knobs, errs):
            kn.record(e)
            if abs(e) > (kn.tolerance or a['target_tolerance']):
                kn.step()
    if accepted is None:
        raise RuntimeError(f"{label}the targets were not met in {a['max_trials']} runs "
                           f"(see {trials}; raise analysis.max_trials, or narrow a target's range)")
    folder, result, deck, stats, resolved = accepted
    strict = a['solver'].get('reltol', 1e-4)
    reported = a['search_reltol']
    if a['strict'] and strict < a['search_reltol']:
        final = trials / 'final'
        try:
            n = int(a['final_cycles'])
            for _ in range(3):
                log(label + f"reported run: {n} periods at reltol {strict:g}")
                r2, d2, s2, _retries = simulate(resolved, state, n, final, a, strict, log, label, final=True)
                if (max(periodic_error(r2, d2, d2.period, a.get('ignore', ())).values(), default=0.0) <= a['periodic_tolerance']
                        and unsteadiness(r2, resolved, d2, s2)[1] <= a['energy_tolerance']):
                    folder, result, deck, stats, reported = final, r2, d2, s2, strict
                    break
                state = end_state(r2, resolved, d2)
                n *= 2
            else:
                notes.append(f"the strict rerun (reltol {strict:g}) did not settle: reported at reltol {a['search_reltol']:g}")
        except RuntimeError as exc:
            if 'ngspice transient failed' not in str(exc):
                raise
            notes.append(f"ngspice could not hold reltol {strict:g}: reported at reltol {a['search_reltol']:g}")
    for name in ('transient.npz', 'circuit.cir', 'ngspice.log', 'signals.json', 'measurements.json', 'performance.json'):
        if (folder / name).exists():
            shutil.copy2(folder / name, out / name)
    for f in trials.glob('*/transient.npz'):
        f.unlink()
    (out / 'model.json').write_text(json.dumps(resolved, indent=1))
    notes += plausibility(stats)
    row = dict(
        fingerprint=key, sweep=point or {}, name=M.case_name(point), folder=str(out), period_s=deck.period,
        knobs={kn.path: kn.x for kn in knobs},
        targets=[dict(adjust=kn.path, measure=kn.measure, value=kn.value, reached=measured_value(kn.measure, stats))
                 for kn in knobs],
        **stats, reltol=reported, settled=True,
        periodic_error=max(periodic_error(result, deck, deck.period, a.get('ignore', ())).values(), default=0.0),
        energy_drift_relative=unsteadiness(result, resolved, deck, stats)[1],
        runs=len(history), periods=total, notes=notes, history=history, runtime_s=time.perf_counter() - started)
    try:
        from .report import waveforms
        waveforms(out, row, model, deck)
    except Exception as exc:                                        # noqa: BLE001  (a picture is not the result)
        row['notes'].append(f"waveforms.png was not drawn: {type(exc).__name__}: {exc}")
    saved.write_text(json.dumps(row, indent=1))
    return row


def run(model, out, jobs=1, log=print):
    """Every point of the model's sweep, `jobs` at a time; <out>/circuit.json has them all:
    {format, name, complete, cases: [run_case's rows], failed: [{sweep, error}]}. A point that fails is reported
    and the others still run."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    M.validate(model)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'model.json').write_text(json.dumps(model, indent=1))
    points = M.cases(model)
    folders = [out / M.case_name(p) for p in points]
    if len({str(f) for f in folders}) != len(folders):
        raise M.ModelError('two sweep points have the same folder name: make the swept values distinct')
    reset_runs()
    rows, failed = {}, {}

    def write(complete):
        (out / 'circuit.json').write_text(json.dumps(dict(
            format=M.FORMAT, name=model.get('name', ''), description=model.get('description', ''), complete=complete,
            sweep=model.get('sweep', {}), cases=[rows[i] for i in sorted(rows)],
            failed=[dict(sweep=points[i], name=M.case_name(points[i]), error=failed[i]) for i in sorted(failed)]), indent=1))

    try:
        with ThreadPoolExecutor(max_workers=max(1, min(int(jobs), len(points)))) as pool:
            pending = {pool.submit(run_case, model, p, f, log): i for i, (p, f) in enumerate(zip(points, folders))}
            try:
                for future in as_completed(pending):
                    i = pending[future]
                    try:
                        rows[i] = future.result()
                    except (RuntimeError, ValueError) as exc:
                        failed[i] = str(exc)
                        log(f"{M.case_name(points[i])}: failed: {exc}")
                    write(False)
            except BaseException:
                cancel_active_runs()
                for future in pending:
                    future.cancel()
                raise
    finally:
        reset_runs()
    write(True)
    return json.loads((out / 'circuit.json').read_text())
