"""
The ngspice deck of a resolved model: every part and coupling it lists, and nothing else.

    deck = build(model, state=None)     # model: model.resolve()'s; state: engine.end_state()'s, to continue a run
    result = deck.run(folder, 1 / period, cycles=..., ...)

It is power_spice.Circuit (the passives with their parasitics, the switch models, the run) with the parts that
were missing: a diode, a source, a load, a coupling between two inductors, and a switch driven by a drive of any
duty, phase and dead time. Every two-terminal part is sensed the same way: <ref>.V is the voltage from its first
node to its second, <ref>.I the current entering its first node, so V x I is the power it absorbs.
"""
import math

import numpy as np

from ..power_spice import Circuit, _positive

GROUND = ('0', 'GND', 'gnd')
VT = 0.025852                           # kT/q at 27 C, ngspice's nominal temperature


def period_of(model):
    """The analysis period (s): analysis.period_s when given, else the common period of the drives and of the
    periodic sources (frequencies in whole-number ratio; 200 kHz and 300 kHz share 10 us)."""
    a = model.get('analysis', {})
    if a.get('period_s'):
        return float(a['period_s'])
    fs = [d['frequency_Hz'] for d in model.get('drives', {}).values()]
    fs += [p['wave']['frequency_Hz'] for p in model['parts'] if p['type'] == 'source' and 'wave' in p
           and p['wave'].get('frequency_Hz')]
    if not fs:
        raise ValueError('nothing in the model is periodic: give analysis.period_s (the time one period of the '
                         'results covers)')
    for f in fs:
        _positive(f, 'frequency')
    from fractions import Fraction
    from math import gcd
    common = None                                                   # the greatest frequency dividing them all
    for f in fs:
        q = Fraction(f).limit_denominator(1000)
        common = q if common is None else Fraction(gcd(common.numerator * q.denominator, q.numerator * common.denominator),
                                                   common.denominator * q.denominator)
    if max(fs) / float(common) > 1e5:
        raise ValueError(f"the frequencies {sorted(set(fs))} share no practical period: give analysis.period_s")
    return float(1 / common)


def gate_timing(drive, invert, model, period):
    """(delay, width, T) of a switch's gate pulse from its drive: on for duty x T from phase (its complement: the
    rest of the period), each edge held back by half the dead time, so a switch and its complement are never on
    together. T is the drive's own period."""
    T = 1 / drive['frequency_Hz']
    duty = drive.get('duty', 0.5)
    if not 0 < duty < 1:
        raise ValueError(f"a drive's duty must be between 0 and 1 (got {duty:g})")
    dead = model.get('dead_time_s', drive.get('dead_time_s', 0.0))
    tr, tf = model.get('rise_s', drive.get('rise_s', 20e-9)), model.get('fall_s', drive.get('fall_s', 20e-9))
    start = drive.get('phase_deg', 0.0) / 360 * T + (duty * T if invert else 0.0)
    on = (1 - duty) * T if invert else duty * T
    width = on - dead - (tr + tf) / 2
    return (start + dead / 2) % T, width, T


class Deck(Circuit):
    """power_spice.Circuit and the general parts."""

    def __init__(self):
        super().__init__()
        self.nodes.update({g: '0' for g in GROUND})
        self.kinds = {}                                             # ref: its part type
        self.coupled = []                                           # the sets of inductors coupled to each other
        self.step_labels = ()

    def add(self, part, drives, period, state=None):
        ref, kind, (a, b) = part['ref'], part['type'], part['nodes']
        if not part.get('enabled', True):
            return
        self.kinds[ref] = kind
        saved = (state or {}).get(ref, {})
        if kind in ('resistor', 'capacitor'):
            self.passive(ref, a, b, {**part, **saved})
            inner = next((k for k in ('L:' + ref, 'R:' + ref) if k in self.nodes), None)
            if kind == 'capacitor' and inner:                       # the capacitance's own voltage: the state it
                self.voltage(ref + '.cap_V', inner, b)              # holds, without its ESR's and ESL's drops
        elif kind == 'inductor':
            if any(k in part for k in ('SRF_Hz', 'Cp_F', 'core_R_ohm', 'loss_model')):
                coil = dict(part, L_H=part['value'], R_ohm=part.get('DCR_ohm', part.get('R_ohm', 0.0)), **saved)
                self.coil(ref, a, b, coil)
            else:
                self.passive(ref, a, b, {**part, **saved})
        elif kind == 'diode':
            self.diode(ref, a, b, part.get('model', {}), saved.get('initial_V'))
        elif kind == 'switch':
            m = dict(part.get('model', {}))
            m.setdefault('diode_TT_s', 0.0)                         # reverse recovery only where the model asks
            if 'initial_A' in saved:
                m['initial_A'] = saved['initial_A']
            drive = drives[part['drive']]
            delay, width, T = gate_timing(drive, bool(part.get('invert', False)), m, period)
            for key in ('gate_V', 'rise_s', 'fall_s'):               # the drive's, unless the switch has its own
                if key in drive:
                    m.setdefault(key, drive[key])
            self.switch(ref, a, b, m, 1 / T, 0.0, True, timing=(delay, width))
            if 'initial_V' in saved:                                # continuing: its capacitance and its body diode
                v = saved['initial_V']                              # start at the voltage the last run left on it
                for i, line in enumerate(self.lines):
                    if line.startswith(f"C_{ref}_Cds_F "):
                        self.lines[i] = line + f" IC={v:.17g}"
                    elif line.startswith(f"D_{ref} "):
                        self.lines[i] = line + f" IC={-v:.17g}"
        elif kind == 'source':
            self.source(ref, a, b, part, saved)
        elif kind == 'load':
            self.load(ref, a, b, part)

    def diode(self, ref, anode, cathode, model, initial_V=None):
        """Vf_V at If_A sets the saturation current for the emission coefficient n; R_ohm in series, Cj_F the
        junction capacitance at zero bias, TT_s the transit time (reverse recovery)."""
        vf, i_f, n = model.get('Vf_V', 0.7), model.get('If_A', 1.0), model.get('n', 1.0)
        rs = model.get('R_ohm', 0.01)
        for k, v in (('Vf_V', vf), ('If_A', i_f), ('n', n)):
            _positive(v, f"{ref}.{k}")
        for k in ('R_ohm', 'Cj_F', 'TT_s'):
            _positive(model.get(k, 0.0), f"{ref}.{k}", True)
        junction = max(vf - rs * i_f, 0.05)                         # the drop left for the junction itself
        sat = i_f / math.expm1(min(junction / (n * VT), 600.0))
        d = self.sense(ref, anode)
        k = self.node(cathode)
        self.lines.extend([
            f".model DM_{ref} D(IS={sat:.17g} N={n:.17g} RS={rs:.17g} CJO={model.get('Cj_F', 0.0):.17g} "
            f"TT={model.get('TT_s', 0.0):.17g})",
            f"D_{ref} {d} {k} DM_{ref}" + ('' if initial_V is None else f" IC={initial_V:.17g}")])
        self.power_refs.append(ref)
        self.voltage(ref + '.V', anode, cathode)

    def source(self, ref, plus, minus, part, saved=None):
        """A voltage source (dc_V, or a wave) with R_ohm and L_H in series, or a current source (dc_A) pushing
        its current out of its + node."""
        n = self.sense(ref, plus)
        end = self.node(minus)
        self.voltage(ref + '.V', plus, minus)
        self.power_refs.append(ref)
        if 'dc_A' in part:
            self.lines.append(f"I_{ref} {end} {n} DC {part['dc_A']:.17g}")
            return
        r, l = part.get('R_ohm', 0.0), part.get('L_H', 0.0)
        _positive(r, ref + '.R_ohm', True)
        _positive(l, ref + '.L_H', True)
        if r:
            nxt = self.node('R:' + ref)
            self.lines.append(f"R_{ref}_series {n} {nxt} {r:.17g}")
            n = nxt
        if l:
            nxt = self.node('L:' + ref)
            ic = (saved or {}).get('initial_A')
            self.lines.append(f"L_{ref}_series {n} {nxt} {l:.17g}" + (f" IC={ic:.17g}" if ic is not None else ''))
            n = nxt
        self.lines.append(f"V_{ref}_emf {n} {end} " + self.wave(ref, part))

    @staticmethod
    def wave(ref, part):
        if 'dc_V' in part:
            return f"DC {part['dc_V']:.17g}"
        w = part['wave']
        kind = w['type']
        if kind == 'sine':
            _positive(w['frequency_Hz'], ref + '.wave.frequency_Hz')
            return (f"SIN({w.get('offset_V', 0.0):.17g} {w['amplitude_V']:.17g} {w['frequency_Hz']:.17g} 0 0 "
                    f"{w.get('phase_deg', 0.0):.17g})")
        if kind == 'pulse':
            T = 1 / w['frequency_Hz']
            tr, tf = w.get('rise_s', T / 1000), w.get('fall_s', T / 1000)
            duty = w.get('duty', 0.5)
            if not 0 < duty < 1:
                raise ValueError(ref + ': a pulse source needs a duty between 0 and 1')
            width = duty * T - (tr + tf) / 2
            return (f"PULSE({w.get('low_V', 0.0):.17g} {w['high_V']:.17g} {w.get('delay_s', 0.0):.17g} {tr:.17g} "
                    f"{tf:.17g} {width:.17g} {T:.17g})")
        pts = w['points']                                           # pwl: [[t, v], ...], repeated every period_s
        flat = ' '.join(f"{t:.17g} {v:.17g}" for t, v in pts)
        return f"PWL({flat})" + (f" r=0" if w.get('repeat', True) else '')

    def load(self, ref, a, b, part):
        """A resistance; or a constant power or a constant current, drawn by a source of current that is a
        resistor below min_V (a tenth of a volt by default), so the circuit can start from nothing."""
        n = self.sense(ref, a)
        end = self.node(b)
        self.voltage(ref + '.V', a, b)
        self.power_refs.append(ref)
        if 'resistance_ohm' in part:
            _positive(part['resistance_ohm'], ref + '.resistance_ohm')
            self.lines.append(f"R_{ref} {n} {end} {part['resistance_ohm']:.17g}")
            return
        vmin = part.get('min_V', 0.1)
        _positive(vmin, ref + '.min_V')
        v = f"V({n},{end})" if end != '0' else f"V({n})"
        if 'power_W' in part:
            _positive(part['power_W'], ref + '.power_W', True)
            # P / V above min_V, P V / min_V^2 below it: the same value and slope where they meet is not needed,
            # only that the current falls to nothing with the voltage
            self.lines.append(f"B_{ref} {n} {end} I={part['power_W']:.17g}*{v}/max({v}*{v},{vmin * vmin:.17g})")
        else:
            self.lines.append(f"B_{ref} {n} {end} I={part['current_A']:.17g}*tanh({v}/{vmin:.17g})")

    def coupling(self, index, c):
        k = c['k']
        if not 0 < abs(k) < 1:
            raise ValueError(f"couplings[{index}]: k must be between 0 and 1 (got {k:g})")
        a, b = c['inductors']
        self.lines.append(f"K_{index} L_{a} L_{b} {k:.17g}")
        joined = [g for g in self.coupled if a in g or b in g]
        self.coupled = [g for g in self.coupled if g not in joined] + [sorted({a, b}.union(*joined))]


def build(model, state=None):
    """The deck of a resolved model. state: {ref: {initial_V, initial_A, initial_I}} from a previous run's end
    (engine.end_state): every capacitor, inductor and switch starts where that run stopped, and the run uses
    them (uic); None: ngspice works out the DC solution at time zero, where every gate is still low."""
    deck = Deck()
    period = period_of(model)
    drives = model.get('drives', {})
    for part in model['parts']:
        deck.add(part, drives, period, state)
    for i, c in enumerate(model.get('couplings', [])):
        deck.coupling(i, c)
    for name, pr in model.get('probes', {}).items():
        if 'voltage' in pr:
            deck.voltage(name, pr['voltage'][0], pr['voltage'][1])
        else:
            ref = next(p['ref'] for p in model['parts'] if p['ref'].lower() == str(pr['current']).lower())
            deck.signals[name] = deck.signals[ref + '.I']
    deck.uic = state is not None
    deck.period = period
    return deck
