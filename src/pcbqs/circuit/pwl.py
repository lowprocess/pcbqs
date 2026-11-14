"""
The piecewise-linear engine: a switched circuit's periodic steady state without a transient to wait for.

    result, deck, stats, _ = simulate(resolved_model, None, cycles, folder, settings, None, log)   # engine.simulate's
    plant = Plant(resolved_model); x, samples = plant.steady_state()                              # the solver alone

Between two switching instants a converter of ideal switches is a linear circuit: its capacitor voltages and
inductor currents x obey x' = A x + c, with A and c those of the switches' and diodes' states. That has an exact
solution, x(t + h) = exp(A h) x + ..., so there is no time step to choose and no tolerance to hold: a period is a
handful of matrix exponentials, and the steady state is the x that a period maps onto itself, found by Newton's
method on that map (its Jacobian is the product of the period's exponentials) in a few periods where ngspice
runs hundreds to let a filter settle.

What it models (the rest is ngspice's: analysis.engine = "ngspice", the default):
    switch      Ron_ohm when its gate is on, open otherwise, with its body diode (types switch and ideal, as
                ngspice's deck has them; vdmos is taken as switch: its Ron_ohm, its Cds_F as the capacitance, its
                package's inductances, package_L_H and source_L_H, in series as ngspice's); its gate's edges where the drive puts them (deck.gate_timing), the
                dead time included; its drain-source capacitance Cds_F (type switch, as ngspice's), a state of
                its own, so turning on discharges it through Ron and the loop's inductance rings against it;
                no gate charge (Cgd_F and Cgs_F are left out, and the results' notes say so)
    diode       a threshold and a resistance: on above Vth with R_on (the model's Vf_V at If_A, its R_ohm), off
                below; no junction capacitance (Cj_F: ngspice's falls with the reverse voltage, and a constant
                one put a bridge's floating AC side on the junctions' charge, whose steady state the search could
                not pin down), no recovery; the results' notes say what was left out
    capacitor   value x bias_factor with its ESR, ESL and leakage
    inductor    value (with its ESL) and its DCR, the model's couplings; a winding (SRF_Hz, Cp_F, core_R_ohm or a
                loss_model: power_spice.Circuit.coil's) as its series resistance, its inductance, the core's
                resistance across it and its self-capacitance across the whole
    source      dc_V with its series R and L, or dc_A; a sine (two states that turn at its frequency) or a
                pulse (two levels, its edges at their instants); no pwl wave
    load        resistance_ohm, current_A, or power_W (a resistance that takes that power at its mean voltage)
    resistor
The board's copper (engine.run(copper=True): circuit/copper.py's R-L model of it) is in the circuit as the ngspice
deck has it: every part's pin on its own pad, and each pad joined to its net's node by a port of the copper's
model, Z(s) = R + s L + sum of s/(s + p) R_m. Its port currents are states (the matrix L their coupled
inductance, R in series), and each Foster section's every mode a state of its own (its share of the current
through the section's inductance).

A diode's state follows the circuit: it turns on when its voltage reaches the threshold and off when its current
reaches zero, at the instant that happens (found by bisection inside the step), so discontinuous conduction and
a rectifier's commutation are the circuit's own. An open switch or diode is G_OFF, so a current with nowhere to
go finds the diode that takes it at once.

The result is the signals ngspice's run gives (every part's V and I, the nodes, the probes), over a few
identical periods, and the same measures, so everything after a run (the targets' search, the parts' figures,
the supply's currents, the report) is engine.py's as it is.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from . import model as M
from .deck import GROUND, build, gate_timing, period_of

G_OFF = 1e-8            # S: an open switch, a diode that does not conduct
G_MIN = 1e-12           # S: every node to ground (a node held only by inductors is an island: Plant.matrices)
R_MIN = 1e-6            # ohm: the least series resistance of a capacitor or a voltage source
VT = 0.025852           # V: the thermal voltage, as ngspice's diodes at 27 C
SAMPLES = 8192          # a period's samples in the result (analysis.samples_per_cycle): a signal that jumps at a
                        # switching instant is averaged to a sample's width, so a product's mean (a part's power) to
                        # about its jump over this
SEARCH = 512            # and while the steady state is looked for: the instants are exact whatever this is
PERIODS = 4             # the periods the result holds (identical: engine.py judges a run over several)
HOLD = 2000             # an island's current is held to its law at this many times over a period (Plant.matrices):
                        # fast, as while it is off the law a current goes nowhere (held over a switch's 20 ns edge
                        # instead, the buck's loss rose by a seventh)
PER_RING = 16           # samples in a cycle of the circuit's fastest ringing (Plant.resolution), at least
GROW = 1.3              # after a switching instant: samples from a quarter of the fastest time constant, each this
                        # much further on (Plant.fine_period)
MOST = 1 << 18          # and never more than this a period


class Unsupported(M.ModelError):
    """A model the piecewise-linear engine does not take: says what, and that ngspice does."""


def diode_line(vf, i_f, n=1.0, rs=0.01):
    """(threshold V, on resistance ohm) of a diode that drops vf at i_f: the junction's own drop less one thermal
    voltage as the threshold, and that thermal voltage over i_f with the series resistance, so the line meets
    the diode's curve at i_f with its slope there."""
    junction = max(vf - rs * i_f, 0.05)
    return junction - n * VT, rs + n * VT / i_f


class Plant:
    """A resolved model as a switched linear system. States: every capacitor's own voltage (a switch's Cds and a
    diode's Cj among them), then every inductor's current (a source's series inductance and the copper's ports
    among them), then the copper's Foster modes, then a sine and a cosine for every sine source (they turn at its
    frequency: the source is its amplitude times the sine, and the period's map leaves them where they were, so
    the search for the steady state does not move them). copper: circuit/copper.prepare's, or None."""

    def __init__(self, model, copper=None):
        """copper: circuit/copper.prepare's (each part's pins on its own pads, the ports between them and their
        nets' nodes), or {model: the R-L fit, ends: {port name: (node, node)}, pins: {(ref, k): node}}."""
        from . import copper as C
        self.model, self.period, self._noted = model, period_of(model), {}
        on_board = bool(copper and (copper.get('ports') or copper.get('ends')))
        if on_board and 'ends' not in copper:
            copper = dict(model=copper['model'], ends=C.port_nodes(model, copper), pins=C.nodes(model, copper))
        pins = (copper or {}).get('pins') or {}
        self.index = {}                                     # node name -> row (ground: -1)
        self.g, self.vs, self.cs = [], [], []               # (a, b, S); (ref, a, b, S, V); (ref, a, b, A a -> b)
        self.caps, self.inds, self.sw, self.di, self.pl = [], [], [], [], []
        self.sines, self.pulses = [], []                    # sources with a wave (their entry of vs, and the wave)
        self.parts = []                                     # (ref, kind, a, b, how its current is found)
        drives = model.get('drives', {})
        for p in model['parts']:
            if not p.get('enabled', True):
                continue
            ref, kind = p['ref'], p['type']
            a, b = (self.node(pins.get((ref, k), n)) for k, n in enumerate(p['nodes']))
            if kind == 'resistor':
                self.g.append((a, b, 1 / p['value']))
                self.parts.append((ref, kind, a, b, ('g', 1 / p['value'])))
            elif kind == 'capacitor':
                g = 1 / max(p.get('ESR_ohm', 0.0), R_MIN)
                inner, esl = a, None
                if p.get('ESL_H'):                          # its ESL: an inductance in series, behind a node
                    inner = self.node('L:' + ref)
                    self.inds.append(dict(ref=ref + ':ESL', a=a, b=inner, L=p['ESL_H'], r=0.0))
                    esl = len(self.inds) - 1
                self.caps.append(dict(ref=ref, a=inner, b=b, g=g, C=p['value'] * p.get('bias_factor', 1.0)))
                if p.get('leakage_ohm'):
                    self.g.append((a, b, 1 / p['leakage_ohm']))
                self.parts.append((ref, kind, a, b, ('cap', len(self.caps) - 1, 1 / p['leakage_ohm'] if p.get('leakage_ohm') else 0.0, esl)))
            elif kind == 'inductor':
                if any(p.get(k) for k in ('SRF_Hz', 'Cp_F', 'core_R_ohm', 'loss_model')):
                    self._winding(ref, a, b, p)
                    continue
                self.inds.append(dict(ref=ref, a=a, b=b, L=p['value'] + p.get('ESL_H', 0.0), r=p.get('DCR_ohm', 0.0)))
                self.parts.append((ref, kind, a, b, ('ind', len(self.inds) - 1)))
            elif kind == 'diode':
                m = p.get('model', {})
                vth, ron = diode_line(m.get('Vf_V', 0.7), m.get('If_A', 1.0), m.get('n', 1.0), m.get('R_ohm', 0.01))
                self.di.append(dict(ref=ref, a=a, b=b, vth=vth, g=1 / ron))
                left = [k for k in ('Cj_F', 'TT_s') if m.get(k)]
                if left:
                    self._note(ref, f"its {' and '.join(dict(Cj_F='junction capacitance', TT_s='reverse recovery')[k] for k in left)} "
                                    f"left out (the piecewise-linear engine)")
                self.parts.append((ref, kind, a, b, ('diode', len(self.di) - 1, None)))
            elif kind == 'switch':
                m = p.get('model', {})
                kind_m = m.get('type', 'switch')
                if kind_m == 'vdmos':
                    kind_m = 'switch'
                    self._note(ref, "its vdmos model is taken as a switch: Ron_ohm, Cds_F and the body diode "
                                    "(the piecewise-linear engine)")
                if kind_m not in ('switch', 'ideal'):
                    raise Unsupported(f"{ref}: a {kind_m} switch model is ngspice's (analysis.engine = \"ngspice\")")
                pins_ = (a, b)                              # its package's inductances: the channel behind them
                drain = None
                if m.get('package_L_H'):
                    da = self.node('d:' + ref)
                    self.inds.append(dict(ref=ref + ':Ld', a=a, b=da, L=m['package_L_H'], r=0.0))
                    drain, a = len(self.inds) - 1, da
                if m.get('source_L_H'):
                    sb = self.node('s:' + ref)
                    self.inds.append(dict(ref=ref + ':Ls', a=sb, b=b, L=m['source_L_H'], r=0.0))
                    b = sb
                if p.get('drive') not in drives:
                    raise M.ModelError(f"{ref}: its drive {p.get('drive')!r} is not one of the model's")
                d = drives[p['drive']]
                delay, width, T = gate_timing(d, bool(p.get('invert')), m, self.period)
                tr, tf = m.get('rise_s', d.get('rise_s', 20e-9)), m.get('fall_s', d.get('fall_s', 20e-9))
                body = None
                if m.get('body_diode', True):               # source to drain, as the deck's
                    vth, ron = diode_line(VT * math.log(1e12), 1.0, 1.0, m.get('diode_R_ohm', 0.02))
                    self.di.append(dict(ref=ref + ':body', a=b, b=a, vth=vth, g=1 / ron))
                    body = len(self.di) - 1
                self.sw.append(dict(ref=ref, a=a, b=b, g=1 / m.get('Ron_ohm', 0.008 if kind_m != 'ideal' else 1e-6),
                                    on=(delay + tr / 2) % T, span=width + (tr + tf) / 2, T=T, body=body,
                                    gate=m.get('gate_V', 10.0)))
                cap = self._junction(ref, a, b, m.get('Cds_F', 0.0)) if kind_m == 'switch' else None
                if kind_m == 'switch' and (m.get('Cgd_F') or m.get('Cgs_F')):
                    self._note(ref, "its gate capacitances (Cgd_F, Cgs_F) are left out: the gate is ideal "
                                    "(the piecewise-linear engine)")
                self.parts.append((ref, kind, *pins_, ('switch', len(self.sw) - 1, cap, a, b, drain)))
            elif kind == 'source':
                wave = p.get('wave')
                if wave and wave['type'] not in ('sine', 'pulse'):
                    raise Unsupported(f"{ref}: a source with a {wave['type']} wave is ngspice's "
                                      f"(analysis.engine = \"ngspice\")")
                if 'dc_A' in p:
                    self.cs.append((ref, a, b, -p['dc_A']))            # out of its + node
                    self.parts.append((ref, kind, a, b, ('cs', -p['dc_A'])))
                    continue
                plus = a
                if p.get('L_H'):                             # its series inductance: a state, behind a node
                    plus = self.node('L:' + ref)
                    self.inds.append(dict(ref=ref + ':L', a=a, b=plus, L=p['L_H'], r=0.0))
                g = 1 / max(p.get('R_ohm', 0.0), R_MIN)
                if not wave:
                    self.vs.append((ref, plus, b, g, p['dc_V']))
                elif wave['type'] == 'sine':                # offset + amplitude x (the first of its two states)
                    self.vs.append((ref, plus, b, g, wave.get('offset_V', 0.0)))
                    self.sines.append(dict(vs=len(self.vs) - 1, A=wave['amplitude_V'], w=2 * math.pi * wave['frequency_Hz'],
                                           phase=math.radians(wave.get('phase_deg', 0.0))))
                else:                                       # low or high: a level that changes at its edges
                    T = 1 / wave['frequency_Hz']
                    duty = wave.get('duty', 0.5)
                    self.vs.append((ref, plus, b, g, wave.get('low_V', 0.0)))
                    self.pulses.append(dict(vs=len(self.vs) - 1, high=wave['high_V'], T=T, span=duty * T,
                                            on=(wave.get('delay_s', 0.0) + wave.get('rise_s', T / 1000) / 2) % T))
                self.parts.append((ref, kind, a, b, ('vs', len(self.vs) - 1)))
            elif kind == 'load':
                if 'resistance_ohm' in p:
                    self.g.append((a, b, 1 / p['resistance_ohm']))
                    self.parts.append((ref, kind, a, b, ('g', 1 / p['resistance_ohm'])))
                elif 'current_A' in p:
                    self.cs.append((ref, a, b, p['current_A']))
                    self.parts.append((ref, kind, a, b, ('cs', p['current_A'])))
                else:                                       # constant power: a resistance, set by its mean voltage
                    self.pl.append(dict(ref=ref, a=a, b=b, P=p['power_W'], g=p['power_W'] / max(p.get('min_V', 0.1), 1e-3) ** 2,
                                        vmin=p.get('min_V', 0.1)))
                    self.parts.append((ref, kind, a, b, ('pload', len(self.pl) - 1)))
        self.cu, self.modes = len(self.inds), []            # the copper's first port among the inductors; its
        cu_L = cu_R = None                                  # Foster modes: (gain over its ports, pole rad/s)
        if on_board:
            ends = copper['ends']
            cm = copper['model']
            for i, name in enumerate(cm['names']):          # PCB<i>: the pad's end first, as the deck's
                a, b = (self.node(n) for n in ends[name])
                self.inds.append(dict(ref=f"PCB{i}", a=a, b=b, L=0.0, r=0.0))
                self.parts.append((f"PCB{i}", 'copper', a, b, ('ind', len(self.inds) - 1)))
            cu_R = np.asarray(cm['R_ohm'], float)
            cu_L = np.asarray(cm['L_H'], float)
            e, V = np.linalg.eigh((cu_L + cu_L.T) / 2)
            floor = max(1e-3 * e.max(), 1e-13)              # a direction the fit left no inductance in (it took
            cu_L = (V * np.maximum(e, floor)) @ V.T         # out what changed the impedance by under its tolerance)
            for f in cm.get('foster', []):
                Rm = np.asarray(f['R_ohm'], float)
                e, V = np.linalg.eigh((Rm + Rm.T) / 2)
                for j, val in enumerate(e):
                    if val > max(e.max() * 1e-8, 1e-20):
                        self.modes.append((V[:, j] * math.sqrt(val), float(f['pole_rad_s'])))
        self.notes = [f"{', '.join(refs)}: {text}" for text, refs in self._noted.items()]   # (one line a kind)
        self.n, self.nc, self.nl, self.nf = len(self.index), len(self.caps), len(self.inds), len(self.modes)
        self.s0 = self.nc + self.nl + self.nf               # the first sine state
        self.nx = self.s0 + 2 * len(self.sines)
        self.free = np.arange(self.s0)                      # the states the steady state is looked for in
        self.x0 = np.zeros(self.nx)
        for k, q in enumerate(self.sines):
            self.x0[self.s0 + 2 * k:self.s0 + 2 * k + 2] = math.sin(q['phase']), math.cos(q['phase'])
        if not self.nc + self.nl:
            raise Unsupported('a circuit with no capacitor and no inductor has no state: nothing for this engine')
        L = np.diag([q['L'] for q in self.inds]) if self.nl else np.zeros((0, 0))
        self.R = np.diag([q['r'] for q in self.inds]) if self.nl else np.zeros((0, 0))
        if cu_L is not None:
            k = slice(self.cu, self.cu + len(cu_L))
            L[k, k], self.R[k, k] = cu_L, cu_R
        where = {q['ref']: i for i, q in enumerate(self.inds)}
        for c in model.get('couplings', []):
            i, j = (where[r] for r in c['inductors'])
            L[i, j] = L[j, i] = c['k'] * math.sqrt(L[i, i] * L[j, j])
        self.Linv = np.linalg.inv(L) if self.nl else L
        # the gates' edges within the period: [(time, switch, on)], and each switch's state at its start
        self.edges = []
        for k, s in enumerate(self.sw):
            for m in range(int(round(self.period / s['T']))):
                t_on = (s['on'] + m * s['T']) % self.period
                self.edges += [(t_on, k, True), ((t_on + s['span']) % self.period, k, False)]
        for k, q in enumerate(self.pulses):                 # a pulse's level: a gate after the switches'
            for m in range(int(round(self.period / q['T']))):
                t_on = (q['on'] + m * q['T']) % self.period
                self.edges += [(t_on, len(self.sw) + k, True), ((t_on + q['span']) % self.period, len(self.sw) + k, False)]
        self.edges.sort(key=lambda e: e[0])
        self._mats, self._phi = {}, {}

    def resolution(self):
        """The samples a period needs so that the fastest ringing of the circuit (the copper's inductance against
        a switch's capacitance: hundreds of MHz) is seen PER_RING times a cycle: a period's averages taken
        between samples further apart alias it, and move the power between a switch and the copper by more
        than either has. The ringings: each topology's underdamped eigenvalues (gates as they come in the
        period, the diodes all off and all on). 0: none faster than the period's own samples see."""
        tops = {self.gates_at(t) for t, *_ in self.edges} | {self.gates_at(0.0)}
        fastest = 0.0
        for gates in tops:
            for on in (False, True):
                lam = np.linalg.eigvals(self.matrices(gates, (on,) * len(self.di))['A'][:self.s0, :self.s0])
                ring = lam[np.abs(lam.imag) > np.abs(lam.real)]
                if len(ring):
                    fastest = max(fastest, float(np.abs(ring.imag).max()) / (2 * math.pi))
        return int(min(2 ** math.ceil(math.log2(max(fastest * self.period * PER_RING, 1.0))), MOST))

    def _note(self, ref, text):
        """What the engine leaves out of a part: a note, the parts it is said of together."""
        self._noted.setdefault(text, []).append(ref)

    def _winding(self, ref, a, b, p):
        """A winding as power_spice.Circuit.coil has it: its series resistance to a node of its own, its
        inductance from there (a state: <ref>.winding_I), the core's resistance across that, and the
        self-capacitance across the whole."""
        from ..wpt import coil_capacitance, core_resistance, series_resistance
        coil = dict(p, L_H=p['value'], R_ohm=p.get('DCR_ohm', p.get('R_ohm', 0.0)))
        R, C, rc = series_resistance(coil), coil_capacitance(coil), core_resistance(coil)
        mid = self.node('RL:' + ref)
        g = 1 / max(R, 1e-12)
        self.g.append((a, mid, g))
        self.inds.append(dict(ref=ref, a=mid, b=b, L=p['value'], r=0.0))
        if rc:
            self.g.append((mid, b, 1 / rc))
        cap = self._junction(ref, a, b, C)
        self.parts.append((ref, 'inductor', a, b, ('winding', len(self.inds) - 1, mid, g, cap)))

    def _junction(self, ref, a, b, value):
        """A part's own capacitance between its pins (a switch's Cds, a diode's Cj): a capacitor of the plant, its
        index; None where there is none."""
        value = float(value or 0.0)
        if value <= 0:
            return None
        self.caps.append(dict(ref=ref + ':C', a=a, b=b, g=1 / R_MIN, C=value))
        return len(self.caps) - 1

    def node(self, name):
        if name in GROUND:
            return -1
        return self.index.setdefault(name, len(self.index))

    def gates_at(self, t):
        """Each switch's gate just after time t of the period, then each pulse source's level (high: True)."""
        return tuple(((t - s['on']) % s['T']) < s['span'] for s in (*self.sw, *self.pulses))

    # ---- one topology: the gates' and the diodes' states ----
    def matrices(self, gates, diodes):
        """A, c (x' = A x + c) and Vx, v0 (the nodes' voltages, Vx x + v0) with these switches and diodes on."""
        key = (gates, diodes)
        if key in self._mats:
            return self._mats[key]
        n, nx = self.n, self.nx
        G = np.zeros((n, n))
        Jx, j0 = np.zeros((n, nx)), np.zeros(n)
        parent, grounded = list(range(n)), set()           # the nodes joined by conductance, and which reach ground
        weak = []                                           # an open switch's or diode's G_OFF: (a, b, g)

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def stamp(a, b, g):
            for p, q in ((a, a), (b, b)):
                if p >= 0:
                    G[p, q] += g
            if a >= 0 and b >= 0:
                G[a, b] -= g
                G[b, a] -= g
                parent[root(a)] = root(b)
            elif max(a, b) >= 0:
                grounded.add(max(a, b))

        def inject(vec, a, b, amount):                      # a current `amount` from a to b inside an element:
            if a >= 0:                                      # the element's own g v part is in G; what is left is
                vec[a] -= amount                            # a source, into the right-hand side
            if b >= 0:
                vec[b] += amount
        for a, b, g in self.g:
            stamp(a, b, g)
        for q in self.pl:
            stamp(q['a'], q['b'], q['g'])
        for k, q in enumerate(self.caps):                   # i = g (v_a - v_b - x)
            stamp(q['a'], q['b'], q['g'])
            col = np.zeros(n)
            inject(col, q['a'], q['b'], -q['g'])
            Jx[:, k] += col
        for k, q in enumerate(self.inds):                   # i = x
            col = np.zeros(n)
            inject(col, q['a'], q['b'], 1.0)
            Jx[:, self.nc + k] += col
        high = {q['vs']: q['high'] for k, q in enumerate(self.pulses) if gates[len(self.sw) + k]}
        for k, (ref, a, b, g, volts) in enumerate(self.vs):   # i = g (v_a - v_b - V)
            stamp(a, b, g)
            inject(j0, a, b, -g * high.get(k, volts))
        for k, q in enumerate(self.sines):                  # V = offset + A x (its sine state)
            ref, a, b, g, volts = self.vs[q['vs']]
            col = np.zeros(n)
            inject(col, a, b, -g * q['A'])
            Jx[:, self.s0 + 2 * k] += col
        for ref, a, b, amps in self.cs:
            inject(j0, a, b, amps)
        for k, s in enumerate(self.sw):
            if gates[k]:
                stamp(s['a'], s['b'], s['g'])
            else:
                weak.append((s['a'], s['b'], G_OFF))
        for k, d in enumerate(self.di):                     # on: i = g (v_a - v_b - Vth)
            if diodes[k]:
                stamp(d['a'], d['b'], d['g'])
                inject(j0, d['a'], d['b'], -d['g'] * d['vth'])
            else:
                weak.append((d['a'], d['b'], G_OFF))
        # An island: nodes joined by conductance with no path to ground, reached from the rest only through
        # inductors, current sources and open switches and diodes (a pad between the copper's port and a part's
        # winding; a capacitor between two pads; a switching node while everything on it is open). Its
        # potential is not the conductances' to set. First, it is what keeps the current the inductors and the
        # current sources bring it at zero; what that leaves free (an island no inductor reaches: a bridge's
        # AC side while its diodes are all off), the open parts' leakage (G_OFF) sets, as it does the rest of
        # the circuit's. An open part between two islands is open: its leakage stamped in, the current an
        # inductor drives through it would have to be a state of a nanoamp and a 1e-17 s time constant, whose
        # rounding is the volts across it. The others: G_MIN to ground, as the deck's GMIN.
        islands = {}
        for i in range(n):
            islands.setdefault(root(i), []).append(i)
        reach = {root(i) for i in grounded}
        islands = [nodes for r, nodes in islands.items() if r not in reach]
        N = np.zeros((n, len(islands)))
        for f, nodes in enumerate(islands):
            N[nodes, f] = 1.0
        held = lambda a: a < 0 or root(a) in reach
        Gw = np.zeros((n, n))                               # the leakage that joins an island to anything
        for a, b, g in weak:
            inside = (held(a) and held(b)) or (a >= 0 and b >= 0 and root(a) == root(b))
            M_ = G if inside else Gw
            for p_ in (a, b):
                if p_ >= 0:
                    M_[p_, p_] += g
            if a >= 0 and b >= 0:
                M_[a, b] -= g
                M_[b, a] -= g
        G += np.diag(np.where(N.any(axis=1), 0.0, G_MIN))
        for f, nodes in enumerate(islands):                 # (a potential for the solve: mu sets the real one)
            G[np.ix_(nodes, nodes)] += max(np.abs(np.diag(G)[nodes]).max(), 1.0)
        Gi = np.linalg.inv(G)
        Vx, v0 = Gi @ Jx, Gi @ j0
        row = lambda a: (Vx[a], v0[a]) if a >= 0 else (np.zeros(nx), 0.0)
        if islands:
            F = len(islands)
            mux, mu0 = np.zeros((F, nx)), np.zeros(F)
            S = np.zeros((F, F))
            if self.nl:
                # the current into each island, r = N^T j0 - E^T I (the current sources' and the inductors'), is
                # zero: held so as r' = -HOLD / period r (so a state off it, as the search may step to, comes back
                # within the period; in the steady state r = 0 and the term is nothing), with I' = Linv (VL x + v
                # + E mu)
                E = np.array([(N[q['a']] if q['a'] >= 0 else 0) - (N[q['b']] if q['b'] >= 0 else 0)
                              for q in self.inds]).reshape(self.nl, F)
                VL, v = self._inductor_rows(row, nx)
                W = E.T @ self.Linv
                S = W @ E
                Sp = np.linalg.pinv(S, rcond=1e-10, hermitian=True)
                alpha = HOLD / self.period
                EI = np.zeros((F, nx))
                EI[:, self.nc:self.nc + self.nl] = E.T
                mux, mu0 = Sp @ (-alpha * EI - W @ VL), Sp @ (alpha * (N.T @ j0) - W @ v)
            e, Q = np.linalg.eigh(S)                        # what the inductors leave free: by the leakage
            Z = Q[:, e <= max(e.max(initial=0.0), 0.0) * 1e-10 + 1e-300]
            if Z.shape[1] and Gw.any():
                NG = N.T @ Gw
                K = np.linalg.pinv(Z.T @ NG @ N @ Z)
                mux = mux - Z @ (K @ (Z.T @ NG @ (Vx + N @ mux)))
                mu0 = mu0 - Z @ (K @ (Z.T @ NG @ (v0 + N @ mu0)))
            Vx, v0 = Vx + N @ mux, v0 + N @ mu0
        A, c = np.zeros((nx, nx)), np.zeros(nx)
        for k, q in enumerate(self.caps):                   # C x' = g (v_a - v_b - x)
            (ra, ca), (rb, cb) = row(q['a']), row(q['b'])
            A[k] = q['g'] * (ra - rb) / q['C']
            A[k, k] -= q['g'] / q['C']
            c[k] = q['g'] * (ca - cb) / q['C']
        if self.nl:
            VL, v = self._inductor_rows(row, nx)
            A[self.nc:self.nc + self.nl], c[self.nc:self.nc + self.nl] = self.Linv @ VL, self.Linv @ v
            if self.modes:
                ports = slice(self.nc + self.cu, self.nc + self.cu + len(self.modes[0][0]))
                for m, (gain, pole) in enumerate(self.modes):   # its inductance (1 / pole) has its section's
                    z = self.nc + self.nl + m                   # voltage across it: z' = pole (gain . I - z)
                    A[z, ports], A[z, z] = pole * gain, -pole
        for k, q in enumerate(self.sines):                  # s' = w c, c' = -w s
            i = self.s0 + 2 * k
            A[i, i + 1], A[i + 1, i] = q['w'], -q['w']
        # each diode's own voltage, as a row over x and a constant: what its state is judged by
        Dx = np.array([row(d['a'])[0] - row(d['b'])[0] for d in self.di]).reshape(len(self.di), nx)
        d0 = np.array([row(d['a'])[1] - row(d['b'])[1] for d in self.di])
        Mx = np.zeros((nx + 1, nx + 1))
        Mx[:nx, :nx], Mx[:nx, nx] = A, c
        self._mats[key] = dict(A=A, c=c, Vx=Vx, v0=v0, Dx=Dx, d0=d0, M=Mx)
        return self._mats[key]

    def _inductor_rows(self, row, nx):
        """L I' = VL x + v: each inductor's voltage (row(node): that node's voltage over x and its constant) less
        its resistance's drop, and across the copper's ports their Foster modes'."""
        VL, v = np.zeros((self.nl, nx)), np.zeros(self.nl)
        for k, q in enumerate(self.inds):                   # L x' = v_a - v_b - R x
            (ra, ca), (rb, cb) = row(q['a']), row(q['b'])
            VL[k] = ra - rb
            v[k] = ca - cb
        VL[:, self.nc:self.nc + self.nl] -= self.R
        for m, (gain, pole) in enumerate(self.modes):       # a Foster mode: its section's resistance carries what
            z = self.nc + self.nl + m                       # of the mode's current its inductance does not, and
            ports = slice(self.nc + self.cu, self.nc + self.cu + len(gain))     # puts gain (gain . I - z) across
            rows = slice(self.cu, self.cu + len(gain))      # the ports
            VL[rows, ports] -= np.outer(gain, gain)
            VL[rows, z] += gain
        return VL, v

    def step(self, key, h):
        """exp(M h) of a topology (x and a 1 under it), kept for the steps that repeat."""
        k = (key, round(h, 18))
        if k not in self._phi:
            if len(self._phi) > 4000:
                self._phi.clear()
            self._phi[k] = expm(self.matrices(*key)['M'] * h)
        return self._phi[k]

    def wrong(self, x, gates, diodes, tol=1e-7):
        """How far each diode is from where its state says: an on diode's reverse current (A, as volts over its
        resistance) or an off diode's voltage over its threshold; <= tol where its state holds."""
        m = self.matrices(gates, diodes)
        v = m['Dx'] @ x + m['d0']
        return np.array([(d['vth'] - v[k]) if diodes[k] else (v[k] - d['vth']) for k, d in enumerate(self.di)]) - tol

    def settle(self, x, gates, diodes):
        """The diodes' states that hold at x with these gates: the worst one flipped until none is wrong."""
        diodes, seen = tuple(diodes), set()
        for _ in range(4 * len(self.di) + 4):
            w = self.wrong(x, gates, diodes)
            if not len(w) or w.max() <= 0 or diodes in seen:
                break
            seen.add(diodes)
            k = int(w.argmax())
            diodes = diodes[:k] + (not diodes[k],) + diodes[k + 1:]
        return diodes

    def advance(self, xa, gates, diodes, h, P=None, hook=None, t0=0.0):
        """From xa (x and a 1) over h with no gate edge inside: (xa at the end, the diodes' states there, P x the
        steps' matrices). A diode that changes state does so at its own instant, found by bisection; hook(time,
        xa, gates, the diodes before, after) is told of each (time: t0 and the time into h)."""
        left = h
        for _ in range(64):
            key = (gates, diodes)
            Phi = self.step(key, left)
            xb = Phi @ xa
            if not self.di or self.wrong(xb[:-1], gates, diodes).max() <= 0:
                return xb, diodes, (Phi @ P if P is not None else None)
            lo, hi = 0.0, left                              # the first instant a diode's state stops holding
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if self.wrong((expm(self.matrices(*key)['M'] * mid) @ xa)[:-1], gates, diodes).max() <= 0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-9 * h + 1e-15:
                    break
            Phi = expm(self.matrices(*key)['M'] * hi)
            xa = Phi @ xa
            if P is not None:
                P = Phi @ P
            new = self.settle(xa[:-1], gates, diodes)
            if new == diodes:                               # (at the tolerance's edge: the worst one, flipped)
                k = int(self.wrong(xa[:-1], gates, diodes).argmax())
                new = diodes[:k] + (not diodes[k],) + diodes[k + 1:]
            if hook is not None:
                hook(t0 + h - left + hi, xa, gates, diodes, new)
            diodes, left = new, left - hi
            if left <= 0:
                return xa, diodes, P
        raise RuntimeError('a diode changes state without end inside one step: a circuit this engine cannot settle '
                           '(analysis.engine = "ngspice")')

    def one_period(self, x0, samples=SAMPLES, diodes=None, keep=False):
        """A period from x0: (x at its end, the product of its steps' matrices, the diodes' states at the end,
        the samples: [(time, x, gates, diodes)] when keep)."""
        T, dt = self.period, self.period / samples
        xa = np.append(x0, 1.0)
        gates = self.gates_at(0.0)
        diodes = self.settle(x0, gates, diodes if diodes is not None else (False,) * len(self.di))
        P = np.eye(self.nx + 1)
        out = []
        edges = [e for e in self.edges if 0.0 < e[0] < T]
        e = 0
        for i in range(samples):
            t, t1 = i * dt, (i + 1) * dt
            if keep:
                out.append((t, xa[:-1].copy(), gates, diodes))
            while e < len(edges) and edges[e][0] <= t1:
                te = edges[e][0]
                if te > t:
                    xa, diodes, P = self.advance(xa, gates, diodes, te - t, P)
                    t = te
                gates = self.gates_at(te + 1e-12 * T)
                diodes = self.settle(xa[:-1], gates, diodes)
                e += 1
            if t1 > t:
                xa, diodes, P = self.advance(xa, gates, diodes, t1 - t, P)
        return xa[:-1], P, diodes, out

    def fine_period(self, x0, samples, diodes=None):
        """A period from x0, sampled for its averages: the regular samples, every switching instant twice (the
        state just before it, in the topology before, and just after, in the one after) and after each instant
        samples from a quarter of the new topology's fastest time constant, GROW times further on each, up to the
        next instant or regular sample. A capacitance a switch discharges as it turns on (its Cds through Ron:
        tens of picoseconds) puts its energy in that time: the regular samples step over it. [(time, x, gates,
        diodes)] in time order, from 0 to the period's end."""
        T, dt = self.period, self.period / samples
        xa = np.append(x0, 1.0)
        gates = self.gates_at(0.0)
        diodes = self.settle(x0, gates, diodes if diodes is not None else (False,) * len(self.di))
        out, events = [], []

        def hook(t, xa_, gates_, before, after):
            events.append((t, xa_.copy(), gates_, before, gates_, after))
        edges = [e for e in self.edges if 0.0 < e[0] < T]
        e = 0
        for i in range(samples):
            t, t1 = i * dt, (i + 1) * dt
            out.append((t, 0, xa[:-1].copy(), gates, diodes))
            while e < len(edges) and edges[e][0] <= t1:
                te = edges[e][0]
                if te > t:
                    xa, diodes, _ = self.advance(xa, gates, diodes, te - t, None, hook, t)
                    t = te
                new_g = self.gates_at(te + 1e-12 * T)
                new_d = self.settle(xa[:-1], new_g, diodes)
                if (new_g, new_d) != (gates, diodes):
                    events.append((te, xa.copy(), gates, diodes, new_g, new_d))
                gates, diodes = new_g, new_d
                e += 1
            if t1 > t:
                xa, diodes, _ = self.advance(xa, gates, diodes, t1 - t, None, hook, t)
        out.append((T, 0, xa[:-1].copy(), gates, diodes))
        times = sorted({s_[0] for s_ in out} | {ev[0] for ev in events})
        for k, (te, xe, g0, d0, g1, d1) in enumerate(events):
            out.append((te, -2, xe[:-1].copy(), g0, d0))                 # (before it, then after it: the jump)
            out.append((te, -1, xe[:-1].copy(), g1, d1))
            end = min(t_ for t_ in times + [T] if t_ > te)
            lam = self._fastest((g1, d1))
            h = 0.25 / lam if lam > 0 else end - te
            M_ = self.matrices(g1, d1)['M']
            while te + h < end:
                out.append((te + h, 1, (expm(M_ * h) @ xe)[:-1], g1, d1))
                h *= GROW
        out.sort(key=lambda s_: (s_[0], s_[1]))
        return [(t_, x_, g_, d_) for t_, _, x_, g_, d_ in out]

    def copper_loss(self, fine):
        """Each copper port's share of what the copper dissipates over a period of fine samples (fine_period's),
        W, in the ports' order. What it dissipates: I^T R I and each Foster mode's (gain . I - z)^2 through its
        section's resistance, averaged by the trapezoid (no reactive power in it to cancel: each port's V x I is
        mostly L dI/dt, and those cancel only in the sum). A mode's dissipation is shared among the ports by
        their part of its current."""
        n = len(self.modes[0][0]) if self.modes else self.nl - self.cu
        if n <= 0:
            return []
        t = np.array([s_[0] for s_ in fine])
        X = np.array([s_[1] for s_ in fine])
        I = X[:, self.nc + self.cu:self.nc + self.cu + n]
        mean = lambda y: float(np.sum(np.diff(t) * (y[1:] + y[:-1])) / 2 / (t[-1] - t[0]))
        R = self.R[self.cu:self.cu + n, self.cu:self.cu + n]
        share = np.array([mean(I[:, i] * (I @ R[i])) for i in range(n)])
        for m, (gain, pole) in enumerate(self.modes):
            through = I @ gain - X[:, self.nc + self.nl + m]                # its resistance's current
            part = np.array([mean(gain[i] * I[:, i] * through) for i in range(n)])
            w = part / part.sum() if abs(part.sum()) > 1e-30 else gain ** 2 / max(float(gain @ gain), 1e-30)
            share += w * mean(through * through)
        return share.tolist()

    def _fastest(self, key):
        """The fastest decay of a topology (1/s): its eigenvalues' largest |real part|, kept."""
        cache = self.__dict__.setdefault('_rates', {})
        if key not in cache:
            lam = np.linalg.eigvals(self.matrices(*key)['A'][:self.s0, :self.s0]) if self.s0 else np.zeros(0)
            cache[key] = float(np.abs(lam.real).max(initial=0.0))
        return cache[key]

    def steady_state(self, samples=SAMPLES, tol=1e-9, log=None, search=SEARCH):
        """(the state a period maps onto itself, that period's samples). Newton's method on the period's map, its
        Jacobian the product of the period's matrices; a constant-power load's resistance is set from its mean
        voltage between whiles. The search steps through the period on `search` samples (a diode's state is
        looked at that often), the period kept on `samples`."""
        search = min(search, samples)
        x = self.x0.copy()
        free = self.free
        diodes, periods = None, 0
        for outer in range(12 if self.pl else 1):
            self._mats.clear()
            self._phi.clear()
            for it in range(80):
                x1, P, diodes, _ = self.one_period(x, search, diodes)
                periods += 1
                size = max(np.abs(x1[free]).max(), 1e-9)
                err = np.abs(x1 - x)[free].max() / size
                if err < tol:
                    break
                J = P[:-1, :-1][np.ix_(free, free)]
                try:                                        # the fixed point of x -> x1 + J (x' - x)
                    dx = np.linalg.lstsq(np.eye(len(free)) - J, (x1 - x)[free], rcond=1e-12)[0]
                except np.linalg.LinAlgError:
                    dx = (x1 - x)[free]
                step = dx if np.isfinite(dx).all() and np.abs(dx).max() < 1e3 * size + 1e3 else (x1 - x)[free]
                x = x.copy()
                x[free] += step
            else:
                raise RuntimeError(f"no periodic steady state in 80 periods (the state still moves by {err:.2g} of "
                                   f"itself a period): a circuit this engine cannot settle (analysis.engine = \"ngspice\")")
            x1, P, diodes, kept = self.one_period(x, samples if not self.pl else search, diodes, keep=True)
            if not self.pl:
                break
            moved = 0.0
            for q in self.pl:                               # P / <v>^2, a resistor below min_V
                v = np.mean([self._voltage(s, q['a'], q['b']) for s in kept])
                g = q['P'] / max(v * v, q['vmin'] ** 2)
                moved = max(moved, abs(g - q['g']) / max(g, q['g']))
                q['g'] = g
            if moved < 1e-6:
                x1, P, diodes, kept = self.one_period(x, samples, diodes, keep=True)
                break
        if log:
            log(f"piecewise-linear: {self.nx} states, {periods} periods to the steady state")
        self.periods = periods
        return x, kept

    def _voltage(self, sample, a, b):
        t, x, gates, diodes = sample
        m = self.matrices(gates, diodes)
        v = m['Vx'] @ x + m['v0']
        return (v[a] if a >= 0 else 0.0) - (v[b] if b >= 0 else 0.0)

    def signals(self, kept):
        """The period's signals as ngspice's run names them: time_s, every part's <ref>.V and <ref>.I (the
        current entering at its first node), a capacitor's <ref>.cap_V, a switch's gate, node.<name>."""
        n = len(kept)
        t = np.array([s[0] for s in kept])
        X = np.array([s[1] for s in kept])
        V = np.zeros((n, self.n))
        gates = np.array([s[2] for s in kept], bool).reshape(n, len(self.sw) + len(self.pulses))
        diodes = np.array([s[3] for s in kept], bool).reshape(n, len(self.di))
        for i, s in enumerate(kept):
            m = self.matrices(s[2], s[3])
            V[i] = m['Vx'] @ s[1] + m['v0']
        at = lambda a: V[:, a] if a >= 0 else np.zeros(n)
        out = {'time_s': t}
        named = {n for p in self.model['parts'] for n in p['nodes']}
        for name, k in self.index.items():
            if name in named:                               # (not a pad's own node, nor a part's inner one)
                out['node.' + name] = V[:, k]
        own = lambda cap, v: 0.0 if cap is None else self.caps[cap]['g'] * (v - X[:, cap])
        for ref, kind, a, b, how in self.parts:
            v = at(a) - at(b)
            out[ref + '.V'] = v
            if how[0] == 'g':
                i = how[1] * v
            elif how[0] == 'cap':
                q = self.caps[how[1]]
                out[ref + '.cap_V'] = X[:, how[1]]
                if how[3] is not None:                      # behind its ESL: that inductance's current
                    i = X[:, self.nc + how[3]] + how[2] * v
                else:
                    i = q['g'] * (v - X[:, how[1]]) + how[2] * v
            elif how[0] == 'winding':
                out[ref + '.winding_I'] = X[:, self.nc + how[1]]
                i = how[3] * (at(a) - at(how[2])) + own(how[4], v)
            elif how[0] == 'ind':
                i = X[:, self.nc + how[1]]
            elif how[0] == 'diode':
                d = self.di[how[1]]
                i = np.where(diodes[:, how[1]], d['g'] * (v - d['vth']), G_OFF * v) + own(how[2], v)
            elif how[0] == 'switch':
                s = self.sw[how[1]]
                vc = at(how[3]) - at(how[4])                # across the channel (inside its package)
                i = np.where(gates[:, how[1]], s['g'], G_OFF) * vc + own(how[2], vc)
                if s['body'] is not None:                   # its body diode, source to drain
                    d = self.di[s['body']]
                    i = i - np.where(diodes[:, s['body']], d['g'] * (-vc - d['vth']), G_OFF * -vc)
                if how[5] is not None:                      # (its drain's inductance carries it all)
                    i = X[:, self.nc + how[5]]
                out[ref + '.gate_V'] = np.where(gates[:, how[1]], s['gate'], 0.0)
                out[ref + '.gate_I'] = np.zeros(n)
            elif how[0] == 'cs':
                i = np.full(n, how[1])
            elif how[0] == 'vs':
                _, plus, minus, g, volts = self.vs[how[1]]
                emf = np.full(n, volts)
                for k, q in enumerate(self.pulses):
                    if q['vs'] == how[1]:
                        emf = np.where(gates[:, len(self.sw) + k], q['high'], volts)
                for k, q in enumerate(self.sines):
                    if q['vs'] == how[1]:
                        emf = volts + q['A'] * X[:, self.s0 + 2 * k]
                i = g * (at(plus) - at(minus) - emf)
            else:
                i = self.pl[how[1]]['g'] * v
            out[ref + '.I'] = i
        return out


STATS = dict(AVG=lambda t, y: float(np.mean(y)), RMS=lambda t, y: float(np.sqrt(np.mean(y * y))),
             PP=lambda t, y: float(np.max(y) - np.min(y)), MAX=lambda t, y: float(np.max(y)),
             MIN=lambda t, y: float(np.min(y)))


def measure(deck, signals, fine=None):
    """The deck's measures (engine.add_measures: a statistic of an expression over @signal@s) on a period's
    signals. fine (Plant.fine_period's signals, from 0 to the period's end): the averages (a part's power among
    them) are taken over it by the trapezoid, so a switching instant's energy is in them; the RMS values and the
    peaks stay the regular samples' (an instant edge's spike, hundreds of amps for picoseconds, is the ideal
    switch's, not the circuit's)."""
    names = dict(abs=np.abs, sqrt=np.sqrt, min=np.minimum, max=np.maximum, exp=np.exp, log=np.log, pi=np.pi)
    out = {}
    for name, (stat, expr) in deck.measures.items():
        pieces = expr.split('@')
        if any(p not in signals for p in pieces[1::2]):
            missing = next(p for p in pieces[1::2] if p not in signals)
            raise Unsupported(f"{name}: the signal {missing} is not one the piecewise-linear engine has "
                              f"(analysis.engine = \"ngspice\")")
        code = ''.join(f"S[{p!r}]" if k % 2 else p for k, p in enumerate(pieces))
        use = fine if (fine is not None and stat == 'AVG') else signals
        y = eval(code, {'__builtins__': {}}, dict(names, S=use))        # noqa: S307 (the model's own expression)
        y = np.broadcast_to(np.asarray(y, float), use['time_s'].shape)
        if use is fine:
            t = fine['time_s']
            out[name] = float(np.sum(np.diff(t) * (y[1:] + y[:-1])) / 2 / (t[-1] - t[0]))
        else:
            out[name] = STATS[stat](use['time_s'], y)
    return out


def simulate(model, state, cycles, folder, a, reltol, log, label='', final=False, copper=None):
    """engine.simulate with no transient: the model's periodic steady state as the run's result (PERIODS identical
    periods of every signal), the deck for its names, and what the run measured."""
    from .engine import add_measures, collect, count_out_storage
    deck = build(model, None, copper)
    add_measures(deck, model)
    plant = Plant(model, copper)
    samples = int(a.get('samples_per_cycle') or SAMPLES)
    if final and a.get('final_samples_per_cycle'):
        samples = int(a['final_samples_per_cycle'])
    # the reported run is sampled as finely as the circuit rings (a diode the ringing turns on between two steps
    # is missed: the search for its steady state steps as finely); while the targets are looked for, the period's
    # own samples (what a target measures is the circuit's averages, which the ringing barely moves)
    ringing = plant.resolution() if final else 0
    samples = max(samples, ringing)
    x, kept = plant.steady_state(samples, log=(lambda s: log(label + s)) if log else None, search=max(SEARCH, ringing))

    def probed(sig):
        for name, pr in model.get('probes', {}).items():
            if 'voltage' in pr:
                p, q = (plant.node(n) for n in pr['voltage'])
                sig[name] = (sig['node.' + pr['voltage'][0]] if p >= 0 else 0.0) - (sig['node.' + pr['voltage'][1]] if q >= 0 else 0.0)
            else:
                ref = next(p['ref'] for p in model['parts'] if p['ref'].lower() == str(pr['current']).lower())
                sig[name] = sig[ref + '.I']
        return sig
    one = probed(plant.signals(kept))
    period = plant.fine_period(x, samples, kept[0][3])
    fine = probed(plant.signals(period))
    measured = measure(deck, one, fine)
    for i, w in enumerate(plant.copper_loss(period)):              # the copper's own dissipation, not its V x I
        measured[f"copper.{i}.power_W"] = w
    n = len(one['time_s'])
    result = {k: np.tile(v, PERIODS) for k, v in one.items() if k != 'time_s'}
    result['time_s'] = np.arange(PERIODS * n) * (plant.period / n)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'transient.npz', **result)
    (folder / 'signals.json').write_text(json.dumps({k: 'piecewise-linear' for k in one if k != 'time_s'}, indent=2))
    (folder / 'measurements.json').write_text(json.dumps(measured, indent=2))
    (folder / 'performance.json').write_text(json.dumps(dict(engine='pwl', states=plant.nx, periods=plant.periods,
                                                             samples_per_cycle=samples, ringing_samples=ringing), indent=2))
    if deck.ports:                                                  # the copper's currents, harmonic by harmonic
        port_spectrum(folder, one, plant.period, max(31, int((model.get('fields') or {}).get('harmonics', 16))))
    result['_measures'] = measured
    deck.engine_notes = list(plant.notes)
    stats = count_out_storage(collect(measured, deck), result, model, deck)
    gap = (stats.get('power_balance_relative') or 0.0) * stats.get('source_power_W', 0.0)
    if deck.ports and gap > 1e-3 * max(stats.get('source_power_W', 0.0), 1e-12):
        deck.engine_notes.append(
            f"{1e3 * gap:.3g} mW is lost at the switching edges and in no part: the copper's inductance carries a "
            f"current an opening switch or diode cuts off (with no capacitance across it), and its energy goes with "
            f"the edge. It is in the loss and the efficiency; ngspice puts it in the switch, over the gate's edge")
    return result, deck, stats, 0


def port_spectrum(folder, one, period, harmonics=31):
    """<folder>/port-spectrum.npz from a period's PCB<i>.I (as power_waveforms.save_port_spectrum writes it from
    ngspice's steps): what circuit/fields.py maps the copper's fields from."""
    from ..power_waveforms import periodic_coefficients
    labels = sorted((k for k in one if k.startswith('PCB') and k.endswith('.I')), key=lambda k: int(k[3:-2]))
    if not labels:
        return None
    t = np.append(one['time_s'], period)                            # the period closed: its first sample again
    I = np.stack([np.append(one[k], one[k][0]) for k in labels], axis=1)
    coefficients, info = periodic_coefficients(t, I, 1 / period, harmonics)
    np.savez_compressed(Path(folder) / 'port-spectrum.npz', coefficients_A=coefficients, port_labels=np.array(labels),
                        frequencies_Hz=np.arange(harmonics + 1) / period, **info)
    return info
