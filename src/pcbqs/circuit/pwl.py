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
                ngspice's deck has them); its gate's edges where the drive puts them (deck.gate_timing), the
                dead time included; no capacitance, no gate charge
    diode       a threshold and a resistance: on above Vth with R_on (the model's Vf_V at If_A, its R_ohm), off
                below; no junction capacitance, no recovery
    capacitor   value x bias_factor with its ESR and leakage (its ESL is left out: said in the result's notes)
    inductor    value with its DCR, and the model's couplings (a winding's self-capacitance and core loss: no)
    source      dc_V with its series R and L, or dc_A; no waves
    load        resistance_ohm, current_A, or power_W (a resistance that takes that power at its mean voltage)
    resistor
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
G_MIN = 1e-12           # S: every node to ground (a node between two inductors has a voltage)
R_MIN = 1e-6            # ohm: the least series resistance of a capacitor or a voltage source
VT = 0.025852           # V: the thermal voltage, as ngspice's diodes at 27 C
SAMPLES = 8192          # a period's samples in the result (analysis.samples_per_cycle): a signal that jumps at a
                        # switching instant is averaged to a sample's width, so a product's mean (a part's power) to
                        # about its jump over this
SEARCH = 512            # and while the steady state is looked for: the instants are exact whatever this is
PERIODS = 4             # the periods the result holds (identical: engine.py judges a run over several)


class Unsupported(M.ModelError):
    """A model the piecewise-linear engine does not take: says what, and that ngspice does."""


def diode_line(vf, i_f, n=1.0, rs=0.01):
    """(threshold V, on resistance ohm) of a diode that drops vf at i_f: the junction's own drop less one thermal
    voltage as the threshold, and that thermal voltage over i_f with the series resistance, so the line meets
    the diode's curve at i_f with its slope there."""
    junction = max(vf - rs * i_f, 0.05)
    return junction - n * VT, rs + n * VT / i_f


class Plant:
    """A resolved model as a switched linear system. States: every capacitor's own voltage, then every inductor's
    current (a source's series inductance among them)."""

    def __init__(self, model):
        self.model, self.period, self.notes = model, period_of(model), []
        self.index = {}                                     # node name -> row (ground: -1)
        self.g, self.vs, self.cs = [], [], []               # (a, b, S); (ref, a, b, S, V); (ref, a, b, A a -> b)
        self.caps, self.inds, self.sw, self.di, self.pl = [], [], [], [], []
        self.parts = []                                     # (ref, kind, a, b, how its current is found)
        drives = model.get('drives', {})
        for p in model['parts']:
            if not p.get('enabled', True):
                continue
            ref, kind = p['ref'], p['type']
            a, b = (self.node(n) for n in p['nodes'])
            if kind == 'resistor':
                self.g.append((a, b, 1 / p['value']))
                self.parts.append((ref, kind, a, b, ('g', 1 / p['value'])))
            elif kind == 'capacitor':
                if p.get('ESL_H'):
                    self.notes.append(f"{ref}: its ESL is left out (the piecewise-linear engine)")
                g = 1 / max(p.get('ESR_ohm', 0.0), R_MIN)
                self.caps.append(dict(ref=ref, a=a, b=b, g=g, C=p['value'] * p.get('bias_factor', 1.0)))
                if p.get('leakage_ohm'):
                    self.g.append((a, b, 1 / p['leakage_ohm']))
                self.parts.append((ref, kind, a, b, ('cap', len(self.caps) - 1, 1 / p['leakage_ohm'] if p.get('leakage_ohm') else 0.0)))
            elif kind == 'inductor':
                if any(k in p for k in ('SRF_Hz', 'Cp_F', 'core_R_ohm', 'loss_model')):
                    raise Unsupported(f"{ref}: a winding with its self-capacitance or core loss is ngspice's "
                                      f"(analysis.engine = \"ngspice\")")
                self.inds.append(dict(ref=ref, a=a, b=b, L=p['value'], r=p.get('DCR_ohm', 0.0)))
                self.parts.append((ref, kind, a, b, ('ind', len(self.inds) - 1)))
            elif kind == 'diode':
                m = p.get('model', {})
                vth, ron = diode_line(m.get('Vf_V', 0.7), m.get('If_A', 1.0), m.get('n', 1.0), m.get('R_ohm', 0.01))
                self.di.append(dict(ref=ref, a=a, b=b, vth=vth, g=1 / ron))
                self.parts.append((ref, kind, a, b, ('diode', len(self.di) - 1)))
            elif kind == 'switch':
                m = p.get('model', {})
                kind_m = m.get('type', 'switch')
                if kind_m not in ('switch', 'ideal'):
                    raise Unsupported(f"{ref}: a {kind_m} switch model is ngspice's (analysis.engine = \"ngspice\")")
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
                self.parts.append((ref, kind, a, b, ('switch', len(self.sw) - 1)))
            elif kind == 'source':
                if 'wave' in p:
                    raise Unsupported(f"{ref}: a source with a wave is ngspice's (analysis.engine = \"ngspice\")")
                if 'dc_A' in p:
                    self.cs.append((ref, a, b, -p['dc_A']))            # out of its + node
                    self.parts.append((ref, kind, a, b, ('cs', -p['dc_A'])))
                    continue
                plus = a
                if p.get('L_H'):                             # its series inductance: a state, behind a node
                    plus = self.node('L:' + ref)
                    self.inds.append(dict(ref=ref + ':L', a=a, b=plus, L=p['L_H'], r=0.0))
                g = 1 / max(p.get('R_ohm', 0.0), R_MIN)
                self.vs.append((ref, plus, b, g, p['dc_V']))
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
        self.n, self.nc, self.nl = len(self.index), len(self.caps), len(self.inds)
        self.nx = self.nc + self.nl
        if not self.nx:
            raise Unsupported('a circuit with no capacitor and no inductor has no state: nothing for this engine')
        L = np.diag([q['L'] for q in self.inds]) if self.nl else np.zeros((0, 0))
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
        self.edges.sort(key=lambda e: e[0])
        self._mats, self._phi = {}, {}

    def node(self, name):
        if name in GROUND:
            return -1
        return self.index.setdefault(name, len(self.index))

    def gates_at(self, t):
        """Each switch's gate just after time t of the period."""
        out = []
        for s in self.sw:
            out.append(((t - s['on']) % s['T']) < s['span'])
        return tuple(out)

    # ---- one topology: the gates' and the diodes' states ----
    def matrices(self, gates, diodes):
        """A, c (x' = A x + c) and Vx, v0 (the nodes' voltages, Vx x + v0) with these switches and diodes on."""
        key = (gates, diodes)
        if key in self._mats:
            return self._mats[key]
        n, nx = self.n, self.nx
        G = np.eye(n) * G_MIN
        Jx, j0 = np.zeros((n, nx)), np.zeros(n)

        def stamp(a, b, g):
            for p, q in ((a, a), (b, b)):
                if p >= 0:
                    G[p, q] += g
            if a >= 0 and b >= 0:
                G[a, b] -= g
                G[b, a] -= g

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
        for ref, a, b, g, volts in self.vs:                 # i = g (v_a - v_b - V)
            stamp(a, b, g)
            inject(j0, a, b, -g * volts)
        for ref, a, b, amps in self.cs:
            inject(j0, a, b, amps)
        for k, s in enumerate(self.sw):
            stamp(s['a'], s['b'], s['g'] if gates[k] else G_OFF)
        for k, d in enumerate(self.di):                     # on: i = g (v_a - v_b - Vth)
            if diodes[k]:
                stamp(d['a'], d['b'], d['g'])
                inject(j0, d['a'], d['b'], -d['g'] * d['vth'])
            else:
                stamp(d['a'], d['b'], G_OFF)
        Gi = np.linalg.inv(G)
        Vx, v0 = Gi @ Jx, Gi @ j0
        row = lambda a: (Vx[a], v0[a]) if a >= 0 else (np.zeros(nx), 0.0)
        A, c = np.zeros((nx, nx)), np.zeros(nx)
        for k, q in enumerate(self.caps):                   # C x' = g (v_a - v_b - x)
            (ra, ca), (rb, cb) = row(q['a']), row(q['b'])
            A[k] = q['g'] * (ra - rb) / q['C']
            A[k, k] -= q['g'] / q['C']
            c[k] = q['g'] * (ca - cb) / q['C']
        if self.nl:
            VL, v = np.zeros((self.nl, nx)), np.zeros(self.nl)
            for k, q in enumerate(self.inds):               # L x' = v_a - v_b - r x
                (ra, ca), (rb, cb) = row(q['a']), row(q['b'])
                VL[k] = ra - rb
                VL[k, self.nc + k] -= q['r']
                v[k] = ca - cb
            A[self.nc:], c[self.nc:] = self.Linv @ VL, self.Linv @ v
        # each diode's own voltage, as a row over x and a constant: what its state is judged by
        Dx = np.array([row(d['a'])[0] - row(d['b'])[0] for d in self.di]).reshape(len(self.di), nx)
        d0 = np.array([row(d['a'])[1] - row(d['b'])[1] for d in self.di])
        Mx = np.zeros((nx + 1, nx + 1))
        Mx[:nx, :nx], Mx[:nx, nx] = A, c
        self._mats[key] = dict(A=A, c=c, Vx=Vx, v0=v0, Dx=Dx, d0=d0, M=Mx)
        return self._mats[key]

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

    def advance(self, xa, gates, diodes, h, P=None):
        """From xa (x and a 1) over h with no gate edge inside: (xa at the end, the diodes' states there, P x the
        steps' matrices). A diode that changes state does so at its own instant, found by bisection."""
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

    def steady_state(self, samples=SAMPLES, tol=1e-9, log=None, search=SEARCH):
        """(the state a period maps onto itself, that period's samples). Newton's method on the period's map, its
        Jacobian the product of the period's matrices; a constant-power load's resistance is set from its mean
        voltage between whiles. The search steps through the period on `search` samples (a diode's state is
        looked at that often), the period kept on `samples`."""
        search = min(search, samples)
        x = np.zeros(self.nx)
        diodes, periods = None, 0
        for outer in range(12 if self.pl else 1):
            self._mats.clear()
            self._phi.clear()
            for it in range(80):
                x1, P, diodes, _ = self.one_period(x, search, diodes)
                periods += 1
                size = max(np.abs(x1).max(), 1e-9)
                err = np.abs(x1 - x).max() / size
                if err < tol:
                    break
                J = P[:-1, :-1]
                try:                                        # the fixed point of x -> x1 + J (x' - x)
                    dx = np.linalg.lstsq(np.eye(self.nx) - J, x1 - x, rcond=1e-12)[0]
                except np.linalg.LinAlgError:
                    dx = x1 - x
                x = x + dx if np.isfinite(dx).all() and np.abs(dx).max() < 1e3 * size + 1e3 else x1
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
        gates = np.array([s[2] for s in kept], bool).reshape(n, len(self.sw))
        diodes = np.array([s[3] for s in kept], bool).reshape(n, len(self.di))
        for i, s in enumerate(kept):
            m = self.matrices(s[2], s[3])
            V[i] = m['Vx'] @ s[1] + m['v0']
        at = lambda a: V[:, a] if a >= 0 else np.zeros(n)
        out = {'time_s': t}
        for name, k in self.index.items():
            if ':' not in name:
                out['node.' + name] = V[:, k]
        for ref, kind, a, b, how in self.parts:
            v = at(a) - at(b)
            out[ref + '.V'] = v
            if how[0] == 'g':
                i = how[1] * v
            elif how[0] == 'cap':
                q = self.caps[how[1]]
                out[ref + '.cap_V'] = X[:, how[1]]
                i = q['g'] * (v - X[:, how[1]]) + how[2] * v
            elif how[0] == 'ind':
                i = X[:, self.nc + how[1]]
            elif how[0] == 'diode':
                d = self.di[how[1]]
                i = np.where(diodes[:, how[1]], d['g'] * (v - d['vth']), G_OFF * v)
            elif how[0] == 'switch':
                s = self.sw[how[1]]
                i = np.where(gates[:, how[1]], s['g'], G_OFF) * v
                if s['body'] is not None:                   # its body diode, source to drain
                    d = self.di[s['body']]
                    i = i - np.where(diodes[:, s['body']], d['g'] * (-v - d['vth']), G_OFF * -v)
                out[ref + '.gate_V'] = np.where(gates[:, how[1]], s['gate'], 0.0)
                out[ref + '.gate_I'] = np.zeros(n)
            elif how[0] == 'cs':
                i = np.full(n, how[1])
            elif how[0] == 'vs':
                _, plus, minus, g, volts = self.vs[how[1]]
                i = g * (at(plus) - at(minus) - volts)
            else:
                i = self.pl[how[1]]['g'] * v
            out[ref + '.I'] = i
        return out


STATS = dict(AVG=lambda t, y: float(np.mean(y)), RMS=lambda t, y: float(np.sqrt(np.mean(y * y))),
             PP=lambda t, y: float(np.max(y) - np.min(y)), MAX=lambda t, y: float(np.max(y)),
             MIN=lambda t, y: float(np.min(y)))


def measure(deck, signals):
    """The deck's measures (engine.add_measures: a statistic of an expression over @signal@s) on a period's
    signals."""
    names = dict(abs=np.abs, sqrt=np.sqrt, min=np.minimum, max=np.maximum, exp=np.exp, log=np.log, pi=np.pi)
    out = {}
    for name, (stat, expr) in deck.measures.items():
        pieces = expr.split('@')
        if any(p not in signals for p in pieces[1::2]):
            missing = next(p for p in pieces[1::2] if p not in signals)
            raise Unsupported(f"{name}: the signal {missing} is not one the piecewise-linear engine has "
                              f"(analysis.engine = \"ngspice\")")
        code = ''.join(f"S[{p!r}]" if k % 2 else p for k, p in enumerate(pieces))
        y = eval(code, {'__builtins__': {}}, dict(names, S=signals))    # noqa: S307 (the model's own expression)
        out[name] = STATS[stat](signals['time_s'], np.broadcast_to(np.asarray(y, float), signals['time_s'].shape))
    return out


def simulate(model, state, cycles, folder, a, reltol, log, label='', final=False, copper=None):
    """engine.simulate with no transient: the model's periodic steady state as the run's result (PERIODS identical
    periods of every signal), the deck for its names, and what the run measured."""
    from .engine import add_measures, collect, count_out_storage
    if copper:
        raise Unsupported("the board's copper in the circuit is ngspice's (analysis.engine = \"ngspice\")")
    deck = build(model, None)
    add_measures(deck, model)
    plant = Plant(model)
    samples = int(a.get('samples_per_cycle') or SAMPLES)
    if final and a.get('final_samples_per_cycle'):
        samples = int(a['final_samples_per_cycle'])
    x, kept = plant.steady_state(samples, log=(lambda s: log(label + s)) if log else None)
    one = plant.signals(kept)
    for name, pr in model.get('probes', {}).items():
        if 'voltage' in pr:
            p, q = (plant.node(n) for n in pr['voltage'])
            one[name] = (one['node.' + pr['voltage'][0]] if p >= 0 else 0.0) - (one['node.' + pr['voltage'][1]] if q >= 0 else 0.0)
        else:
            ref = next(p['ref'] for p in model['parts'] if p['ref'].lower() == str(pr['current']).lower())
            one[name] = one[ref + '.I']
    measured = measure(deck, one)
    n = len(one['time_s'])
    result = {k: np.tile(v, PERIODS) for k, v in one.items() if k != 'time_s'}
    result['time_s'] = np.arange(PERIODS * n) * (plant.period / n)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'transient.npz', **result)
    (folder / 'signals.json').write_text(json.dumps({k: 'piecewise-linear' for k in one if k != 'time_s'}, indent=2))
    (folder / 'measurements.json').write_text(json.dumps(measured, indent=2))
    (folder / 'performance.json').write_text(json.dumps(dict(engine='pwl', states=plant.nx, periods=plant.periods,
                                                             samples_per_cycle=samples), indent=2))
    result['_measures'] = measured
    deck.engine_notes = list(plant.notes)
    return result, deck, count_out_storage(collect(measured, deck), result, model, deck), 0
