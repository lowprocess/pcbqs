"""
pcbqs's tools as kicad2ems analysis plugins (the 'kicad2ems.analyses' entry points of pyproject.toml): power
integrity, near field, circuit and thermal. kicad2ems lists them as engines of the questions they answer (a
rail's drop and its impedance, a converter's near field, a circuit, the temperature: kicad2ems.study), a front end
builds each one's settings from its options table, runs it as its command and shows what its result folder holds
(reports.py); the circuit examples come as configs beside their models (circuit/examples/*.toml).

Only this module imports kicad2ems; the tools themselves run without it.

    from pcbqs.analyses import PowerIntegrity
    PowerIntegrity().command(dict(rail='+3V3', load='U1=2'), 'board.kicad_pcb', 'out/pdn')
"""
import json
import os
import re
import sys

from kicad2ems.api.analysis import Analysis

API = (2, 6)


def argv(module, options, values, board, out):
    """[python, -m, module, (board), --out, out, ...] for values {option key: text, bool or list}: an option left
    empty (or at its default) is not passed; a list is lines, or REF=... items joined by commas (a comma inside
    one, CA1=47e-6,0.02,1e-9, stays in it)."""
    opts = {o['key']: o for o in options}
    out_argv = [sys.executable, '-m', module, *([board] if board else []), '--out', out]
    for key, v in values.items():
        o = opts.get(key)
        if o is None:
            continue
        if o['kind'] == 'bool':
            if v is True or str(v).strip().lower() in ('1', 'true', 'yes', 'on'):
                out_argv.append(o['flag'])
        elif o['kind'] == 'list':
            text = '\n'.join(map(str, v)) if isinstance(v, (list, tuple)) else str(v or '')
            for item in [s.strip() for s in re.split(r'\n|,(?=\s*[^\s,=]+\s*=)', text) if s.strip()]:
                out_argv += [o['flag'], item]
        else:
            s = '' if v is None else str(v).strip()
            if s and s != str(o['default']):
                out_argv += [o['flag'], s]
    return out_argv


# the settings that name a net or a part of the board (kicad2ems API 2.11: an option's `pick`; plural: several)
PICK = dict(rail='net', ground='nets', vin='net', source='part', ground_source='part', converter='part',
            inductor='part', through='parts', stop='parts', capacitors='parts')


class _Tool(Analysis):
    """One of pcbqs's command-line tools: its options table (pcbqs._cli's form), its command, its results."""
    api = API
    engine = 'pcbqs'
    module = ''
    pick = {}                               # a tool's own word over PICK's

    def check(self):
        from . import __version__
        return True, f"pcbqs {__version__} ({os.path.dirname(os.path.abspath(__file__))})"

    def table(self):
        return []

    def options(self):
        from ._cli import describe
        opts = describe(self.table())['options']
        pick = {**PICK, **self.pick}
        for o in opts:                                              # what a front end may pick on the board
            if o['key'] in pick:
                o['pick'] = pick[o['key']]
        return opts

    def report(self, folder):
        from . import reports
        try:
            return reports.report(self.name, folder)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def curves(self, folder):
        from . import reports
        try:
            return reports.curves(self.name, folder)
        except (OSError, ValueError, KeyError, TypeError):
            return []

    def _board(self, board):
        if not board or not os.path.exists(board):
            raise ValueError('choose a board first' if not board else f'{board}: no such board')
        return board

    def _read(self, folder):
        """Its result file's JSON, or {} (not there, not readable)."""
        try:
            with open(os.path.join(folder or '', self.result), encoding='utf-8') as fh:
                d = json.load(fh)
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}


_PADS = {}


def _pad_nets(board):
    """{reference: the nets of its pads} of a KiCad board file (kept until the file changes)."""
    stamp = os.path.getmtime(board)
    if _PADS.get(board, (None,))[0] != stamp:
        from kicad2ems.importers import open_board           # (as pcbqs._cli reads a board)
        _PADS[board] = (stamp, {fp['ref']: {p.get('net') for p in fp.get('pads', [])} for fp in open_board(board).footprints})
    return _PADS[board][1]


def _floating_loads(values, board):
    """The loads typed as REF=AMPS (no @RETURN) that are on the board with no pad on a ground net."""
    if not board or not str(board).lower().endswith('.kicad_pcb') or not os.path.isfile(board):
        return []
    ground = {x.strip() for x in str(values.get('ground') or 'GND').split(',') if x.strip()}
    loads = values.get('load') or []
    loads = loads.splitlines() if isinstance(loads, str) else loads       # (a form's list: one a line)
    nets = _pad_nets(board)
    out = []
    for item in loads:
        ref = str(item).partition('=')[0].strip()
        if '@' not in str(item) and ref in nets and not nets[ref] & ground:
            out.append(ref)
    return out


def _return_of(ref, values, board):
    """The part a load with no pad on ground can return through (pdn.rail.return_part), or None."""
    from .pdn.rail import return_part
    ground = {x.strip() for x in str(values.get('ground') or 'GND').split(',') if x.strip()}
    return return_part(_pad_nets(board), ref, ground)


class PowerIntegrity(_Tool):
    name, label, tool, module, order = 'pdn', 'Power integrity', 'pcbqs.pdn', 'pcbqs.pdn', 20
    about = "a rail's DC drop and the impedance its loads see"
    result = 'pdn.json'
    provides = ('dc', 'pdn')                # one run: the drop alone when the impedance is not asked for
    several = 'rail'                        # a study may hold several rails, a run each (kicad2ems 2.10)
    several_keys = ('rail', 'volts', 'source', 'ground_source', 'load', 'load_spectrum')    # what differs by rail

    def for_questions(self, values, asked):
        return values if 'pdn' in asked else dict(values, no_pdn=True)

    def figures(self, folder):
        """Each load's drop in the rail's own copper (what an engine of that copper alone gives too), then this
        tool's own: the drop with the ground's rise, and the loss of both."""
        r = self._read(folder)
        loads = {ref: l for ref, l in (r.get('loads') or {}).items() if isinstance(l, dict)}
        out = [dict(name=f"drop {ref}", value=l['power_drop'] * 1e3, unit='mV') for ref, l in loads.items()
               if isinstance(l.get('power_drop'), (int, float))]
        out += [dict(name=f"drop with return {ref}", value=l['drop'] * 1e3, unit='mV') for ref, l in loads.items()
                if isinstance(l.get('drop'), (int, float))]
        if isinstance(r.get('loss_W'), (int, float)):
            out.append(dict(name='loss with return', value=r['loss_W'] * 1e3, unit='mW'))
        return out

    def table(self):
        from .pdn.cli import OPTIONS
        return OPTIONS

    def tag(self, values):
        return str(values.get('rail') or 'rail').strip()

    def command(self, values, board, out, preview=False, results=None):
        board = self._board(board)
        if not str(values.get('rail') or '').strip():
            raise ValueError('name the rail (its net)')
        return argv(self.module, self.options(), values, board, out)

    def missing(self, values, board, results=None):
        """What command() raises, and a load with no pad on ground whose return is not named and that the rail
        cannot find by itself (pdn.rail.return_part: a bridge's high-side switch returns through its low-side
        one): its return would be left out of the drop."""
        out = super().missing(values, board, results) if hasattr(super(), 'missing') else []
        try:
            path = self._board(board)
            out += [dict(key='load', text=f"{ref} has no pad on ground, and the board has no one part of its kind "
                         f"to return through: name it, {ref}=AMPS@PART")
                    for ref in _floating_loads(values, path) if not _return_of(ref, values, path)]
        except Exception:                                    # noqa: BLE001 (the run says what is wrong with a board)
            pass
        return out

    def status(self, line):
        return line[4:].strip()[:160] if line.startswith('pdn:') else None


class NearField(_Tool):
    name, label, tool, module, order = 'mqs', 'Near field', 'pcbqs.mqs', 'pcbqs.mqs', 30
    about = "a converter's input loop and its magnetic near field"
    previewed = True
    finds = 'converters'
    result, preview_result = 'mqs.json', 'mqs-preview.json'
    provides = ('nearfield',)
    several = 'converter'                   # and several converters
    pick = dict(ground='net')               # (one ground net: the loop's return)

    def table(self):
        from .mqs_cli import OPTIONS
        return OPTIONS

    def tag(self, values):
        return str(values.get('converter') or 'auto').strip()

    def command(self, values, board, out, preview=False, results=None):
        return argv(self.module, self.options(), values, self._board(board), out) + (['--preview'] if preview else [])

    def status(self, line):
        return line[4:].strip()[:160] if line.startswith('mqs:') else None

    def discover(self, values, board):
        return [sys.executable, '-m', self.module, self._board(board), '--list-converters', '--ground',
                str(values.get('ground') or 'GND')]

    def choices(self, text):
        """One choice per converter and input net: a regulator with several candidate inputs (VCC, VDD, VDRV on a
        buck) would otherwise stop the run."""
        try:
            found = json.loads(text[text.index('['):])
        except ValueError:
            return []
        out = []
        for c in found:
            for i in c.get('inputs', []):
                caps = i.get('capacitors', [])
                part = f"{c['converter']} and {c['low']}" if c.get('low') else c['converter']     # a discrete cell's two
                what = f"inductor {c['inductor']}, " if c.get('inductor') else ''
                label = (f"{part}: {i['vin']} ({what}switch node {c.get('switch', '?')}; "
                         f"{len(caps)} capacitor{'s' if len(caps) != 1 else ''}: {', '.join(caps[:6])}"
                         f"{'…' if len(caps) > 6 else ''})")
                out.append((label, dict(converter=c['converter'], vin=i['vin'], capacitors=','.join(caps))))
        return out

    def suggest(self, folder):
        from . import reports
        try:
            res = reports.result(self.name, folder)
            return reports.mqs_suggest(res, folder) if res else []
        except (OSError, ValueError, KeyError, TypeError):
            return []


def circuit_run(folder):
    """A general circuit run's folder (model.json and circuit.json): the board its model names, '' when it names
    none or it is gone; None: not such a folder."""
    folder = str(folder or '')
    if not folder or not all(os.path.isfile(os.path.join(folder, f)) for f in ('model.json', 'circuit.json')):
        return None
    try:
        with open(os.path.join(folder, 'model.json'), encoding='utf-8') as fh:
            board = (json.load(fh).get('board') or {}).get('file') or ''
    except (OSError, ValueError):
        board = ''
    return board if board and os.path.isfile(board) else ''


# the circuit examples' groups in a front end's menu: their configs (<path>.toml in this package), in order
EX = 'circuit/examples/'
EXAMPLE_GROUPS = (
    ('Circuits/Passive and rectifiers', (EX + '01_rc_filter', EX + '02_half_wave_rectifier', EX + '03_bridge_rectifier')),
    ('Circuits/DC-DC converters', (EX + '04_buck', EX + '05_buck_synchronous', EX + '06_boost', EX + '07_buck_boost',
                                   EX + '08_flyback')),
    ('Circuits/Bridges', (EX + '09_half_bridge_resonant', EX + '10_full_bridge_phase_shift', 'data/wpthb_bridge')),
)


class Circuit(_Tool):
    name, label, tool, module, order = 'circuit', 'Circuit', 'pcbqs.circuit', 'pcbqs.circuit', 40
    about = ("a circuit model run until it repeats, with its targets met: any circuit of parts, nodes and drives, "
             "by ngspice (with the board's copper between the parts' pads when asked) or, for ideal switches, "
             "the piecewise-linear steady state")
    result = 'circuit.json'
    provides = ('circuit',)
    fields = 'fields'                                   # the option that adds the H and E fields over the board
    paths = ('model',)
    page = True                                         # the GUI's schematic editor
    env = dict(OPENBLAS_NUM_THREADS='8')

    def owns(self, folder, preview=False):
        """A run of either tool: pcbqs.circuit's circuit.json, the full bridge's power-circuit.json."""
        from .reports import ALSO
        return not preview and any(os.path.isfile(os.path.join(folder or '', f))
                                   for f in (self.result, *ALSO['circuit']))

    JOBS = 4                                            # cases run at once, unless set

    def options(self):
        """The model, then the options of the two tools that run one (their own tables: python -m pcbqs.circuit
        --describe, python -m pcbqs.power_circuit --describe). `only` says which kind of model an option is for:
        'general' (pcbqs.circuit's, format 2), 'bridge' (the full bridge's, format 1), '' for either. The board is
        a front end's to give."""
        from ._cli import describe
        from .circuit.__main__ import OPTIONS as general
        from .power_circuit import OPTIONS as bridge
        out = [dict(o, only='') for o in describe(
            [('model', 'str', '', 'The circuit model (JSON); a path beside the config', 'Circuit')])['options']]
        at = {'model': out[0]}
        for table, only in ((general, 'general'), (bridge, 'bridge')):
            for o in describe(table)['options']:
                if o['key'] == 'board':
                    continue
                if o['key'] in at:                              # both tools have it: for either kind
                    at[o['key']]['only'] = ''
                    continue
                at[o['key']] = dict(o, only=only, default=self.JOBS if o['key'] == 'jobs' else o['default'])
                out.append(at[o['key']])
        order = ['Circuit', 'Board', 'Operating points', 'Switches', 'Overrides', 'Model', 'Run']
        return sorted(out, key=lambda o: order.index(o['group']) if o['group'] in order else len(order))

    def tag(self, values):
        return os.path.splitext(os.path.basename(str(values.get('model') or 'circuit')))[0]

    def command(self, values, board, out, preview=False, results=None):
        model = str(values.get('model') or '').strip()
        if not model or not os.path.isfile(model):
            raise ValueError('name the circuit model (a JSON file)' if not model else f'{model}: no such model')
        with open(model, encoding='utf-8') as fh:
            kind = 'general' if json.load(fh).get('format') == 2 else 'bridge'
        on = lambda k: values.get(k) is True or str(values.get(k)).strip().lower() in ('1', 'true', 'yes', 'on')
        values = dict(values, copper=on('copper') or on('fields'), jobs=values.get('jobs') or self.JOBS)
        opts = [dict(o, default=None) if o['key'] == 'jobs' else o for o in self.options()
                if o['key'] != 'model' and o['only'] in ('', kind)]
        cmd = argv('pcbqs.circuit' if kind == 'general' else 'pcbqs.power_circuit', opts,
                   {k: v for k, v in values.items() if k != 'model'}, model, out)
        cmd.insert(1, '-u')
        if values['copper']:
            cmd += ['--board', self._board(board)]
        return cmd

    def suggest(self, folder):
        """What the run draws from its supply, for the power-integrity and near-field analyses (reports.
        circuit_suggest); only for parts that are on the model's board."""
        from . import reports
        try:
            res = reports.result(self.name, folder)
            if not res:
                return []
            board = circuit_run(folder)
            pads = _pad_nets(board) if board and str(board).lower().endswith('.kicad_pcb') else None
            return reports.circuit_suggest(res, folder, pads)
        except Exception:                                    # noqa: BLE001 (a proposal, never a fault)
            return []

    def examples(self):
        """The example circuits as configs: [dict(name, config, group, title, files)]; a config's model (and its
        board) are its files, copied beside it."""
        root = os.path.dirname(os.path.abspath(__file__))
        out = []
        for group, paths in EXAMPLE_GROUPS:
            for path in paths:
                cfg = os.path.join(root, *path.split('/')) + '.toml'
                if not os.path.isfile(cfg):
                    continue
                here, stem = os.path.dirname(cfg), os.path.basename(path)
                text = open(cfg, encoding='utf-8').read()
                files = re.findall(r'^\s*(?:board|model)\s*=\s*"([^"]+)"', text, re.M)
                title = ''
                for f in files:
                    if f.endswith('.json') and os.path.isfile(os.path.join(here, f)):
                        with open(os.path.join(here, f), encoding='utf-8') as fh:
                            m = json.load(fh)
                        title = (m.get('name') or m.get('description') or '') if m.get('format') == 2 else ''
                name = 'circuit_' + (stem.split('_', 1)[1] if stem[:2].isdigit() else stem)
                out.append(dict(name=name, config=cfg, group=group, title=title, files=files))
        return out


class Thermal(_Tool):
    name, label, tool, module, order = 'thermal', 'Thermal', 'pcbqs.thermal', 'pcbqs.thermal', 50
    about = ("the board's steady temperature from its parts' losses: typed (REF=W), or the latest circuit run's "
             "when none are; each part's temperature, and each layer's map in the viewer")
    result = 'thermal.json'
    provides = ('thermal',)
    paths = ('circuit',)
    uses = ('circuit',)                     # its losses, when none are typed (kicad2ems 2.9: Analysis.uses)

    def owns(self, folder, preview=False):
        """A thermal.json of pcbqs's: another engine's (Elmer's, of the same name) says whose it is."""
        if preview or not super().owns(folder):
            return False
        try:
            with open(os.path.join(folder, self.result), encoding='utf-8') as fh:
                return json.load(fh).get('tool', self.tool) == self.tool
        except (OSError, ValueError, AttributeError):
            return False

    def figures(self, folder):
        """The copper's hottest point and the board under each part (the grid's, where both solvers ran)."""
        from . import reports
        try:
            ms = reports._methods(self._read(folder))
            m = ms.get('grid') or next(iter(ms.values()))
            return [dict(name='hottest', value=float(m['hottest_C']), unit='C')] + \
                   [dict(name=ref, value=float(p['board_C']), unit='C') for ref, p in m['parts'].items()]
        except (KeyError, TypeError, ValueError, StopIteration):
            return []

    def table(self):
        from .thermal.__main__ import OPTIONS
        return OPTIONS

    def _plan(self, values, board, results):
        """(values, board) of a run: the losses typed, else a circuit run's (the one named, else the latest in
        `results`); the board that run's model names (its losses are that board's parts'), else the one given."""
        values = dict(values)
        typed = values.get('loss')
        typed = [s for s in (typed if isinstance(typed, (list, tuple)) else str(typed or '').replace(',', '\n').splitlines())
                 if str(s).strip()]
        circuit = str(values.get('circuit') or '').strip()
        if circuit and circuit_run(circuit) is None:
            raise ValueError(f'{circuit}: not a circuit run (model.json and circuit.json)')
        if not circuit and not typed:
            circuit = str((results or {}).get('circuit') or '')
            if circuit_run(circuit) is None:
                raise ValueError("give the parts' losses (REF=W), or run the Circuit analysis first")
            values['circuit'] = circuit
        board = (circuit_run(circuit) if circuit else '') or board
        return values, self._board(board)

    def tag(self, values):
        """The circuit case named, else where the losses are from: typed ('losses'), or a circuit run ('circuit')."""
        typed = values.get('loss')
        typed = typed if isinstance(typed, (list, tuple)) else str(typed or '').strip()
        return str(values.get('case') or '').strip() or ('circuit' if values.get('circuit') or not typed else 'losses')

    def command(self, values, board, out, preview=False, results=None):
        values, board = self._plan(values, board, results)
        return argv(self.module, self.options(), values, board, out)

    def status(self, line):
        keep = line.startswith(('thermal ', 'losses of', 'results in')) or ' W into the board' in line
        return line.strip()[:160] if keep else None


# ---- a solver: S-parameters at low frequency, from the copper's quasi-static currents (sparams.py) ----
from kicad2ems.api.solver import Deviation, Option, Result, Solver, mixed_mode   # noqa: E402

G_QS = 'pcbqs'


class QuasiStatic(Solver):
    """pcbqs as an S-parameter solver of kicad2ems (`--solvers pcbqs`): the board's ports from the resistance and
    inductance of its copper, with the return current where it flows, and the copper's charge too where the band
    asks for it (R, L and C). For the low end of a band, to about a tenth of a wavelength, where the full-wave
    solvers are slow or lose accuracy; it is never picked by `--solvers auto`."""
    api = API
    name, aliases, label = 'pcbqs', ('qs', 'quasistatic'), 'pcbqs (quasi-static, low frequency)'
    in_auto = in_all = False                            # asked for by name: a low band's solver, not a full-wave one
    basic = ('qs_dmax',)                                # what a run is set up with (kicad2ems 2.9: Solver.basic)
    consumes, previewed, rank, order = 'board', False, 1, 60
    dumps = False                                       # its current maps are always written: nothing to ask for
    POINTS = 31                                         # solved on at most this many frequencies, a log scale
    capabilities = dict(
        ports='yes: a source and its termination between the trace and the copper under it', gap_ports='no',
        diff_pairs="yes: the legs as single-ended ports, the modes from them",
        parts='yes: a series R, L and C between their pads', vias='yes: filaments between the layers they join',
        outline='ignored: the copper alone', planes='yes: meshed, the return current where it flows',
        edge_walls='ignored', thickness="approximated: one sheet; its two faces with --qs-thickness",
        conductor_loss="yes: the copper's DC sheet resistance; the skin effect through it with --qs-thickness",
        roughness='no', finish='no', dispersion='no: constant er and tan d (with the capacitance); none without', radiation='no',
        cavity='no', antenna='no')
    options = [
        Option('qs_dmax', None, "pcbqs: the largest mesh cell, mm (default: a twenty-fifth of the board, 0.5 to 3)",
               float, name='dmax', gui='Largest cell (mm)', group=G_QS),
        Option('qs_fine', None, "pcbqs: the cell at and around the ports' nets, mm (default: the narrowest port's "
               "width)", float, name='fine', gui="Cell at the ports' nets (mm)", group=G_QS),
        Option('qs_capacitance', 'auto', "pcbqs: the copper's charge with its currents (R, L and C): on, off, or "
               "auto (when the band reaches above where resistance and inductance alone hold)",
               choices=('auto', 'on', 'off'), name='capacitance', gui="Copper's capacitance", group=G_QS),
        Option('qs_thickness', False, "pcbqs: the copper's thickness, as its two faces with the skin effect between "
               "them, instead of one sheet with its DC resistance (twice the unknowns)", bool, name='thickness',
               gui="Copper's thickness", group=G_QS),
    ]

    def check(self, s=None):
        from . import __version__
        return True, f"pcbqs {__version__} ({os.path.dirname(os.path.abspath(__file__))})"

    def assess(self, run):
        why = []
        if run.antenna:
            why.append('antenna mode')
        gaps = [p.name for p in run.m.ports if getattr(p, 'kind', 'z') != 'z']
        if gaps:
            why.append(f"gap ports ({', '.join(gaps)}): only ports between a trace and the copper under it")
        if run.freq[0] <= 0:
            why.append('fmin has to be above 0')
        return why

    def solve(self, run, prep):
        import numpy as np
        from . import sparams
        board = run.m.to_dict()
        f = np.unique(np.geomspace(run.freq[0], run.freq[-1], min(len(run.freq), self.POINTS)))
        r = sparams.solve(board, f, dmax=getattr(run.s, 'qs_dmax', None), fine=getattr(run.s, 'qs_fine', None),
                          capacitance=getattr(run.s, 'qs_capacitance', None) or 'auto',
                          thickness=bool(getattr(run.s, 'qs_thickness', False)), log=lambda s: run.log(f"  {s}"))
        lf, want = np.log(f), np.log(run.freq)              # smooth in log f: onto the run's own frequencies
        S = np.empty((len(run.freq),) + r['S'].shape[1:], complex)
        for j in range(S.shape[1]):
            for k in range(S.shape[2]):
                S[:, j, k] = np.interp(want, lf, r['S'][:, j, k].real) + 1j * np.interp(want, lf, r['S'][:, j, k].imag)
        self._last = (board, r)
        info = r['info']
        over = int((run.freq > info['valid_to_Hz']).sum())
        apart = [sparams.apart_note(p) for p in sparams.paths(f, r['S'], r['names'], r['zref']) if not p['joined']]
        if over:
            run.log(f"  pcbqs: {over} of {len(run.freq)} frequencies are above about {info['valid_to_Hz'] / 1e6:.3g} MHz, "
                    + ("where the ports' copper is over a tenth of a wavelength long: quasi-static, no radiation"
                       if info['capacitance'] else "where the copper's capacitance counts: resistance and inductance only"))
        for note in apart:
            run.log(f"  pcbqs: {note}")
        Smm, names, zref = mixed_mode(S, run.m)
        about = ("The ports from pcbqs's quasi-static solve of the copper: every layer's sheet current, the vias and "
                 "their mutual inductance, the return current where it flows"
                 + (", and the copper's charge in the board's dielectrics (R, L and C). No radiation: "
                    if info['capacitance'] else ". Resistance and inductance alone, no capacitance and no radiation: ")
                 + f"valid to about {info['valid_to_Hz'] / 1e6:.3g} MHz on this board.")
        return Result(Smm, names, zref, info['seconds'], comment=f"pcbqs quasi-static ({run.name})",
                      info=dict({k: v for k, v in info.items() if k != 'model'},
                                kpi=f"{info['triangles']:,} triangles, {info['seconds']:.1f} s", about=about,
                                not_joined=apart))

    def outputs(self, run, prep, res):
        """The current of each layer with the first port driven, at three frequencies of the band, for the viewer."""
        import numpy as np
        from . import sparams
        board, r = getattr(self, '_last', (None, None))
        if r is None:
            return
        try:
            got = sparams.write_maps(run.s.out, board, r, r['names'][0], np.geomspace(r['f'][0], r['f'][-1], 3),
                                     name='pcbqs', log=lambda s: run.log(f"  {s}"))
            run.wrote('pcbqs: the current of each layer', os.path.join(run.s.out, got['dumps'][0]['file']))
        except Exception as exc:                            # noqa: BLE001 (the maps are extra: the answer stands)
            run.log(f"  pcbqs: current maps not written: {exc}")

    def deviations(self, run, prep):
        from . import sparams
        return [Deviation(f"part:{ref}", 'parts', f"left out: {why}", dropped=True)
                for ref, why in sparams.part_elements(run.m.to_dict())[1]]

