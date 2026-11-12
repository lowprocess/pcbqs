"""
pcbqs's tools as kicad2ems analysis plugins (the 'kicad2ems.analyses' entry points of pyproject.toml): power
integrity, near field, circuit and thermal. kicad2ems lists them, a front end builds each one's settings from its
options table, runs it as its command and shows what its result folder holds (reports.py); the circuit examples
come as configs beside their models (circuit/examples/*.toml).

Only this module imports kicad2ems; the tools themselves run without it.

    from pcbqs.analyses import PowerIntegrity
    PowerIntegrity().command(dict(rail='+3V3', load='U1=2'), 'board.kicad_pcb', 'out/pdn')
"""
import json
import os
import re
import sys

from kicad2ems.api.analysis import Analysis

API = (2, 5)


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


class _Tool(Analysis):
    """One of pcbqs's command-line tools: its options table (pcbqs._cli's form), its command, its results."""
    api = API
    module = ''

    def check(self):
        from . import __version__
        return True, f"pcbqs {__version__} ({os.path.dirname(os.path.abspath(__file__))})"

    def table(self):
        return []

    def options(self):
        from ._cli import describe
        return describe(self.table())['options']

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


class PowerIntegrity(_Tool):
    name, label, tool, module, order = 'pdn', 'Power integrity', 'pcbqs.pdn', 'pcbqs.pdn', 20
    about = "a rail's DC drop and the impedance its loads see"
    result = 'pdn.json'

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

    def status(self, line):
        return line[4:].strip()[:160] if line.startswith('pdn:') else None


class NearField(_Tool):
    name, label, tool, module, order = 'mqs', 'Near field', 'pcbqs.mqs', 'pcbqs.mqs', 30
    about = "a converter's input loop and its magnetic near field"
    previewed = True
    finds = 'converters'
    result, preview_result = 'mqs.json', 'mqs-preview.json'

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
                label = (f"{c['converter']}: {i['vin']} (inductor {c.get('inductor', '?')}, switch {c.get('switch', '?')}; "
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
    about = ("ngspice runs of a circuit model: any circuit of parts, nodes and drives, settled, with its targets "
             "met; with the board's copper between the parts' pads when asked")
    result = 'circuit.json'
    paths = ('model',)
    page = True                                         # the GUI's schematic editor
    env = dict(OPENBLAS_NUM_THREADS='8')

    def table(self):
        return [('model', 'str', '', 'The circuit model (JSON); a path beside the config', 'Circuit'),
                ('copper', 'bool', False, "With the board's copper between the parts' pads", 'Circuit'),
                ('fields', 'bool', False, 'With the copper, and the H and E fields over the board', 'Circuit'),
                ('jobs', 'int', 4, 'Sweep points run at once', 'Circuit')]

    def tag(self, values):
        return os.path.splitext(os.path.basename(str(values.get('model') or 'circuit')))[0]

    def command(self, values, board, out, preview=False, results=None):
        model = str(values.get('model') or '').strip()
        if not model or not os.path.isfile(model):
            raise ValueError('name the circuit model (a JSON file)' if not model else f'{model}: no such model')
        with open(model, encoding='utf-8') as fh:
            general = json.load(fh).get('format') == 2
        cmd = [sys.executable, '-u', '-m', 'pcbqs.circuit' if general else 'pcbqs.power_circuit', model, '--out', out,
               '--jobs', str(values.get('jobs') or 4)]
        on = lambda k: values.get(k) is True or str(values.get(k)).strip().lower() in ('1', 'true', 'yes', 'on')
        if on('copper') or on('fields'):
            cmd += ['--copper', '--board', self._board(board)]
        if on('fields'):
            cmd += ['--fields']
        return cmd

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
    paths = ('circuit',)

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
