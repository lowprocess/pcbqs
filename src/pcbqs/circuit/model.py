"""
The circuit model: its format, the values that are expressions, the sweep.

    model = load(path)                  # read and validate
    validate(model)                     # ModelError naming what is wrong and where
    for point in cases(model):          # the sweep's points: {path: value}
        concrete = resolve(model, point)        # numbers everywhere: parameters and expressions worked out
    get(model, 'drives.pwm.duty'), put(model, 'R1.value', 4.7)      # a number of the model by its path

format 2. A value may be a number, engineering notation ('22u', '4.7k', '100n') or an expression over the model's
parameters ('1/(2*pi*fsw*sqrt(L*C))'): + - * / **, parentheses, pi, sqrt, abs, min, max, exp, log.

    {"format": 2, "name": "...", "description": "...",
     "parameters": {"fsw": 200e3},
     "parts": [{"ref": "L1", "type": "inductor", "nodes": ["in", "sw"], "value": "22u", "DCR_ohm": 0.03}, ...],
     "couplings": [{"inductors": ["L1", "L2"], "k": 0.98}],
     "drives": {"pwm": {"frequency_Hz": "fsw", "duty": 0.5, "phase_deg": 0, "dead_time_s": 0}},
     "probes": {"vout": {"voltage": ["out", "0"]}, "il": {"current": "L1"}},
     "measures": {"vout_avg": {"avg": "vout"}},
     "targets": [{"adjust": "drives.pwm.duty", "range": [0.05, 0.95], "measure": "vout_avg", "value": 24}],
     "sweep": {"R1.power_W": [6, 12, 24]},
     "analysis": {"cycles": 200}}

Parts, by type (nodes: two names; '0' or 'GND' is ground; the first is where a part's current is taken as
entering it, so its voltage times its current is the power it absorbs):
    resistor    value
    capacitor   value, ESR_ohm, ESL_H, leakage_ohm, bias_factor
    inductor    value, DCR_ohm, ESL_H; or, with SRF_Hz / Cp_F / core_R_ohm, a winding with its self-capacitance
                and core loss
    diode       nodes anode, cathode; model: Vf_V at If_A (0.7 V at 1 A), n, R_ohm, Cj_F, TT_s
    switch      nodes drain, source; drive (a drive's name), invert (its complement); model: as power_spice's
                switches (type switch | ideal | mosfet | vdmos | subcircuit, Ron_ohm, ...)
    source      nodes +, -; dc_V, or wave: {type: pulse | sine | pwl, ...}; or dc_A (a current source); R_ohm, L_H
                in series
    load        resistance_ohm, or power_W (constant power), or current_A (constant current); below min_V it is
                a resistor, so it starts
A path names a number: 'parameters.fsw', 'drives.pwm.duty', '<ref>.<field>' or 'parts.<ref>.<field>',
'<ref>.model.<field>', 'couplings.0.k', 'analysis.cycles'.
"""
import ast
import copy
import itertools
import json
import math
import re
from pathlib import Path

FORMAT = 2
PART_TYPES = ('resistor', 'capacitor', 'inductor', 'diode', 'switch', 'source', 'load')
STATS = ('avg', 'rms', 'pp', 'max', 'min')
WAVES = ('pulse', 'sine', 'pwl')
TEXT_KEYS = {'ref', 'type', 'nodes', 'drive', 'name', 'description', 'library', 'subcircuit', 'pin_order', 'adjust',
             'measure', 'notes', 'method', 'loss_model', 'initialization', 'ignore', 'inductors', 'enabled', 'invert',
             'body_diode', 'external_capacitance', 'linear_solver', 'strict'}
SUFFIX = dict(f=1e-15, p=1e-12, n=1e-9, u=1e-6, m=1e-3, k=1e3, K=1e3, M=1e6, G=1e9)
FUNCTIONS = dict(sqrt=math.sqrt, abs=abs, min=min, max=max, exp=math.exp, log=math.log)
REF = re.compile(r'[A-Za-z][A-Za-z0-9_]*$')


class ModelError(ValueError):
    """A model that cannot be run: says what and where."""


def number(value, names=None, where='value'):
    """A value as a float: a number, engineering notation, or an expression over `names`."""
    if isinstance(value, bool):
        raise ModelError(f"{where}: expected a number, got {value!r}")
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        text = str(value).strip().replace('μ', 'u').replace('µ', 'u')
        m = re.fullmatch(r'([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([fpnumkKMG])?', text)
        try:
            out = float(m[1]) * SUFFIX.get(m[2], 1.0) if m else _evaluate(ast.parse(text, mode='eval').body, names or {})
        except ModelError:
            raise
        except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError, KeyError) as e:
            raise ModelError(f"{where}: cannot work out {value!r} ({e})") from None
    if not math.isfinite(out):
        raise ModelError(f"{where}: {value!r} is not a finite number")
    return out


def _evaluate(node, names):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id == 'pi':
            return math.pi
        if node.id not in names:
            raise ModelError(f"unknown name {node.id!r} (parameters: {', '.join(sorted(names)) or 'none'})")
        return float(names[node.id])
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _evaluate(node.operand, names)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        a, b = _evaluate(node.left, names), _evaluate(node.right, names)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if isinstance(node.op, ast.Div):
            return a / b
        return a ** b
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCTIONS and not node.keywords:
        return float(FUNCTIONS[node.func.id](*[_evaluate(a, names) for a in node.args]))
    raise ModelError('only numbers, parameters, + - * / **, pi and ' + ', '.join(FUNCTIONS) + ' are allowed')


# ---- paths ----
def _walk(model, path):
    """The container and key a path names: (dict or list, key)."""
    keys = path.split('.')
    if keys[0] not in ('parameters', 'drives', 'parts', 'couplings', 'analysis', 'targets', 'probes', 'measures'):
        keys = ['parts'] + keys                                     # '<ref>.<field>'
    at = model
    for depth, key in enumerate(keys[:-1]):
        if at is model.get('parts') and depth == 1:
            found = next((p for p in at if p.get('ref') == key), None)
            if found is None:
                raise ModelError(f"{path}: no part {key!r}")
            at = found
        elif isinstance(at, list):
            if not key.isdigit() or int(key) >= len(at):
                raise ModelError(f"{path}: no item {key!r}")
            at = at[int(key)]
        elif isinstance(at, dict):
            if key not in at:
                if key in ('model',):                               # a part's model: made when first written to
                    at[key] = {}
                else:
                    raise ModelError(f"{path}: no {key!r}")
            at = at[key]
        else:
            raise ModelError(f"{path}: {key!r} is not a section")
    return at, (int(keys[-1]) if isinstance(at, list) else keys[-1])


def get(model, path):
    at, key = _walk(model, path)
    try:
        return at[key]
    except (KeyError, IndexError):
        raise ModelError(f"{path}: not in the model") from None


def put(model, path, value):
    at, key = _walk(model, path)
    at[key] = value


# ---- validation ----
def _need(cond, message):
    if not cond:
        raise ModelError(message)


def validate(model):
    """Check a model's shape: every part has a known type, two nodes and what its type needs; every drive, probe,
    measure, target, coupling and sweep path refers to something that exists. Values are checked when resolved
    (resolve), since a sweep may set them. Returns the model."""
    _need(isinstance(model, dict), 'the model must be an object')
    _need(model.get('format') == FORMAT, f"format must be {FORMAT} (a format 1 model is the full-bridge tool's: "
          'python -m pcbqs.power_circuit)')
    parts = model.get('parts')
    _need(isinstance(parts, list) and parts, 'parts: a list of at least one part')
    refs = {}
    for i, p in enumerate(parts):
        _need(isinstance(p, dict), f"parts[{i}]: must be an object")
        ref = p.get('ref')
        _need(isinstance(ref, str) and REF.match(ref), f"parts[{i}]: ref must be letters, digits and _ (got {ref!r})")
        _need(ref.lower() not in refs, f"{ref}: the reference is used twice")
        refs[ref.lower()] = p
        _need(p.get('type') in PART_TYPES, f"{ref}: type must be one of {', '.join(PART_TYPES)} (got {p.get('type')!r})")
        nodes = p.get('nodes')
        _need(isinstance(nodes, list) and len(nodes) == 2 and all(isinstance(n, str) and n for n in nodes),
              f"{ref}: nodes must be two node names")
        _need(nodes[0] != nodes[1] or p['type'] == 'source', f"{ref}: both nodes are {nodes[0]!r}")
        kind = p['type']
        if kind in ('resistor', 'capacitor', 'inductor'):
            _need('value' in p, f"{ref}: a {kind} needs a value")
        elif kind == 'switch':
            _need(p.get('drive') in model.get('drives', {}), f"{ref}: drive must name one of the drives "
                  f"({', '.join(model.get('drives', {})) or 'none defined'})")
        elif kind == 'source':
            given = [k for k in ('dc_V', 'dc_A', 'wave') if k in p]
            _need(len(given) == 1, f"{ref}: a source needs one of dc_V, dc_A or wave")
            if 'wave' in p:
                _need(isinstance(p['wave'], dict) and p['wave'].get('type') in WAVES,
                      f"{ref}: wave.type must be one of {', '.join(WAVES)}")
        elif kind == 'load':
            given = [k for k in ('resistance_ohm', 'power_W', 'current_A') if k in p]
            _need(len(given) == 1, f"{ref}: a load needs one of resistance_ohm, power_W or current_A")
    for name, d in model.get('drives', {}).items():
        _need(isinstance(d, dict) and 'frequency_Hz' in d, f"drives.{name}: needs frequency_Hz")
    for i, c in enumerate(model.get('couplings', [])):
        ls = c.get('inductors') if isinstance(c, dict) else None
        _need(isinstance(ls, list) and len(ls) == 2 and 'k' in c, f"couplings[{i}]: needs inductors (two) and k")
        for r in ls:
            _need(str(r).lower() in refs and refs[str(r).lower()]['type'] == 'inductor',
                  f"couplings[{i}]: {r!r} is not an inductor of the model")
    signals = signal_names(model)
    for name, pr in model.get('probes', {}).items():
        _need(REF.match(name), f"probes.{name}: a name of letters, digits and _")
        _need(isinstance(pr, dict) and len(pr) == 1 and next(iter(pr)) in ('voltage', 'current'),
              f"probes.{name}: one of voltage: [a, b] or current: <ref>")
        if 'voltage' in pr:
            v = pr['voltage']
            _need(isinstance(v, list) and len(v) == 2, f"probes.{name}: voltage needs two nodes")
            known = {n for p in parts for n in p['nodes']} | {'0', 'GND', 'gnd'}
            for n in v:
                _need(n in known, f"probes.{name}: no part is on node {n!r}")
        else:
            _need(str(pr['current']).lower() in refs, f"probes.{name}: no part {pr['current']!r}")
    for name, ms in model.get('measures', {}).items():
        _need(isinstance(ms, dict) and len(ms) == 1 and next(iter(ms)) in STATS,
              f"measures.{name}: one of {', '.join(STATS)}: <expression of probes>")
        for token in re.findall(r'[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_]+)?', str(next(iter(ms.values())))):
            _need(token in signals or token in FUNCTIONS or token == 'pi',
                  f"measures.{name}: {token!r} is not a probe or a part's signal (<ref>.V, <ref>.I)")
    for i, t in enumerate(model.get('targets', [])):
        _need(isinstance(t, dict) and all(k in t for k in ('adjust', 'range', 'measure', 'value')),
              f"targets[{i}]: needs adjust, range, measure and value")
        _need(t['measure'] in model.get('measures', {}) or t['measure'] in AUTOMATIC,
              f"targets[{i}]: measure {t['measure']!r} is not one of the model's measures or "
              f"{', '.join(AUTOMATIC)}")
        _need(isinstance(t['range'], list) and len(t['range']) == 2, f"targets[{i}]: range is [low, high]")
        get(model, t['adjust'])
    board = model.get('board')
    if board is not None:
        _need(isinstance(board, dict), 'board: an object (file, parts, nets, reference, mesh)')
        _need(not board.get('file') or isinstance(board['file'], str), 'board.file: a path')
        for key in ('parts', 'nets', 'reference', 'mesh'):
            _need(isinstance(board.get(key, {}), dict), f"board.{key}: an object")
        for ref in board.get('parts', {}):
            _need(str(ref).lower() in refs, f"board.parts: {ref!r} is not a part of the model")
    for path, values in model.get('sweep', {}).items():
        _need(isinstance(values, list) and values, f"sweep.{path}: a list of values")
        if not path.startswith('targets.'):
            try:
                get(model, path)
            except ModelError:
                at, key = _walk(model, path)                        # a field the part does not have yet: allowed
                _need(isinstance(at, dict), f"sweep.{path}: not in the model")
    return model


AUTOMATIC = ('source_power_W', 'load_power_W', 'efficiency', 'loss_W')        # measures every run has


def signal_names(model):
    """The signals a measure may use: the probes' names and every part's <ref>.V and <ref>.I."""
    out = set(model.get('probes', {}))
    for p in model.get('parts', []):
        if isinstance(p, dict) and isinstance(p.get('ref'), str):
            out |= {p['ref'] + '.V', p['ref'] + '.I'}
    return out


# ---- the sweep and the numbers ----
def cases(model):
    """The sweep's points, every combination in the order given: [{path: value}] ([{}] with no sweep)."""
    sweep = model.get('sweep', {})
    if not sweep:
        return [{}]
    paths = list(sweep)
    return [dict(zip(paths, combo)) for combo in itertools.product(*[sweep[p] for p in paths])]


def case_name(point):
    """A folder name for a sweep point: 'R1-power-W=12_fsw=200k' ('run' with no sweep); a parameter by its name,
    anything else by its path."""
    def short(v):
        if isinstance(v, (int, float)):
            for s, m in (('G', 1e9), ('M', 1e6), ('k', 1e3)):
                if abs(v) >= m:
                    return f"{v / m:g}{s}"
            return f"{v:g}"
        return re.sub(r'[^A-Za-z0-9.+-]+', '', str(v))
    return '_'.join(f"{re.sub(r'[^A-Za-z0-9]+', '-', path.removeprefix('parameters.'))}={short(v)}"
                    for path, v in point.items()) or 'run'


def _numbers(obj, names, where):
    """Every value of a section as a number, in place; text fields (TEXT_KEYS) and booleans are left."""
    items = obj.items() if isinstance(obj, dict) else enumerate(obj)
    for key, v in list(items):
        if isinstance(obj, dict) and key in TEXT_KEYS:
            continue
        here = f"{where}.{key}"
        if isinstance(v, (dict, list)):
            _numbers(v, names, here)
        elif isinstance(v, str) or (isinstance(v, (int, float)) and not isinstance(v, bool)):
            obj[key] = number(v, names, here)


def resolve(model, point=None):
    """A copy of the model at one sweep point with numbers everywhere: the point's values set, the parameters
    worked out (they may refer to each other), then every value of the parts, couplings, drives, targets and
    analysis. The probes and measures are left as written."""
    m = copy.deepcopy(model)
    for path, value in (point or {}).items():
        put(m, path, value)
    names, todo = {}, dict(m.get('parameters', {}))
    while todo:
        done = []
        for k, v in todo.items():
            try:
                names[k] = number(v, names, f"parameters.{k}")
                done.append(k)
            except ModelError:
                pass
        if not done:
            k, v = next(iter(todo.items()))
            number(v, names, f"parameters.{k}")                     # raises: says which name is missing
        for k in done:
            del todo[k]
    m['parameters'] = names
    for section in ('parts', 'couplings', 'drives', 'analysis'):
        if section in m:
            _numbers(m[section], names, section)
    for i, t in enumerate(m.get('targets', [])):
        for key in ('value', 'initial', 'tolerance'):
            if key in t:
                t[key] = number(t[key], names, f"targets[{i}].{key}")
        t['range'] = [number(v, names, f"targets[{i}].range") for v in t['range']]
        _need(t['range'][0] < t['range'][1], f"targets[{i}]: range must be [low, high], low below high")
    return m


def load(path):
    path = Path(path)
    try:
        model = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        raise ModelError(f"{path.name}: not JSON ({e})") from None
    return validate(_beside(model, path.parent))


def _beside(model, folder):
    """The model with its board's file as a full path: one written relative is beside the model."""
    board = model.get('board')
    if isinstance(board, dict) and board.get('file') and not Path(board['file']).expanduser().is_absolute():
        board['file'] = str((Path(folder) / board['file']).resolve())
    return model


def templates():
    """The example circuits shipped with pcbqs: {name: description}, in a teaching order."""
    from importlib.resources import files
    out = {}
    for f in sorted(files('pcbqs.circuit').joinpath('examples').iterdir(), key=lambda f: f.name):
        if f.name.endswith('.json'):
            m = json.loads(f.read_text(encoding='utf-8'))
            out[f.name[:-5].split('_', 1)[1] if f.name[:2].isdigit() else f.name[:-5]] = m.get('description', m.get('name', ''))
    return out


def template(name):
    from importlib.resources import files
    for f in files('pcbqs.circuit').joinpath('examples').iterdir():
        stem = f.name[:-5]
        if f.name.endswith('.json') and name in (stem, stem.split('_', 1)[-1]):
            return _beside(json.loads(f.read_text(encoding='utf-8')), Path(str(f)).parent)
    raise ModelError(f"no template {name!r}: {', '.join(templates())}")
