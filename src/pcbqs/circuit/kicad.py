"""
KiCad into the general model: the parts, their values and their nodes from a schematic, a netlist or a board.

    document = read(path)                   # .kicad_sch (through kicad-cli), a KiCad XML netlist, or a .kicad_pcb
    model = model_from(document)            # a new model of everything that can be taken over
    model = apply(model, document, refs)    # or into a model that exists: add these parts, or bring them up to date
    python -m pcbqs.circuit --from-kicad board.kicad_pcb > model.json

What is taken over, each a part between the two nets its pins are on:
    R, C, L         by its symbol, its reference (R1) or its footprint's library; the value as KiCad writes it (4k7, 22u, 0R01, 47uH); ESR, ESL
                    and DCR from fields of those names
    a diode         reference D<n> or a D symbol: anode to cathode by its pins' names (A, K), else pad 2 to pad 1 as
                    KiCad numbers them; Vf from a field Vf, 0.45 V for a Schottky, else 0.7 V
    a MOSFET        pins named D, G, S: a switch from its drain's net to its source's, on the model's first drive
A node is the net's name (without KiCad's leading '/'); the board's own name is kept in board.nets where it
differs, and a board that was read becomes the model's board, so its copper can be put in (circuit/copper.py).

What KiCad does not say is left to add: which connector is the source and which the load, what drives the
switches and how fast, and what to measure. status() says for each part what would be done with it and why not.
"""
import copy
import hashlib
import re
from pathlib import Path

from ..schematic import PASSIVES, engineering_value, read_schematic, _parasitics

IMPORTED = ('resistor', 'capacitor', 'inductor', 'diode', 'switch')


def _blocks(text, head):
    """The parenthesised blocks of a KiCad file that start '(<head> ', quoted text skipped: [str]."""
    out, start, depth, quoted, k = [], None, 0, False, 0
    mark = '(' + head + ' '
    while k < len(text):
        ch = text[k]
        if quoted:
            if ch == '\\':
                k += 1
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
        elif ch == '(':
            if start is None and text.startswith(mark, k):
                start, depth = k, 0
            if start is not None:
                depth += 1
        elif ch == ')' and start is not None:
            depth -= 1
            if depth == 0:
                out.append(text[start:k + 1])
                start = None
        k += 1
    return out


def pad_functions(text):
    """{reference: {pad: its pin's name}} from a board file's text: what KiCad writes as (pinfunction "...")."""
    out = {}
    for fp in _blocks(text, 'footprint') + _blocks(text, 'module'):
        ref = re.search(r'\(property "Reference" "([^"]*)"', fp) or re.search(r'\(fp_text reference "?([^\s")]+)', fp)
        if not ref:
            continue
        for pad in _blocks(fp, 'pad'):
            number = re.match(r'\(pad "?([^\s")]*)"?', pad)
            fn = re.search(r'\(pinfunction "([^"]*)"\)', pad)
            if number and fn:
                out.setdefault(ref[1], {})[number[1]] = fn[1]
    return out


def read_board(path):
    """A board as the document read_schematic gives: {components: [{ref, value, kind, value_si, pins: [{number,
    name, net}], dnp, exclude_from_sim, fields, ...}], ...}, from its footprints, their pads' nets and their
    pads' pin names."""
    from .._cli import load_board
    path = Path(path).expanduser().resolve()
    data = path.read_bytes()
    board = load_board(str(path))
    functions = pad_functions(data.decode('utf-8', 'replace')) if path.suffix == '.kicad_pcb' else {}
    comps = []
    for fp in board['prims']['footprints']:
        ref = fp.get('ref', '')
        if not ref:
            continue
        pins = {}
        for pad in fp.get('pads', []):
            number = str(pad.get('name', ''))
            pin = pins.setdefault(number, dict(number=number, name=functions.get(ref, {}).get(number, ''), type='',
                                               net=pad.get('net') or ''))
            if not pin['net']:
                pin['net'] = pad.get('net') or ''
        value = str(fp.get('value', ''))
        library = str(fp.get('lib', '')).lower()                    # KiCad's own libraries say what a footprint is for
        kind = ({'R': 'resistor', 'C': 'capacitor', 'L': 'inductor'}.get(ref[0]) if re.match(r'[RCL]\d', ref) else
                next((k for word, k in (('resistor_', 'resistor'), ('capacitor_', 'capacitor'), ('inductor_', 'inductor'))
                      if library.startswith(word)), None))
        if {'D', 'G', 'S'} <= {p['name'].upper() for p in pins.values()}:
            kind = 'mosfet'
        c = dict(ref=ref, value=value, footprint=fp.get('lib', ''), description='', lib_id=fp.get('lib', ''),
                 fields=dict(fp.get('props', {})), properties={}, pins=sorted(pins.values(), key=lambda p: (len(p['number']), p['number'])),
                 dnp=bool(fp.get('dnp')), exclude_from_sim=bool(fp.get('exclude_from_sim')), sheet='', kind=kind or 'unsupported',
                 value_si=None)
        if kind in PASSIVES:
            try:
                c['value_si'] = engineering_value(value)
            except ValueError:
                pass
        nets = {p['net'] for p in c['pins'] if p['net']}
        c['supported_passive'] = kind in PASSIVES and len(nets) == 2 and bool(c['value_si']) and c['value_si'] > 0
        comps.append(c)
    return dict(format=1, path=str(path), kind='kicad_pcb', sha256=hashlib.sha256(data).hexdigest(), design_source=path.name,
                tool='pcbqs', components=comps, nets=[], sheets=[])


def read(path):
    """The document of a KiCad file: a board (.kicad_pcb), a schematic (.kicad_sch, needs kicad-cli) or a KiCad
    XML netlist."""
    return read_board(path) if str(path).lower().endswith('.kicad_pcb') else read_schematic(path)


def node_of(net):
    """A net's name as a node: KiCad's leading '/' (a net named on the root sheet) left off."""
    return net[1:] if net.startswith('/') and len(net) > 1 else net


def part_of(c, drives=()):
    """A document's component as a part of the general model, or why it is not one: (part | None, reason)."""
    ref = c['ref']
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', ref):
        return None, 'its reference is not letters, digits and _'
    pins = [p for p in c['pins'] if p['net'] and not p['net'].startswith('unconnected-')]
    names = {p['name'].upper(): p for p in pins}
    text = ' '.join([c.get('value', ''), c.get('lib_id', ''), c.get('description', '')]).lower()
    fields = {k.lower(): v for k, v in c.get('fields', {}).items()}

    def field(*keys):
        for k in keys:
            if fields.get(k):
                try:
                    return engineering_value(fields[k])
                except ValueError:
                    pass
        return None
    if c['kind'] == 'mosfet' or (re.match(r'Q\d', ref) and {'D', 'S'} <= set(names)):
        ends = []
        for role in ('D', 'S'):
            on = {p['net'] for p in pins if p['name'].upper() == role}
            if len(on) != 1:
                return None, f"its {role} pins are on {len(on)} nets"
            ends.append(node_of(on.pop()))
        if ends[0] == ends[1]:
            return None, 'its drain and source are on one net'
        model = dict(type='switch', Ron_ohm=field('rds', 'rdson', 'ron', 'sim.ron') or 0.02, Cds_F=field('coss', 'cds') or '300p')
        return dict(ref=ref, type='switch', nodes=ends, drive=next(iter(drives), 'pwm'), model=model), ''
    nets = list(dict.fromkeys(p['net'] for p in pins))
    if len(nets) != 2:
        return None, f"its pins are on {len(nets)} net{'s' if len(nets) != 1 else ''}, not two"
    if c['kind'] in PASSIVES:
        if not c.get('value_si') or c['value_si'] <= 0:
            return None, f"its value {c.get('value', '')!r} is not a number"
        part = dict(ref=ref, type=c['kind'], value=c['value_si'], nodes=[node_of(n) for n in nets])
        part.update(_parasitics(c))
        return part, ''
    symbol = c.get('lib_id', '').split(':')[-1].lower()
    if (re.match(r'D\d', ref) or symbol == 'd' or symbol.startswith('d_') or fields.get('sim.device', '').lower() == 'd'
            or c.get('lib_id', '').lower().startswith('diode_')):
        anode = next((p for p in pins if p['name'].upper() in ('A', 'ANODE')), None)
        cathode = next((p for p in pins if p['name'].upper() in ('K', 'C', 'CATHODE')), None)
        if anode is None or cathode is None:                        # KiCad's diodes: pad 1 the cathode, pad 2 the anode
            by = {p['number']: p for p in pins}
            anode, cathode = by.get('2'), by.get('1')
        if anode is None or cathode is None or anode['net'] == cathode['net']:
            return None, 'which pin is its anode is not told (pins A and K, or pads 2 and 1)'
        vf = field('vf', 'sim.vf') or (0.45 if 'schottky' in text or re.search(r'\b(ss|sk|mbr|b[0-9]{3})', text) else 0.7)
        return dict(ref=ref, type='diode', nodes=[node_of(anode['net']), node_of(cathode['net'])],
                    model=dict(Vf_V=vf, Cj_F=field('cj') or '100p')), ''
    return None, 'not a resistor, capacitor, inductor, diode or MOSFET: add it as a source or a load if it is one'


def status(c, model):
    """What importing a component would do: (text, whether it can be taken over, whether to tick it)."""
    part, why = part_of(c, model.get('drives', {}))
    if part is None:
        return 'Not taken over: ' + why, False, False
    existing = next((p for p in model.get('parts', []) if p['ref'] == c['ref']), None)
    off = c.get('dnp') or c.get('exclude_from_sim')
    tail = ' (not fitted: left out of the simulation)' if off else ''
    if existing is None:
        return f"New {part['type']}" + tail, True, not off
    if existing['type'] != part['type']:
        return f"The model has it as a {existing['type']}: review before applying", True, False
    return 'In the model: its value and nodes brought up to date' + tail, True, True


def apply(model, document, refs, update_connections=True):
    """The model with the document's components `refs` added, or brought up to date where the model has them
    (value, parasitic fields, and the nodes when update_connections). The model given is left as it is."""
    from . import model as M
    out = copy.deepcopy(model)
    out.setdefault('format', M.FORMAT)
    out.setdefault('parts', [])
    source = {c['ref']: c for c in document['components']}
    names = {}
    for ref in refs:
        if ref not in source:
            raise ValueError('the KiCad file has no ' + ref)
        c = source[ref]
        part, why = part_of(c, out.get('drives', {}))
        if part is None:
            raise ValueError(f"{ref}: {why}")
        for p in c['pins']:
            if p['net'] and node_of(p['net']) != p['net']:
                names[node_of(p['net'])] = p['net']
        existing = next((p for p in out['parts'] if p['ref'] == ref), None)
        if existing is None:
            out['parts'].append(part)
            existing = part
        else:
            if existing['type'] != part['type']:
                raise ValueError(f"{ref} is a {existing['type']} in the model and a {part['type']} in KiCad: change one first")
            for key, value in part.items():
                if key == 'nodes':
                    if update_connections:
                        existing['nodes'] = value
                elif key in ('model', 'drive'):
                    continue                                        # the model's own choices stand
                else:
                    existing[key] = value
        if c.get('dnp') or c.get('exclude_from_sim'):
            existing['enabled'] = False
        elif 'enabled' in existing:
            existing['enabled'] = True
        if part['type'] == 'switch' and not out.get('drives'):
            out['drives'] = {part['drive']: dict(frequency_Hz=100e3, duty=0.5)}
    if document.get('kind') == 'kicad_pcb':
        board = out.setdefault('board', {})
        board['file'] = document['path']
        used = {n for p in out['parts'] for n in p['nodes']}
        mapped = {**board.get('nets', {}), **{k: v for k, v in names.items() if k in used}}
        if mapped:
            board['nets'] = mapped
    out.setdefault('schematic', {})['source'] = copy.deepcopy(document)
    return M.validate(out)


def model_from(document, name=None):
    """A new model of every component of the document that can be taken over and is fitted."""
    refs = [c['ref'] for c in document['components'] if part_of(c)[0] is not None and not (c.get('dnp') or c.get('exclude_from_sim'))]
    if not refs:
        raise ValueError('nothing in the KiCad file can be taken over (two-pin R, C, L, diodes and MOSFETs with pins D, G, S)')
    left = [c['ref'] for c in document['components'] if c['ref'] not in refs]
    base = dict(format=2, name=name or Path(document.get('path', 'circuit')).stem,
                description='From KiCad' + (': to add, a source, a load, the drives and what to measure'
                                            + (f" ({', '.join(left)} not taken over)" if left else '')), parts=[])
    return apply(base, document, refs)
