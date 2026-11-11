"""
A model's board: the copper between its parts' pads, as part of the circuit.

    "board": {"file": "06_boost.kicad_pcb",          # beside the model; a .kicad_pcb, or kicad2ems's board.json
              "parts": {"V1": "J1", "R1": "J2"},     # a model part's footprint, when its reference differs
              "nets": {"vbus": "+24V"},              # a model node's net on the board, when its name differs
              "reference": {"in": "C1"},             # the part whose pad is a net's node itself (else its first)
              "mesh": {"coarse_mm": 3, "fine_mm": 1.6, "fit_tolerance": 0.03}}

A part's pin is the pads of its footprint on its node's net ('0' is the board's GND), so nothing is drawn or named
twice: the model says which part is on which node, the board which pad is on which net. Without copper every pin
on a node is one point. With it, each net's pads are joined by what the board has between them: one pad of the
net is the node itself (its reference), and every other pad reaches it through a port of a passive R-L model
of the copper (all the nets' ports in one model, so their mutual inductance is in it too), extracted by the
quasi-static solver (pcbqs.mqs) and fitted over frequency.

    cu = prepare(model, out, log)                    # extract, or read back <out>/copper-model.json
    deck = build(resolved, state, cu)                # circuit/deck.py

A part with no footprint on the board (a coil on a cable, a bench supply) is at its net's node, and the results'
notes say so.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

GROUND = ('0', 'GND', 'gnd')
FIT_TOLERANCE = 0.03
BASE_FREQUENCIES = (0.0, 1e3, 1e4, 3e4, 1e5, 3e5, 3e6, 7e6, 2e7)


class BoardError(ValueError):
    """A model and a board that do not fit: says which part, pad or net."""


def board_file(model):
    b = model.get('board') or {}
    if not b.get('file'):
        raise BoardError('the model names no board: add "board": {"file": "<layout>.kicad_pcb"}')
    path = Path(b['file']).expanduser()
    if not path.is_file():
        raise BoardError(f"board: no file {path}")
    return path


def net_name(model, node, nets):
    """The board's net of a model node: board.nets' when given, the board's ground for '0', else the same name."""
    named = (model.get('board') or {}).get('nets', {})
    if node in named:
        return named[node]
    if node in GROUND:
        return next((n for n in ('GND', 'gnd', '0', 'GNDPWR', 'PGND') if n in nets), 'GND')
    return node


def pins(model, board):
    """Where each pin of the model is on the board: ({(ref, 0 | 1): terminal}, {terminal: (footprint, net)},
    [parts with no footprint]). A terminal is a footprint's pads on one net, named '<footprint>.<net>'."""
    from ..pdn.rail import parts_of
    footprints = {p.ref: p for p in parts_of(board)}
    known = {pad.net for p in footprints.values() for pad in p.pads if pad.net}
    names = (model.get('board') or {}).get('parts', {})
    where, terminals, absent = {}, {}, []
    for part in model['parts']:
        ref = part['ref']
        if not part.get('enabled', True):
            continue
        fp = names.get(ref, ref)
        if fp not in footprints:
            if ref in names:
                raise BoardError(f"board.parts: {ref} is given footprint {fp!r}, which the board does not have")
            absent.append(ref)
            continue
        for k, node in enumerate(part['nodes']):
            net = net_name(model, node, known)
            if not footprints[fp].on({net}):
                have = sorted({pad.net for pad in footprints[fp].pads if pad.net})
                raise BoardError(f"{ref}: footprint {fp} has no pad on net {net!r} (node {node!r}); its pads are on "
                                 f"{', '.join(have) or 'no net'}")
            where[(ref, k)] = f"{fp}.{net}"
            terminals[f"{fp}.{net}"] = (fp, net)
    if not terminals:
        raise BoardError('none of the model\'s parts is a footprint of the board (board.parts names them when their '
                         'references differ)')
    return where, terminals, absent


def frequencies(model):
    """The frequencies the copper model is fitted at: DC to 20 MHz, and every switching and source frequency of
    every sweep point with its tenth harmonic."""
    from . import model as M
    out = set(BASE_FREQUENCIES)
    for point in M.cases(model):
        r = M.resolve(model, point)
        fs = [d['frequency_Hz'] for d in r.get('drives', {}).values()]
        fs += [p['wave']['frequency_Hz'] for p in r['parts'] if p['type'] == 'source' and p.get('wave', {}).get('frequency_Hz')]
        out |= {float(f) for f in fs} | {10.0 * f for f in fs}
    return sorted(out)


def geometry(model, board, log=print):
    """The solver's model of the board's copper with a port at every pad but one of each net:
    (MQS, ports [{name, terminal, reference, net}], same {terminal: the terminal it is one mesh node with})."""
    from shapely.ops import unary_union
    from ..mqs import MQS
    from ..pdn.rail import parts_of
    from ..contacts import contact_terminal
    mesh = (model.get('board') or {}).get('mesh', {})
    where, terminals, _ = pins(model, board)
    footprints = {p.ref: p for p in parts_of(board)}
    terms = {name: contact_terminal(footprints[fp], footprints[fp].on({net}), mesh.get('pad_contact', 'termination'))
             for name, (fp, net) in terminals.items()}
    for name, shapes in terms.items():
        if not shapes:
            raise BoardError(f"board: {name} has no contact area")
    nets = {net for _, net in terminals.values()}
    extent = unary_union([p for vv in terms.values() for _, p in vv])
    q = MQS(board, nets, terminals=terms, region=extent.buffer(mesh.get('margin_mm', 2)).envelope,
            dmax=mesh.get('coarse_mm', 3), fine=extent.buffer(0.6).envelope, dmax_fine=mesh.get('fine_mm', 1.6),
            conductor_loss='slab', conductor_faces=2, thickness_sheets=int(mesh.get('thickness_sheets', 1)),
            matrix_backend=mesh.get('matrix_backend', 'hierarchical'), fill_backend=mesh.get('backend', 'auto'),
            fill_workers=mesh.get('workers', 8), log=log)
    order = [t for part in model['parts'] for k in (0, 1) if (t := where.get((part['ref'], k)))]     # the model's order
    chosen = (model.get('board') or {}).get('reference', {})
    reference = {}
    for net in nets:
        mine = [t for t in dict.fromkeys(order) if terminals[t][1] == net]
        want = chosen.get(net) or next((chosen[n] for n in chosen if net_name(model, n, nets) == net), None)
        if want:
            fp = (model.get('board') or {}).get('parts', {}).get(want, want)
            if f"{fp}.{net}" not in mine:
                raise BoardError(f"board.reference: {want} has no pad on net {net!r}")
            reference[net] = f"{fp}.{net}"
        else:                                                       # where the board meets the outside: a source's
            kind = {(model.get('board') or {}).get('parts', {}).get(p['ref'], p['ref']): p['type'] for p in model['parts']}
            rank = lambda t: dict(source=0, load=1).get(kind.get(terminals[t][0]), 2)   # pad, else a load's, else the first
            reference[net] = min(mine, key=rank)
    ports, same = [], {}
    for t in dict.fromkeys(order):
        net = terminals[t][1]
        if t == reference[net]:
            continue
        if q.term_node[t] == q.term_node[reference[net]]:           # one piece of copper too small to tell apart
            same[t] = reference[net]
            continue
        name = f"T{len(ports)}"
        q.source(name, t, reference[net], part_h=0)
        ports.append(dict(name=name, terminal=t, reference=reference[net], net=net))
    return q, ports, same


def fingerprint(model, path, fs):
    h = hashlib.sha256(Path(path).read_bytes())
    b = dict(model.get('board') or {})
    b.pop('file', None)
    h.update(json.dumps(dict(board=b, parts=[(p['ref'], p['nodes'], p.get('enabled', True)) for p in model['parts']],
                             frequencies=fs), sort_keys=True).encode())
    from .. import mqs, mqs_ports
    for module in (mqs, mqs_ports, __import__(__name__, fromlist=['x'])):
        h.update(Path(module.__file__).read_bytes())
    return h.hexdigest()


def prepare(model, out, log=print):
    """The copper of the model's board, ready for the deck: {file, ports, model (R_ohm, L_H, foster: the passive
    fit), pins {(ref, k): terminal}, same, absent, fit_error, fingerprint}. Extracted once for a model and
    its board and kept in <out>/copper-model.json."""
    from .._cli import load_board
    from ..mqs_ports import CopperPorts, foster_equivalent
    from ..power_circuit import reduce_copper
    path = board_file(model)
    fs = frequencies(model)
    key = fingerprint(model, path, fs)
    cache = Path(out) / 'copper-model.json'
    board = load_board(str(path))
    where, terminals, absent = pins(model, board)
    saved = json.loads(cache.read_text()) if cache.exists() else {}
    if saved.get('fingerprint') != key:
        log(f"Extracting the copper of {path.name}: {len(terminals)} pads on {len({n for _, n in terminals.values()})} nets")
        q, ports, same = geometry(model, board, log)
        if ports:
            response = CopperPorts(q).extract(fs, log=log)
            fitted = reduce_copper(foster_equivalent(response), response,
                                   (model.get('board') or {}).get('mesh', {}).get('fit_tolerance', FIT_TOLERANCE))
            fitted['check_frequencies_Hz'] = fs
        else:
            fitted = None
        saved = dict(fingerprint=key, file=str(path), ports=ports, same=same, model=fitted)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(saved, indent=1))
    else:
        log(f"The copper of {path.name}: as extracted before")
    fitted = saved['model']
    return dict(saved, pins=where, terminals=terminals, absent=absent,
                fit_error=None if fitted is None else float(np.max(fitted.get('relative_impedance_error', 0.0))))


def nodes(model, cu):
    """The circuit node of every pin: {(ref, k): node}. A net's reference pad is the model's node; any other pad
    is a node of its own ('<footprint>.<net>'), joined to it by its port."""
    port_of = {p['terminal'] for p in cu['ports']}
    out = {}
    for part in model['parts']:
        for k, node in enumerate(part['nodes']):
            t = cu['pins'].get((part['ref'], k))
            t = cu['same'].get(t, t)
            out[(part['ref'], k)] = t if t in port_of else node
    return out


def port_nodes(model, cu):
    """{port name: (its pad's node, its net's node)} for Circuit.copper."""
    net_node = {}
    for part in model['parts']:
        for k, node in enumerate(part['nodes']):
            t = cu['pins'].get((part['ref'], k))
            if t is not None:
                net_node[cu['terminals'][t][1]] = node
    return {p['name']: (p['terminal'], net_node[p['net']]) for p in cu['ports']}


def port_currents(model, cu, currents):
    """The current each port carries when its part carries `currents` ({ref: A entering its first node}): what
    leaves a part's second pin into its pad's copper, and what its first pin draws out of it. [A], in the ports'
    order, as Circuit.copper's PCB<i>.I counts it (into the pad's end of the port)."""
    at = {}
    for part in model['parts']:
        i = currents.get(part['ref'])
        if i is None:
            continue
        for k in (0, 1):
            t = cu['pins'].get((part['ref'], k))
            t = cu['same'].get(t, t)
            at[t] = at.get(t, 0.0) + (-i if k == 0 else i)
    return [at.get(p['terminal'], 0.0) for p in cu['ports']]
