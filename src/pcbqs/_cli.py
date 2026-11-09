"""Shared command-line plumbing of pcbqs's tools (pdn, mqs): options tables, --describe, board import.

An options table is [(key, kind, default, help, group)]; the flag is '--' + key with '-' for '_', kind one of
'str', 'int', 'float', 'bool' (a switch) or 'list' (repeatable). --describe --json gives the table to the window,
which builds its form from it.
"""
import json
from pathlib import Path

DESCRIBE_FORMAT = 1


def flag(key):
    return '--' + key.replace('_', '-')


def describe(options, **extra):
    from . import __version__
    return dict(format=DESCRIBE_FORMAT, version=__version__, **extra,
                options=[dict(key=k, flag=flag(k), kind=t, default=d, help=h, group=g, choices=None)
                         for k, t, d, h, g in options])


def add_options(ap, options):
    for k, t, d, h, _ in options:
        if t == 'bool':
            ap.add_argument(flag(k), dest=k, action='store_true', help=h)
        elif t == 'list':
            ap.add_argument(flag(k), dest=k, action='append', default=[], help=h)
        else:
            ap.add_argument(flag(k), dest=k, type={'float': float, 'int': int, 'str': str}[t], default=d, help=h)


def print_describe(d, as_json):
    print(json.dumps(d, indent=1) if as_json else '\n'.join(f"{o['flag']:18s} {o['help']}" for o in d['options']))


def load_board(path, nets=None):
    """board.json's dict from a .kicad_pcb (through kicad2ems; only `nets` when given) or a kicad2ems board.json."""
    path = str(path)
    if path.endswith('.json'):
        return json.loads(Path(path).read_text())
    try:
        from kicad2ems.importers import open_board
        from kicad2ems.board import Board
    except ImportError as exc:
        raise RuntimeError('KiCad input requires pcbqs[kicad2ems]; JSON input works independently') from exc
    only = dict(nets=sorted(nets), parts='none') if nets else {}
    return Board(open_board(path), need_ports=False, log=lambda *a: None, **only).to_dict()
