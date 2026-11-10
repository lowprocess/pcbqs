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


def cache_dir():
    """Where imported boards are cached: PCBQS_CACHE_DIR, else ~/.cache/pcbqs/boards (on disk: /tmp may be RAM)."""
    import os
    return Path(os.environ.get('PCBQS_CACHE_DIR') or Path.home() / '.cache' / 'pcbqs' / 'boards')


def _cache_key(path, nets):
    """The board file's content hash, the net subset and both packages' versions: an edited board or an upgraded
    importer misses the cache."""
    import hashlib
    from . import __version__
    try:
        from kicad2ems import __version__ as k2e
    except ImportError:
        k2e = '?'
    h = hashlib.sha256(Path(path).read_bytes())
    h.update(json.dumps([sorted(nets) if nets else None, __version__, k2e, BOARD_CACHE_FORMAT]).encode())
    return h.hexdigest()[:32]


BOARD_CACHE_FORMAT = 1


def load_board(path, nets=None, cache=None):
    """board.json's dict from a .kicad_pcb (through kicad2ems; only `nets` when given) or a kicad2ems board.json.
    A KiCad import is cached by content hash (cache_dir(); PCBQS_NO_CACHE=1 or cache=False to bypass), so the PDN,
    near-field and circuit tools and the window share one import of a board (Jetson: 25 s the first time)."""
    import os
    import pickle
    path = str(path)
    if path.endswith('.json'):
        return json.loads(Path(path).read_text())
    try:
        from kicad2ems.importers import open_board
        from kicad2ems.board import Board
    except ImportError as exc:
        raise RuntimeError('KiCad input requires pcbqs[kicad2ems]; JSON input works independently') from exc
    use = (cache if cache is not None else not os.environ.get('PCBQS_NO_CACHE'))
    hit = cache_dir() / f'{_cache_key(path, nets)}.pickle' if use else None
    if hit is not None and hit.is_file():
        try:
            return pickle.loads(hit.read_bytes())
        except Exception:                                   # a damaged entry: import again and replace it
            pass
    only = dict(nets=sorted(nets), parts='none') if nets else {}
    board = Board(open_board(path), need_ports=False, log=lambda *a: None, **only).to_dict()
    if hit is not None:
        try:
            hit.parent.mkdir(parents=True, exist_ok=True)
            tmp = hit.with_suffix(f'.{os.getpid()}.tmp')
            tmp.write_bytes(pickle.dumps(board, protocol=pickle.HIGHEST_PROTOCOL))
            os.replace(tmp, hit)                            # atomic: parallel tools never read half a file
        except OSError:
            pass                                            # a read-only cache is only slower
    return board
