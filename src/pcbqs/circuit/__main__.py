"""python -m pcbqs.circuit: run a circuit model, or print one of the example circuits."""
import argparse
import json
import signal
import sys

from . import model as M
from .engine import run
from .copper import BoardError


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m pcbqs.circuit', description=__doc__)
    ap.add_argument('model', nargs='?', help='a circuit model (JSON, format 2)')
    ap.add_argument('--out', default='circuit-results', help='the results folder (default: circuit-results)')
    ap.add_argument('--jobs', type=int, default=1, help='sweep points run at once')
    ap.add_argument('--set', action='append', default=[], metavar='PATH=VALUE',
                    help="set a value of the model before running, e.g. --set drives.pwm.duty=0.4 (repeatable)")
    ap.add_argument('--copper', action='store_true', help="with the copper of the model's board between its parts' pads")
    ap.add_argument('--board', metavar='FILE', help='the board (.kicad_pcb), when the model names none or another')
    ap.add_argument('--from-kicad', metavar='FILE', help='print a model of the parts of a KiCad board, schematic or '
                    'XML netlist, to complete with a source, a load and drives')
    ap.add_argument('--templates', action='store_true', help='list the example circuits')
    ap.add_argument('--template', metavar='NAME', help='print an example circuit to start from')
    a = ap.parse_args(argv)
    if a.templates:
        for name, text in M.templates().items():
            print(f"{name:22s} {text}")
        return 0
    if a.from_kicad:
        from . import kicad
        try:
            model = kicad.model_from(kicad.read(a.from_kicad))
        except (ValueError, RuntimeError, OSError) as e:
            ap.error(str(e))
        model.get('schematic', {}).pop('source', None)              # the file itself is there to read again
        print(json.dumps(model, indent=1))
        return 0
    if a.template:
        try:
            print(json.dumps(M.template(a.template), indent=1))
        except M.ModelError as e:
            ap.error(str(e))
        return 0
    if not a.model:
        ap.error('a model is required (or --templates, --template NAME)')
    if a.jobs < 1:
        ap.error('--jobs must be positive')

    def stop(*_):
        from ..power_spice import cancel_active_runs
        cancel_active_runs()
        raise SystemExit('Circuit run stopped')
    signal.signal(signal.SIGTERM, stop)
    try:
        model = M.load(a.model)
        for item in a.set:
            path, _, value = item.partition('=')
            M.put(model, path.strip(), M.number(value) if not value.strip().startswith('[') else json.loads(value))
        if a.board:
            model.setdefault('board', {})['file'] = a.board
        M.validate(model)
        summary = run(model, a.out, jobs=a.jobs, log=lambda s: print(s, flush=True), copper=a.copper)
    except (M.ModelError, BoardError) as e:
        print(f"circuit: error: {e}", file=sys.stderr)
        return 2
    for row in summary['cases']:
        eff = row.get('efficiency')
        knobs = ', '.join(f"{k} = {v:.5g}" for k, v in row['knobs'].items())
        print(f"{row['name']}: " + (f"{knobs}; " if knobs else '') + f"in {row['source_power_W']:.4g} W, out "
              f"{row['load_power_W']:.4g} W" + (f", efficiency {100 * eff:.2f} %" if eff else '')
              + (f", copper {row['copper']['loss_W']:.3g} W" if row.get('copper') else ''), flush=True)
        for note in row['notes']:
            print(f"  note: {note}", flush=True)
    for f in summary['failed']:
        print(f"{f['name']}: FAILED: {f['error']}", flush=True)
    print(f"Completed circuit cases: {len(summary['cases'])} of {len(summary['cases']) + len(summary['failed'])}", flush=True)
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
