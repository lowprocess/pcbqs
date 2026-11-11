"""python -m pcbqs.circuit: run a circuit model, or print one of the example circuits."""
import argparse
import json
import signal
import sys

from . import model as M
from .engine import run


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m pcbqs.circuit', description=__doc__)
    ap.add_argument('model', nargs='?', help='a circuit model (JSON, format 2)')
    ap.add_argument('--out', default='circuit-results', help='the results folder (default: circuit-results)')
    ap.add_argument('--jobs', type=int, default=1, help='sweep points run at once')
    ap.add_argument('--set', action='append', default=[], metavar='PATH=VALUE',
                    help="set a value of the model before running, e.g. --set drives.pwm.duty=0.4 (repeatable)")
    ap.add_argument('--templates', action='store_true', help='list the example circuits')
    ap.add_argument('--template', metavar='NAME', help='print an example circuit to start from')
    a = ap.parse_args(argv)
    if a.templates:
        for name, text in M.templates().items():
            print(f"{name:22s} {text}")
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
        M.validate(model)
        summary = run(model, a.out, jobs=a.jobs, log=lambda s: print(s, flush=True))
    except M.ModelError as e:
        print(f"circuit: error: {e}", file=sys.stderr)
        return 2
    for row in summary['cases']:
        eff = row.get('efficiency')
        knobs = ', '.join(f"{k} = {v:.5g}" for k, v in row['knobs'].items())
        print(f"{row['name']}: " + (f"{knobs}; " if knobs else '') + f"in {row['source_power_W']:.4g} W, out "
              f"{row['load_power_W']:.4g} W" + (f", efficiency {100 * eff:.2f} %" if eff else ''), flush=True)
        for note in row['notes']:
            print(f"  note: {note}", flush=True)
    for f in summary['failed']:
        print(f"{f['name']}: FAILED: {f['error']}", flush=True)
    print(f"Completed circuit cases: {len(summary['cases'])} of {len(summary['cases']) + len(summary['failed'])}", flush=True)
    return 1 if summary['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())
