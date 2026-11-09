"""Solver-option trials on a real pcbqs power-circuit netlist (wpthb with its 24-port copper model).

    python tools/benchmark_circuit_options.py <power-circuit case folder with circuit.cir> [variant ...] [--out DIR]

Each variant rewrites only the .options line; measurements (last switching cycle) are compared with the
unmodified netlist's.
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pcbqs.power_spice import options_line  # noqa: E402  (pcbqs's defaults, SOLVER_DEFAULTS)

VARIANTS = {                                     # each: SOLVER_DEFAULTS with these changes
    'default': options_line(),
    'reltol1e-3': options_line(dict(reltol=1e-3)),
    'reltol3e-4': options_line(dict(reltol=3e-4)),
    'chgtol1e-14': options_line(dict(chgtol=1e-14)),
    'trap': options_line(dict(method='trap')),
    'trap-reltol1e-3': options_line(dict(method='trap', reltol=1e-3)),
    'abstol1e-9': options_line(dict(abstol=1e-9)),
    'reltol1e-3-abstol1e-9': options_line(dict(reltol=1e-3, abstol=1e-9)),
}
KEYS = ['input_power_W', 'load_power_W', 'primary_rms_A', 'secondary_rms_A', 'primary_peak_A', 'bus_ripple_V']


def run(case, tag, opts, root):
    work = root / tag
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    text = (case / 'circuit.cir').read_text()
    text = re.sub(r'^\.options .*$', '.options ' + opts, text, count=1, flags=re.M)
    text = re.sub(r'^write adaptive-ports\.raw .*$', '', text, flags=re.M)        # benchmark the solve, not the export
    (work / 'circuit.cir').write_text(text)
    t0 = time.perf_counter()
    p = subprocess.run(['ngspice', '-b', 'circuit.cir'], cwd=work, capture_output=True, text=True, timeout=7200)
    wall = time.perf_counter() - t0
    log = p.stdout + p.stderr
    (work / 'ngspice.log').write_text(log)
    labels = json.loads((case / 'measurements.json').read_text())
    meas = {}
    for i, key in enumerate(labels):
        m = re.search(rf'^m{i}\s*=\s*([-+0-9.eE]+)', log, re.M)
        if m:
            meas[key] = float(m.group(1))
    stats = {}
    for key, pat in (('accepted', r'Accepted timepoints\s*=\s*(\d+)'), ('rejected', r'Rejected timepoints\s*=\s*(\d+)'),
                     ('iterations', r'Transient iterations\s*=\s*(\d+)'), ('factor_s', r'Matrix factor time\s*=\s*([0-9.]+)')):
        m = re.search(pat, log)
        stats[key] = float(m.group(1)) if m else None
    return dict(tag=tag, wall_s=wall, ok=p.returncode == 0, **stats, measures={k: meas.get(k) for k in KEYS})


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case'); ap.add_argument('variants', nargs='*')
    ap.add_argument('--out', default='circuit-options-bench')
    args = ap.parse_args()
    case = Path(args.case); tags = args.variants or list(VARIANTS)
    root = Path(args.out) / case.parent.name / case.name
    out = []
    for tag in tags:
        r = run(case, tag, VARIANTS[tag], root)
        out.append(r)
        ref = out[0]['measures']
        dev = {k: (r['measures'][k] / ref[k] - 1) * 100 for k in KEYS if r['measures'].get(k) and ref.get(k)}
        print(f"{tag:24s} {r['wall_s']:7.1f} s  steps {r['accepted']:.0f}+{r['rejected']:.0f}  iters {r['iterations']:.0f}  "
              f"| dev % " + ' '.join(f'{k.split("_")[0]}={v:+.2f}' for k, v in dev.items()), flush=True)
    json.dump(out, open(root / 'results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
