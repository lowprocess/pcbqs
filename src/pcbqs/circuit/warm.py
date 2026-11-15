"""pcbqs.circuit --warm N (R-E9): the circuit run at its parts' own temperatures. The run's losses go to the board's
thermal model (pcbqs.thermal, each case its own map); each part is set to the temperature the map gives it (its
pads', or its junction's where its footprint gives thermal.theta) and the board's copper to its copper's mean (the
mean under the copper's own loss where the run has its map, --fields), and the circuit is run again, up to N
times, until no part moves by more than TOLERANCE_K. Their resistances follow their temperatures as
circuit/temperature.py has it; nothing else in the circuit does (a vendor model's own temperature is its own).

<out>/circuit.json gets 'warm': each pass's largest move and its hottest part, and whether it settled; each case
its 'temperatures_C'. <out>/thermal is the thermal model of the last run's losses (thermal --case all)."""
import contextlib
import io
import json
import shlex
from pathlib import Path

import numpy as np

from . import model as M
from .temperature import REFERENCE_C

TOLERANCE_K = 1.0


def copper_temperature(case, folder):
    """The board copper's temperature for a case from its thermal results folder: the mean under the copper's own
    loss where the case has its map (copper-loss.npz) and the thermal model's grid; else the copper layers' mean."""
    r = json.loads((folder / 'thermal.json').read_text())
    loss = Path(case.get('folder', '')) / 'copper-loss.npz'
    maps = folder / 'thermal.npz'
    if loss.is_file() and maps.is_file():
        with np.load(maps, allow_pickle=False) as z, np.load(loss, allow_pickle=False) as c:
            if 'grid_T' in z.files:
                layers = [str(x) for x in z['layers']]
                (x0, y0), pitch = z['grid_origin'], float(z['grid_pitch'])
                T = z['grid_T']
                total = weighted = 0.0
                for lay, x, y, w in zip(c['layer'], c['x_mm'], c['y_mm'], c['W']):
                    if str(lay) not in layers or w <= 0:
                        continue
                    t = T[layers.index(str(lay))]
                    i = min(max(int((y - y0) / pitch), 0), t.shape[0] - 1)
                    j = min(max(int((x - x0) / pitch), 0), t.shape[1] - 1)
                    if np.isfinite(t[i, j]):
                        total += w
                        weighted += w * t[i, j]
                if total > 0:
                    return weighted / total
    means = [v['mean_C'] for v in r['layers'].values() if np.isfinite(v['mean_C'])]
    return float(np.mean(means)) if means else None


def temperatures(model, summary, thermal_out):
    """{case name: {'parts': {ref: C}, 'board': C}} from thermal --case all's results in thermal_out."""
    table = json.loads((thermal_out / 'thermal-cases.json').read_text())
    bound = (model.get('board') or {}).get('parts', {})
    out = {}
    for c in summary['cases']:
        parts = {}
        for p in model.get('parts', []):
            given = bound.get(p['ref'], p['ref'])
            if isinstance(given, list):
                given = given[0] if len(set(given)) == 1 else None
            at = (table['parts'].get(given) or {}).get(c['name']) if given else None
            if at and at.get('part_C') is not None:
                parts[p['ref']] = round(float(at['part_C']), 2)
        out[c['name']] = dict(parts=parts, board=copper_temperature(c, thermal_out / c['name'])
                              if c.get('copper') else None)
    return out


def _moved(new, old, start):
    """The largest move (K) of any part or the copper from old to new (start: {ref: C}, the parts' own before
    the first), and where: (K, 'case: ref')."""
    worst = (0.0, '')
    for name, t in new.items():
        before = (old or {}).get(name) or dict(parts={}, board=None)
        for ref, c in t['parts'].items():
            d = abs(c - before['parts'].get(ref, start.get(ref, REFERENCE_C)))
            if d > worst[0]:
                worst = (d, f"{name}: {ref}")
        if t.get('board') is not None:
            d = abs(t['board'] - (before.get('board') if before.get('board') is not None else
                                  float((start.get('board') or REFERENCE_C))))
            if d > worst[0]:
                worst = (d, f"{name}: the copper")
    return worst


def run(model, out, passes, log=print, thermal_args='', **run_kw):
    """The circuit run, then up to `passes` runs again at the temperatures the thermal model gives the last one's
    losses: the last run's summary (circuit.json's), with 'warm'."""
    from ..thermal.__main__ import main as thermal_main
    from .engine import run as run_circuit
    if not (model.get('board') or {}).get('file'):
        raise M.ModelError('--warm needs the board (board.file, or --board) for the thermal model')
    out = Path(out).resolve()
    thermal_out = out / 'thermal'
    extra = shlex.split(thermal_args or '')
    temps, history = None, []
    start = {p['ref']: float(p.get('temperature_C', REFERENCE_C)) for p in model.get('parts', [])}
    start['board'] = (model.get('board') or {}).get('temperature_C')
    for k in range(passes + 1):
        summary = run_circuit(model, out, log=log, temperatures=temps, **run_kw)
        if summary['failed'] or not summary['cases']:
            break
        log(f"warm pass {k + 1}: the thermal model of the run's losses")
        said = io.StringIO()                                        # (its tables: in thermal/thermal.log)
        with contextlib.redirect_stdout(said):
            code = thermal_main(['--circuit', str(out), '--case', 'all', '--out', str(thermal_out), *extra])
        (thermal_out / 'thermal.log').write_text(said.getvalue())
        if code not in (0, None):
            raise RuntimeError(f"the thermal model failed (python -m pcbqs.thermal exit {code})")
        new = temperatures(model, summary, thermal_out)
        moved, where = _moved(new, temps, start)
        hottest = max(((c, f"{n}: {r}") for n, t in new.items() for r, c in t['parts'].items()), default=(None, ''))
        history.append(dict(run=k + 1, largest_move_K=round(moved, 3), where=where, hottest_C=hottest[0],
                            hottest=hottest[1]))
        log(f"warm pass {k + 1}: parts moved by up to {moved:.2f} K ({where}); the hottest {hottest[1]} at "
            f"{hottest[0]} C" if hottest[0] is not None else f"warm pass {k + 1}: no part on the board")
        if moved <= TOLERANCE_K or k == passes:
            break
        temps = new
    settled = bool(history) and history[-1]['largest_move_K'] <= TOLERANCE_K
    for c in summary['cases']:
        c['temperatures_C'] = (temps or {}).get(c['name'], dict(parts={}, board=None))
    summary['warm'] = dict(passes=history, settled=settled, tolerance_K=TOLERANCE_K, thermal=str(thermal_out))
    (out / 'circuit.json').write_text(json.dumps(summary, indent=1))
    if history and not settled:
        log(f"warm: still moving by {history[-1]['largest_move_K']:.2f} K after {len(history)} runs (raise --warm)")
    return summary
