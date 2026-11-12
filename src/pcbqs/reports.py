"""
What a result folder of each tool holds, in the forms a front end shows without knowing the tool: a report to read
(a title, a line, sections of text and tables, notes, pictures) and curve sets to plot. Plain dicts and lists, read
from the tool's own result file; kicad2ems's Analysis.report and Analysis.curves are these (analyses.py), and
nothing here needs kicad2ems.

    report('pdn', folder)       dict(title, summary, sections=[dict(title, text, table=dict(columns, rows, formats),
                                notes)], notes, images, footer), or None: no result there
    curves('pdn', folder)       [dict(title, xlabel, ylabel, logx, logy, info, curves=[dict(name, x, y, role, unit)])]

A table's cells are numbers or strings; formats give a column's number format (None: '.4g'). The maps of a result
are not pictures here: each tool writes them as field dumps, which a viewer shows (pdn/viewer.py, thermal/viewer.py,
the near field's own); images holds only what has no dump (a circuit's fields picture).
"""
import json
import os

RESULTS = dict(pdn='pdn.json', mqs='mqs.json', thermal='thermal.json', circuit='circuit.json')
ALSO = dict(circuit=('power-circuit.json',))    # the full bridge's runs (pcbqs.power_circuit): the circuit's too
PREVIEWS = dict(mqs='mqs-preview.json')


def _load(folder, name):
    path = os.path.join(folder or '', name)
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _hz(f):
    return f"{f / 1e9:.3g} GHz" if f >= 1e9 else (f"{f / 1e6:.3g} MHz" if f >= 1e6 else f"{f / 1e3:.3g} kHz")


def _ohm(z):
    return f"{z:.3g} Ω" if z >= 1 else f"{z * 1e3:.3g} mΩ"


def _table(columns, rows, formats=None):
    return dict(columns=list(columns), rows=[list(r) for r in rows], formats=list(formats or [None] * len(columns)))


# ---- power integrity -----------------------------------------------------------------------------------
def pdn_report(res, folder=''):
    worst = max(res['loads'].items(), key=lambda kv: kv[1]['drop'], default=None)
    out = dict(title=f"{res['rail']}: {res['volts']:g} V", sections=[], notes=list(res.get('notes') or []), images=[],
               show=dict(field='V', run='power'))               # a viewer starts on the layer that drops most
    out['summary'] = f"Power integrity, rail {res['rail']}" + (
        f": worst DC drop {worst[1]['drop'] * 1e3:.2f} mV (at {worst[0]})" if worst else '')
    if res.get('no_return'):                                    # a drop without its ground's share: said first
        out['summary'] += (f"; without the return of {', '.join(res['no_return'])} (no pad on ground: "
                           f"REF=AMPS@PART names the part it returns through)")
    text = (f"Rail tree: {', '.join(res['nets'])}; source {', '.join(res['source'])}; "
            f"{res['loss_W'] * 1e3:.1f} mW in the copper.")
    if res.get('assumed_loads'):
        text += " The loads' currents are assumed (shared by pad count), not known."
    if res.get('field_loads'):
        text += f" From the parts' pdn.load fields: {', '.join(res['field_loads'])}."
    if res.get('returns'):
        text += " Returning through another part's ground pads: " + ', '.join(
            f"{r} through {b}" for r, b in sorted(res['returns'].items())) + '.'
    out['sections'].append(dict(title='', text=text))
    if res.get('tree'):
        rows = []
        for t in res['tree']:
            s = res['series'].get(t['ref'], {})
            rows.append([t['ref'], t['kind'] + (' (DCR assumed)' if t.get('defaulted') else ''), t['R'],
                         t['L'] * 1e6 if t.get('L') else '', s.get('I', 0.0), s.get('drop', 0.0) * 1e3,
                         f"{t['a']} → {t['b']}"])
        out['sections'].append(dict(title='Series parts', table=_table(
            ['part', 'kind', 'R (Ω)', 'L (µH)', 'current (A)', 'drop (mV)', 'between'], rows,
            [None, None, '.3g', '.3g', '.3g', '.2f', None])))
    rows = [[r, l['I'], l['V'], l['drop'] * 1e3, l['power_drop'] * 1e3, l['ground_rise'] * 1e3]
            for r, l in sorted(res['loads'].items(), key=lambda kv: -kv[1]['drop'])]
    out['sections'].append(dict(title='Loads at DC', table=_table(
        ['load', 'I (A)', 'sees (V)', 'drop (mV)', 'supply side (mV)', 'ground rise (mV)'], rows,
        [None, '.3g', '.4f', '.2f', '.2f', '.2f'])))
    vias = (res.get('vias') or {}).get('power') or []
    if vias:
        out['sections'].append(dict(title='Hottest supply vias', table=_table(
            ['current (A)', 'x (mm)', 'y (mm)'], [[v['I'], v['x'], v['y']] for v in vias[:10]], ['.2f', '.1f', '.1f'])))
    p = res.get('pdn')
    if p:
        out['sections'].append(dict(
            title='Plane pairs' + (f" (target {p['target'] * 1e3:.1f} mΩ)" if p.get('target') else ''),
            table=_table(['plane pair', 'd (mm)', 'εr', 'area (mm²)', 'C (nF)'],
                         [[f"{c['rail']} over {c['ground']}", c['d'], c['er'], c['area'], c['C'] * 1e9]
                          for c in p['cavities']], [None, '.3g', '.3g', '.0f', '.3g'])))
        rows = []
        for r, z in p['z'].items():
            t = z.get('target', p.get('target'))
            rows.append([r, t if t else '', ('; '.join(f"{_hz(a)} – {_hz(b)}" for a, b in z['over']) or 'never') if t
                         else 'no target', max(z['mag'])])
        out['sections'].append(dict(title='Impedance each load sees', table=_table(
            ['load', 'target (Ω)', 'over the target', 'at most (Ω)'], rows, [None, '.3g', None, '.3g'])))
        rows = []
        for n, i in sorted(p['parts'].items(), key=lambda kv: (kv[1]['role'], kv[0])):
            el = i['element']
            what = (f"{el['C'] * 1e6:.3g} µF, {el.get('ESR', 0) * 1e3:.3g} mΩ, {el.get('ESL', 0) * 1e9:.2f} nH"
                    if 'C' in el else (f"{el.get('R', 0) * 1e3:.3g} mΩ, {el.get('L', 0) * 1e9:.3g} nH" if el else ''))
            rows.append([n, i['role'], what + (' (ESR/ESL by package)' if i.get('defaulted') else ''), i['mount_nH'],
                         f"node {i['node']}" if i['node'] else i['rail']])
        out['sections'].append(dict(title='Parts', table=_table(
            ['part', 'role', 'element', 'mount (nH)', 'connection'], rows, [None, None, None, '.2f', None])))
        top = max(p['z'].items(), key=lambda kv: max(kv[1]['mag']))
        out['summary'] += f"; {len(p['z'])} load{'s' if len(p['z']) != 1 else ''}, at most {_ohm(max(top[1]['mag']))} (at {top[0]})"
    s = res.get('seconds', {})
    out['footer'] = (f"{res['vertices']['power']:,} + {res['vertices']['ground']:,} DC vertices; "
                     + ', '.join(f"{k} {v} s" for k, v in s.items()))
    return out


def pdn_curves(res, folder=''):
    p = res.get('pdn')
    if not p or not p.get('z'):
        return []
    f = [float(x) for x in p['f']]
    cs = dict(title='Impedance each load sees', xlabel='frequency (Hz)', ylabel='|Z| (Ω)', logx=True, logy=True,
              curves=[], info=pdn_report(res, folder)['summary'])
    for ref, z in p['z'].items():
        cs['curves'].append(dict(name=ref, x=f, y=z['mag'], role='data', unit='Ω'))
        cs['curves'].append(dict(name=f"{ref}, bare planes", x=f, y=z['bare'], role='reference', unit='Ω'))
    for t in sorted({t for t in [p.get('target'), *[z.get('target') for z in p['z'].values()]] if t}):
        cs['curves'].append(dict(name=f"target {_ohm(t)}", x=[f[0], f[-1]], y=[t, t], role='limit', unit='Ω'))
    return [cs]


# ---- near field ----------------------------------------------------------------------------------------
def mqs_report(res, folder=''):
    s = res['selection']
    if res.get('preview'):
        return dict(title=f"{s['converter']}: mesh preview", summary=f"Near field, {s['converter']}: mesh preview",
                    sections=[dict(title='', text=f"{res['triangles']:,} triangles. No matrix fill or solve."),
                              dict(title='Mesh', table=_table(
                                  ['layer', 'triangles', 'max edge (mm)', 'min angle (°)'],
                                  [[l, v['triangles'], v['max_edge_mm'], v['min_angle_deg']]
                                   for l, v in res['layers'].items()], [None, 'd', '.3g', '.2f']))],
                    notes=[], images=[], footer='', mesh='mqs')
    n = len(res['frequencies_Hz'])
    rows = [[f, a, r * 1e3, l * 1e9, max(hp)] for f, a, r, l, hp in zip(
        res['frequencies_Hz'], res['harmonic_peak_A'], res['copper_R_ohm'], res['copper_L_H'], res['peak_H_Am'])]
    return dict(
        title=f"{s['converter']}: {s['vin']} / {s['ground']}",
        summary=f"Near field, {s['converter']} on {s['vin']}: {n} harmonic{'s' if n != 1 else ''}",
        sections=[dict(title='', text=f"Input capacitors: {', '.join(s['capacitors'])}; {res['triangles']:,} triangles."),
                  dict(title='Harmonics', table=_table(
                      ['frequency (Hz)', 'peak (A)', 'copper R (mΩ)', 'copper L (nH)', 'peak H (A/m)'], rows,
                      ['g', '.4g', '.4g', '.4g', '.4g'])),
                  dict(title='Model assumptions', notes=list(res.get('notes') or []))],
        notes=[], images=[], mesh='mqs',
        footer=', '.join(f"{k} {v:g} s" for k, v in (res.get('seconds') or {}).items()
                         if isinstance(v, (int, float))))


def mqs_curves(res, folder=''):
    if res.get('preview') or not res.get('frequencies_Hz'):
        return []
    return [dict(title='Peak magnetic field per harmonic', xlabel='frequency (Hz)', ylabel='peak |H| (A/m)',
                 logx=True, logy=True, info=mqs_report(res, folder)['summary'],
                 curves=[dict(name='peak |H|', x=res['frequencies_Hz'], y=[max(row) for row in res['peak_H_Am']],
                              role='data', unit='A/m')])]


def mqs_suggest(res, folder=''):
    """The converter's input rail, for the power-integrity analysis: its net, its ground and its voltage; the
    source and the loads are not the near field's to know."""
    s = res['selection']
    volts = (res.get('settings') or {}).get('vin_volts') or ''
    return [dict(to='pdn', label='Copy the input rail to Power integrity',
                 values=dict(rail=s['vin'], ground=s['ground'], ground_source=s['converter'], volts=volts, load=''),
                 need=[] if volts else ['volts'],
                 note='The input rail is copied. Set its source' + ('' if volts else ', its voltage')
                      + ' and its DC loads before Run.')]


# ---- temperature ---------------------------------------------------------------------------------------
def _methods(res):
    if 'method' in res:                                     # one solver: its summary alone
        return {res['method']: res}
    return {k: v for k, v in res.items() if k in ('grid', 'mesh')}


def thermal_report(res, folder=''):
    ms = _methods(res)
    if not ms:
        return None
    first = next(iter(ms.values()))
    names = list(ms)
    hot = max(m['hottest_C'] for m in ms.values())
    src = os.path.basename(first.get('board') or '')
    if first.get('circuit'):
        src += f"; losses of circuit case {first.get('case')} ({os.path.basename(str(first['circuit']).rstrip('/'))})"
    text = (f"{src}. " if src else '') + (
        f"{first['heat_in_W']:.3g} W in, {first['heat_out_W']:.3g} W given off "
        f"({first.get('heat_out_edges_W', 0):.3g} W of it through the edges); air {first['ambient_C']:g} °C, films "
        f"{first['film_W_m2K'][1]:g} / {first['film_W_m2K'][0]:g} W/m²K top / bottom, "
        f"{first.get('edge_film_W_m2K', 0):g} at the edges.")
    parts = sorted(first['parts'].items(), key=lambda kv: -kv[1]['part_C'])
    rows = [[ref, p['power_W'], *[ms[n]['parts'].get(ref, {}).get('board_C', '') for n in names], p['part_C'],
             p['part_C'] - first['ambient_C']] for ref, p in parts]
    out = dict(title=f"Hottest copper {hot:.1f} °C", sections=[dict(title='', text=text)], images=[],
               show=dict(field='T'),                            # a viewer starts on the hottest layer
               summary=f"Thermal: hottest copper {hot:.1f} °C" + (
                   f", hottest part {parts[0][0]} at {parts[0][1]['part_C']:.1f} °C" if parts else ''))
    out['sections'].append(dict(title='Parts', table=_table(
        ['part', 'loss (W)', *[f"board °C ({n})" for n in names], 'part °C', 'rise (K)'], rows,
        [None, '.3g', *['.1f'] * len(names), '.1f', '.1f'])))
    out['sections'].append(dict(title='Layers', table=_table(
        ['layer', 'hottest °C', 'mean °C'],
        [[lay, v['hottest_C'], v['mean_C']] for lay, v in first['layers'].items()], [None, '.1f', '.1f'])))
    d = res.get('difference')
    if d:
        out['sections'].append(dict(title='Mesh against grid', text=(
            f"Parts within {max(abs(v) for v in d['parts_K'].values()):.2f} K, the maps within {d['largest_K']:.2f} K.")))
    out['notes'] = list(dict.fromkeys(first.get('notes') or []))
    out['footer'] = ('; '.join(f"{n}: {ms[n]['unknowns']:,} unknowns at {ms[n]['size_mm']:g} mm" for n in names) + '; '
                     + ', '.join(f"{k} {v:g} s" for k, v in (first.get('seconds') or {}).items()))
    return out


# ---- circuits ------------------------------------------------------------------------------------------
def _num(v):
    return v if isinstance(v, (int, float)) else ''


def circuit_report(res, folder=''):
    """A general circuit run (format 2): a row a point, then for each point where its loss goes and its parts'
    stress; the full bridge's (format 1): its operating points."""
    if res.get('format') != 2:
        rows = [[c['frequency_Hz'] / 1e3, c['load_power_W'], _num(c.get('input_power_W')), 100 * c['efficiency'],
                 c['phase_deg'], c['primary_rms_A'], _num(c.get('bus_ripple_V')), _num(c.get('bus_spike_pp_V')),
                 _num(c.get('runtime_s'))] for c in res.get('cases', [])]
        best = max((c for c in res.get('cases', []) if c.get('efficiency')), key=lambda c: c['efficiency'], default=None)
        name = res.get('name') or 'Full bridge'
        notes = ["Bus ripple: peak to peak of the final cycle sampled at T/256, as the waveforms are by default. Bus "
                 "spikes: peak to peak at every solver timestep, switching-edge spikes included (the curve sets "
                 "named 'solver steps' show them).",
                 "The receiver, its load and unmeasured parasitics are model assumptions: model.json and each "
                 "case.json have the matching, the settling and the power checks."]
        slow = next((c for c in res.get('cases', []) if (c.get('averaged_cycles') or 1) > 1), None)
        if slow:
            notes.append(f"Powers and efficiency: averaged over whole periods of the supply filter's slow mode "
                         f"({slow['slow_mode_Hz'] / 1e3:.3g} kHz: {slow['averaged_cycles']} switching cycles at "
                         f"{slow['frequency_Hz'] / 1e3:g} kHz), so a mode still ringing does not shift them.")
        if res.get('matching_definition'):
            notes.append(res['matching_definition'])
        if any('field_inference' in c for c in res.get('cases', [])):
            notes.append("Fields: band-limited inference from the copper's currents; external coils, package "
                         "filaments and coupling to the surroundings are left out. Each point's are in its folder, "
                         "which the viewer opens as a run.")
        notes += [str(a) for a in res.get('assumptions') or []]
        return dict(title=name, summary=f"Circuit, {name}: {len(rows)} operating point{'s' if len(rows) != 1 else ''}"
                    + (f", efficiency up to {100 * best['efficiency']:.1f} %" if best else ''),
                    sections=[dict(title='Operating points', table=_table(
                        ['kHz', 'load (W)', 'input (W)', 'efficiency %', 'phase °', 'primary RMS (A)', 'bus ripple (V)',
                         'bus spikes p-p (V)', 'run (s)'], rows,
                        ['g', '.3f', '.3f', '.2f', '.2f', '.3f', '.2f', '.2f', '.0f']))],
                    notes=notes, images=[], footer='Completed.' if res.get('complete') else 'Partial results.')
    cases = res.get('cases', [])
    knobs = sorted({k for c in cases for k in c.get('knobs', {})})
    measures = sorted({k for c in cases for k in c.get('measures', {})})
    rows = [[c['name'], *[_num(c['knobs'].get(k)) for k in knobs], _num(c.get('source_power_W')),
             _num(c.get('load_power_W')), _num(c.get('loss_W')), 100 * c['efficiency'] if c.get('efficiency') else '',
             *[_num(c['measures'].get(m)) for m in measures], c.get('runs', ''), _num(c.get('runtime_s'))] for c in cases]
    name = res.get('name') or 'Circuit'
    best = max((c for c in cases if c.get('efficiency')), key=lambda c: c['efficiency'], default=None)
    out = dict(title=name, sections=[], images=[],
               notes=[f"{f['name']}: failed. {f['error']}" for f in res.get('failed', [])],
               summary=f"Circuit, {name}: {len(cases)} point{'s' if len(cases) != 1 else ''}" + (
                   f", efficiency up to {100 * best['efficiency']:.1f} %" if best else ''))
    out['sections'].append(dict(title='Points', table=_table(
        ['point', *[k.split('.', 1)[-1] for k in knobs], 'in (W)', 'out (W)', 'loss (W)', 'efficiency %', *measures,
         'runs', 'time (s)'], rows,
        [None, *['.4g'] * len(knobs), '.4g', '.4g', '.4g', '.2f', *['.4g'] * len(measures), 'd', '.0f'])))
    for c in cases:
        together = {r for grp in c.get('coupled', []) for r in grp['inductors']}
        loss = [(r, p['power_W']) for r, p in c['parts'].items()
                if p['type'] not in ('source', 'load') and r not in together]
        loss += [(' + '.join(grp['inductors']), grp['power_W']) for grp in c.get('coupled', [])]
        if c.get('copper'):
            loss.append((f"board copper ({c['copper'].get('board', '')})", c['copper']['loss_W']))
        if c.get('gate_drive_power_W'):
            loss.append(('gate drives', -c['gate_drive_power_W']))
        total = max(c.get('loss_W') or 0.0, 1e-30)
        out['sections'].append(dict(title=f"{c['name']}: where the {c.get('loss_W', 0):.4g} W goes", table=_table(
            ['part', 'loss (W)', '%'], [[r, w, 100 * w / total] for r, w in sorted(loss, key=lambda x: -x[1])
                                        if abs(w) > 1e-4 * total], [None, '.4g', '.1f'])))
        stress = [(r, p) for r, p in c['parts'].items() if p['type'] in ('switch', 'diode', 'capacitor', 'inductor')]
        out['sections'].append(dict(
            title=f"{c['name']}: stress",
            table=_table(['part', 'avg (A)', 'RMS (A)', 'peak (A)', 'peak (V)', 'avg (V)', 'ripple (V)', 'ripple (A)'],
                         [[r, *[_num(p.get(k)) for k in ('avg_A', 'rms_A', 'peak_A', 'peak_V', 'avg_V', 'ripple_V',
                                                         'ripple_A')]] for r, p in stress], [None] + ['.4g'] * 7),
            notes=list(c.get('notes') or []),
            text=f"Settled to {100 * c.get('periodic_error', 0):.2g} % period to period; solver tolerance "
                 f"{c.get('reltol', '–')}; {c.get('periods', '–')} periods simulated."))
        if c.get('copper') and c['copper'].get('ports'):
            out['sections'].append(dict(title=f"{c['name']}: board copper", table=_table(
                ['pad', 'net', 'to', 'RMS (A)'],
                [[p['terminal'].rsplit('.', 1)[0], p['net'], p['reference'].rsplit('.', 1)[0], _num(p.get('rms_A'))]
                 for p in c['copper']['ports']], [None, None, None, '.4g'])))
        f = c.get('fields')
        if f:
            out['sections'].append(dict(
                title=f"{c['name']}: fields over the board (AC RMS, the largest on each plane)",
                table=_table(['height (mm)', 'H (A/m)', 'E (V/m)'], list(zip(
                    f['heights_mm'], f['H_ac_rms_max_A_per_m'], f['E_ac_rms_max_V_per_m'])), ['.4g'] * 3),
                text=f"The copper and its vias only: no part's own field. H to {f['highest_Hz'] / 1e6:.4g} MHz; "
                     "E from the nets' voltages."))
            if f.get('picture') and os.path.isfile(f['picture']):
                out['images'].append(f['picture'])
    out['footer'] = 'Completed.' if res.get('complete') else 'Partial results.'
    return out


def _case_folder(case, folder):
    """A point's folder: the one it names, else its name under the run's (a run moved since)."""
    d = case.get('folder') or ''
    return d if os.path.isdir(d) else os.path.join(folder, case.get('name', ''))


BRIDGE_VIEWS = (('winding currents', ('PRIMARY.winding_I', 'SECONDARY.winding_I'), 'A'),
                ('bridge and coil voltages', ('bridge.V', 'PRIMARY.V', 'SECONDARY.V'), 'V'),
                ('supply filter currents', None, 'A'),              # the series parts between the supply's rails
                ('bus voltages', ('bus.V', 'input.V', 'supply.V'), 'V'))


def _supply_series(model):
    """The parts in series between the supply's rails (VIN, VBUS, VHB), by what they join: the input filter's
    inductor and its damping resistor on wpthb, whatever a model has there."""
    net = lambda t: t[4:] if str(t).startswith('NET:') else (model.get('terminals') or {}).get(t, '')
    rails = ({'VIN', 'VBUS'}, {'VBUS', 'VHB'})
    return [p['ref'] for p in model.get('components', []) if p.get('type') in ('inductor', 'resistor')
            and {net(p.get('a', '')), net(p.get('b', ''))} in rails]


def bridge_curves(res, folder=''):
    """The full bridge's waveforms over the last two periods, per operating point: the windings' currents, the
    bridge's and the coils' voltages, the currents of the series parts between the supply's rails, the bus
    voltages; and each again at the solver's own timesteps where the run saved them (the switching edges' spikes,
    which the samples at T/256 miss)."""
    import numpy as np
    info = circuit_report(res, folder)['summary']
    model = _load(folder, 'model.json') or {}
    series = tuple(r + '.I' for r in _supply_series(model))
    sets = []
    for c in res.get('cases', []):
        d = _case_folder(c, folder)
        try:
            with np.load(os.path.join(d, 'transient.npz')) as data:
                v = {k: data[k] for k in data.files}
        except (OSError, ValueError):
            continue
        name = f"{c['frequency_Hz'] / 1e3:g} kHz, {c.get('target_W', c.get('load_power_W', 0)):g} W"
        two = 2 / c['frequency_Hz']
        for stepped in (False, True):
            if stepped and 'steps.time_s' not in v:
                continue
            w = {**v, 'time_s': v['steps.time_s'], **{k[6:]: x for k, x in v.items() if k.startswith('steps.')}} if stepped else v
            t = w['time_s']
            take = t >= t[-1] - two
            x = ((t[take] - (t[-1] - two)) * 1e6).tolist()
            for what, keys, unit in BRIDGE_VIEWS:
                keys = keys if keys is not None else (series or tuple(
                    k for k in w if k.endswith('.I') and not k.startswith(('PRIMARY', 'SECONDARY', 'steps.'))))
                have = [k for k in keys if k in w and len(w[k]) == len(t)]
                if have:
                    sets.append(dict(title=f"{name}: {what}" + (', solver steps' if stepped else ''),
                                     xlabel='time in the final two periods (µs)', ylabel=unit, logx=False, logy=False,
                                     info=info, curves=[dict(name=k, x=x, y=w[k][take].tolist(), role='data', unit=unit)
                                                        for k in have]))
    return sets


def circuit_curves(res, folder=''):
    """A general run's waveforms over the last two periods: per point, its probes' voltages and currents (a model
    without probes: its capacitors' voltages and inductors' currents). The full bridge's: bridge_curves."""
    if res.get('format') != 2:
        return bridge_curves(res, folder)
    import numpy as np
    sets = []
    for c in res.get('cases', []):
        d = _case_folder(c, folder)
        try:
            with np.load(os.path.join(d, 'transient.npz')) as data:
                v = {k: data[k] for k in data.files}
            model = _load(d, 'model.json') or {}
        except (OSError, ValueError):
            continue
        probes = model.get('probes', {})
        kinds = {p['ref']: p.get('type') for p in model.get('parts', [])}
        volts = [n for n, p in probes.items() if 'voltage' in p] or [r + '.V' for r, k in kinds.items() if k == 'capacitor']
        amps = [n for n, p in probes.items() if 'current' in p] or [r + '.I' for r, k in kinds.items() if k == 'inductor']
        T, t = c['period_s'], v['time_s']
        take = t >= t[-1] - 2 * T
        scale, unit = (1e3, 'ms') if T >= 1e-3 else (1e6, 'µs')
        x = ((t[take] - (t[-1] - 2 * T)) * scale).tolist()
        for what, keys, u in (('voltages', volts, 'V'), ('currents', amps, 'A')):
            have = [k for k in keys if k in v]
            if have:
                sets.append(dict(title=f"{c['name']}: {what}", xlabel=f"time in the final two periods ({unit})",
                                 ylabel=u, logx=False, logy=False, info=circuit_report(res, folder)['summary'],
                                 curves=[dict(name=k, x=x, y=v[k][take].tolist(), role='data', unit=u) for k in have]))
    return sets


BUILDERS = dict(pdn=(pdn_report, pdn_curves), mqs=(mqs_report, mqs_curves), thermal=(thermal_report, None),
                circuit=(circuit_report, circuit_curves))


def result(kind, folder):
    """A folder's result of a tool (its preview's when that is all there is), or None."""
    for name in (RESULTS[kind], *ALSO.get(kind, ()), PREVIEWS.get(kind)):
        res = _load(folder, name) if name else None
        if res is not None:
            if name == PREVIEWS.get(kind):
                res.setdefault('preview', True)
            return res
    return None


def report(kind, folder):
    res = result(kind, folder)
    return None if res is None else BUILDERS[kind][0](res, folder or '')


def curves(kind, folder):
    res = result(kind, folder)
    make = BUILDERS[kind][1]
    return [] if res is None or make is None else make(res, folder or '')
