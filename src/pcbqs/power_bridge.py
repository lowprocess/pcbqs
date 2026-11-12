"""Configurable full-bridge example: electrical circuit, matching and PCB terminals."""

import copy, json
from pathlib import Path
import numpy as np
from .power_spice import Circuit, periodic_samples
from .wpt import match_receiver, link, impedance


def validate(config):
    if config.get('format') != 1:
        raise ValueError('unsupported power-circuit config format')
    for key in ['primary', 'secondary']:
        c = config[key]
        for k in ['L_H', 'R_ohm']:
            if not np.isfinite(c[k]) or c[k] <= 0:
                raise ValueError(key + '.' + k + ' must be positive')
    if not 0 < config['coupling'] < 1:
        raise ValueError('coupling must be between 0 and 1')
    for k in ['voltage_V', 'primary_C_F']:
        if not np.isfinite(config[k]) or config[k] <= 0:
            raise ValueError(k + ' must be positive')
    for k in ['frequencies_Hz', 'powers_W']:
        if not config[k] or not all(np.isfinite(v) and v > 0 for v in config[k]):
            raise ValueError(k + ' must be positive')
    return config


def load_config(path):
    path = Path(path).resolve()
    c = validate(json.loads(path.read_text()))
    for model in [c.get('switch_model', {}), *[s.get('model', {}) for s in c['switches']]]:
        if model.get('library'):
            model['library'] = str((path.parent / model['library']).resolve())
    return c


def matching(config, f):
    return match_receiver(
        f,
        config['primary'],
        config['secondary'],
        config['coupling'],
        config['primary_C_F'],
        config.get('primary_cap_ESR_ohm', 0),
        config.get('secondary_cap_ESR_ohm', 0),
        C2=config.get('secondary_C_F'),
        load=config.get('load_ohm'),
    )


def predicted_phase(config, f, power, match):
    r = link(
        f,
        config['primary'],
        config['secondary'],
        config['coupling'],
        config['primary_C_F'],
        match['C2_F'],
        match['load_ohm'],
        config.get('primary_cap_ESR_ohm', 0),
        config.get('secondary_cap_ESR_ohm', 0),
    )
    # Fundamental sinusoidal estimate only; ngspice adjusts against actual load power.
    V = np.sqrt(power / match['load_ohm']) * abs(r['input_Z_ohm']) / abs(r['secondary_per_primary'])
    vfull = 2 * np.sqrt(2) * config['voltage_V'] / np.pi
    return float(np.degrees(2 * np.arcsin(min(V / vfull, 0.999))))


def end_state(result, config, ports=None):
    """The circuit's state at the end of a run (a whole number of switching periods: the start's phase).

    Capacitors: the voltage on the capacitance itself (terminal voltage less the ESR drop); inductors,
    the source inductance, windings: their currents; coil self-capacitance: the terminal voltage; the
    copper model: its port currents and modal currents. build(..., state=) starts the next run from it (a warm start).
    """
    last = lambda key: float(result[key][-1])
    stores = [p for p in config['components'] if p.get('enabled', True)]
    stores += [
        dict(ref='LSOURCE', type='inductor'),
        dict(ref='CSECONDARY', type='capacitor', ESR_ohm=config.get('secondary_cap_ESR_ohm', 0)),
    ]
    parts = {}
    for p in stores:
        ref = p['ref']
        if ref + '.I' not in result:
            continue
        if p['type'] == 'capacitor':
            parts[ref] = dict(
                initial_V=last(ref + '.V') - p.get('ESR_ohm', 0) * last(ref + '.I'), initial_I=last(ref + '.I')
            )
        elif p['type'] == 'inductor':
            parts[ref] = dict(initial_A=last(ref + '.I'))
    switches = {sw['ref']: last(sw['ref'] + '.I') for sw in config.get('switches', []) if sw['ref'] + '.I' in result}
    modes = None  # the copper model's modal inductor currents
    if 'MODE0.I' in result:
        modes = []
        k = 0
        while f'MODE{k}.I' in result:
            modes.append(last(f'MODE{k}.I'))
            k += 1
    coils = {
        name: dict(initial_A=last(name + '.winding_I'), initial_V=last(name + '.V'))
        for name in ('PRIMARY', 'SECONDARY')
        if name + '.winding_I' in result
    }
    port_currents = (
        [last(f'PCB{i}.I') for i in range(len(ports or []))]
        if ports and all(f'PCB{i}.I' in result for i in range(len(ports)))
        else None
    )
    return dict(parts=parts, coils=coils, ports=port_currents, switches=switches, modes=modes)


def net_of(config, terminal):
    """The net a component terminal is on: NET:name's name, else the model's terminals' ('' when unknown)."""
    return terminal[4:] if terminal.startswith('NET:') else config.get('terminals', {}).get(terminal, '')


GROUPS = ('tank', 'bridge-rail', 'input-rail', 'filter')     # the parts a setting can be given for together
SLOW_FIT = 0.06                         # of a slow mode's period: how far an averaging window may be from whole ones


def groups(config):
    """{group: [refs]}: the model's parts by what they join, not by their references. tank: the capacitors
    between the bridge's output and the primary coil; bridge-rail and input-rail: the capacitors from VHB and
    from VBUS to ground; filter: the inductors and resistors in series between the supply's rails (VIN, VBUS,
    VHB). On wpthb: C25/C27/C30, C19 to C24, CA1 to CA4, and L3 with R9."""
    tank = {net_of(config, config['bridge_terminals'][0]), net_of(config, config['primary_terminals'][0])}
    out = {g: [] for g in GROUPS}
    for p in config.get('components', []):
        ends = {net_of(config, p['a']), net_of(config, p['b'])}
        if p['type'] == 'capacitor' and ends == tank and len(tank) == 2:
            out['tank'].append(p['ref'])
        elif p['type'] == 'capacitor' and ends == {'VHB', 'GND'}:
            out['bridge-rail'].append(p['ref'])
        elif p['type'] == 'capacitor' and ends == {'VBUS', 'GND'}:
            out['input-rail'].append(p['ref'])
        elif p['type'] in ('inductor', 'resistor') and ends in ({'VIN', 'VBUS'}, {'VBUS', 'VHB'}):
            out['filter'].append(p['ref'])
    return out


def apply_overrides(config, overrides):
    """GROUP.KEY=VALUE items onto a model: the value for KEY of every part of the group (groups()), where the
    part's type has it (a capacitor's ESR_ohm and ESL_H, an inductor's DCR_ohm, any key the part already has);
    'switch.KEY': the switch model's (rise_s, fall_s, dead_time_s, Cds_F ...). Returns {item: the parts it set}.
    ValueError for an item of another form, a group nobody is in, or a key no part of the group takes."""
    takes = dict(capacitor=('ESR_ohm', 'ESL_H', 'value'), inductor=('DCR_ohm', 'value'), resistor=('value',))
    by, done = groups(config), {}
    for item in overrides or []:
        head, sep, value = str(item).partition('=')
        group, dot, key = head.strip().partition('.')
        try:
            number = float(value)
        except ValueError:
            number = None
        if not sep or not dot or not key or number is None:
            raise ValueError(f"override {item!r}: GROUP.KEY=VALUE, e.g. tank.ESR_ohm=0.005")
        if group == 'switch':
            config.setdefault('switch_model', {})[key] = number
            done[item] = ['switch model']
            continue
        if group not in by:
            raise ValueError(f"override {item!r}: no group {group!r}: one of {', '.join(GROUPS)}, switch")
        refs = [p['ref'] for p in config['components'] if p['ref'] in by[group]
                and (key in takes.get(p['type'], ()) or key in p)]
        if not refs:
            raise ValueError(f"override {item!r}: " + (f"no part of {group} takes {key}" if by[group] else
                                                       f"the model has no part in {group}"))
        for p in config['components']:
            if p['ref'] in refs:
                p[key] = number
        done[item] = refs
    return done


def slow_mode(config):
    """The frequency (Hz) of the slowest mode of the supply's filter, or None when it has none: the inductors in
    series between the supply's rails (VIN, VBUS, VHB) against the rails' capacitors to ground, the supply itself
    stiff, a series resistor a connection. wpthb's L3 against C19 to C24: 10.4 kHz. Lightly damped, it rings for
    many switching cycles after anything disturbs the input current."""
    rails = ['VBUS', 'VHB']
    tied = {'VIN': 'VIN', 'VBUS': 'VBUS', 'VHB': 'VHB'}
    parts = [p for p in config.get('components', []) if p.get('enabled', True)]
    ends = lambda p: (net_of(config, p['a']), net_of(config, p['b']))
    for p in parts:                                             # a resistor between two rails: one node
        a, b = ends(p)
        if p['type'] == 'resistor' and a in tied and b in tied and a != b:
            keep, gone = (a, b) if a == 'VIN' or (b != 'VIN' and a < b) else (b, a)
            tied = {k: (keep if v == tied[gone] else v) for k, v in tied.items()}
    nodes = sorted({tied[r] for r in rails} - {tied['VIN']})
    if not nodes:
        return None
    at = {n: i for i, n in enumerate(nodes)}
    gamma, cap = np.zeros((len(nodes), len(nodes))), np.zeros(len(nodes))
    for p in parts:
        a, b = ends(p)
        if p['type'] == 'inductor' and a in tied and b in tied and tied[a] != tied[b] and p.get('value', 0) > 0:
            for x, y in ((tied[a], tied[b]), (tied[b], tied[a])):
                if x in at:
                    gamma[at[x], at[x]] += 1 / p['value']
                    if y in at:
                        gamma[at[x], at[y]] -= 1 / p['value']
        elif p['type'] == 'capacitor' and {a, b} & set(tied) and 'GND' in (a, b):
            rail = tied[a if a in tied else b]
            if rail in at:
                cap[at[rail]] += p.get('value', 0)
    if not gamma.any() or (cap <= 0).any():
        return None
    w2 = np.linalg.eigvals(gamma / cap[:, None]).real
    w2 = w2[w2 > 1e-6 * w2.max()]
    return float(np.sqrt(w2.min()) / (2 * np.pi)) if len(w2) else None


def average_cycles(config, f):
    """The switching cycles the powers of a run at f are averaged over: one, or with a slow mode of the supply's
    filter (slow_mode) a whole number of cycles that is a whole number of its periods to within SLOW_FIT of one
    (the fewest such; none within two and a half periods: the nearest). A mode that has not died out then adds
    next to nothing to the average, where over one cycle it shifts the input power by its swing at that moment
    (C8: 0.66 percent on wpthb). transient.slow_average false: one."""
    slow = slow_mode(config) if config.get('transient', {}).get('slow_average', True) else None
    if not slow or slow > f / 2:
        return 1
    period = f / slow                                           # in switching cycles
    off = lambda n: abs(n / period - max(round(n / period), 1))
    span = range(max(int(period), 2), int(2.5 * period) + 2)
    good = [n for n in span if off(n) <= SLOW_FIT]
    return int(min(good[0] if good else min(span, key=lambda n: (round(off(n), 6), n)), 64))


def supply_drops(config):
    """The DC resistance from the supply to the input rail (VBUS) and on to the bridge rail (VHB), ohm: the
    source's, then the series resistors' and inductors' between the rails. They set the rails' starting voltages
    under the input current. wpthb's (R9 and L3 behind a 50 mΩ source) are 0.055 and 0.085."""
    def series(x, y):
        total = 0.0
        for p in config['components']:
            if p.get('enabled', True) and {net_of(config, p['a']), net_of(config, p['b'])} == {x, y}:
                if p['type'] == 'resistor':
                    total += p['value']
                elif p['type'] == 'inductor':
                    total += p.get('DCR_ohm', 0)
        return total

    bus = max(config.get('supply', {}).get('R_ohm', 0.05), 1e-9) + series('VIN', 'VBUS')
    hb = bus + series('VBUS', 'VHB')
    return bus, hb


def build(config, f, phase, match, copper=None, ports=None, state=None):
    """The ngspice circuit at `phase`. state (end_state of a previous run) warm-starts it: every capacitor,
    inductor, winding and copper mode begins where that run ended, instead of the precharge estimate.
    The config is validated by load_config/run_case, not here (build runs once per phase trial)."""
    if not np.isfinite(phase) or not 0 <= phase <= 180:
        raise ValueError('phase must be in [0,180]')
    config = copy.deepcopy(config)
    c = Circuit()
    ports = ports or []
    precharged = config.get('transient', {}).get('initialization', 'ramp') == 'precharged' or state is not None
    initial_input = 0.0
    if precharged:
        primary_initial = secondary_initial = primary_v = secondary_v = cap1_v = cap2_v = 0.0
        for n in range(1, 32, 2):
            vc = -1j * config['voltage_V'] / (np.pi * n) * (1 - np.exp(-1j * n * np.deg2rad(phase)))
            r = link(
                f * n,
                config['primary'],
                config['secondary'],
                config['coupling'],
                config['primary_C_F'],
                match['C2_F'],
                match['load_ohm'],
                config.get('primary_cap_ESR_ohm', 0),
                config.get('secondary_cap_ESR_ohm', 0),
            )
            ip = vc / r['input_Z_ohm']
            isec = r['secondary_per_primary'] * ip
            z = impedance(f * n, config['primary'], config['secondary'], config['coupling'])
            primary_initial += 2 * ip.real
            secondary_initial += 2 * isec.real
            primary_v += 2 * (z[0, 0] * ip + z[0, 1] * isec).real
            secondary_v += 2 * (z[1, 0] * ip + z[1, 1] * isec).real
            cap1_v += 2 * (-ip / (2j * np.pi * f * n * config['primary_C_F'])).real
            cap2_v += 2 * (isec / (2j * np.pi * f * n * match['C2_F'])).real
            initial_input += 2 * (abs(ip) ** 2 * r['input_Z_ohm'].real) / config['voltage_V']
        initial_input = config.get('transient', {}).get('initial_input_A', initial_input)
        config['primary'].update(initial_A=primary_initial, initial_V=primary_v)
        config['secondary'].update(initial_A=secondary_initial, initial_V=secondary_v)
        drop_bus, drop_hb = supply_drops(config)
        tank = {net_of(config, config['bridge_terminals'][0]), net_of(config, config['primary_terminals'][0])}
        for p in config['components']:
            if not p.get('enabled', True):
                continue
            a, b = net_of(config, p['a']), net_of(config, p['b'])
            # The starting state by what a part joins, not by its reference: the tank's capacitors, the
            # capacitors from each supply rail to ground charged to the rail, the series inductors carrying the
            # input current (groups(): wpthb's C25/C27/C30, C19 to C24 and CA1 to CA4, L3).
            if p['type'] == 'capacitor' and {a, b} == tank and len(tank) == 2:
                p['initial_V'] = cap1_v if a == net_of(config, config['primary_terminals'][0]) else -cap1_v
            elif p['type'] == 'capacitor' and {a, b} in ({'VHB', 'GND'}, {'VBUS', 'GND'}):
                rail = 'VHB' if 'VHB' in (a, b) else 'VBUS'
                v = config['voltage_V'] - (drop_hb if rail == 'VHB' else drop_bus) * initial_input
                p['initial_V'] = v if a == rail else -v
            elif p['type'] == 'inductor' and {a, b} in ({'VIN', 'VBUS'}, {'VBUS', 'VHB'}):
                p['initial_A'] = initial_input if b in ('VBUS', 'VHB') and a != 'VHB' else -initial_input
        if state is not None:  # a warm start overrides the estimates
            for p in config['components']:
                p.update(state['parts'].get(p['ref'], {}))
            for name, key in (('PRIMARY', 'primary'), ('SECONDARY', 'secondary')):
                config[key].update(state['coils'].get(name, {}))
            cap2_v = state['parts'].get('CSECONDARY', {}).get('initial_V', cap2_v)
        c.uic = True
    # Every selected pad is explicit. Unselected upstream parts connect at net references.
    nodes = {t: 'NET:' + net for t, net in config.get('terminals', {}).items()}
    for p in ports:
        nodes[p['terminal']] = 'PAD:' + p['terminal']

    def node(t):
        if t.startswith('NET:') or t == '0':
            return t
        if t not in nodes:
            raise ValueError('unknown component terminal ' + t)
        return nodes[t]

    if copper is not None:
        initial_ports = None
        if precharged:
            currents = {}
            for sw in config['switches']:
                currents[sw['a']] = currents.get(sw['a'], 0) - initial_input / 2
                currents[sw['b']] = currents.get(sw['b'], 0) + initial_input / 2
            initial_ports = [currents.get(p['terminal'], 0) for p in ports]
            if state is not None and state.get('ports') is not None and len(state['ports']) == len(ports):
                initial_ports = list(state['ports'])
        c.copper(
            copper,
            {p['name']: (nodes[p['terminal']], 'NET:' + p['net']) for p in ports},
            initial_ports,
            (state or {}).get('modes'),
        )
    supply = config.get('supply', {})
    ramp = supply.get('ramp_s', 100e-6)
    c.lines.append(
        f'VSUP {c.node("NET:SOURCE")} 0 '
        + (f'{config["voltage_V"]:.17g}' if precharged else f'PWL(0 0 {ramp:.17g} {config["voltage_V"]:.17g})')
    )
    c.signals['supply.I'] = 'i(VSUP)'
    c.voltage('supply.V', 'NET:SOURCE')
    c.passive('RSOURCE', 'NET:SOURCE', 'NET:SRC_R', dict(type='resistor', value=max(supply.get('R_ohm', 0.05), 1e-9)))
    source_A = state['parts'].get('LSOURCE', {}).get('initial_A', initial_input) if state is not None else initial_input
    c.passive(
        'LSOURCE',
        'NET:SRC_R',
        'NET:VIN',
        dict(
            type='inductor',
            value=max(supply.get('L_H', 100e-9), 1e-15),
            **({'initial_A': source_A} if precharged else {}),
        ),
    )
    for p in config['components']:
        c.passive(p['ref'], node(p['a']), node(p['b']), p)
    for sw in config['switches']:
        m = {**config.get('switch_model', {}), **sw.get('model', {})}
        if state is not None and sw['ref'] in state.get('switches', {}):
            m['initial_A'] = state['switches'][sw['ref']]
        c.switch(sw['ref'], node(sw['a']), node(sw['b']), m, f, phase if sw['leg'] == 'B' else 0, sw['high'])
    a, b = config['primary_terminals']
    c.coil('PRIMARY', node(a), node(b), config['primary'])
    c.nodes['SEC:A'] = '0'  # receiver voltage gauge; no environmental coupling is modeled
    c.coil('SECONDARY', 'SEC:A', 'SEC:B', config['secondary'])
    c.lines.append(f'KLINK L_PRIMARY L_SECONDARY {config["coupling"]:.17g}')
    secondary_cap = dict(
        type='capacitor',
        value=match['C2_F'],
        ESR_ohm=config.get('secondary_cap_ESR_ohm', 0),
        ESL_H=config.get('secondary_cap_ESL_H', 0),
    )
    if precharged:
        secondary_cap['initial_V'] = cap2_v
    if state is not None:
        secondary_cap.update({k: v for k, v in state['parts'].get('CSECONDARY', {}).items() if k == 'initial_I'})
    c.passive('CSECONDARY', 'SEC:B', 'SEC:LOAD', secondary_cap)
    c.passive('LOAD', 'SEC:LOAD', 'SEC:A', dict(type='resistor', value=match['load_ohm']))
    c.voltage('load.V', 'SEC:LOAD', 'SEC:A')
    c.voltage('bus.V', 'NET:VHB')
    c.voltage('input.V', 'NET:VBUS')
    c.voltage('bridge.V', node(config['bridge_terminals'][0]), node(config['bridge_terminals'][1]))
    if precharged and state is None:  # a warm start's capacitors already set these nodes
        for net, drop in zip(('VBUS', 'VHB'), supply_drops(config)):
            node = c.node('NET:' + net)
            # A bus that only inductors and the parts' current senses touch has no conductance of its own, and the
            # solver (KLU) cannot then hold a starting voltage on it: 1 GΩ to ground gives it one (24 nA at 24 V).
            # Only then: a deck that runs is left exactly as it was (wpthb's strict run is sensitive to less).
            anchored = any(line[:1] in 'RCDBMQrcdbmq' and node in line.split()[1:3] for line in c.lines)
            if not anchored:
                c.lines.append(f'R_IC_{net} {node} 0 1e9')
            c.lines.append(f'.ic V({node})={config["voltage_V"] - drop * initial_input:.17g}')
    c.measures = dict(
        input_power_W=('AVG', '-@supply.V@*@supply.I@'),
        load_power_W=('AVG', f'@load.V@^2/{match["load_ohm"]:.17g}'),
        primary_rms_A=('RMS', '@PRIMARY.winding_I@'),
        secondary_rms_A=('RMS', '@SECONDARY.winding_I@'),
        primary_peak_A=('MAX', 'abs(@PRIMARY.winding_I@)'),
        bus_ripple_V=('PP', '@bus.V@'),
    )
    gate_expr = '+'.join(f'-@{s["ref"]}.gate_V@*@{s["ref"]}.gate_I@' for s in config['switches'])
    c.measures['gate_drive_power_W'] = ('AVG', gate_expr)
    for ref in c.power_refs:
        c.measures['part_power.' + ref] = ('AVG', f'@{ref}.V@*@{ref}.I@')
    c.average_cycles = average_cycles(config, f)                # the powers: over whole periods of a slow mode
    return c


def metrics(result, f, match, samples=512):
    """Results of a run: ngspice's measurements over the final cycle (on its adaptive grid, so narrow
    input/diode pulses count) and the cycle-to-cycle change of the main signals. The bus voltage is reported
    twice (I1): bus_ripple_V, the settled ripple, peak to peak of the final cycle sampled at T/256 as the saved
    waveforms are; bus_spike_pp_V, peak to peak on the solver's own steps, switching-edge spikes included."""
    _, _, errors = periodic_samples(result, f, samples=samples)
    _, waves, _ = periodic_samples(result, f, samples=256)
    measured = result['_measures']
    power = measured['load_power_W']
    pin = measured['input_power_W']
    gate = measured['gate_drive_power_W']
    part_power = {k.split('.', 1)[1]: v for k, v in measured.items() if k.startswith('part_power.')}
    return dict(
        component_power_W=part_power,
        terminal_power_balance_relative=abs(pin - sum(part_power.values())) / max(abs(pin), 1e-12)
        if part_power
        else None,
        load_power_W=power,
        input_power_W=pin,
        gate_drive_power_W=gate,
        efficiency=power / (pin + gate) if pin + gate > 0 else None,
        **{k: measured[k] for k in ('primary_rms_A', 'secondary_rms_A', 'primary_peak_A')},
        bus_ripple_V=float(np.ptp(waves['bus.V'][-1])),
        bus_spike_pp_V=measured['bus_ripple_V'],
        periodic_error={
            k: errors[k] for k in ['PRIMARY.winding_I', 'SECONDARY.winding_I', 'bus.V', 'supply.I', 'load.V']
        },
    )


def make_board_ports(board, config, log=print):
    from shapely.ops import unary_union
    from .mqs import MQS
    from .pdn.rail import parts_of
    from .contacts import contact_terminal

    selected = config['mesh']
    nets = set(selected['nets'])
    parts = {p.ref: p for p in parts_of(board)}
    terms = {}
    for terminal, net in config['terminals'].items():
        if net not in nets:
            continue
        ref = terminal.split(':', 1)[0]
        if ref not in parts:
            raise ValueError('missing PCB component ' + ref)
        pads = parts[ref].on({net})
        terms[terminal] = contact_terminal(parts[ref], pads, selected.get('pad_contact', 'termination'))
        if not terms[terminal]:
            raise ValueError('missing PCB terminal ' + terminal)
    extent = unary_union([p for vv in terms.values() for _, p in vv])
    crop = extent.buffer(selected.get('margin_mm', 2)).envelope
    q = MQS(
        board,
        nets,
        terminals=terms,
        region=crop,
        dmax=selected.get('coarse_mm', 3),
        fine=extent.buffer(0.6).envelope,
        dmax_fine=selected.get('fine_mm', 1.6),
        conductor_loss='slab',
        conductor_faces=2,
        # one sheet by default: two face sheets moved wpthb's settled bus ripple 6% (power, efficiency, phase, currents
        # under 0.03%) at 2.6x the time on the compressed backend (dense: 4.7x and 25 GB); mesh.thickness_sheets: 2
        thickness_sheets=int(selected.get('thickness_sheets', 1)),
        # mesh.matrix_backend: 'hierarchical' extracts the ports from the compressed copper (mqs_iterative.port_method:
        # a direct solve when the loops are few, else GMRES; two face sheets fit in memory) and is the default: on
        # wpthb the same time as 'dense' (dense loop matrices from a dense inductance) at half the memory
        matrix_backend=selected.get('matrix_backend', 'hierarchical'),
        fill_backend=selected.get('backend', 'auto'),
        fill_workers=selected.get('workers', 8),
        log=log,
    )
    refs = {
        net: config.get('net_references', {}).get(net, next(t for t in terms if config['terminals'][t] == net))
        for net in nets
    }
    ports = []
    for term in sorted(terms):
        net = config['terminals'][term]
        ref = refs[net]
        if q.term_node[term] == q.term_node[ref]:
            continue
        name = 'T' + str(len(ports))
        q.source(name, term, ref, part_h=0)
        ports.append(dict(name=name, terminal=term, reference=ref, net=net))
    return q, ports, crop
