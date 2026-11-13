"""
A converter found on a board, as a circuit model: what the near-field tool selects (pcbqs.mqs: a regulator, its
input net and capacitors, the inductor on its switch net), with what the board has on the inductor's far side.

    python -m pcbqs.mqs board.kicad_pcb --converter U702 --vin-volts 5 --vout 3.3 --fsw 1e6 --current 2 --circuit > u702.json
    python -m pcbqs.circuit u702.json --copper --out out/u702
    model = model_of(board, 'board.kicad_pcb', a)        # a: the near-field tool's settings (mqs_cli.parser)

The model is a buck as the board draws it:

    V1      the supply, on the input net: on the board's one connector between that net and ground, when it
            has just one (else not on the board: at the net's node)
    C...    the input capacitors of the near-field selection, with its ESR and ESL
    QH, QL  the regulator's two switches, input to switch node and switch node to ground, both on the regulator's
            footprint (board.parts), the lower the complement of the upper with a dead time; where the board has
            a diode from ground to the switch node, that diode instead of QL. A discrete cell: QH on its upper
            transistor, QL on its lower one (or its diode)
    L...    the inductor, its value from the board
    C...    the capacitors on the inductor's far net within reach of the regulator or the inductor; none there:
            COUT, all that net's capacitance in one (not on the board)
    RL      the load, vout / iout: what the output net feeds is not the model's to know (on the output net's
            one connector, when it has just one)

The duty is adjusted until the output is vout. The nodes are in, sw and out, their nets in board.nets, so the
board's copper between the pads can be put in (--copper) and the fields over it (--fields). What a board cannot
say is assumed and listed in the model's description: the voltages when the near-field settings give a duty
alone, the switches' on resistance, the inductor's DCR.

A discrete cell whose part up from the switch node is a diode, or whose settings say an output above its input,
is a boost (boost_of): the capacitors the near-field tool selected are its output's (its hot loop: those
capacitors, the upper part, the lower switch), and its input is the net on the inductor's far side:

    V1      the supply, on the inductor's far net (the board's one connector there, when it has just one)
    C...    the capacitors on that net within reach of the inductor or the switch, where there are any
    L...    the inductor, from the input to the switch node
    Q...    the lower switch, as the board names it; D... (or the upper switch, the lower's complement) up to the
            output
    C...    the output capacitors: the near-field selection's
    RL      the load, vout / iout, iout what the hot loop's pulse leaves the output: its amplitude x vin / vout
"""
import re

RON, DCR, DEAD = 0.05, 0.03, 20e-9          # assumed: a switch's on resistance, the inductor's DCR, the dead time
VIN = 12.0                                  # assumed input voltage when the settings give a duty alone


def _ref(name, taken):
    """A reference the model takes (letters, digits, _) that the board's own parts do not have."""
    name = re.sub(r'[^A-Za-z0-9_]', '_', name)
    while name in taken:
        name += '_'
    return name


def model_of(board, path, a):
    """The circuit model (format 2) of the converter the settings `a` select on `board` (pcbqs._cli.load_board's;
    path: its file, the model's board). ValueError: no converter, an ambiguous one (as the near-field tool says
    it), or one whose inductor is not found."""
    from shapely.ops import unary_union
    from ..mqs_cli import discover, select
    from ..pdn.rail import parts_of, part_value, field_value
    selected, _ = select(board, a, require_values=True)
    ref, vin, ground = selected['converter'], selected['vin'], selected['ground']
    parts = {p.ref: p for p in parts_of(board)}
    found = [c for c in discover(board, ground) if c['converter'] == ref and any(i['vin'] == vin for i in c['inputs'])
             and (not getattr(a, 'inductor', '') or c['inductor'] == a.inductor)]
    if not found:
        raise ValueError(f"{ref}: no inductor on a net it shares with the regulator (its switch node): name it, "
                         f"--inductor")
    if len({(c['inductor'], c['switch']) for c in found}) > 1:
        raise ValueError(f"{ref}: several inductors on its nets ({', '.join(sorted(c['inductor'] for c in found))}): "
                         f"name one, --inductor")
    cell = found[0].get('low')                                      # a discrete switching cell: its own parts
    if not found[0]['inductor']:
        raise ValueError(f"{ref}: no one inductor on its switch node {found[0]['switch']}: only a buck or a boost is "
                         f"made into a circuit (a bridge's tank, a resonant stage: a model of your own)")
    up_v, out_v = float(getattr(a, 'vin_volts', 0) or 0), float(getattr(a, 'vout', 0) or 0)
    if cell and (not re.match(r'(Q|M)\d', ref.upper()) or (up_v and out_v > up_v)):
        return boost_of(board, path, a, selected, found[0], parts)  # (a diode up from the switch node, or stepping up)
    ind, sw = parts[found[0]['inductor']], found[0]['switch']
    far = sorted(ind.nets() - {sw})
    if len(far) != 1 or far[0] in (vin, ground):
        raise ValueError(f"{ind.ref}: its far side is not one net of its own ({', '.join(far) or 'none'})")
    out = far[0]
    henries = part_value(ind)
    if not henries:
        raise ValueError(f"{ind.ref}: its value {ind.value!r} is not an inductance")
    reg = parts[ref]
    assumed = []
    volts, vout = float(getattr(a, 'vin_volts', 0) or 0), float(getattr(a, 'vout', 0) or 0)
    if not volts:
        volts, vout = VIN, VIN * float(a.duty)
        assumed.append(f"{volts:g} V in and {vout:g} V out (the settings give a duty alone)")
    taken = set(parts)
    qh, ql, v1, rl = (_ref(n, taken) for n in ('QH', 'QL', 'V1', 'RL'))
    esr, esl = float(getattr(a, 'esr', 0.003)), float(getattr(a, 'esl', 0.4e-9))
    cap = lambda r, node, value: dict(ref=r, type='capacitor', nodes=[node, '0'], value=value, ESR_ohm=esr, ESL_H=esl)
    model_parts = [dict(ref=v1, type='source', nodes=['in', '0'], dc_V='vin', R_ohm=0.02)]
    model_parts += [cap(c, 'in', selected['values'][c]) for c in selected['capacitors']]
    switch = dict(type='switch', Ron_ohm=RON, Cds_F='100p')
    model_parts.append(dict(ref=qh, type='switch', nodes=['in', 'sw'], drive='pwm', model=dict(switch)))
    on_board = {qh: ref}
    diodes = [p for p in parts.values() if re.match(r'D\d', p.ref) and p.nets() == {sw, ground}]
    if len(diodes) == 1:                                            # a non-synchronous regulator's own diode
        low = diodes[0].ref
        model_parts.append(dict(ref=low, type='diode', nodes=['0', 'sw'], model=dict(Vf_V=0.45, If_A=2, Cj_F='150p')))
        assumed.append(f"{low} as a Schottky diode (0.45 V at 2 A)")
    else:
        low = ql
        model_parts.append(dict(ref=ql, type='switch', nodes=['sw', '0'], drive='pwm', invert=True, model=dict(switch)))
        on_board[ql] = cell or ref                                  # the cell's lower switch, else the regulator's
    assumed.append(f"{RON * 1e3:g} mOhm switches")
    dcr = field_value(ind, 'DCR', 'dcr')
    model_parts.append(dict(ref=ind.ref, type='inductor', nodes=['sw', 'out'], value=henries, DCR_ohm=dcr or DCR))
    if not dcr:
        assumed.append(f"{DCR * 1e3:g} mOhm DCR for {ind.ref}")
    centre = unary_union([p.geom for p in reg.pads + ind.pads])
    on_out = [p for p in parts.values() if p.ref.upper().startswith('C') and p.nets() == {out, ground}
              and (part_value(p) or 0) > 0]
    near = sorted((p for p in on_out if min(pad.geom.distance(centre) for pad in p.pads) <= float(a.cap_radius)),
                  key=lambda p: p.ref)
    if near:
        model_parts += [cap(p.ref, 'out', part_value(p)) for p in near]
        rest = len(on_out) - len(near)
        output = ', '.join(p.ref for p in near) + (f" ({rest} more on {out} further away left out)" if rest else '')
    elif on_out:
        cout = _ref('COUT', taken)
        model_parts.append(cap(cout, 'out', sum(part_value(p) for p in on_out)))
        output = f"{cout}, the {len(on_out)} capacitors on {out} in one (none within {float(a.cap_radius):g} mm)"
    else:
        raise ValueError(f"{out}: no capacitor from the inductor's far net to {ground}")
    model_parts.append(dict(ref=rl, type='load', nodes=['out', '0'], resistance_ohm='vout/iout'))
    # where the board meets the outside, when it shows it: the one connector on the input net and ground is the
    # supply's footprint, the one on the output net and ground the load's (their copper then in the circuit, and
    # the supply's as the rail's source for a power-integrity run)
    plug = lambda net: [p.ref for p in parts.values() if re.match(r'(J|P|CN|X)\d', p.ref) and {net, ground} <= p.nets()]
    for model_ref, net in ((v1, vin), (rl, out)):
        found_plug = plug(net)
        if len(found_plug) == 1:
            on_board[model_ref] = found_plug[0]
    nets = {'in': vin, 'sw': sw, 'out': out}
    if ground not in ('GND', 'gnd', '0', 'GNDPWR', 'PGND'):         # (those copper.net_name finds itself)
        nets['0'] = ground
    return dict(
        format=2, name=f"{ref}: {vin} to {out}",
        description=(f"The converter {ref} as its board has it: input capacitors {', '.join(selected['capacitors'])}, "
                     f"inductor {ind.ref}, output {output}; the duty is adjusted until the output is vout. Assumed: "
                     + '; '.join(assumed) + '.'),
        parameters=dict(fsw=float(a.fsw), vin=volts, vout=vout, iout=float(a.current)),
        parts=model_parts,
        drives=dict(pwm=dict(frequency_Hz='fsw', duty=round(vout / volts, 4), dead_time_s=DEAD)),
        probes=dict(vout=dict(voltage=['out', '0']), vsw=dict(voltage=['sw', '0']), il=dict(current=ind.ref),
                    iin=dict(current=qh)),
        measures=dict(vout_avg=dict(avg='vout'), vout_ripple=dict(pp='vout'), il_ripple=dict(pp='il'),
                      iin_avg=dict(avg='iin'), iin_rms=dict(rms='iin')),
        targets=[dict(adjust='drives.pwm.duty', range=[0.02, 0.98], initial='vout/vin', measure='vout_avg',
                      value='vout')],
        board=dict(file=str(path), parts=on_board, nets=nets))


def boost_of(board, path, a, selected, found, parts):
    """The circuit model of a boost: the discrete cell `found` (mqs_cli.discover: its upper part the converter, its
    lower switch `low`, the inductor on its switch node) whose selected capacitors are the output's. ValueError:
    an inductor whose far side is not one net of its own, or settings that do not step up."""
    from shapely.ops import unary_union
    from ..pdn.rail import part_value, field_value
    up, low, sw, ground, out = found['converter'], found['low'], found['switch'], selected['ground'], selected['vin']
    ind = parts[found['inductor']]
    far = sorted(ind.nets() - {sw})
    if len(far) != 1 or far[0] in (out, ground):
        raise ValueError(f"{ind.ref}: its far side is not one net of its own ({', '.join(far) or 'none'})")
    vin = far[0]
    henries = part_value(ind)
    if not henries:
        raise ValueError(f"{ind.ref}: its value {ind.value!r} is not an inductance")
    assumed = []
    volts, vout = float(getattr(a, 'vin_volts', 0) or 0), float(getattr(a, 'vout', 0) or 0)
    if not volts or not vout:                           # a duty alone: the hot loop's pulse, the upper part's share
        volts, vout = VIN, VIN / float(a.duty)
        assumed.append(f"{volts:g} V in and {vout:g} V out (the settings give a duty alone: the upper part's)")
    if vout <= volts:
        raise ValueError(f"{up}: a boost's output ({vout:g} V) is above its input ({volts:g} V)")
    iout = float(a.current) * volts / vout              # the pulse is the inductor's current; the output gets its share
    taken = set(parts)
    v1, rl = (_ref(n, taken) for n in ('V1', 'RL'))
    esr, esl = float(getattr(a, 'esr', 0.003)), float(getattr(a, 'esl', 0.4e-9))
    cap = lambda r, node, value: dict(ref=r, type='capacitor', nodes=[node, '0'], value=value, ESR_ohm=esr, ESL_H=esl)
    model_parts = [dict(ref=v1, type='source', nodes=['in', '0'], dc_V='vin', R_ohm=0.02)]
    centre = unary_union([p.geom for p in ind.pads + parts[low].pads])
    on_in = sorted((p for p in parts.values() if p.ref.upper().startswith('C') and p.nets() == {vin, ground}
                    and (part_value(p) or 0) > 0
                    and min(pad.geom.distance(centre) for pad in p.pads) <= float(a.cap_radius)), key=lambda p: p.ref)
    model_parts += [cap(p.ref, 'in', part_value(p)) for p in on_in]
    dcr = field_value(ind, 'DCR', 'dcr')
    model_parts.append(dict(ref=ind.ref, type='inductor', nodes=['in', 'sw'], value=henries, DCR_ohm=dcr or DCR))
    if not dcr:
        assumed.append(f"{DCR * 1e3:g} mOhm DCR for {ind.ref}")
    switch = dict(type='switch', Ron_ohm=RON, Cds_F='100p')
    model_parts.append(dict(ref=low, type='switch', nodes=['sw', '0'], drive='pwm', model=dict(switch)))
    if re.match(r'(Q|M)\d', up.upper()):                # a synchronous boost: the upper switch, the complement
        model_parts.append(dict(ref=up, type='switch', nodes=['sw', 'out'], drive='pwm', invert=True, model=dict(switch)))
    else:
        model_parts.append(dict(ref=up, type='diode', nodes=['sw', 'out'], model=dict(Vf_V=0.45, If_A=2, Cj_F='150p')))
        assumed.append(f"{up} as a Schottky diode (0.45 V at 2 A)")
    assumed.append(f"{RON * 1e3:g} mOhm switches")
    # the output capacitors with their ESR alone: the inductor's current commutes into them at once (the model
    # has no stray inductance to slow it: the copper's is --copper's), and their ESL rang with the junction's
    # capacitance, 27 V peak to peak on a 24 V output whose ripple is 65 mV
    model_parts += [dict(ref=c, type='capacitor', nodes=['out', '0'], value=selected['values'][c], ESR_ohm=esr)
                    for c in selected['capacitors']]
    assumed.append("the output capacitors without their ESL (the commutation into them is ideal)")
    model_parts.append(dict(ref=rl, type='load', nodes=['out', '0'], resistance_ohm='vout/iout'))
    plug = lambda net: [p.ref for p in parts.values() if re.match(r'(J|P|CN|X)\d', p.ref) and {net, ground} <= p.nets()]
    on_board = {}
    for model_ref, net in ((v1, vin), (rl, out)):
        found_plug = plug(net)
        if len(found_plug) == 1:
            on_board[model_ref] = found_plug[0]
    nets = {'in': vin, 'sw': sw, 'out': out}
    if ground not in ('GND', 'gnd', '0', 'GNDPWR', 'PGND'):
        nets['0'] = ground
    return dict(
        format=2, name=f"{low} and {up}: {vin} up to {out}",
        description=(f"The boost of {low} and {up} as its board has it: inductor {ind.ref} from {vin}"
                     + (f" (capacitors {', '.join(p.ref for p in on_in)})" if on_in else '')
                     + f", output capacitors {', '.join(selected['capacitors'])}; the duty is adjusted until the "
                     f"output is vout. The load draws {iout:.3g} A: the settings' {float(a.current):g} A is the "
                     f"inductor's current, the hot loop's pulse. Assumed: " + '; '.join(assumed) + '.'),
        parameters=dict(fsw=float(a.fsw), vin=volts, vout=vout, iout=round(iout, 6)),
        parts=model_parts,
        drives=dict(pwm=dict(frequency_Hz='fsw', duty=round(1 - volts / vout, 4), dead_time_s=DEAD)),
        probes=dict(vout=dict(voltage=['out', '0']), vsw=dict(voltage=['sw', '0']), il=dict(current=ind.ref),
                    iup=dict(current=up)),
        measures=dict(vout_avg=dict(avg='vout'), vout_ripple=dict(pp='vout'), il_ripple=dict(pp='il'),
                      iin_avg=dict(avg='il'), iup_rms=dict(rms='iup')),
        targets=[dict(adjust='drives.pwm.duty', range=[0.02, 0.95], initial='1 - vin/vout', measure='vout_avg',
                      value='vout')],
        board=dict(file=str(path), parts=on_board, nets=nets))
