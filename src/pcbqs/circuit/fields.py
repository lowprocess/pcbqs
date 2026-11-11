"""
The fields over a board while its circuit runs: H from the currents in its copper, E from the voltages on it.

    python -m pcbqs.circuit model.json --out result --fields          # implies --copper
    export(model, summary, copper, out, log)                          # engine.run(..., fields=True) calls it

For each case, on planes 1 and 5 mm above the top copper (fields.heights_mm), over the last period:

    H   The copper's port currents of the reported run are taken apart into harmonics of the period (the first 16,
        fields.harmonics; from ngspice's own time steps). The solver that gave the circuit its copper model
        (pcbqs.mqs) gives, at each harmonic, the current everywhere in the copper for those port currents, and
        H from that by Biot-Savart. Put back together over the period.
    E   Each net's copper is a conductor at its node's voltage, the ground at zero. The charge that puts on
        the copper comes from the board's capacitance (the same mesh, with its dielectric), once for a volt on
        each net; E at any instant is those fields weighted by the nets' voltages at that instant. Nothing
        is band-limited here: E has the edges as the circuit has them.

What that is, and is not:
    - the copper and its vias only. No part's own field is in it: not an inductor's stray field, not the E of a
      part's body or a heatsink tab above the board. Those need a model of each part (docs/circuit.md).
    - quasi-static: the Coulomb E of the charges, without the induced E of changing currents (-dA/dt), and no
      radiation. It is what a near-field probe a few millimetres over the board sees, not a far field.
    - the board taken as grounded: its ground copper at zero volts against its surroundings. A board that floats
      has a common-mode voltage on top of this that no circuit on the board alone decides.
    - H up to the last harmonic kept: the ringing at the edges (hundreds of MHz) is above it, and case['fields']
      says what share of the port currents' AC was left out.

<case>/fields.npz: x_mm, y_mm, heights_mm, time_s, H_A_per_m and E_V_per_m (time, x, y, height, 3), their RMS maps
with and without the part that does not change over the period, H's harmonics; <case>/fields.png: the AC RMS maps
over the board's copper.
"""
import json
from pathlib import Path

import numpy as np

DEFAULTS = dict(harmonics=16, heights_mm=(1.0, 5.0), pitch_mm=0.75, time_samples=64, margin_mm=3.0)


def settings(model):
    return {**DEFAULTS, **(model.get('fields') or {})}


def grid(q, s):
    """The planes' points: x, y (mm, the board's frame) over the meshed region and a margin, z (mm) above the
    top copper."""
    nodes = q.nodes * 1000
    lo, hi = nodes[:, :2].min(axis=0) - s['margin_mm'], nodes[:, :2].max(axis=0) + s['margin_mm']
    n = np.maximum(np.ceil((hi - lo) / s['pitch_mm']).astype(int), 8) + 1
    x, y = np.linspace(lo[0], hi[0], n[0]), np.linspace(lo[1], hi[1], n[1])
    return x, y, q.z_of.max() * 1000 + np.asarray(s['heights_mm'], float)


def unit_fields(model, board, x, y, z, log=print):
    """E at the planes' points for one volt on each net and none on the others: ({net: E (nx, ny, nh, 3) V/m per
    V}, {net: its capacitance to everything else, F}). The copper's charge for those voltages comes from its
    coefficients of potential, in the board's dielectric stack."""
    from scipy.linalg import solve
    from ..mqs import MQS, Result
    from ..pdn.rail import parts_of
    from ..contacts import contact_terminal
    from shapely.ops import unary_union
    from .copper import pins
    mesh = (model.get('board') or {}).get('mesh', {})
    _, terminals, _ = pins(model, board)
    footprints = {p.ref: p for p in parts_of(board)}
    terms = {name: contact_terminal(footprints[fp], footprints[fp].on({net}), mesh.get('pad_contact', 'termination'))
             for name, (fp, net) in terminals.items()}
    nets = {net for _, net in terminals.values()}
    extent = unary_union([p for vv in terms.values() for _, p in vv])
    stack = bool(board.get('stack', {}).get('layers'))             # the board's dielectrics, when it says them
    q = MQS(board, nets, terminals=terms, region=extent.buffer(mesh.get('margin_mm', 2)).envelope,
            dmax=mesh.get('coarse_mm', 3), fine=extent.buffer(0.6).envelope, dmax_fine=mesh.get('fine_mm', 1.6),
            capacitance=True, dielectric='stack' if stack else 'homogeneous', thickness_sheets=1, matrix_backend='dense',
            fill_backend=mesh.get('backend', 'auto'), fill_workers=mesh.get('workers', 8), log=log)
    C, labels, of_terminal = q.capacitance_matrix()
    conductor = {}
    for name, k in of_terminal.items():
        net = terminals[name][1]
        if conductor.setdefault(net, k) != k:
            raise ValueError(f"the copper of net {net} is in more than one piece inside the meshed region")
    order = sorted(conductor)
    B = np.zeros((len(labels), len(order)))
    for j, net in enumerate(order):
        B[labels == conductor[net], j] = 1.0
    charge = solve(q._ensure_potential(), B, assume_a='sym')          # C per volt on each net, node by node
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]) * 1e-3
    res = Result(np.zeros(len(order)), None, {}, [], charge=np.asarray(charge).T)
    E = q.e_field(res, list(range(len(order))), pts)                   # (points, 3, nets)
    E = np.real(E).reshape(len(x), len(y), len(z), 3, len(order))
    idx = [conductor[n] for n in order]
    cap = {n: float(np.real(C[k, k])) for n, k in zip(order, idx)}
    return {n: E[..., j] for j, n in enumerate(order)}, cap


def harmonics_of(result, labels, period, count):
    """The complex Fourier coefficients c_n (n = 0..count) of signals over the last period, from uniform samples:
    x(t) = c_0 + 2 Re sum c_n exp(2 pi i n t / period). (count + 1, signals)."""
    t = np.asarray(result['time_s'])
    n = max(int(round(period / (t[1] - t[0]))), 2 * count + 2)
    at = t[-1] - period + np.arange(n) / n * period
    samples = np.stack([np.interp(at, t, result[k]) for k in labels], axis=1)
    return (np.fft.fft(samples, axis=0) / n)[:count + 1], at


def export(model, summary, copper, out, log=print):
    """Every case's fields, written beside it; each case of `summary` gets its 'fields' (what was kept, the
    largest values, the files). The copper's solver model is made once, its response once for each period."""
    from .._cli import load_board
    from ..mqs_ports import CopperPorts
    from . import model as M
    from .copper import board_file, geometry, net_name
    s = settings(model)
    board = load_board(str(board_file(model)))
    log("Fields: the copper's currents and charges")
    q, ports, _ = geometry(model, board, log)
    if [p['name'] for p in ports] != [p['name'] for p in copper['ports']]:
        raise ValueError("the copper's ports are not the ones the circuit ran with: run again")
    x, y, z = grid(q, s)
    unit, capacitance = unit_fields(model, board, x, y, z, log)
    known = set(unit)
    node_net = {}
    for part in model['parts']:
        for node in part['nodes']:
            net = net_name(model, node, known)
            if net in unit and node not in ('0', 'GND', 'gnd'):
                node_net[node] = net
    responses = {}
    count, nt = int(s['harmonics']), int(s['time_samples'])
    theta = np.arange(nt) / nt
    weights = np.r_[1.0, np.full(count, 2.0)]
    for case in summary['cases']:
        folder = Path(case['folder'])
        case['notes'] = [n for n in case.get('notes', []) if not n.startswith(('H keeps ', 'fields.png '))]   # a rerun's own
        with np.load(folder / 'transient.npz') as data:
            result = {k: data[k] for k in data.files}
        period = case['period_s']
        labels = [f"PCB{i}.I" for i in range(len(ports))]
        spectrum = folder / 'port-spectrum.npz'
        if spectrum.exists():                                       # from ngspice's own steps: the edges as solved
            with np.load(spectrum) as sp:
                full = sp['coefficients_A']
                total = float(np.sum(sp['AC_mean_square_A2']))
            coef = full[:count + 1]
        else:
            coef, _ = harmonics_of(result, labels, period, count)
            ac = np.stack([result[k][result['time_s'] >= result['time_s'][-1] - period] for k in labels], axis=1)
            total = float(np.sum(np.var(ac, axis=0)))
        if len(coef) < count + 1:
            raise ValueError(f"{case['name']}: the port currents were kept to {len(coef) - 1} harmonics, fewer than fields.harmonics")
        key = round(1 / period, 6)
        if key not in responses:
            responses[key] = CopperPorts(q).extract(np.arange(count + 1) / period, log=None)
        driven = responses[key].drive(coef)
        try:
            H, info = q.h_grid(driven, list(range(count + 1)), x * 1e-3, y * 1e-3, z * 1e-3, rtol=0.02, return_info=True)
        except RuntimeError as exc:                                 # the grid's own check: said, not hidden
            H = q.h_grid(driven, list(range(count + 1)), x * 1e-3, y * 1e-3, z * 1e-3, refine=4)
            info = dict(converged=False, note=str(exc)[:200])
        Ht = np.einsum('xyvcn,nt,n->txyvc', H, np.exp(2j * np.pi * np.arange(count + 1)[:, None] * theta), weights).real
        t = result['time_s'][-1] - period + theta * period
        Et = np.zeros((nt, len(x), len(y), len(z), 3))
        swing = {}
        for node, net in node_net.items():
            key_v = 'node.' + node
            if key_v not in result:
                continue
            v = np.interp(t, result['time_s'], result[key_v])
            Et += v[:, None, None, None, None] * unit[net][None]
            swing[net] = float(v.max() - v.min())
        mag = lambda F: np.linalg.norm(F, axis=-1)
        rms = lambda F: np.sqrt((mag(F) ** 2).mean(axis=0))            # over the period, at each point
        ac = lambda F: rms(F - F.mean(axis=0))                       # without what does not change: the DC current's
        kept = float(2 * np.sum(np.abs(coef[1:]) ** 2))              # H, the rails' E
        fields = dict(
            file=str(folder / 'fields.npz'), picture=str(folder / 'fields.png'), heights_mm=[float(h) for h in s['heights_mm']],
            harmonics=count, highest_Hz=count / period,
            port_current_AC_left_out=float(max(0.0, total - kept) / max(total, 1e-30)),
            H_ac_rms_max_A_per_m=[float(ac(Ht)[:, :, k].max()) for k in range(len(z))],
            H_rms_max_A_per_m=[float(rms(Ht)[:, :, k].max()) for k in range(len(z))],
            H_peak_A_per_m=[float(mag(Ht)[:, :, :, k].max()) for k in range(len(z))],
            E_ac_rms_max_V_per_m=[float(ac(Et)[:, :, k].max()) for k in range(len(z))],
            E_rms_max_V_per_m=[float(rms(Et)[:, :, k].max()) for k in range(len(z))],
            E_peak_V_per_m=[float(mag(Et)[:, :, :, k].max()) for k in range(len(z))],
            net_voltage_swing_V=swing, net_capacitance_F=capacitance, grid=info,
            scope='copper and vias only, no part bodies; quasi-static, Coulomb E, the ground copper at zero volts; '
                  'H band-limited to the harmonics kept')
        np.savez_compressed(
            folder / 'fields.npz', x_mm=x, y_mm=y, heights_mm=np.asarray(s['heights_mm'], float), z_mm=z, time_s=theta * period,
            H_A_per_m=Ht.astype(np.float32), E_V_per_m=Et.astype(np.float32),
            H_rms_A_per_m=rms(Ht).astype(np.float32), E_rms_V_per_m=rms(Et).astype(np.float32),
            H_ac_rms_A_per_m=ac(Ht).astype(np.float32), E_ac_rms_V_per_m=ac(Et).astype(np.float32),
            H_harmonics_A_per_m=np.linalg.norm(np.abs(H), axis=3).astype(np.float32) * weights / np.sqrt(2),
            frequencies_Hz=np.arange(count + 1) / period)
        try:
            picture(folder / 'fields.png', board, model, case, x, y, ac(Ht), ac(Et), s['heights_mm'])
        except Exception as exc:                                    # noqa: BLE001  (a picture is not the result)
            fields['picture'] = None
            case.setdefault('notes', []).append(f"fields.png was not drawn: {type(exc).__name__}: {exc}")
        if fields['port_current_AC_left_out'] > 0.25:
            case.setdefault('notes', []).append(
                f"H keeps {count} harmonics (to {count / period / 1e6:.3g} MHz): {100 * fields['port_current_AC_left_out']:.0f} % "
                f"of the copper currents' AC is above that")
        case['fields'] = fields
        (folder / 'case.json').write_text(json.dumps(case, indent=1))
        log(f"{case['name']}: fields at {s['heights_mm'][0]:g} mm: H to {fields['H_ac_rms_max_A_per_m'][0]:.3g} A/m, "
            f"E to {fields['E_ac_rms_max_V_per_m'][0]:.3g} V/m (AC RMS)")
    return summary


def picture(path, board, model, case, x, y, H, E, heights):
    """The AC RMS maps of H and E on each plane (what changes over the period: the DC current's H and the rails'
    steady E left out), the board's top copper and its parts' references drawn over them."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from shapely.geometry import shape
    from ..pdn.rail import parts_of
    top = next((l for l in ('F.Cu',) if l in board['copper']), next(iter(board['copper'])))
    outlines = [shape(c['polygon']) for c in board['copper'][top]]
    fig, axs = plt.subplots(len(heights), 2, figsize=(11, 4.2 * len(heights)), squeeze=False, constrained_layout=True)
    for k, h in enumerate(heights):
        for ax, F, unit, cmap, name in ((axs[k, 0], H, 'A/m', 'inferno', 'H'), (axs[k, 1], E, 'V/m', 'viridis', 'E')):
            im = ax.pcolormesh(x, y, F[:, :, k].T, shading='gouraud', cmap=cmap)
            for g in outlines:
                for piece in (g.geoms if hasattr(g, 'geoms') else [g]):
                    ax.plot(*piece.exterior.xy, color='w', lw=0.5, alpha=0.7)
            for p in parts_of(board):
                c = np.mean([pad.xy for pad in p.pads], axis=0)
                ax.text(c[0], c[1], p.ref, color='w', fontsize=6, ha='center', va='center')
            ax.set_aspect('equal')
            ax.set(xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
            ax.set_title(f"AC RMS |{name}| {h:g} mm over the board, to {F[:, :, k].max():.3g} {unit}", fontsize=9)
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, shrink=0.8).set_label(unit, fontsize=8)
    fig.suptitle(model.get('name', '') + (f"  ({case['name']})" if case.get('sweep') else ''), fontsize=10)
    fig.savefig(path, dpi=100)
    plt.close(fig)
