"""DC drop: pcbqs.pdn.dc.Network (2D copper sheets, via barrels) against Elmer (3D conduction in the copper and
solid barrels of the plated annulus's conductance).

    python validation/dc_elmer.py [case ...]        # cases: strip, via_single, via_hop, wpthb_vhb, jetson_1v15

Each case: one net's copper, a source terminal held at 0 V and loads drawing their currents as equipotential pads.
Compared: each load's voltage (the drop) and the copper loss, over a ladder of mesh densities for each tool; a
figure per case (dc_<case>.png) shows both converging.
"""
import sys
import time
from pathlib import Path

from shapely.geometry import box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from common import OUT, IMAGES, SIGMA_CU, copper_model, elmergrid, elmer, save_scalars, write_json  # noqa: E402

PLATING = 0.025


def two_layer_board(copper, vias=(), t=0.035, h=1.6):
    """A board.json-shaped dict: F.Cu at z=h over B.Cu at z=0, copper {layer: [polygon]} of net 'N'."""
    return dict(stack=dict(copper=[dict(name='B.Cu', z=0.0, t=t), dict(name='F.Cu', z=h, t=t)],
                           layers=[dict(name='core', z0=0.0, z1=h, er=4.5, tand=0.02)]),
                copper={l: [dict(net='N', polygon=mapping(p)) for p in ps] for l, ps in copper.items()},
                vias=[dict(x=x, y=y, r=r, z0=0.0, z1=h, net='N') for x, y, r in vias], outline=mapping(box(-5, -5, 60, 10)))


def case_strip():
    """A 50 x 5 mm strip on F.Cu, 5 x 5 mm pads at its ends: R = 40 mm / (sigma t w) between the pads."""
    b = two_layer_board({'F.Cu': [box(0, 0, 50, 5)]})
    terms = {'src': [('F.Cu', box(0, 0, 5, 5))], 'load': [('F.Cu', box(45, 0, 50, 5))]}
    return b, 'N', terms, {'load': 1.0}, dict(load=-0.040 / (SIGMA_CU * 35e-6 * 0.005))


def case_via_single():
    """Two 10 x 10 mm plates joined by one via (r 0.15 mm): the barrel (1.277 mOhm) and the spreading into each plate."""
    b = two_layer_board({'F.Cu': [box(0, 0, 10, 10)], 'B.Cu': [box(0, 0, 10, 10)]}, [(5, 5, 0.15)])
    terms = {'src': [('F.Cu', box(0, 0, 1, 10))], 'load': [('B.Cu', box(9, 0, 10, 10))]}
    return b, 'N', terms, {'load': 1.0}, None


def case_via_hop():
    """F.Cu strip to a 2 x 2 via array, then B.Cu strip back: the barrels' resistance in series (no closed form)."""
    vias = [(25 + dx, 2.5 + dy, 0.15) for dx in (-0.6, 0.6) for dy in (-0.6, 0.6)]
    b = two_layer_board({'F.Cu': [box(0, 0, 27, 5)], 'B.Cu': [box(23, 0, 50, 5)]}, vias)
    terms = {'src': [('F.Cu', box(0, 0, 5, 5))], 'load': [('B.Cu', box(45, 0, 50, 5))]}
    return b, 'N', terms, {'load': 1.0}, None


def board_case(path, net, source, loads, nets_for_board=None):
    from pcbqs._cli import load_board
    from pcbqs.pdn.rail import parts_of, terminal
    board = load_board(path, set(nets_for_board or [net]))
    parts = {p.ref: p for p in parts_of(board)}
    layers = [c['name'] for c in board['stack']['copper']]
    terms = {'src': terminal(parts[source].on({net}), layers)}
    for ref in loads:
        terms[ref] = terminal(parts[ref].on({net}), layers)
    return board, net, terms, loads, None


CASES = dict(
    strip=case_strip,
    via_single=case_via_single,
    via_hop=case_via_hop,
    wpthb_vhb=lambda: board_case(Path.home() / 'Desktop/wpthb/wpthb.kicad_pcb', 'VHB', 'L3', {'Q1': 0.67, 'Q2': 0.67}),
    jetson_1v15=lambda: board_case(Path('/usr/share/kicad/demos/jetson-agx-thor-baseboard/jetson-agx-thor-baseboard.kicad_pcb'),
                                   '+1V15', 'R7', {'U21': 2.0}),
)
# (pcbqs dmax, Elmer h, Elmer h at vias and pad edges), mm
LADDERS = dict(
    strip=((1.0, 0.4, 0.12), (0.5, 0.25, 0.06), (0.25, 0.15, 0.03)),
    via_single=((0.5, 0.3, 0.075), (0.25, 0.2, 0.04), (0.125, 0.12, 0.02)),
    via_hop=((1.0, 0.4, 0.12), (0.5, 0.25, 0.06), (0.25, 0.15, 0.03), (0.125, 0.1, 0.02)),
    wpthb_vhb=((1.0, 0.4, 0.12), (0.5, 0.25, 0.06), (0.25, 0.15, 0.03), (0.125, 0.1, 0.02)),
    jetson_1v15=((0.5, 0.25, 0.03), (0.25, 0.15, 0.02), (0.125, 0.1, 0.015), (0.0625, 0.07, 0.01)),
)


def pcbqs_dc(board, net, terms, loads, dmax):
    from pcbqs.pdn.dc import Network
    t = time.perf_counter()
    nw = Network(board, net, terms, dmax=dmax, plating=PLATING, fine=list(terms), log=lambda *a: None)
    r = nw.solve({'src': 0.0}, {k: -i for k, i in loads.items()})
    return dict(V={k: r.terminals[k] for k in loads}, loss=r.loss, vertices=nw.n_mesh, seconds=time.perf_counter() - t)


def elmer_dc(board, net, terms, loads, folder, h, h_fine):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    m = copper_model(board, {net}, terms, folder / 'copper.msh', h=h, h_fine=h_fine, plating=PLATING,
                     log=lambda s: print('  ' + s, flush=True))
    elmergrid(folder / 'copper.msh', folder / 'mesh')
    bc = [f"Boundary Condition 1\n  Target Boundaries(1) = {m['boundaries']['src']}\n  Potential = 0.0\nEnd\n"]
    for k, (name, amps) in enumerate(loads.items()):
        # an equipotential pad drawing `amps` out of the copper
        bc.append(f"Boundary Condition {k + 2}\n  Name = \"{name}\"\n  Target Boundaries(1) = {m['boundaries'][name]}\n"
                  f"  Potential Constant = Logical True\n  Potential Constant Flux = Real {-amps:.12g}\n"
                  f"  load{k} = Logical True\nEnd\n")
    bodies = "Body 1\n  Target Bodies(1) = 1\n  Equation = 1\n  Material = 1\nEnd\n"
    if m['bodies']['barrel']:
        bodies += "Body 2\n  Target Bodies(1) = 2\n  Equation = 1\n  Material = 2\nEnd\n"
    masks = ''.join(f'  Variable {k + 2} = Potential\n  Operator {k + 2} = boundary mean\n  Mask Name {k + 2} = load{k}\n'
                    for k in range(len(loads)))
    sif = f"""Header
  Mesh DB "." "mesh"
End
Simulation
  Coordinate System = Cartesian 3D
  Coordinate Scaling = 1.0e-3
  Simulation Type = Steady State
  Steady State Max Iterations = 1
  Max Output Level = 4
End
{bodies}Equation 1
  Active Solvers(2) = 1 2
End
Material 1
  Electric Conductivity = {SIGMA_CU:.6g}
End
Material 2
  Electric Conductivity = {m['barrel_sigma']:.6g}
End
Solver 1
  Equation = "StatCurrent"
  Variable = "Potential"
  Procedure = "StatCurrentSolveVec" "StatCurrentSolver"
  Calculate Joule Heating = Logical True
  Linear System Solver = "Direct"
  Linear System Direct Method = "Umfpack"
End
Solver 2
  Exec Solver = After All
  Equation = "SaveScalars"
  Procedure = "SaveData" "SaveScalars"
  Filename = "scalars.dat"
  Variable 1 = Joule Heating
  Operator 1 = int
{masks}End
{''.join(bc)}"""
    elmer(folder, sif)
    s = save_scalars(folder / 'scalars.dat')
    V = {}
    for k, name in enumerate(loads):
        V[name] = float(next(v for c, v in s.items() if 'potential' in c.lower() and f'load{k}' in c.lower()))
    loss = float(next(v for c, v in s.items() if 'joule' in c.lower()))
    return dict(V=V, loss=loss, tets=m['tets'], seconds=time.perf_counter() - t)


def run(name, meshes=None):
    board, net, terms, loads, analytic = CASES[name]()
    meshes = meshes or LADDERS[name]
    folder = OUT / 'dc' / name
    rows = []
    for dmax, h, hf in meshes:
        print(f'{name}: pcbqs dmax {dmax}, Elmer h {h}/{hf}', flush=True)
        p = pcbqs_dc(board, net, terms, loads, dmax)
        e = elmer_dc(board, net, terms, loads, folder / f'h{h:g}', h, hf)
        rows.append(dict(dmax=dmax, h=h, h_fine=hf, pcbqs=p, elmer=e))
        for k in loads:
            print(f'  {k}: pcbqs {p["V"][k] * 1e3:.4f} mV, Elmer {e["V"][k] * 1e3:.4f} mV'
                  + (f', analytic {analytic[k] * 1e3:.4f} mV' if analytic else '')
                  + f' ({(p["V"][k] / e["V"][k] - 1) * 100:+.2f}%)', flush=True)
        print(f'  loss: pcbqs {p["loss"] * 1e3:.4f} mW, Elmer {e["loss"] * 1e3:.4f} mW; '
              f'{p["vertices"]:,} vertices {p["seconds"]:.1f} s, {e["tets"]:,} tets {e["seconds"]:.1f} s', flush=True)
    write_json(folder / 'result.json', dict(case=name, net=net, loads=loads, analytic=analytic, rows=rows))
    figure(name, rows, analytic)
    return rows


def figure(name, rows, analytic=None):
    """Each load's drop against mesh size, both tools: dc_<case>.png in OUT."""
    from plots import style, mesh_axis, PCBQS, REFERENCE, ANALYTIC
    plt = style()
    loads = list(rows[0]['pcbqs']['V'])
    fig, axs = plt.subplots(1, len(loads), figsize=(4.2 * len(loads), 3.4), squeeze=False)
    for ax, k in zip(axs[0], loads):
        ax.plot([r['dmax'] for r in rows], [-r['pcbqs']['V'][k] * 1e3 for r in rows], 'o-', color=PCBQS, label='pcbqs (dmax)')
        ax.plot([r['h'] for r in rows], [-r['elmer']['V'][k] * 1e3 for r in rows], 's-', color=REFERENCE, label='Elmer 3D (h)')
        if analytic:
            ax.axhline(-analytic[k] * 1e3, color=ANALYTIC, ls='--', lw=1, label='closed form')
        mesh_axis(ax, [r['dmax'] for r in rows] + [r['h'] for r in rows])
        ax.set(xlabel='mesh size (mm): pcbqs dmax, Elmer h', ylabel='drop (mV)', title=f'{name}: {k}')
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES / f'dc_{name}.png', dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--plot']:                 # redraw from the saved results
        import json
        for name in sys.argv[2:] or list(CASES):
            r = json.loads((OUT / 'dc' / name / 'result.json').read_text())
            figure(name, r['rows'], r['analytic'])
    else:
        for name in sys.argv[1:] or list(CASES):
            run(name)
