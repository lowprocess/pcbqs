"""The thermal trial: the layered steady-state model on a grid and on the triangle mesh, against exact solutions,
against each other, and from the command line."""
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.special import i0, i1, k0, k1
from shapely.geometry import Point, box, mapping

from pcbqs.thermal import Problem, solve, losses_of
from pcbqs.thermal import grid as G

EXAMPLES = Path(__file__).resolve().parents[1] / 'src' / 'pcbqs' / 'circuit' / 'examples'
quiet = lambda *a: None


def plate(w, h=None, outline=None, vias=()):
    sq = box(0, 0, w, h or w)
    out = outline or sq
    return dict(stack=dict(layers=[], copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='F.Cu', z=1.5, t=0.035)],
                           thickness=1.5),
                copper={n: [dict(net='GND', polygon=mapping(out))] for n in ('B.Cu', 'F.Cu')},
                vias=[dict(x=x, y=y, r=0.2, z0=0.0, z1=1.5, net='GND') for x, y in vias], outline=mapping(out))


def disc_problem(side=120.0, a=5.0, watts=1.0, film=50.0):
    p = Problem.of(plate(side), film=film)
    p.heat('disc', [('F.Cu', Point(side / 2, side / 2).buffer(a, quad_segs=32))], watts)
    return p


def disc_exact(p, a, watts, film):
    """(top, bottom)(r mm) for a uniformly heated disc on an unbounded two-layer plate: the sum and the difference
    of the two layers each obey a screened Poisson equation."""
    g = p.k_cu * 0.035e-3 + p.layers[0].laminate
    c, q, A = p.gaps[0], watts / (np.pi * (a * 1e-3) ** 2), a * 1e-3

    def f(r, L):
        r = np.asarray(r, float) * 1e-3
        return np.where(r < A, q * L * L / g * (1 - A / L * k1(A / L) * i0(np.minimum(r, A) / L)),
                        q * L * L / g * A / L * i1(A / L) * k0(np.maximum(r, A) / L))
    Ls, Ld = np.sqrt(g / film), np.sqrt(g / (film + 2 * c))
    return lambda r: (p.ambient + (f(r, Ls) + f(r, Ld)) / 2, p.ambient + (f(r, Ls) - f(r, Ld)) / 2)


@pytest.mark.parametrize('method,size', [('grid', 1.0), ('mesh', 2.0)])
def test_a_face_heated_all_over_is_uniform_and_exact(method, size):
    p = Problem.of(plate(40), film=10.0, edge_film=0.0)              # the infinite plate's answer: no edges
    p.heat('all', [('F.Cu', box(0, 0, 40, 40))], 2.0)
    q, h, c = 2.0 / 0.04 ** 2, 10.0, p.gaps[0]
    top = q / (h + c - c * c / (h + c))
    r = solve(p, method, size, log=quiet)
    assert np.nanmax(np.abs(r.T[1] - (25 + top))) < 1e-6
    assert np.nanmax(np.abs(r.T[0] - (25 + c * top / (h + c)))) < 1e-6
    assert r.out_W == pytest.approx(2.0, rel=1e-9)


@pytest.mark.parametrize('method,size', [('grid', 1.0), ('grid', 0.7), ('mesh', 2.0)])
@pytest.mark.parametrize('hole', [False, True])
def test_the_edges_give_heat_over_the_board_s_thickness(method, size, hole):
    """A plate that conducts so well it is one temperature: the rise is the loss over the faces' and the edges'
    conductances, the edges being the outline and its cut-out, the board's whole thickness high."""
    out = box(0, 0, 40, 30)
    if hole:
        out = out.difference(box(10.3, 10.3, 22.7, 18.1))
    p = Problem.of(plate(40, 30, outline=out), film=(8.0, 12.0), edge_film=15.0, k_cu=1e7, k_z=1e7)
    assert sum(l.edge for l in p.layers) == pytest.approx(1.5)
    p.heat('u', [('F.Cu', box(30, 3, 34, 7))], 1.0)
    r = solve(p, method, size, log=quiet)
    faces = (8.0 + 12.0) * out.area * 1e-6
    edges = 15.0 * out.length * 1.5e-6
    rise = 1.0 / (faces + edges)
    assert np.nanmax(np.abs(r.T - (25 + rise))) < 1e-3 * rise
    assert r.out_W == pytest.approx(1.0, rel=1e-6)
    assert r.edge_W == pytest.approx(edges / (faces + edges), rel=1e-3)


def test_the_edges_default_to_the_faces_and_can_be_insulated():
    p = Problem.of(plate(40), film=(6.0, 10.0))
    assert p.edge_film == 8.0
    p.heat('u', [('F.Cu', box(18, 18, 22, 22))], 0.5)
    on = solve(p, 'grid', 1.0, log=quiet)
    p.edge_film = 0.0
    off = solve(p, 'grid', 1.0, log=quiet)
    assert off.edge_W == 0.0 and 0 < on.edge_W < 0.1 * 0.5
    assert off.parts['u']['board_C'] > on.parts['u']['board_C']
    with pytest.raises(ValueError):
        Problem.of(plate(40), edge_film=-1.0)


@pytest.mark.parametrize('method,size,tol', [('grid', 1.0, 0.1), ('grid', 0.5, 0.06), ('mesh', 2.0, 0.15),
                                             ('mesh', 1.0, 0.06)])
def test_a_heated_disc_matches_the_exact_solution(method, size, tol):
    p = disc_problem()
    exact = disc_exact(p, 5.0, 1.0, 50.0)
    r = solve(p, method, size, log=quiet)
    rr = np.array([0.0, 2.5, 7.5, 10, 15, 20, 30])
    ang = np.arange(8) * np.pi / 4 + 0.1
    top = np.mean([r.sample('F.Cu', 60 + rr * np.cos(u), 60 + rr * np.sin(u)) for u in ang], axis=0)
    bot = np.mean([r.sample('B.Cu', 60 + rr * np.cos(u), 60 + rr * np.sin(u)) for u in ang], axis=0)
    et, eb = exact(rr)
    assert np.max(np.abs(top - et)) < tol
    assert np.max(np.abs(bot - eb)) < tol
    rs = np.linspace(0, 5, 2001)
    mean = np.trapezoid(exact(rs)[0] * rs, rs) / np.trapezoid(rs, rs)
    assert r.parts['disc']['board_C'] == pytest.approx(mean, abs=tol)
    assert r.out_W == pytest.approx(1.0, rel=1e-9)


def test_the_two_solvers_agree_on_the_disc():
    p = disc_problem()
    g, m = solve(p, 'grid', 0.5, log=quiet), solve(p, 'mesh', 1.0, log=quiet)
    assert g.parts['disc']['board_C'] == pytest.approx(m.parts['disc']['board_C'], abs=0.05)
    assert g.hottest == pytest.approx(m.hottest, abs=0.05)


def test_a_board_that_ends_inside_a_cell_keeps_its_edge_strip():
    """20.5 mm at a 1 mm pitch: the last row of cells is half board. Its conduction to the row before must not be
    lost (it was: every chain to its centre ran off the board, and the board read 1 K hot). The extra half
    millimetre must cool the part by as much on the coarse grid as on a fine one."""
    def at(pitch, height):
        p = Problem.of(plate(30, height), film=10.0)
        p.heat('u', [('F.Cu', box(12, 8, 16, 12))], 0.5)
        return solve(p, 'grid', pitch, log=quiet).parts['u']['board_C']
    fine = at(0.25, 20.0) - at(0.25, 20.5)
    assert fine > 0.5
    assert at(1.0, 20.0) - at(1.0, 20.5) == pytest.approx(fine, abs=0.05)
    assert at(1.0, 20.0) - at(1.0, 20.3) == pytest.approx(at(0.25, 20.0) - at(0.25, 20.3), abs=0.05)


def test_a_slot_narrower_than_a_cell_still_parts_the_board():
    out = box(0, 0, 40, 20).difference(box(19.9, -1, 20.3, 21))
    p = Problem.of(plate(40, 20, outline=out), film=10.0)
    p.heat('u', [('F.Cu', box(5, 8, 9, 12))], 0.5)
    for pitch in (1.0, 0.7):
        r = solve(p, 'grid', pitch, log=quiet)
        far = r.sample('F.Cu', np.array([25.0, 30.0, 38.0]), np.array([10.0, 10.0, 10.0]))
        assert np.all(np.abs(far - 25.0) < 1e-6)
        assert r.out_W == pytest.approx(0.5, rel=1e-9)
    m = solve(p, 'mesh', 1.0, log=quiet)
    assert abs(m.sample('F.Cu', np.array([30.0]), np.array([10.0]))[0] - 25.0) < 1e-6


def test_a_gap_in_the_copper_narrower_than_a_cell_holds_heat_back():
    """Two islands 0.2 mm apart on a 1 mm grid: the far island must sit well below the heated one, as on the mesh."""
    b = plate(40, 20)
    isl = box(1, 1, 19.9, 19).union(box(20.1, 1, 39, 19))
    b['copper'] = {'F.Cu': [dict(net='a', polygon=mapping(isl))], 'B.Cu': []}
    p = Problem.of(b, film=10.0)
    p.heat('u', [('F.Cu', box(8, 8, 12, 12))], 0.5)
    g, m = solve(p, 'grid', 1.0, log=quiet), solve(p, 'mesh', 0.5, log=quiet)
    x, y = np.array([15.0, 25.0, 35.0]), np.array([10.0, 10.0, 10.0])
    tg, tm = g.sample('F.Cu', x, y), m.sample('F.Cu', x, y)
    assert tm[0] - tm[1] > 5.0                              # the gap matters
    assert np.max(np.abs(tg - tm)) < 0.5
    assert g.parts['u']['board_C'] == pytest.approx(m.parts['u']['board_C'], abs=0.3)


@pytest.mark.parametrize('method,size', [('grid', 0.5), ('mesh', 1.0)])
def test_vias_bring_the_two_faces_together(method, size):
    def gap(vias):
        p = Problem.of(plate(30, vias=vias), film=10.0)
        p.heat('u', [('F.Cu', box(13, 13, 17, 17))], 1.0)
        r = solve(p, method, size, log=quiet)
        return r.sample('F.Cu', np.array([15.0]), np.array([15.0]))[0] - r.sample('B.Cu', np.array([15.0]), np.array([15.0]))[0]
    none, some = gap([]), gap([(x, y) for x in (14, 15, 16) for y in (14, 15, 16)])
    assert none > 3 * some > 0


def test_the_fine_raster_measures_area():
    g = Point(10.3, 7.7).buffer(4.2, quad_segs=32).difference(Point(11, 8).buffer(1.5, quad_segs=32))
    a = G.raster(g, 0.0, 0.0, 0.0625, 320, 256)
    assert a.sum() * 0.0625 ** 2 == pytest.approx(g.area, rel=2e-3)
    assert a.min() >= 0 and a.max() <= 1


def test_heat_is_shared_over_pads_by_area_and_missing_parts_are_said():
    p = Problem.of(plate(20))
    p.heat('u', [('F.Cu', box(1, 1, 3, 2)), ('F.Cu', box(5, 1, 11, 2)), ('In9.Cu', box(0, 0, 1, 1))], 0.8)
    assert [s.watts for s in p.sources] == pytest.approx([0.2, 0.6])
    assert p.watts == pytest.approx(0.8)
    p.heat('v', [('In9.Cu', box(0, 0, 1, 1))], 0.1)
    assert any(n.startswith('v:') for n in p.notes)


def test_a_part_reads_its_pads_plus_its_own_rise():
    p = Problem.of(plate(20), film=10.0, theta={'u': 20.0})
    p.heat('u', [('F.Cu', box(8, 8, 12, 12))], 0.5)
    r = solve(p, 'grid', 1.0, log=quiet)
    assert r.parts['u']['part_C'] == pytest.approx(r.parts['u']['board_C'] + 10.0)
    assert r.summary()['parts']['u']['power_W'] == pytest.approx(0.5)


def test_circuit_losses_leave_out_what_is_not_on_the_board():
    model = dict(board=dict(parts={'V1': 'J1', 'R1': 'J2', 'LP': 'T1', 'LS': 'T1', 'L9': 'J3'}),
                 parts=[dict(ref='V1', type='source'), dict(ref='R1', type='load'), dict(ref='LP', type='inductor'),
                        dict(ref='LS', type='inductor'), dict(ref='Q1', type='switch'), dict(ref='L9', type='inductor'),
                        dict(ref='C1', type='capacitor')])
    case = dict(parts={'V1': dict(power_W=-6.0), 'R1': dict(power_W=5.0), 'LP': dict(power_W=5.57),
                       'LS': dict(power_W=-5.5), 'Q1': dict(power_W=0.02), 'L9': dict(power_W=0.3),
                       'C1': dict(power_W=-1e-4)})
    losses, left = losses_of(case, model)
    assert losses == pytest.approx({'T1': 0.07, 'Q1': 0.02})
    assert len(left) == 4 and any('connector J3' in s for s in left)


def test_sampling_off_the_board_is_nan():
    p = disc_problem(side=20.0, a=2.0, film=10.0)
    for r in (solve(p, 'grid', 1.0, log=quiet), solve(p, 'mesh', 2.0, log=quiet)):
        t = r.sample(1, np.array([-1.0, 10.0, 25.0]), np.array([10.0, 10.0, 10.0]))
        assert np.isnan(t[0]) and np.isnan(t[2]) and t[1] > 25


def test_bad_sizes_and_methods_are_refused():
    p = disc_problem(side=20.0, a=2.0)
    with pytest.raises(ValueError):
        solve(p, 'grid', -1.0)
    with pytest.raises(ValueError):
        solve(p, 'cubes')


# ---- on a demo board (through kicad2ems)

def demo(name):
    pytest.importorskip('kicad2ems')
    from pcbqs._cli import load_board
    return load_board(str(EXAMPLES / f'{name}.kicad_pcb'))


def test_the_two_solvers_agree_on_a_demo_board():
    board = demo('05_buck_synchronous')
    p = Problem.of(board, {'L1': 0.48, 'Q2': 0.2, 'Q1': 0.1, 'C1': 0.006, 'ZZ9': 1.0})
    assert any(n.startswith('ZZ9:') for n in p.notes)
    g, m = solve(p, 'grid', 0.5, log=quiet), solve(p, 'mesh', 1.0, log=quiet)
    assert set(g.parts) == {'L1', 'Q2', 'Q1', 'C1'}
    for ref in g.parts:
        assert g.parts[ref]['board_C'] == pytest.approx(m.parts[ref]['board_C'], abs=0.4)
    assert g.out_W == pytest.approx(p.watts, rel=1e-9) and m.out_W == pytest.approx(p.watts, rel=1e-9)
    coarse = solve(p, 'grid', 2.0, log=quiet)
    for ref in g.parts:
        assert coarse.parts[ref]['board_C'] == pytest.approx(m.parts[ref]['board_C'], abs=1.0)


def test_the_command_line_takes_a_circuit_run(tmp_path, capsys):
    demo('06_boost')
    from pcbqs.thermal.__main__ import main
    run = tmp_path / 'run'
    run.mkdir()
    model = json.loads((EXAMPLES / '06_boost.json').read_text())
    model['board']['file'] = str(EXAMPLES / '06_boost.kicad_pcb')
    (run / 'model.json').write_text(json.dumps(model))
    parts = {'V1': -12.3, 'C1': 1e-4, 'L1': 0.055, 'Q1': 0.02, 'D1': 0.228, 'C2': 0.003, 'R1': 11.98}
    (run / 'circuit.json').write_text(json.dumps(dict(cases=[
        dict(name='light', loss_W=0.1, parts={k: dict(power_W=v / 2) for k, v in parts.items()}),
        dict(name='full', loss_W=0.3, parts={k: dict(power_W=v) for k, v in parts.items()})])))
    out = tmp_path / 'thermal'
    assert main(['--circuit', str(run), '--method', 'both', '--theta', 'D1=10', '--out', str(out)]) == 0
    said = capsys.readouterr().out
    assert 'losses of case full' in said and 'mesh against grid' in said
    res = json.loads((out / 'thermal.json').read_text())
    assert set(res) == {'grid', 'mesh', 'difference'}
    assert set(res['grid']['parts']) == {'C1', 'L1', 'Q1', 'D1', 'C2'}
    d1 = res['grid']['parts']['D1']
    assert d1['part_C'] == pytest.approx(d1['board_C'] + 2.28, abs=1e-6)
    assert res['grid']['heat_out_W'] == pytest.approx(res['grid']['heat_in_W'], rel=1e-9)
    assert max(abs(v) for v in res['difference']['parts_K'].values()) < 0.5
    for f in ('thermal-grid.png', 'thermal-mesh.png', 'thermal-compare.png', 'thermal.npz'):
        assert (out / f).stat().st_size > 1000
    z = np.load(out / 'thermal.npz')
    assert z['grid_T'].shape[0] == 2 and z['mesh_T'].shape == (2, len(z['mesh_nodes']))
    assert main(['--circuit', str(run), '--case', 'light', '--out', str(tmp_path / 'one')]) == 0
    assert json.loads((tmp_path / 'one' / 'thermal.json').read_text())['method'] == 'grid'


def test_the_command_line_needs_a_loss(tmp_path):
    from pcbqs.thermal.__main__ import main
    with pytest.raises(SystemExit):
        main([str(EXAMPLES / '06_boost.kicad_pcb'), '--out', str(tmp_path)])
