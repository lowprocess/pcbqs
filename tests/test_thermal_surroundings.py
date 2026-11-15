"""The thermal model's surroundings beyond one film (R-E13): a part's own path to the air, a footprint held at a
temperature, moving air, and the heating from cold, against their exact answers on a plate heated all over."""
import json
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box, mapping

from pcbqs.thermal import Problem, solve
from pcbqs.thermal.problem import FILM, RADIATION, RHO_C_CU, RHO_C_FR4, Held, forced_film

EXAMPLES = Path(__file__).resolve().parents[1] / 'src' / 'pcbqs' / 'circuit' / 'examples'
quiet = lambda *a: None


def plate(w, layers=(('B.Cu', 0.0), ('F.Cu', 1.5))):
    sq = box(0, 0, w, w)
    return dict(stack=dict(layers=[], copper=[dict(name=n, z=z, t=0.035) for n, z in layers], thickness=1.5),
                copper={n: [dict(net='GND', polygon=mapping(sq))] for n, _ in layers}, vias=[], outline=mapping(sq))


def board_conductance(p, area):
    """W/K from the top sheet of a uniform two-sheet plate to the air (no edges)."""
    h, c = p.film[1], p.gaps[0]
    return area * (h + c - c * c / (p.film[0] + c))


@pytest.mark.parametrize('method,size', [('grid', 1.0), ('mesh', 2.0)])
def test_a_parts_own_path_to_the_air_takes_its_share(method, size):
    p = Problem.of(plate(40), film=10.0, edge_film=0.0)
    p.heat('Q1', [('F.Cu', box(0, 0, 40, 40))], 2.0)
    p.theta, p.air = {'Q1': 5.0}, {'Q1': 20.0}
    r = solve(p, method, size, log=quiet)
    gb, theta, R, P = board_conductance(p, 0.04 ** 2), 5.0, 20.0, 2.0
    part = P / (gb / (1 + theta * gb) + 1 / R)                     # its node: through theta to the plate, or to air
    q = r.parts['Q1']
    assert q['part_C'] == pytest.approx(25 + part, rel=1e-6)
    assert q['to_air_W'] == pytest.approx(part / R, rel=1e-6)
    assert q['board_C'] == pytest.approx(25 + part / (1 + theta * gb), rel=1e-6)
    assert r.out_W + q['to_air_W'] == pytest.approx(P, rel=1e-9)      # every watt leaves somewhere
    assert r.summary()['heat_out_parts_air_W'] == pytest.approx(q['to_air_W'])


@pytest.mark.parametrize('method,size', [('grid', 1.0), ('mesh', 2.0)])
def test_a_held_footprint_holds_its_pads_and_says_what_leaves_through_it(method, size):
    p = Problem.of(plate(40), film=10.0, edge_film=0.0)
    p.held.append(Held('H1', 1, box(0, 0, 40, 40), 60.0))          # the whole top face at 60 C, nothing heated
    r = solve(p, method, size, log=quiet)
    assert np.nanmax(np.abs(r.T[1] - 60.0)) < 1e-3
    h, c = p.film[0], p.gaps[0]
    assert np.nanmean(r.T[0]) == pytest.approx(25 + 35 * c / (h + c), rel=1e-4)
    assert -r.extra['held_W']['H1'] == pytest.approx(r.out_W, rel=1e-6)   # into the board through it, out by the films


def test_moving_air_is_a_flat_plates_convection_joined_to_the_still_airs():
    assert forced_film(0.0, 100.0) == pytest.approx(FILM)
    natural, forced = FILM - RADIATION, 3.84 * np.sqrt(2.0 / 0.1)
    assert forced_film(2.0, 100.0) == pytest.approx(RADIATION + (natural ** 3 + forced ** 3) ** (1 / 3))
    assert forced_film(1.0, 50.0) < forced_film(4.0, 50.0) and forced_film(2.0, 200.0) < forced_film(2.0, 50.0)
    with pytest.raises(ValueError):
        forced_film(-1.0, 100.0)


def test_the_heating_from_cold_is_the_boards_time_constant():
    p = Problem.of(plate(40, layers=(('F.Cu', 0.0),)), film=10.0, edge_film=0.0)   # one sheet: one time constant
    p.heat('all', [('F.Cu', box(0, 0, 40, 40))], 1.0)
    r = solve(p, 'grid', 1.0, log=quiet, transient=3000.0)
    tr = r.transient
    C = 0.04 ** 2 * (1.5e-3 * RHO_C_FR4 + 0.035e-3 * RHO_C_CU)
    G = 0.04 ** 2 * 2 * 10.0
    assert tr['to_63_s']['all'] == pytest.approx(C / G, rel=0.025)       # backward Euler, 40 steps a decade
    assert tr['parts']['all'][0] == pytest.approx(25.0) and tr['parts']['all'][-1] == pytest.approx(25 + 1 / G, rel=1e-3)
    assert np.all(np.diff(tr['parts']['all']) >= -1e-9) and tr['times_s'][-1] == pytest.approx(3000.0)


def test_the_command_line_takes_the_surroundings(tmp_path, capsys):
    from pcbqs.thermal.__main__ import main
    board = str(EXAMPLES / '04_buck.kicad_pcb')
    assert main([board, '--loss', 'D1=0.4', '--loss', 'L1=0.15', '--air', 'D1=40', '--theta', 'D1=20',
                 '--hold', 'C1=25', '--air-speed', '2', '--transient', '600', '--out', str(tmp_path)]) == 0
    r = json.loads((tmp_path / 'thermal.json').read_text())
    assert r['heat_out_W'] + r['heat_out_parts_air_W'] + r['held']['C1']['out_W'] == pytest.approx(r['heat_in_W'], rel=1e-4)
    assert r['film_W_m2K'][0] > FILM and r['parts']['D1']['to_air_W'] > 0
    assert (tmp_path / 'thermal-transient.json').is_file() and (tmp_path / 'thermal-transient.png').is_file()
    assert r['transient']['to_63_s']['L1'] > 0
    out = capsys.readouterr().out
    assert 'to the air directly' in out and 'held' in out and 'heating from cold' in out
