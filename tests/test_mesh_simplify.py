"""Lone-pad dropping, regular pads/holes and polygonal via rings."""
import math

import numpy as np
import pytest
from shapely.geometry import Point, box, mapping

from pcbqs.mqs import MQS
from pcbqs.pdn.mesh import mesh_sheet, simplify_copper


def test_simplify_copper_rules():
    plane = box(0, 0, 10, 10).difference(Point(5, 5).buffer(.4, quad_segs=16))     # a round clearance hole
    lone = Point(20, 5).buffer(.3, quad_segs=16)                                   # a via pad, one via
    pair = box(30, 4.8, 30.9, 5.2)                                                 # two vias tied by copper
    terminal = Point(40, 5).buffer(.3, quad_segs=16)                               # a terminal's round pad
    copper = plane.union(lone).union(pair).union(terminal)
    vias = [(5, 5, .15), (20, 5, .15), (30.2, 5, .1), (30.7, 5, .1), (40, 5, .15)]
    out, counts = simplify_copper(copper, keep=[terminal], points=vias, area=1.0, sides=8)
    assert counts == dict(dropped_pads=1, regular_pads=0, regular_holes=1)
    assert not out.intersects(Point(20, 5)) and out.contains(Point(30.45, 5)) and out.contains(Point(40, 5))
    hole = [r for p in out.geoms for r in p.interiors][0]
    assert len(hole.coords) - 1 == 8
    assert out.area == pytest.approx(copper.area - lone.area, rel=1e-9)            # equal-area hole
    same, none = simplify_copper(copper, keep=[terminal], points=vias, drop_lone=False, sides=None)
    assert same.equals(copper) and none == dict(dropped_pads=0, regular_pads=0, regular_holes=0)
    kept, counts = simplify_copper(lone, points=[(20, 5, .15)], drop_lone=False, sides=6)
    assert counts['regular_pads'] == 1 and len(kept.exterior.coords) - 1 == 6
    assert kept.area == pytest.approx(lone.area, rel=1e-12)
    assert simplify_copper(lone, points=[(20, 5, .15)])[0] is None


@pytest.mark.parametrize('sides', [4, 6, 8])
def test_via_ring_matches_the_via_perimeter(sides):
    nodes, tris, rings, _ = mesh_sheet(box(0, 0, 4, 4), [(2, 2, .2)], dmax=.5, via_sides=sides)
    ring = nodes[rings[0][1:]] - (2, 2)
    corners = ring[np.argsort(-np.hypot(*ring.T))[:sides]]
    side = np.linalg.norm(corners[0] - corners[np.argsort(np.linalg.norm(corners - corners[0], axis=1))[1]])
    assert sides*side == pytest.approx(2*math.pi*.2, rel=1e-4)          # vertices are snapped to the mesher grid
    with pytest.raises(ValueError, match='via_sides'):
        mesh_sheet(box(0, 0, 4, 4), [(2, 2, .2)], dmax=.5, via_sides=2)


from test_via_segments import stitched_board


def test_lone_pads_are_dropped_and_vias_still_join_the_planes():
    terminals = {'s': [('F.Cu', box(.5, 3.5, 1.5, 4.5))], 'g': [('B.Cu', box(.5, 3.5, 1.5, 4.5))]}
    results = {}
    for simplified in (False, True):
        options = {} if simplified else dict(drop_lone_pads=False, pad_sides=None, via_sides=None)
        q = MQS(stitched_board(), 'N', terminals=terminals, dmax=.8, log=lambda *_: None, **options)
        q.source('I', 's', 'g', part_h=0)
        results[simplified] = q, q.solve([1e3, 1e6])
    plain, simple = results[False][0], results[True][0]
    assert simple.mesh_info['dropped_pads'] == 6 and plain.mesh_info['dropped_pads'] == 0
    assert set(simple.layer_tris) == {'B.Cu', 'F.Cu'} and len(simple.tris) < len(plain.tris)
    assert simple.nv == 3 and plain.nv == 9           # one barrel per via instead of three segments
    # Lone pads carry no current, so dropping them alone changes nothing; the
    # square via ring shifts this deliberately coarse (0.8 mm) case by ~1% in R.
    z0, z1 = results[False][1].z['I'], results[True][1].z['I']
    np.testing.assert_allclose(z1.real, z0.real, rtol=.02)
    np.testing.assert_allclose(z1.imag, z0.imag, rtol=.005)


def test_simplification_options_are_validated():
    with pytest.raises(ValueError, match='pad_sides'):
        MQS(stitched_board(), 'N', dmax=.8, pad_sides=2, log=lambda *_: None)
    with pytest.raises(ValueError, match='pad_area'):
        MQS(stitched_board(), 'N', dmax=.8, pad_area=-1, log=lambda *_: None)
