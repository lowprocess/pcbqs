"""Stacked via segments: their partial inductances add up to the whole barrel."""
import numpy as np
import pytest
from shapely.geometry import Point, box, mapping

from pcbqs.mqs import MQS


def stitched_board(inner_pads=True):
    """Two plane layers joined by three vias; optionally the vias have pads on two inner layers."""
    names = [('B.Cu', 0.), ('In2.Cu', .2), ('In1.Cu', .4), ('F.Cu', .6)]
    plane = box(0, 0, 12, 8)
    pads = [Point(x, 4).buffer(.35, quad_segs=16) for x in (4, 6, 8)]
    copper = {'B.Cu': [dict(net='N', polygon=mapping(plane))], 'F.Cu': [dict(net='N', polygon=mapping(plane))]}
    if inner_pads:
        copper['In1.Cu'] = [dict(net='N', polygon=mapping(p)) for p in pads]
        copper['In2.Cu'] = [dict(net='N', polygon=mapping(p)) for p in pads]
    vias = [dict(x=x, y=4, r=.15, z0=0, z1=.6, net='N') for x in (4, 6, 8)]
    stack = dict(layers=[], copper=[dict(name=n, z=z, t=.035) for n, z in names], thickness=.6)
    return dict(stack=stack, copper=copper, vias=vias, outline=mapping(plane))


TERMINALS = {'s': [('F.Cu', box(.5, 3.5, 1.5, 4.5))], 'g': [('B.Cu', box(.5, 3.5, 1.5, 4.5))]}


@pytest.mark.parametrize('backend', ['dense', 'hierarchical'])
def test_stacked_via_segments_add_up_to_the_whole_barrel(backend):
    # Inner pads split each barrel into three coaxial segments. Their self and
    # mutual terms (same radius) must sum to the undivided barrel's self term.
    options = dict(capacitance=True, matrix_backend='hierarchical', compression_tol=1e-10) if backend == 'hierarchical' else {}
    segmented = MQS(stitched_board(True), 'N', terminals=TERMINALS, dmax=.8, log=lambda *_: None, **options)
    whole = MQS(stitched_board(False), 'N', terminals=TERMINALS, dmax=.8, log=lambda *_: None, **options)
    assert segmented.nv == 9 and whole.nv == 3
    dense = lambda L: L if isinstance(L, np.ndarray) else L @ np.eye(L.shape[0])
    Ls, Lw = dense(segmented._via_matrices()[1]), dense(whole._via_matrices()[1])
    group = np.repeat(np.arange(3), 3)               # segments are ordered per via, bottom to top
    total = np.array([[Ls[np.ix_(group == a, group == b)].sum() for b in range(3)] for a in range(3)])
    np.testing.assert_allclose(total, Lw, rtol=1e-9)


def test_dead_end_inner_pads_do_not_change_the_loop_inductance():
    # Pads that carry no current must not change the loop: before the fix the
    # segment mutuals used 0.1 um separation and these differed by ~3%.
    results = []
    for inner in (True, False):
        q = MQS(stitched_board(inner), 'N', terminals=TERMINALS, dmax=.8, log=lambda *_: None)
        q.source('I', 's', 'g', part_h=0)
        results.append(q.solve([1e3]).z['I'][0])
    assert results[0].imag == pytest.approx(results[1].imag, rel=1e-6)
    assert results[0].real == pytest.approx(results[1].real, rel=1e-8)
