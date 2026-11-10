"""MQS(thickness_sheets=N, conductor_loss='dc'): N graded sheets through the copper, the thick-copper reference."""
import numpy as np
import pytest
from shapely.geometry import box, mapping

from pcbqs.mqs import MQS


def strip(t=0.035):
    """A 20 x 1 mm strip on F.Cu with 0.1 mm terminal slivers at its ends."""
    return dict(stack=dict(copper=[dict(name='F.Cu', z=1.6, t=t)], layers=[], thickness=1.6),
                copper={'F.Cu': [dict(net='S', polygon=mapping(box(0, 0, 20, 1)))]}, vias=[],
                outline=mapping(box(-1, -1, 21, 2)), prims=dict(footprints=[]))


TERMS = {'A': [('F.Cu', box(0, 0, 0.1, 1))], 'B': [('F.Cu', box(19.9, 0, 20, 1))]}


def solve(n, f=(1e3,)):
    q = MQS(strip(), {'S'}, terminals=TERMS, dmax=0.5, conductor_loss='dc', thickness_sheets=n, log=lambda *a: None)
    q.source('port', 'A', 'B', part_h=0.0)
    return q, np.asarray(q.solve(np.asarray(f)).zc['port'])


def test_stacked_sheets_keep_the_dc_resistance_and_lie_in_the_copper():
    _, z1 = solve(1)
    q, z4 = solve(4)
    assert z4.real[0] == pytest.approx(z1.real[0], rel=2e-3)        # the slices' conductances add up
    zs = np.unique(np.round(q.z_of * 1e3, 6))
    assert len(zs) == 4 and zs.min() > 1.6 - 0.0175 and zs.max() < 1.6 + 0.0175
    ts = {round(z, 6): t for z, t in zip(q.z_of * 1e3, q.t_of * 1e3)}
    assert sum(ts.values()) == pytest.approx(0.035, rel=1e-9)        # graded slices fill the thickness
    assert ts[zs.min()] < ts[sorted(zs)[1]]                          # thinner at the faces


def test_stacked_sheets_show_skin_effect():
    """At 100 MHz (skin depth 6.6 um in 35 um copper) the resistance rises well above DC."""
    _, z = solve(4, f=(1e3, 1e8))
    assert z.real[1] > 1.5 * z.real[0]


def test_stacked_sheets_need_dc_loss():
    with pytest.raises(ValueError):
        MQS(strip(), {'S'}, terminals=TERMS, conductor_loss='slab', thickness_sheets=3, log=lambda *a: None)
