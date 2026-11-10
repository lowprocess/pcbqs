"""Capacitance with thickness: charge on the face sheets (each copy its own charge) and on the copper's sides."""
import numpy as np
import pytest
from shapely.geometry import Point, box, mapping

from pcbqs.mqs import MQS


def microstrip():
    """A 0.3 mm trace 10 mm long, 0.2 mm (face to face) over a 6 x 12 mm plane, in air."""
    return dict(stack=dict(copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='F.Cu', z=0.2, t=0.035)],
                           layers=[dict(name='core', z0=0.0, z1=0.2, er=1.0, tand=0.0)], thickness=0.2),
                copper={'F.Cu': [dict(net='T', polygon=mapping(box(-0.15, -5, 0.15, 5)))],
                        'B.Cu': [dict(net='G', polygon=mapping(box(-3, -6, 3, 6)))]},
                vias=[], outline=mapping(box(-4, -7, 4, 7)), prims=dict(footprints=[]))


TERMS = {'T': [('F.Cu', Point(0, 0).buffer(0.05, quad_segs=4))], 'G': [('B.Cu', Point(0, 0).buffer(0.05, quad_segs=4))]}


def c_of(**kw):
    q = MQS(microstrip(), {'T', 'G'}, terminals=TERMS, dmax=0.5, capacitance=True, er=1.0, conductor_loss='slab',
            log=lambda *a: None, **kw)
    C, _, tl = q.capacitance_matrix()
    C = np.real(np.asarray(C))
    return C, tl


def test_thickness_adds_capacitance_step_by_step():
    """One sheet < two charged faces < faces + side panels (FasterCap: copper boxes +17% over sheets here)."""
    c1, t1 = c_of(thickness_sheets=1)
    c2, t2 = c_of(thickness_sheets=2)
    c3, t3 = c_of(thickness_sheets=2, side_charge=True)
    m = [abs(c[t['T'], t['G']]) for c, t in ((c1, t1), (c2, t2), (c3, t3))]
    assert m[0] < m[1] < m[2]
    assert 1.05 < m[2] / m[0] < 1.3
    np.testing.assert_allclose(c3, c3.T, rtol=1e-9)


def test_side_charge_needs_face_sheets_and_capacitance():
    with pytest.raises(NotImplementedError):
        MQS(microstrip(), {'T', 'G'}, terminals=TERMS, conductor_loss='slab', side_charge=True, log=lambda *a: None)
