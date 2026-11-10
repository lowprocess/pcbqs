"""Part contacts on pads (C13): a two-pad chip part touches each pad from its end cap outward (cap and fillet)."""
from types import SimpleNamespace as NS

import numpy as np
import pytest
from shapely.geometry import box, mapping

from pcbqs.contacts import beyond, chip_size, part_contacts


def pad(x0, x1, layers=('F.Cu',), net='A'):
    return NS(geom=box(x0, -0.6, x1, 0.6), layers=list(layers), net=net)


def chip(lib='Capacitor_SMD:C_0805_2012Metric'):
    """0805 pads: 1.0 mm long, centres 1.9 mm apart (KiCad's footprint)."""
    return NS(ref='C1', lib=lib, pads=[pad(-1.45, -0.45), pad(0.45, 1.45, net='B')])


def test_chip_size_from_the_footprint_name():
    assert chip_size('Capacitor_SMD:C_0805_2012Metric') == (2.0, 0.5)
    assert chip_size('CM5IO:C_0402_1005Metric') == (1.0, 0.25)
    assert chip_size('R_0603') == (1.6, 0.35)
    assert chip_size('Inductor_SMD:L_Bourns_SRP1265A') is None


def test_chip_part_contacts_its_pads_from_the_end_cap_outward():
    """0805: body 2.0 mm, cap 0.5 mm, so the cap starts 0.5 mm from the centre; the pads start at 0.45 mm."""
    c = part_contacts(chip())
    for k, (lo, hi) in enumerate([(-1.45, -0.5), (0.5, 1.45)]):
        x0, _, x1, _ = c[k].bounds
        assert (x0, x1) == pytest.approx((lo, hi))
        assert c[k].area == pytest.approx(0.95 * 1.2)


def test_modes():
    whole = part_contacts(chip(), mode='pad')
    assert whole[0].equals(box(-1.45, -0.6, -0.45, 0.6))
    edge = part_contacts(chip(), mode='edge')
    assert edge[1].bounds[0] == pytest.approx(0.45) and edge[1].bounds[2] == pytest.approx(0.55)
    unknown = part_contacts(chip(lib='X:L_Big'))                          # no size code: the whole pad
    assert unknown[0].equals(chip().pads[0].geom)
    with pytest.raises(ValueError):
        part_contacts(chip(), mode='middle')


def test_ics_and_through_hole_keep_whole_pads():
    ic = NS(ref='U1', lib='QFN-16', pads=[pad(0, 1), pad(2, 3), pad(4, 5)])
    assert all(ic.pads[k].geom.equals(g) for k, g in part_contacts(ic).items())
    th = NS(ref='J1', lib='C_0805', pads=[pad(-1.45, -0.45, layers=('*.Cu',)), pad(0.45, 1.45)])
    assert th.pads[1].geom.equals(part_contacts(th)[1])


def test_band_follows_a_rotated_part():
    from shapely import affinity
    g = affinity.rotate(box(0.45, -0.6, 1.45, 0.6), 30, origin=(0, 0))
    u = (np.cos(np.radians(30)), np.sin(np.radians(30)))
    b = beyond(g, (0, 0), u, 0.95)
    assert b.area == pytest.approx(0.5 * 1.2, rel=1e-6)


def test_mqs_loop_inductance_rises_when_parts_no_longer_short_their_pads():
    """A 10 x 4 mm loop of 1 mm trace closed by an 0805-sized part: the less of each pad the contact covers, the
    more copper carries the current and the higher the loop reads."""
    from pcbqs.mqs import MQS
    from shapely.ops import unary_union
    tr = unary_union([box(0, 0, 10, 1), box(9, 0, 10, 4), box(0, 3, 10, 4), box(0, 0, 1, 1.55),
                      box(0, 2.45, 1, 4)])
    pa, pb = box(0, 0.55, 1, 1.55), box(0, 2.45, 1, 3.45)             # 1 mm pads 0.9 mm apart, on the open side
    part = NS(ref='C1', lib='C_0805_2012Metric', pads=[NS(geom=pa, layers=['F.Cu'], net='S'),
                                                        NS(geom=pb, layers=['F.Cu'], net='S')])
    board = dict(stack=dict(copper=[dict(name='F.Cu', z=1.6, t=0.035)], layers=[], thickness=1.6),
                 copper={'F.Cu': [dict(net='S', polygon=mapping(tr.difference(box(0, 1.55, 1, 2.45))))]}, vias=[],
                 outline=mapping(box(-1, -1, 11, 5)), prims=dict(footprints=[]))
    L = {}
    for mode in ('pad', 'termination', 'edge'):
        c = part_contacts(part, mode)
        q = MQS(board, {'S'}, terminals={'a': [('F.Cu', c[0])], 'b': [('F.Cu', c[1])]}, dmax=0.25,
                conductor_loss='dc', thickness_sheets=1, log=lambda *a: None)
        q.source('p', 'a', 'b', part_h=0.0)
        L[mode] = np.asarray(q.solve(np.array([1e6])).zc['p'])[0].imag / (2 * np.pi * 1e6)
    assert L['pad'] < L['termination'] < L['edge']
