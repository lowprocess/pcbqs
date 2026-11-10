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
    assert len(zs) == 4 and zs.min() > 1.6 and zs.max() < 1.6 + 0.035      # in the slab [z, z + t] (copper_centres)
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


def test_thick_copper_keeps_the_dielectric_between_facing_surfaces():
    """copper_centres: each slab starts at its z plus the copper below, so KiCad's dielectric is the face-to-face gap."""
    from pcbqs.pdn.mesh import copper_centres
    b = dict(stack=dict(copper=[dict(name='B', z=0.0, t=0.035), dict(name='In', z=0.1, t=0.015),
                                dict(name='F', z=0.6, t=0.035)]))
    c = copper_centres(b)
    top = lambda l: c[l][0] + c[l][1] / 2
    bot = lambda l: c[l][0] - c[l][1] / 2
    assert bot('In') - top('B') == pytest.approx(0.1) and bot('F') - top('In') == pytest.approx(0.5)



def test_thick_kernel_gives_a_bars_inductance_on_both_backends():
    """thick_kernel: the sheet's close pairs through the kernel averaged over the copper's thickness, so a 20 x 1 mm x
    35 um strip has a bar's DC inductance (Grover, 0.3%; a zero-thickness sheet is ~1% high), the same on the dense
    and compressed backends (the correction is one sparse matrix added to either)."""
    mu = 4e-7 * np.pi
    l, w, t = 19.9e-3, 1e-3, 35e-6                    # between the terminal slivers' centres
    grover = mu * l / (2 * np.pi) * (np.log(2 * l / (w + t)) + 0.5 + 0.2235 * (w + t) / l)
    out = {}
    for back in ('dense', 'hierarchical'):
        q = MQS(strip(), {'S'}, terminals=TERMS, dmax=0.5, conductor_loss='dc', thick_kernel=True, matrix_backend=back,
                log=lambda *a: None)
        q.source('port', 'A', 'B', part_h=0.0)
        out[back] = np.asarray(q.solve(np.array([1e3, 1e8])).zc['port'])
    assert out['dense'][0].imag / (2 * np.pi * 1e3) == pytest.approx(grover, rel=3e-3)
    np.testing.assert_allclose(out['hierarchical'], out['dense'], rtol=1e-5)
