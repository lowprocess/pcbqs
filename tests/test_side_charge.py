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


def test_face_sheets_in_a_layered_dielectric():
    """dielectric='stack' with face sheets: the stack expanded by the copper's thickness (faces on interfaces); in
    FR-4 the thickness adds a little capacitance (Hammerstad-Jensen: +3.3% for this microstrip)."""
    b = microstrip()
    b['stack']['layers'][0]['er'] = 4.4
    cs = []
    for sheets in (1, 2):
        q = MQS(b, {'T', 'G'}, terminals=TERMS, dmax=0.5, capacitance=True, dielectric='stack', conductor_loss='slab',
                thickness_sheets=sheets, log=lambda *a: None)
        C, _, tl = q.capacitance_matrix()
        cs.append(abs(np.real(np.asarray(C))[tl['T'], tl['G']]))
    assert 1.0 < cs[1] / cs[0] < 1.08


def _differential(C, t):
    """Two conductors driven against each other (net charge zero): the capacitance the port sees."""
    a, b = t['T'], t['G']
    return (C[a, a] * C[b, b] - C[a, b] ** 2) / (C[a, a] + C[b, b] + 2 * C[a, b])


@pytest.mark.parametrize('side', [False, True])
def test_ac_solve_with_face_sheets_matches_their_capacitance(side):
    """The AC PEEC solve with face sheets: at 1 kHz the open trace is the capacitance its faces (and sides) carry,
    the same as capacitance_matrix; the charge stays free to move between a node's copies."""
    q = MQS(microstrip(), {'T', 'G'}, terminals=TERMS, dmax=0.5, capacitance=True, er=1.0, conductor_loss='slab',
            thickness_sheets=2, side_charge=side, log=lambda *a: None)
    C, _, tl = q.capacitance_matrix()
    Cd = _differential(np.real(np.asarray(C)), tl)
    q.source('port', 'T', 'G', part_h=0.0)
    f = np.array([1e3])
    z = np.asarray(q.solve(f).z['port'])
    assert (-1 / (2 * np.pi * f * z.imag))[0] == pytest.approx(Cd, rel=1e-6)


def test_face_sheets_ac_solve_on_the_compressed_backend():
    """The compressed backend solves face sheets with capacitance through extra unknowns (the charge moving between
    a node's copies), to the dense backend's answer, from the electrostatic to the inductive regime."""
    out = {}
    for back in ('dense', 'hierarchical'):
        q = MQS(microstrip(), {'T', 'G'}, terminals=TERMS, dmax=0.5, capacitance=True, er=1.0,
                conductor_loss='slab', thickness_sheets=2, matrix_backend=back, log=lambda *a: None)
        q.source('port', 'T', 'G', part_h=0.0)
        r = q.solve(np.array([1e3, 1e8, 2e9]))
        out[back] = (np.asarray(r.z['port']), r.charge)
        if back == 'hierarchical':
            assert max(i['relative_residual'] for i in q.solve_info) < 1e-7
    np.testing.assert_allclose(out['hierarchical'][0], out['dense'][0], rtol=1e-4)
    np.testing.assert_allclose(out['hierarchical'][1], out['dense'][1], rtol=1e-3, atol=1e-18)


@pytest.mark.parametrize('pre', ['auto', 'graph'])
def test_side_charge_on_the_compressed_backend(pre):
    """Side panels join the compressed potential (panel pairs through the any-orientation moments): the
    capacitance and the AC solve match the dense backend's, with either preconditioner."""
    out = {}
    for back in ('dense', 'hierarchical'):
        q = MQS(microstrip(), {'T', 'G'}, terminals=TERMS, dmax=0.5, capacitance=True, er=1.0,
                conductor_loss='slab', thickness_sheets=2, side_charge=True, matrix_backend=back,
                log=lambda *a: None)
        q.solver_preconditioner = pre
        C, _, tl = q.capacitance_matrix()
        q.source('port', 'T', 'G', part_h=0.0)
        out[back] = (_differential(np.real(np.asarray(C)), tl), np.asarray(q.solve(np.array([1e3, 1e8, 2e9])).z['port']))
        if back == 'hierarchical':
            assert max(i['iterations'] for i in q.solve_info) < 200
    assert out['hierarchical'][0] == pytest.approx(out['dense'][0], rel=1e-4)
    np.testing.assert_allclose(out['hierarchical'][1], out['dense'][1], rtol=1e-4)


def test_face_sheets_ac_solve_in_a_layered_dielectric():
    """dielectric='stack' with face sheets: the AC solve's 1 kHz capacitance is capacitance_matrix's."""
    b = microstrip()
    b['stack']['layers'][0].update(er=4.4, tand=0.02)
    q = MQS(b, {'T', 'G'}, terminals=TERMS, dmax=0.5, capacitance=True, dielectric='stack', conductor_loss='slab',
            thickness_sheets=2, log=lambda *a: None)
    C, _, tl = q.capacitance_matrix()
    q.source('port', 'T', 'G', part_h=0.0)
    f = np.array([1e3])
    y = 1 / np.asarray(q.solve(f).z['port'])[0]
    assert (y / (2j * np.pi * f[0])) == pytest.approx(_differential(np.asarray(C), tl), rel=1e-6)
