"""The magneto-quasistatic solver (pcbqs.mqs) and the near field (pcbqs.farfield.near_fields, pcbqs.biot) against
closed forms: a trace over a plane (its return current under it, its inductance, its field by image theory), a
via's partial inductance, the spectral H against Biot-Savart."""
import numpy as np
import pytest
from shapely.geometry import box, mapping, Point

from pcbqs._numerics.base import MU0
from pcbqs.mqs import MQS, filament_mutual, spanning_tree, via_inductance

H_TR, W_TR = 0.5, 0.5                     # mm: the trace's height over the plane, its width


def trace_board(dmax_plane=1.0):
    """A 30 x 20 mm plane (B.Cu) and a 0.5 mm trace 0.5 mm over it (F.Cu) from x 3 to 27 at y 10, one net: a via at
    the far end joins them; at the near end a pad on F.Cu with its via down to the plane, beside the trace's pad."""
    cu = {'B.Cu': [dict(net='N', polygon=mapping(box(0, 0, 30, 20)))],
          'F.Cu': [dict(net='N', polygon=mapping(box(3, 10 - W_TR / 2, 27.5, 10 + W_TR / 2))),
                   dict(net='N', polygon=mapping(box(2.0, 9.5, 3.0, 10.5))),          # the trace's start pad
                   dict(net='N', polygon=mapping(box(2.0, 7.6, 3.0, 8.6)))]}          # the return's pad
    vias = [dict(x=27.2, y=10, r=0.15, z0=0, z1=H_TR, net='N'), dict(x=2.5, y=8.1, r=0.15, z0=0, z1=H_TR, net='N')]
    stack = dict(layers=[dict(name='d', z0=0, z1=H_TR, er=1.0, tand=0, kind='diel')],
                 copper=[dict(name='B.Cu', z=0.0, t=0.035), dict(name='F.Cu', z=H_TR, t=0.035)], thickness=H_TR)
    return dict(stack=stack, copper=cu, vias=vias, outline=mapping(box(0, 0, 30, 20)))


@pytest.fixture(scope='module')
def solved():
    b = trace_board()
    q = MQS(b, 'N', terminals={'s': [('F.Cu', box(2.0, 9.6, 2.6, 10.4))], 'g': [('F.Cu', box(2.0, 7.7, 2.6, 8.5))]},
            dmax=0.6, log=lambda *a: None)
    q.source('I', 's', 'g', part_h=0.0)
    return q, q.solve([1e3, 1e7])


def plane_profile(q, r, fi, x0=15.0):
    """The plane's Jx (A/m) across y at x0 (mm), from its triangles' centroids near x0."""
    cen, J = q.current_density(r, fi)['B.Cu']
    sel = np.abs(cen[:, 0] * 1e3 - x0) < 1.0
    return cen[sel, 1] * 1e3, J[sel, 0]


def test_return_current_comes_under_the_trace_at_high_frequency(solved):
    q, r = solved
    frac = []
    for fi in (0, 1):
        y, jx = plane_profile(q, r, fi)
        order = np.argsort(y)
        y, jx = y[order], np.abs(jx[order])
        tot = np.trapezoid(jx, y)
        near = np.trapezoid(np.where(np.abs(y - 10) < H_TR, jx, 0), y)
        frac.append(near / tot)
    # 10 MHz: inductive, the image's 1 / (1 + (y / h)^2): half within +-h; 1 kHz: resistive, spread over the plane
    assert frac[1] == pytest.approx(0.5, abs=0.08)
    assert frac[0] < 0.3


def test_loop_inductance_and_resistance(solved):
    q, r = solved
    z = r.z['I']
    L = z[1].imag / (2 * np.pi * 1e7)
    # Wheeler, a thin strip over a plane (w / h <= 1): mu0 / 2 pi ln(8 h / w + w / 4 h) per length; 24.5 mm of it,
    # its two vias and the pads' turn besides
    Lw = MU0 / (2 * np.pi) * np.log(8 * H_TR / W_TR + W_TR / (4 * H_TR)) * 24.5e-3
    assert Lw < L < 1.25 * Lw
    # at 1 kHz the loop is its resistance: 24.5 mm of 0.5 mm, 35 um trace (and the plane, the vias)
    Rt = 24.5e-3 / (5.8e7 * 35e-6 * 0.5e-3)
    assert Rt < z[0].real < 1.4 * Rt


def test_field_above_the_trace_by_image_theory(solved):
    q, r = solved
    zp = H_TR + 2.0                               # mm above the plane
    pts = np.array([[15e-3, 10e-3, zp * 1e-3]])
    H = q.h_field(r, 1, pts)[0]
    Himg = 1 / (2 * np.pi) * (1 / ((zp - H_TR) * 1e-3) - 1 / ((zp + H_TR) * 1e-3))   # a long wire and its image
    assert abs(H[1]) == pytest.approx(Himg, rel=0.06)
    assert abs(H[0]) < 0.05 * abs(H[1]) and abs(H[2]) < 0.05 * abs(H[1])


def test_via_partial_inductance_is_rosas():
    h, a = 1.6e-3, 0.15e-3
    L = filament_mutual(0, h, 0, h, a)
    rosa = MU0 / (2 * np.pi) * (h * np.log((h + np.sqrt(h * h + a * a)) / a) - np.sqrt(h * h + a * a) + a)
    assert L == pytest.approx(rosa, rel=1e-9)
    assert filament_mutual(0, h, 0, h, 1e-3) < L


def test_the_via_matrix_is_each_pair_s_filaments():
    """The via segments' matrix in blocks (a Python double loop had taken ~15 s at 2,500 segments): each entry the
    two filaments', one barrel's stacked segments at its radius."""
    v = np.array([(0, 0, 1.5e-4, 0, 3e-4), (0, 0, 1.5e-4, 3e-4, 6e-4), (2e-3, 1e-3, 2e-4, 0, 6e-4)])
    L = via_inductance(v, rows=2)                                   # (two blocks)
    for i, (x, y, r, z0, z1) in enumerate(v):
        for j, (x2, y2, r2, z02, z12) in enumerate(v):
            s = np.hypot(x - x2, y - y2)
            assert L[i, j] == pytest.approx(filament_mutual(z0, z1, z02, z12, s if s >= 1e-9 else max(r, r2)), rel=1e-12)
    assert L == pytest.approx(L.T, rel=1e-12) and via_inductance(np.zeros((0, 5))).shape == (0, 0)


def test_the_loop_tree_is_breadth_first_over_the_copper_and_parts_join_islands():
    """A 12 x 12 grid of copper: the tree spans it; a part beside copper stays out of the tree (a source's loop then
    runs through copper, which its zc is), one joining two islands is in it, and a source is never. (Its loops are
    shorter than the branch order's on a board's mesh, not on a grid: the same there.)"""
    n = 12
    node = lambda i, j: i * n + j
    ends = [(node(i, j), node(i, j + 1)) for i in range(n) for j in range(n - 1)] + \
           [(node(i, j), node(i + 1, j)) for i in range(n - 1) for j in range(n)]
    copper = len(ends)
    island = n * n                                                  # a node of its own, joined by a part alone
    ends += [(node(0, 0), node(0, 1)), (node(5, 5), island), (node(0, 0), node(11, 11))]
    is_src = np.array([False] * (copper + 2) + [True])
    tree, comp, up, upb, ups, depth = spanning_tree(n * n + 1, ends, is_src, copper)
    assert tree.sum() == n * n and not tree[copper] and tree[copper + 1] and not tree[copper + 2]
    assert len(set(comp)) == 1 and depth.min() == 0
    for v in range(n * n + 1):                                      # each node's parent: the branch to it, its sense
        if up[v] >= 0:
            a, b = ends[upb[v]]
            assert (a, b) == ((up[v], v) if ups[v] == 1 else (v, up[v]))


def test_open_boundary_fft_grid_matches_biot_savart(solved):
    q, r = solved
    # Observation window covers only a small part of the copper: deposition
    # must still retain the rest of the trace and its return plane.
    x, y = np.linspace(.012, .018, 25), np.linspace(.007, .013, 25)
    heights = [.0025, -.002]
    H = q.h_grid(r, [0, 1], x, y, heights)
    X, Y, Z = np.meshgrid(x[::6], y[::6], heights, indexing='ij')
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    ref = q.h_field(r, [0, 1], pts).reshape(5, 5, 2, 3, 2)
    assert abs(H[::6, ::6] - ref).max() < .01 * abs(ref).max()
    assert q.h_grid(r, 1, x, y, heights) == pytest.approx(H[..., 1])
    refined, info = q.h_grid(r, [0, 1], x, y, heights, rtol=.01, return_info=True)
    assert info['converged'] and info['refine'] >= 2
    assert abs(refined[::6, ::6]-ref).max() < abs(H[::6, ::6]-ref).max()
    with pytest.raises(RuntimeError, match='did not converge'):
        q.h_grid(r, 1, x, y, heights, rtol=.01, max_refine=1)


def test_spectral_near_field_h_is_biot_savart():
    """Frozen independent RF spectral reference; no RF package is needed."""
    from pathlib import Path
    from pcbqs._numerics.base import RWG
    from pcbqs.biot import rwg_currents, h_sheets
    with np.load(Path(__file__).parent/'fixtures/magnetic-reference.npz') as data:
        nodes,tris,I,pts,H = [data[k] for k in ['nodes','tris','I','pts','H']]
    a,b = rwg_currents(RWG(nodes,tris),I)
    Hb = h_sheets([(nodes,tris,a,b,.5e-3),(nodes,tris,-a,-b,-.5e-3)],pts)
    for h,hb in zip(H,Hb):
        assert np.abs(h-hb).max() < .03*np.abs(hb).max()+1e-3*np.abs(Hb).max()


def test_loop_space_impedance_equals_branch_voltage_kvl():
    """Z from the loop equations equals Kirchhoff's voltage law over branch drops, with two coupled sources,
    parts in the loops and finite-thickness sheet loss."""
    poly = mapping(box(0, 0, 8, 4))
    board = dict(stack=dict(layers=[], copper=[dict(name='F', z=0, t=.035), dict(name='B', z=1.0, t=.035)]),
                 copper={'F': [dict(net='N', polygon=poly)], 'B': [dict(net='N', polygon=poly)]},
                 vias=[dict(net='N', x=4.0, y=3.0, r=0.15, z0=0, z1=1.0)])
    terms = {'a': [('F', box(0, 1, 1, 2))], 'b': [('B', box(0, 1, 1, 2))],
             'c': [('F', box(7, 1, 8, 2))], 'd': [('B', box(7, 1, 8, 2))]}
    q = MQS(board, 'N', terminals=terms, dmax=1.0, log=lambda *_: None, conductor_loss='slab')
    q.source('I1', 'a', 'b'); q.source('I2', 'c', 'd')
    q.element('C1', 'a', 'c', C=1e-6, ESR=3e-3, ESL=.4e-9)
    freqs, amps = [1e4, 1e7], {'I1': 1.0, 'I2': .3j}
    r = q.solve(freqs, amps)
    m = q._mats
    ne, nv = q.ne, q.nv
    for i, f in enumerate(freqs):
        w = 2*np.pi*f
        zp = np.zeros(m['nb'], complex)
        zp[-1] = 3e-3 + 1j*w*.4e-9 + 1/(1j*w*1e-6)
        v = (m['R'] + q._resistance_delta(f)) @ r.i[i] + 1j*w*q._inductive_voltage(r.i[i]) + zp*r.i[i]
        copper = np.r_[np.ones(ne + nv), np.zeros(m['nb'] - ne - nv)]
        for k, c in m['src'].items():
            name = q.parts[k - ne - nv]['name']
            column = m['C'][:, c].toarray().ravel()
            assert r.z[name][i] == pytest.approx((column @ v - m['C'][k, c]*v[k]) / amps[name], rel=1e-10)
            assert r.zc[name][i] == pytest.approx((column*copper) @ v / amps[name], rel=1e-10)


def _strip_model(**kw):
    from shapely.geometry import box, mapping
    board = dict(stack=dict(layers=[], copper=[dict(name='F', z=0, t=.035)]),
                 copper={'F': [dict(net='N', polygon=mapping(box(0, 0, 5, 2)))]}, vias=[])
    terms = {'a': [('F', box(0, 0, 1, 2))], 'b': [('F', box(4, 0, 5, 2))]}
    q = MQS(board, 'N', terminals=terms, dmax=.7, conductor_loss='slab', log=lambda *_: None, **kw)
    q.source('I', 'a', 'b', part_h=0)
    return q


def test_conductor_faces_is_the_one_sheet_slab_s():
    """conductor_faces=1: the one-sheet slab, its current on one face (it had been ignored for the face sheets,
    so 1 and 2 gave the same Z and every comparison of the two computed one model twice); with face sheets asked
    for too: refused."""
    one, two = _strip_model(conductor_faces=1), _strip_model()
    assert one.thickness_sheets == 1 and two.thickness_sheets == 2
    r1, r2 = (q.solve([30e6]).z['I'].real[0] for q in (one, two))
    assert r1 > 1.5 * r2                                            # one face carries it all: more resistance
    with pytest.raises(ValueError, match="one-sheet slab"):
        _strip_model(conductor_faces=1, thickness_sheets=2)


def test_shared_kernels_keep_the_faces_skin_loss():
    """A model that takes another's kernels (share_kernels) gives its answer: copying them one by one had left the
    face sheets' terms behind (R at 30 MHz 0.898 mOhm for 1.195). Another mesh: refused."""
    q = _strip_model()
    z = q.solve([1e3, 30e6]).z['I']
    other = _strip_model().share_kernels(q)
    assert np.allclose(other.solve([1e3, 30e6]).z['I'], z, rtol=1e-9)
    with pytest.raises(ValueError, match='another mesh'):
        _strip_model(conductor_faces=1).share_kernels(q)


def test_a_long_sweep_by_one_eigendecomposition_is_the_lu_s(monkeypatch):
    """R-D13.2: the dense sweep with the copper's DC resistance and a capacitor as a part, by Pencil and by an LU a
    frequency, to round-off."""
    out = {}
    for mode in ('never', 'always'):
        monkeypatch.setenv('PCBQS_PENCIL', mode)
        q = MQS(trace_board(), 'N', terminals={'s': [('F.Cu', box(2.0, 9.6, 2.6, 10.4))],
                                               'g': [('F.Cu', box(2.0, 7.7, 2.6, 8.5))]}, dmax=1.0, log=lambda *a: None)
        q.source('I', 's', 'g', part_h=0.0)
        q.element('C1', 's', 'g', C=1e-6, ESR=0.01, ESL=1e-9, part_h=0.0)
        r = q.solve(np.geomspace(1e3, 1e8, 9))
        assert {i['method'] for i in q.solve_info} == {'pencil' if mode == 'always' else 'direct'}
        out[mode] = r
    a, b = out['never'], out['always']
    assert np.max(np.abs(a.i - b.i)) < 1e-10 * np.max(np.abs(a.i))
    assert np.max(np.abs(a.zc['I'] - b.zc['I'])) < 1e-10 * np.max(np.abs(a.zc['I']))
