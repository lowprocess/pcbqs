"""Compressed operators and PEEC checked against the dense reference."""
import numpy as np
import pytest
from scipy.sparse.linalg import LinearOperator
from shapely.geometry import box,mapping
from pcbqs.hmatrix import HMatrix
from pcbqs.mqs import MQS


def test_hmatrix_reciprocal_complex_kernel_and_storage():
    rng=np.random.default_rng(7);pts=rng.random((512,3));pts[:,2]*=.01
    dense=(1-.02j)/np.sqrt(((pts[:,None]-pts[None,:])**2).sum(-1)+.005**2)
    H=HMatrix(lambda a,b:dense[np.ix_(a,b)],pts,pts,dtype=complex,leaf=16,tol=1e-7)
    x=rng.normal(size=(512,3))+1j*rng.normal(size=(512,3))
    np.testing.assert_allclose(H@x,dense@x,rtol=1e-6,atol=1e-5)
    assert H.info['low_rank_blocks']>0
    assert H.storage_bytes<dense.nbytes
    np.testing.assert_allclose(H.diagonal,np.diag(dense))
    np.testing.assert_allclose(H.H@x,dense.conj().T@x,rtol=1e-6,atol=1e-5)


def make(backend,layered=False,closed=False,whole=False):
    g=box(0,0,6,4)
    b=dict(stack=dict(copper=[dict(name='B',z=0,t=.035),dict(name='F',z=.3,t=.035)],
                      layers=[dict(z0=0,z1=.3,er=4.,tand=.02)]),
           copper={l:[dict(net=l,polygon=mapping(g))] for l in ('B','F')},vias=[])
    terms={'a':[('F',g if whole else box(0,0,1,1))],
           'b':[('B',g if whole else box(5,3,6,4))]}
    q=MQS(b,{'B','F'},terminals=terms,dmax=1.2,capacitance=True,matrix_backend=backend,
          dielectric='stack' if layered else 'homogeneous',conductor_loss='slab',conductor_faces=1,
          compression_leaf=16,compression_tol=1e-8,solver_tol=1e-10,solver_maxiter=2000,
          fill_workers=1,log=lambda *_:None)
    q.source('I','a','b',part_h=0)
    if closed:q.element('C','a','b',C=100e-9,ESR=.005,ESL=.5e-9,part_h=0)
    return q


@pytest.mark.parametrize('layered,closed,whole',[(False,False,False),(True,False,True),(True,True,False)])
def test_compressed_peec_matches_dense_and_continuity(layered,closed,whole):
    dense=make('dense',layered,closed,whole);q=make('hierarchical',layered,closed,whole)
    f=[1e-3,1e5,30e6];a=dense.solve(f);r=q.solve(f)
    np.testing.assert_allclose(r.z['I'],a.z['I'],rtol=1e-5,atol=1e-9)
    np.testing.assert_allclose(r.i,a.i,rtol=2e-4,atol=2e-8)
    np.testing.assert_allclose(r.charge,a.charge,rtol=2e-4,atol=1e-16)
    for k,freq in enumerate(f):
        residual=q._peec['A']@r.i[k]+2j*np.pi*freq*r.charge[k]
        assert np.max(abs(residual))<1e-8
        np.testing.assert_allclose(r.potential[k],q._potential@r.charge[k],rtol=1e-7,atol=1e-7)
    assert isinstance(q._potential,LinearOperator)
    assert isinstance(q._copper[1],HMatrix)
    assert not any(key in q._peec for key in ('PT','electro','L','R'))
    assert max(info['relative_residual'] for info in q.solve_info)<1e-8
    np.testing.assert_allclose(q.solve([1e5],amps={'I':2j}).z['I'],r.z['I'][1:2],rtol=1e-7)


def test_nonconvergence_is_reported_and_capacitance_is_available():
    q=make('hierarchical')
    C,_,_=q.capacitance_matrix()
    Cd,_,_=make('dense').capacitance_matrix()
    np.testing.assert_allclose(C,Cd,rtol=1e-5,atol=1e-18)
    q.solver_maxiter=1;q.solver_tol=1e-14
    with pytest.raises(RuntimeError,match='GMRES failed'):q.solve([30e6])


def test_grading_preserves_terminals_and_reduces_bulk():
    from pcbqs.pdn.mesh import mesh_sheet
    g=box(0,0,20,12);pads=[box(1,5,2,6),box(17,5,18,6)]
    n0,t0,_,_=mesh_sheet(g,dmax=1.,regions=pads)
    nd,tr,rings,regs=mesh_sheet(g,dmax=1.,regions=pads,coarsen=4,edge_rings=4,w_ref=.2,edge_zone=box(0,4,3,7))
    assert len(tr)<len(t0)
    assert all(len(ids)>0 for ids in regs)
    for kw in [dict(coarsen=.5),dict(grow=0),dict(edge_rings=0),dict(w_ref=-1)]:
        with pytest.raises(ValueError):mesh_sheet(g,**kw)


def test_via_operator_matches_dense_without_triangle_square_fill(monkeypatch):
    import pcbqs.mqs as module
    q=make('hierarchical',layered=True,closed=True)
    board=q.board
    board['copper']['B']=[dict(net='B',polygon=mapping(box(0,0,8,4)))]
    board['copper']['F'].append(dict(net='B',polygon=mapping(box(7,0,8,4))))
    board['vias']=[dict(net='B',x=7.5,y=y,r=.15,z0=0,z1=.3) for y in (1.,3.)]
    answers=[]
    for backend in ('dense','hierarchical'):
        m=MQS(board,{'B','F'},terminals=q.terms,dmax=1.2,capacitance=True,matrix_backend=backend,
              dielectric='stack',compression_leaf=16,compression_tol=1e-8,solver_tol=1e-10,
              solver_maxiter=2000,fill_workers=1,log=lambda *_:None)
        m.source('I','a','b',part_h=0);m.element('C','a','b',C=100e-9,ESR=.005,ESL=.5e-9,part_h=0)
        if backend=='hierarchical':
            def fail(*a,**kw):raise AssertionError('dense assembly was used')
            monkeypatch.setattr(module,'static_moments',fail)
            monkeypatch.setattr(m,'_project_inductance',fail)
        answers.append(m.solve([1e5,30e6]).z['I'])
        assert m.nv==2
        if backend=='hierarchical':assert isinstance(m._copper[3],HMatrix)
    np.testing.assert_allclose(answers[1],answers[0],rtol=2e-5,atol=1e-9)


def test_graph_preconditioner_avoids_large_projected_fill(monkeypatch):
    import pcbqs.peec_iterative as module
    monkeypatch.setattr(module,'_PROJECTED_PRECONDITIONER_BYTES',0)
    q=make('hierarchical',layered=True,closed=True)
    dense=make('dense',layered=True,closed=True)
    f=[1e-3,1e5,30e6]
    np.testing.assert_allclose(q.solve(f).z['I'],dense.solve(f).z['I'],rtol=1e-5,atol=1e-9)
    assert all(s['preconditioner']=='tree_diagonal' for s in q.solve_info)
