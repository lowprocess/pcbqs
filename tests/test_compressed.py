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
    q.solver_preconditioner='invalid'
    with pytest.raises(ValueError,match='solver_preconditioner'):q.solve([1e5])
    q.solver_preconditioner='auto';q.solver_restart=0
    with pytest.raises(ValueError,match='solver_restart'):q.solve([1e5])


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


@pytest.mark.parametrize('closed',[False,True])
def test_graph_preconditioner_avoids_large_projected_fill(monkeypatch,closed):
    import pcbqs.peec_iterative as module
    monkeypatch.setattr(module,'_PROJECTED_PRECONDITIONER_BYTES',0)
    q=make('hierarchical',layered=True,closed=closed)
    q.solver_restart=50
    dense=make('dense',layered=True,closed=closed)
    f=[1e-3,1e5,30e6]
    np.testing.assert_allclose(q.solve(f).z['I'],dense.solve(f).z['I'],rtol=1e-5,atol=1e-9)
    assert all(s['preconditioner']=='graph_ilu' for s in q.solve_info)


def test_serialized_operators_retain_complex_products():
    import pickle
    q=make('hierarchical',layered=True,closed=True);q._assemble();P=q._ensure_potential()
    rng=np.random.default_rng(81)
    for op in (P,q._copper[1],q._copper[3]):
        restored=pickle.loads(pickle.dumps(op,protocol=5))
        x=rng.normal(size=op.shape[1])+1j*rng.normal(size=op.shape[1])
        np.testing.assert_allclose(restored@x,op@x,rtol=1e-14,atol=0)


@pytest.mark.parametrize('near',[0.,3.,np.inf])
def test_batched_layer_images_match_dense_with_tiny_chunks(near):
    import copy
    from pcbqs.compressed import _LayeredPairs
    from pcbqs.electrostatic import potential_moments
    from pcbqs.static_backends import FillEngine
    q=copy.copy(make('hierarchical',layered=True))
    ids=np.r_[q.layer_tris['B'][:3],q.layer_tris['F'][:3]]
    q.tris=q.tris[ids];q.z_of=q.z_of[ids]
    q.layer_tris={'B':np.arange(3),'F':np.arange(3,6)}
    q.near=near;q._fill_options=dict(backend='auto',workers=1,chunk=7,device=None)
    expected=potential_moments(q)
    engine=FillEngine(q.nodes,q.tris,q.z_of,near,1,**q._fill_options)
    pairs=_LayeredPairs(q,engine)
    pairs.native=None      # the NumPy batching under test; `expected` uses the compiled kernel when built
    a,b=np.meshgrid(np.arange(6),np.arange(6),indexing='ij')
    actual=pairs(a.ravel(),b.ravel()).reshape(6,6)
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=0.)
    assert pairs.info['image_interactions']>0


def test_benchmark_checkpoints_reuse_kernels_and_invalidate_options(tmp_path):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('operator_cache',Path(__file__).parents[1]/'examples/peec-memory/operator_cache.py')
    cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)
    a=make('hierarchical',layered=True,closed=True)
    assert not cache.checkpoint(a,'inductance',tmp_path,a._assemble)['hit']
    assert not cache.checkpoint(a,'potential',tmp_path,a._ensure_potential)['hit']
    b=make('hierarchical',layered=True,closed=True)
    def fail():raise AssertionError('cached kernels rebuilt')
    assert cache.checkpoint(b,'inductance',tmp_path,fail)['hit']
    assert cache.checkpoint(b,'potential',tmp_path,fail)['hit']
    kernel=b._potential.kernel
    panels=np.arange(min(10,kernel.shape[0]))
    columns=np.eye(kernel.shape[0])[:,panels]
    np.testing.assert_allclose(cache.sampled_block(kernel,panels,panels),(kernel@columns)[panels],rtol=1e-14)
    np.testing.assert_allclose(b.solve([1e5]).z['I'],a.solve([1e5]).z['I'],rtol=1e-12)
    b.solver_tol=1e-9
    assert cache.fingerprint(a,'potential')==cache.fingerprint(b,'potential')
    b._compression_options=dict(b._compression_options,tol=1e-4)
    assert cache.fingerprint(a,'potential')!=cache.fingerprint(b,'potential')


@pytest.mark.parametrize('dtype',[float,complex])
def test_native_hmatrix_product_matches_python_and_is_thread_independent(dtype,monkeypatch):
    import pickle
    import pcbqs.hmatrix as hm
    if hm._native() is None:
        pytest.skip('compiled core without hmatrix_apply')
    rng=np.random.default_rng(7);pts=rng.random((512,3));pts[:,2]*=.01
    dense=(1-.02j if dtype is complex else 1)/np.sqrt(((pts[:,None]-pts[None,:])**2).sum(-1)+.005**2)
    H=HMatrix(lambda a,b:dense[np.ix_(a,b)],pts,pts,dtype=dtype,leaf=16,tol=1e-7)
    assert H.info['low_rank_blocks']>0
    inputs=[rng.normal(size=512),rng.normal(size=(512,3)),rng.normal(size=(512,2))+1j*rng.normal(size=(512,2))]
    monkeypatch.setattr(hm,'_native',lambda:None)
    python=[H@x for x in inputs]
    monkeypatch.undo()
    for x,expected in zip(inputs,python):
        results=[]
        for workers in (1,2,5):
            H.workers=workers;results.append(H@x)
        assert all(np.array_equal(results[0],r) for r in results[1:])
        assert results[0].dtype==expected.dtype and results[0].shape==expected.shape
        np.testing.assert_allclose(results[0],expected,rtol=1e-13,atol=1e-13*np.abs(expected).max())
    # The packed cache is not serialized, and blocks still describe the operator.
    restored=pickle.loads(pickle.dumps(H,protocol=5))
    assert not hasattr(restored,'_packed')
    assert np.array_equal(restored@inputs[2],H@inputs[2])
    monkeypatch.setenv('PCBQS_NO_CORE','1')
    np.testing.assert_allclose(H@inputs[1],python[1],rtol=1e-13,atol=1e-13*np.abs(python[1]).max())


def test_native_hmatrix_rejects_inconsistent_layout():
    import pcbqs.hmatrix as hm
    core=hm._native()
    if core is None:
        pytest.skip('compiled core without hmatrix_apply')
    rng=np.random.default_rng(3);pts=rng.random((64,3))
    H=HMatrix(lambda a,b:np.ones((len(a),len(b))),pts,pts,leaf=8)
    p=H._pack();x=np.ones((64,1));work=np.empty(p['nslots'])
    meta=p['meta'].copy();meta[0,1]+=10**6
    with pytest.raises(ValueError,match='layout'):
        core.hmatrix_apply(meta,p['idx'],p['data'],p['row_ptr'],p['slots'],x,work,p['nslots'],2)
    with pytest.raises(ValueError,match='scratch'):
        core.hmatrix_apply(p['meta'],p['idx'],p['data'],p['row_ptr'],p['slots'],x,work[:1],p['nslots'],2)


@pytest.mark.parametrize('near',[3.,np.inf])
def test_native_entry_blocks_match_numpy_assembly_bitwise(near,monkeypatch):
    import pcbqs.compressed as compressed
    from pcbqs._numerics.base import RWG,MU0
    q=make('hierarchical');q.near=near
    engine=compressed._engine(q)
    native=compressed._native(engine)
    if native is None:
        pytest.skip('compiled core without native entry blocks')
    core,common,_=native;g=engine.geometry;n=len(g.tris)
    rng=np.random.default_rng(5)
    rows=rng.choice(n,40,replace=False);cols=np.r_[rows[:5],rng.choice(n,60,replace=False)]
    p,qq=np.meshgrid(rows,cols,indexing='ij')
    for scalar in (False,True):
        expected=compressed._moments(engine,p.ravel(),qq.ravel(),scalar).reshape(len(rows),len(cols),-1)
        blocks=[core.static_block(*common,rows.astype(np.int64),cols.astype(np.int64),scalar,w) for w in (1,4)]
        assert np.array_equal(blocks[0],expected) and np.array_equal(blocks[1],expected)
    # Whole inductance operators agree exactly with the NumPy entry assembly
    # (both through the NumPy ACA; native ACA draws its own check rows).
    import pcbqs.hmatrix as hm
    monkeypatch.setattr(hm,'_build_core',lambda:None)
    L,_=compressed.edge_inductance(q)
    assert q.fill_info['magnetic']['native_entries']
    monkeypatch.setattr(compressed,'_native',lambda engine:None)
    reference,_=compressed.edge_inductance(q)
    assert not q.fill_info['magnetic']['native_entries']
    x=rng.normal(size=L.shape[0])
    assert np.array_equal(L@x,reference@x)
    monkeypatch.undo()
    assert not L.info['native_build']
    r=RWG(g.nodes,g.tris);support=np.stack((r.tp,r.tm),1).astype(np.int64)
    vertices=np.stack((r.vp,r.vm),1);alpha=np.stack((1/(2*r.area[r.tp]),-1/(2*r.area[r.tm])),1)
    with pytest.raises(ValueError,match='edge index'):
        core.rwg_inductance(*common,support,vertices,alpha,np.array([r.Ne]),np.array([0]),MU0/(4*np.pi),1)
    with pytest.raises(ValueError,match='triangle index'):
        core.static_block(*common,np.array([-1]),np.array([0]),True,1)


def test_entry_blocks_follow_selected_fill_backend():
    import pcbqs.compressed as compressed
    q=make('hierarchical');q._fill_options=dict(q._fill_options,backend='numpy')
    assert compressed._native(compressed._engine(q)) is None


@pytest.mark.parametrize('near',[0.,3.,np.inf])
@pytest.mark.parametrize('lossy',[False,True])
def test_native_layered_images_match_numpy_loops(near,lossy,monkeypatch):
    import pcbqs.compressed as compressed
    import pcbqs.electrostatic as electrostatic
    from pcbqs.static_backends import FillEngine
    q=make('hierarchical',layered=True);q.near=near
    if lossy:
        q._dielectric_stack.eps=q._dielectric_stack.eps*(1-.05j);q._dielectric_stack._cache.clear()
    engine=FillEngine(q.nodes,q.tris,q.z_of,near,q.outer_levels,**q._fill_options)
    native=compressed._LayeredPairs(q,engine)
    if native.native is None:
        pytest.skip('compiled core without layered_pairs')
    reference=compressed._LayeredPairs(q,engine);reference.native=None
    rng=np.random.default_rng(2);n=len(q.tris)
    p,qq=rng.integers(0,n,3000),rng.integers(0,n,3000);p[:50]=qq[:50]
    a,b=native(p,qq),reference(p,qq)
    np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12*np.abs(b).max())
    workers=native.native;native.native=(workers[0],workers[1],3)
    assert np.array_equal(native(p,qq),a)
    # The dense fill: compiled image kernel against the NumPy rows.
    M=electrostatic.potential_moments(q)
    assert q.fill_info['capacitance']['native_images']
    monkeypatch.setattr(electrostatic,'native_layered',lambda engine:None)
    R=electrostatic.potential_moments(q)
    np.testing.assert_allclose(M,R,rtol=1e-12,atol=1e-12*np.abs(R).max())


@pytest.mark.parametrize('layered',[False,True])
def test_native_aca_build_matches_numpy_build_within_tolerance(layered,monkeypatch):
    import pcbqs.compressed as compressed
    import pcbqs.hmatrix as hm
    if hm._build_core() is None:
        pytest.skip('compiled core without native ACA')
    q=make('hierarchical',layered=layered)
    L,_=compressed.edge_inductance(q);P=compressed.potential(q)
    assert L.info['native_build'] and P.kernel.info['native_build']
    rng=np.random.default_rng(4)
    x=rng.normal(size=L.shape[0]);y=rng.normal(size=P.shape[0])+1j*rng.normal(size=P.shape[0])
    # Thread count and scheduling do not change native factors.
    for workers in (1,3):
        q._fill_options=dict(q._fill_options,workers=workers)
        L2,_=compressed.edge_inductance(q);P2=compressed.potential(q)
        assert np.array_equal(L2@x,L@x) and np.array_equal(P2@y,P@y)
    monkeypatch.setattr(hm,'_build_core',lambda:None)
    Lr,_=compressed.edge_inductance(q);Pr=compressed.potential(q)
    assert not Lr.info['native_build']
    tol=q._compression_options.get('tol',1e-6)
    assert np.linalg.norm(L@x-Lr@x)<=50*tol*np.linalg.norm(Lr@x)
    assert np.linalg.norm(P@y-Pr@y)<=50*tol*np.linalg.norm(Pr@y)
