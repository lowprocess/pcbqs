"""The trial adapter's small-signal ports and component termination."""
import numpy as np
import pytest
from shapely.geometry import box, mapping

from pcbqs.pdn.rail import Rail
from pcbqs.pdn.stack import RailPDN
from pcbqs.pdn.peec_trial import PEECPDN


def fixture():
    square=box(0,0,6,6)
    pads=[dict(name='1',net='P',layers=['F.Cu'],xy=[3,3],geom=mapping(square)),
          dict(name='2',net='GND',layers=['B.Cu'],xy=[3,3],geom=mapping(square))]
    fps=[dict(ref=ref,value='100n' if ref=='C1' else 'contact',props={'ESR':'5m','ESL':'0.5n'} if ref=='C1' else {},
              lib='',side='F',pads=pads) for ref in ('VRM','C1','U1','U2')]
    board=dict(outline=mapping(square),copper={'F.Cu':[dict(net='P',polygon=mapping(square))],
        'B.Cu':[dict(net='GND',polygon=mapping(square))]},vias=[],prims=dict(footprints=fps),
        stack=dict(copper=[dict(name='B.Cu',z=0.,t=.07),dict(name='F.Cu',z=.3,t=.07)],
                   layers=[dict(z0=0.,z1=.3,er=4.,tand=.002)],thickness=.3))
    rail=Rail(board,'P',source='VRM',ground_source='VRM',loads={'U1':1.,'U2':3.})
    return RailPDN(rail,dmax=2.,vrm=dict(R=.002,L=20e-9),log=lambda *_:None)


def test_trial_parallel_termination_and_separate_load_excitation():
    legacy=fixture();trial=PEECPDN(legacy,dmax=2.,fill_workers=1,log=lambda *_:None)
    f=np.array([1e3,1e6,10e6]);answer=trial.z(f)
    C,_,terms=trial.q.capacitance_matrix()
    d=np.zeros(len(C));d[terms['U1:power']]=1;d[terms['U1:ground']]=-1
    cf=1/(d@np.linalg.solve(C,d))
    w=2*np.pi*f
    expected=1/(1/(.002+1j*w*20e-9)+1/(.005+1j*w*.5e-9+1/(1j*w*100e-9))+1j*w*cf)
    # Both loads have identical ideal plate contacts. Each measurement drives
    # exactly 1 A, regardless of the configured 1 A / 3 A operating DC loads.
    np.testing.assert_allclose(answer['U1'],expected,rtol=2e-8,atol=1e-12)
    np.testing.assert_allclose(answer['U2'],expected,rtol=2e-8,atol=1e-12)
    bare=trial.z(f,observe=['U2'],bare=True)
    assert list(bare)==['U2']
    np.testing.assert_allclose(bare['U2'],1/(1j*w*cf),rtol=2e-8)
    np.testing.assert_allclose(trial.z(f,observe=['U1'])['U1'],expected,rtol=2e-8)
    reverse=PEECPDN(legacy,dmax=2.,fill_workers=1,log=lambda *_:None)
    np.testing.assert_allclose(reverse.z(f,bare=True)['U1'],bare['U2'],rtol=2e-8)
    np.testing.assert_allclose(reverse.z(f)['U1'],expected,rtol=2e-8)


def test_trial_rejects_unsupported_rails_and_bad_queries():
    from types import SimpleNamespace
    legacy=fixture();legacy.rail.series=[SimpleNamespace(kind='bead')]
    with pytest.raises(NotImplementedError,match='beads'):
        PEECPDN(legacy)
    legacy=fixture();trial=PEECPDN(legacy,dmax=2.,fill_workers=1,log=lambda *_:None)
    for frequencies in ([],[0],[-1],[float('nan')]):
        with pytest.raises(ValueError,match='frequencies'):
            trial.z(frequencies)
    with pytest.raises(ValueError,match='configured rail loads'):
        trial.z([1e6],observe=['missing'])
    with pytest.raises(ValueError,match='contact_spacing'):
        PEECPDN(legacy,contact_spacing=0)


def test_series_sense_resistor_is_retained_at_dc():
    from pcbqs.pdn.rail import Pad,Part,Series
    legacy=fixture();rail=legacy.rail
    # Split the equipotential top plate into two disconnected islands; ideal
    # whole-island contacts isolate the known lumped resistance from spreading.
    a,b=box(0,0,2,6),box(3,0,6,6)
    rail.board['copper']['F.Cu']=[dict(net='P0',polygon=mapping(a)),dict(net='P',polygon=mapping(b))]
    def pad(net,geom):return Pad('R1','1',net,['F.Cu'],geom,tuple(geom.centroid.coords[0]))
    link=Part('R1','10m',{},'','F',[pad('P0',a),pad('P',b)])
    rail.nets={'P0','P'};rail.series=[Series(link,'P0','P','R',.01)]
    rail.source_pads=[pad('P0',a)]
    for name,info in legacy.parts.items():
        if info['role']=='source':continue
        info['part'].pads=[pad('P',b),info['part'].pads[1]]
    legacy.series=[dict(s=rail.series[0],esl=.4e-9)]
    trial=PEECPDN(legacy,dmax=2.,fill_workers=1,log=lambda *_:None)
    z=trial.z([1.])['U1'][0]
    assert z.real==pytest.approx(.012,rel=1e-7)
    with pytest.raises(RuntimeError,match='dense-solver limit'):
        PEECPDN(legacy,dmax=2.,max_triangles=1,fill_workers=1,log=lambda *_:None)
