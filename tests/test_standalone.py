"""Reference parity and independence from both application and RF packages."""
import ast,json,subprocess,sys
from pathlib import Path
import numpy as np
from shapely.geometry import box
from pcbqs.mqs import MQS
from pcbqs.mqs_ports import CopperPorts
from pcbqs._artifacts import Dump,write_npz,write_info
from pcbqs.mqs_cli import parser,run
from test_mqs import trace_board
from test_mqs_cli import board


def test_copper_snapshot_parity():
    q=MQS(trace_board(),'N',terminals={'s':[('F.Cu',box(2,9.6,2.6,10.4))],'g':[('F.Cu',box(2,7.7,2.6,8.5))]},dmax=2,log=lambda *_:None)
    q.source('p','s','g',part_h=0);r=CopperPorts(q).extract([0,1e5,2e5])
    with np.load(Path(__file__).parent/'fixtures/copper-reference.npz') as expected:
        np.testing.assert_array_equal(q.nodes,expected['nodes']);np.testing.assert_array_equal(q.tris,expected['tris'])
        np.testing.assert_allclose(r.impedance,expected['impedance'],rtol=1e-11,atol=1e-14)
        np.testing.assert_allclose(r.currents,expected['currents'],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(q.h_field(r.drive(np.ones((3,1))),1,np.array([[.015,.010,.002]])),expected['H'],rtol=1e-10)


def test_json_fields_without_optional_packages(tmp_path):
    b=tmp_path/'board.json';b.write_text(json.dumps(board()));out=tmp_path/'out'
    code='''import sys,importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] in ('mom25d','kicad2ems'):
            raise AssertionError('forbidden runtime dependency: '+fullname)
sys.meta_path.insert(0,Block())
from pcbqs.mqs_cli import main
main()
'''
    subprocess.run([sys.executable,'-c',code,str(b),'--out',str(out),'--harmonics','1,2','--heights','1,2','--scan-points','5','--dmax','1','--fine','1','--fill-backend','numpy'],check=True,capture_output=True,text=True)
    info=json.loads(next(out.glob('mqs/exc_*/dump_info.json')).read_text())
    assert len(info['dumps'])==3 and info['components']
    for file in out.glob('mqs/exc_*/*.npz'):
        with np.load(file) as data:
            assert {'x','y','z','data','freqs'}<=set(data.files)
            assert data['data'].shape[0]==2 and np.isfinite(data['data']).all()


def test_no_rf_or_application_dependency_in_core():
    root=Path(__file__).parents[1]/'src/pcbqs'
    for file in root.rglob('*.py'):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node,ast.ImportFrom) and node.module:
                assert not node.module.startswith('mom25d'),file
                if node.module.startswith('kicad2ems'):assert file.name=='cli.py' or file.name=='mqs_cli.py',file
            elif isinstance(node,ast.Import):
                assert not any(n.name.startswith('mom25d') for n in node.names),file


def test_automatic_fallback_without_native_core(tmp_path):
    file=tmp_path/'board.json';file.write_text(json.dumps(trace_board()))
    code='''import sys,importlib.abc,json,numpy as np
from shapely.geometry import box
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname=='pcbqs._core' or fullname.split('.')[0] in ('mom25d','kicad2ems'):
            raise ImportError('unavailable: '+fullname)
sys.meta_path.insert(0,Block())
from pcbqs.mqs import MQS
from pcbqs.mqs_ports import CopperPorts
q=MQS(json.load(open(sys.argv[1])),'N',terminals={'s':[('F.Cu',box(2,9.6,2.6,10.4))],'g':[('F.Cu',box(2,7.7,2.6,8.5))]},dmax=2,fill_workers=2,log=lambda *_:None)
q.source('p','s','g',part_h=0)
r=CopperPorts(q).extract([0,1e5,2e5])
with np.load(sys.argv[2]) as ref:np.testing.assert_allclose(r.impedance,ref['impedance'],rtol=1e-8,atol=1e-13)
assert q.fill_info['magnetic']['backend']=='threaded'
'''
    subprocess.run([sys.executable,'-c',code,str(file),str(Path(__file__).parent/'fixtures/copper-reference.npz')],check=True,capture_output=True,text=True)
