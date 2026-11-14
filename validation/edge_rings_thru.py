"""The thru with the charge solved: |S11| against the line of the 2D field solver's L' and C', with edge rings: the
CHANGELOG's 8.8 % to 0.1 % (R-C10).

    python validation/edge_rings_thru.py
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "tests"))
import time
import numpy as np
import pcbqs.mqs as M
from pcbqs import sparams as SP
import test_sparams as T
f = np.array([1e8, 3e8])
Lp, Cp, length, z0 = 487.4e-9, 68.3e-12, 29.5e-3, 50.0
zc, g = np.sqrt(Lp / Cp), 2j * np.pi * f * np.sqrt(Lp * Cp)
A, B, C = np.cosh(g * length), zc * np.sinh(g * length), np.sinh(g * length) / zc
line = np.abs((B / z0 - C * z0) / (2 * A + B / z0 + C * z0))
orig = M.MQS
for kw in ({}, dict(edge_rings=4), dict(edge_rings=6), dict(edge_rings=4, edge_width=0.5), dict(edge_rings=6, edge_width=0.5), dict(edge_rings=6, edge_width=0.25)):
    class Q(orig):
        def __init__(self, *a, **k):
            k.update(kw); super().__init__(*a, **k)
    M.MQS = Q
    t = time.perf_counter()
    r = SP.solve(T.thru(), f, capacitance='on', log=lambda *a: None)
    s11 = np.abs(r['S'][:, 0, 0])
    print(f"{str(kw):45s} {r['info']['triangles']:6d} tri {time.perf_counter() - t:5.1f} s  |S11| {s11[0]:.4f} {s11[1]:.4f}  line {line[0]:.4f} {line[1]:.4f}  ratio {s11[0] / line[0]:.3f} {s11[1] / line[1]:.3f}", flush=True)
M.MQS = orig
