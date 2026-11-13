# Performance trials

Scripts from the 2026-11-10 performance investigation, kept as they were run. They are measurement scaffolding,
not part of the package: most monkeypatch pcbqs internals and hard-code the board paths (`~/Desktop/wpthb` and
KiCad's CM5 demo). Some patch functions that have since changed, so expect to adjust them before reuse. Run from
this directory, one at a time, with the kicad2ems venv's python and `OMP_NUM_THREADS=8`.

Maintained entry points:

| Script | What it measures |
|---|---|
| `../benchmark_ports.py SHEETS [auto\|direct\|iterative\|dense] [auto\|graph\|faces]` | Copper-port extraction for the wpthb circuit: time, peak memory, GMRES iterations per frequency, one JSON line. `--save`/`--compare` check one route against another. |
| `workloads.py CASE` | cProfile of one workload of a tool (S-parameters three ways, the capacitance matrix, the near field, two rails, the temperature, the copper ports, a circuit on copper): wall time, peak memory, and the functions with the most time of their own. What it found on 2026-11-13 is below. |
| `circuit_timing.sh NAME SHEETS BACKEND` | The whole circuit regression case (90 kHz, 10 W, copper ports): wall time, peak memory, headline answers. |

One-off trials:

| Script | What it showed |
|---|---|
| `cli_prof.py mqs\|pdn` | cProfile of the near-field and PDN command lines. The PDN's time was all in the ground mesh. |
| `dcmesh.py` | Per-layer mesh sizes in the PDN: 4.4 million triangles around two vias drilled into one another. |
| `dcfix.py [fix]` | The PDN with one of those vias removed: the case for `pdn.mesh.overlapping_vias`. |
| `dcsolve.py [mmd] [default]` | Sparse orderings for the DC solve. Not worth changing once the mesh is sane. |
| `prof.py SHEETS` | Where lockstep port GMRES spends its time (compressed products, then the preconditioner). |
| `ports.py SHEETS MODE` | Lockstep GMRES against a dense loop matrix built from the compressed inductance: the case for the direct route (`mqs_iterative.loop_matrices`). |
| `twolevel.py`, `twolevel2.py` | Two-level preconditioners for two face sheets: the prototype of `mqs_precond.FaceLoops`. |

`results/` (not tracked) holds logs and saved port matrices from these runs.

## Where each tool's time goes (2026-11-13, `workloads.py`)

On an i7-13700K, before the kernels of `core/dense_kernels.hpp`. "Own Python" is time in pcbqs's own functions,
NumPy expressions included; the rest is the compiled core, LAPACK, SuperLU, shapely (GEOS) or ngspice.

| Case | Wall, peak | Where the time was |
|---|---|---|
| `sparams_rl` | 2.2 s | LU 0.7 s; the loop system's assembly 0.4 s |
| `sparams_charge` | 19.3 s, 3.9 GB | LU 8.0 s; **the system's assembly in `peec.solve` 8.0 s** |
| `sparams_faces` | 205 s, 25 GB | LU 162 s; **the loop system's assembly 26 s**, in four copies of the matrix |
| `capmatrix` | 18.5 s | **one symmetric indefinite solve (`sysv`) 12.9 s**, of a matrix that is positive definite; the layered fill 1.3 s |
| `mqs` | 7.6 s | shapely's unions 2.9 s; the fill 0.6 s |
| `pdn_cm5` | 28.6 s | **shapely `buffer` 7.8 s** (an island grown again for every via); SuperLU 6.8 s (320 frequencies) |
| `pdn_wpthb` | 9.8 s | SuperLU 2.2 s; pictures 1 s |
| `thermal` | 38.6 s | **`Result.sample` 22.2 s** (a Delaunay triangulation per call); SuperLU 5.7 s |
| `ports1` | 19.0 s | **the loop matrix's copies and symmetric solves 7.2 s**; the compressed product 5.8 s |
| `circuit_copper` | 118 s | waiting for ngspice 98 s; reading its text output 9.6 s; compressing results 6.7 s |

In bold: what was pcbqs's own, and is now in the core or gone (docs/static-backends.md has the times after:
`sparams_charge` 14 to 17 s, `sparams_faces` 147 s and 15 GB, `capmatrix` 8.9 s, `pdn_cm5` 21.7 s, `thermal`
15.7 s, `ports1` 12.6 s). Not taken up: the LU itself, which is LAPACK's and now 90 % of `sparams_faces` (the
systems are complex symmetric, but LAPACK's symmetric factorization was no faster than its LU here: 1.6 s for
1.2 s at 6,000 unknowns); ngspice's text output (`wrdata`, read with `np.loadtxt`: a binary raw file would
read much faster, and the piecewise-linear engine needs neither); the compression of field dumps; shapely's
unions in the near field's mesh.
