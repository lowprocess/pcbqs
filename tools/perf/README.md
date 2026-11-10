# Performance trials

Scripts from the 2026-11-10 performance investigation, kept as they were run. They are measurement scaffolding,
not part of the package: most monkeypatch pcbqs internals and hard-code the board paths (`~/Desktop/wpthb` and
KiCad's CM5 demo). Some patch functions that have since changed, so expect to adjust them before reuse. Run from
this directory, one at a time, with the kicad2ems venv's python and `OMP_NUM_THREADS=8`.

Maintained entry points:

| Script | What it measures |
|---|---|
| `../benchmark_ports.py SHEETS [auto\|direct\|iterative\|dense] [auto\|graph\|faces]` | Copper-port extraction for the wpthb circuit: time, peak memory, GMRES iterations per frequency, one JSON line. `--save`/`--compare` check one route against another. |
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
