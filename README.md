# pcbqs

PCB DC/power-integrity analysis, magneto-quasistatic and PEEC extraction, magnetic
near fields, and ngspice operational circuits with passive copper feedback.

```sh
pip install .
python -m pcbqs.pdn --describe --json
python -m pcbqs.mqs --describe --json
python -m pcbqs.power_circuit --template > model.json
python -m pcbqs.power_circuit model.json --out result --jobs 4
python -m pcbqs.circuit --template boost > boost.json
python -m pcbqs.circuit boost.json --out result
```

JSON-based solving, layout/mesh export and field reconstruction work without
`mom25d` or `kicad2ems`. Install `.[kicad2ems]` to import `.kicad_pcb` files.

With kicad2ems installed the four tools are its analyses (`pcbqs/analyses.py`, the
`kicad2ems.analyses` entry points): `kicad2ems --check` lists them, a config keeps a
study's under `[analysis]`, and kicad2ems-gui builds each one's settings from its
options table, runs it and shows what its result folder holds: the maps as field
dumps in its Viewer, the report and the curves (`pcbqs/reports.py`) on its Results
page. The circuit examples are configs beside their models
(`circuit/examples/*.toml`), under its File > Examples > Circuits.
Install ngspice separately for operational circuits. Component models and
parasitics are explicit assumptions; footprints do not imply calibrated models.

On Python versions without a sufficiently recent Triangle wheel (including
Python 3.14), use the kicad2ems installer: its QS option builds the pinned mesher
with the floating-point flags required by Triangle's exact predicates.

The optional CPU backend is built with `pip install '.[core]'` followed by
`python -m pcbqs.core.build`. NumPy/threaded fallbacks remain available. CuPy GPU
fill is optional: select a matching CUDA/ROCm extra and `--fill-backend cupy`.
Execution controls use `PCBQS_FILL_BACKEND`, `PCBQS_FILL_THREADS`,
`PCBQS_OMP_THREADS`, `PCBQS_NO_CORE` and `PCBQS_ARCH`; general OpenMP settings
remain supported. Static-fill plugins use the `pcbqs.static_backends` entry group.

- [Quasistatic guide](docs/quasistatic.md)
- [Circuits](docs/circuit.md): any circuit of parts, nodes and drives; ten examples, each with a board whose
  copper can be put in; KiCad boards, schematics and netlists read into a model
- [Temperature](docs/thermal.md): a trial of a lightweight thermal solver, on a grid and on the triangle mesh
- [Operational circuits](docs/power-circuit.md): the full bridge with the board's copper
- [Validation and limits](docs/quasistatic-validation.md)
- [Migration](docs/migration.md)
- [Copied-code provenance](PROVENANCE.json)
- [PEEC and skin-effect test PCB](examples/peec-skin/README.md): two coupons,
  model comparisons, independent references and a reproducible KiCad layout.
- [Experimental PEEC PDN comparison](examples/pdn-peec-trial/README.md):
  loaded rails, independent DC/AC references, and mesh sensitivity.
- [Jetson AGX Thor PDN check](examples/jetson-pdn/REPORT.md): a real-board DC
  cross-check, cavity limitations, and a bounded local PEEC/skin comparison.
- [PEEC memory reduction](examples/peec-memory/REPORT.md) and the
  [experimental compressed backend](docs/peec-memory.md): graded meshes,
  hierarchical current/charge operators and iterative solving.

Private `_numerics` functions are independently maintained snapshots, not a
shared library or a runtime link to MoM. `emsdev` supplies only existing hardware
and native-build services. RF/full-wave solvers remain in `mom25d`.
