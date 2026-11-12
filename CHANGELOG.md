# Changelog

Notable changes to pcbqs, by day, in the form of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), with
*Decided* for what was looked at and not done. It was started on 2026-11-12 from the finished items of the
roadmap; the git log has every commit, and kicad2ems's CHANGELOG.md has the time these tools lived in mom25d and
kicad2ems (before 2026-11-07). What is planned is in kicad2ems's
[ROADMAP.md](https://github.com/lowprocess/kicad2ems/blob/main/ROADMAP.md#power-tools-pcbqs). C, I and U numbers
are findings of [docs/power-tools-review.md](docs/power-tools-review.md).

## Unreleased

### Added
- **kicad2ems analysis plugins.** `pcbqs/analyses.py` registers power integrity, near field, circuit and thermal
  as `kicad2ems.analyses` entry points (kicad2ems API 2.5). Each gives its options (its tool's `--describe`), the
  command that runs it, and what a result holds to read and to plot (`pcbqs/reports.py`: a report of text and
  tables, curve sets), so the kicad2ems window builds its Study, its settings and its Results page from them and
  has no page per tool. Near field lists the board's converters and proposes its input rail to power integrity;
  thermal starts from the latest circuit run's losses.
- **Temperatures in the viewer.** `thermal/viewer.py` writes each layer's map as a field dump in °C (nothing off
  the board, the colours over the data's own range; with both solvers, their difference in K) and
  `thermal_board.json`, where kicad2ems's Viewer reads any run.
- **Circuit examples as configs.** The ten general circuits and the WPT bridge each have a TOML config beside the
  model and board (`[analysis] run`, `[analysis.circuit] model`), shipped through the plugin as Circuits > Passive
  and rectifiers, DC-DC converters, Bridges.
- **PDN loads from the parts' fields.** A part's `pdn.load` field is its current: `0.5`, `500m`, `2 A`;
  `NET=AMPS, ...` for a part on several rails; `AMPS@RETURN`. A rail then runs with no `--load`; `--load`
  overrides one part, `--no-part-loads` ignores the fields, and the result names the loads that came from them
  (`pdn.rail.field_loads`).

### Changed
- kicad2ems may be imported in `_cli.py` and `analyses.py` only, and nothing in pcbqs imports `analyses.py`
  (`tests/test_standalone.py`): the tools still run without kicad2ems.

### Decided
- **The copper model's order stays at a 3% fit tolerance.** Measured on wpthb at 90 kHz and 10 W: 3% keeps 39 of
  146 modes and takes 57 s where all of them take 123 s, with the settled ripple 1.4% off and everything else
  under 0.4%. At 6% (26 modes, 34 s) the ripple is 19% low, at 10% it is 73% low (docs/power-circuit.md).
- **No GPU for a circuit's transient.** 369 equations: the launch latency dominates. It would only pay for
  batched sweeps on a piecewise-linear engine.

## 2026-11-11

### Added
- **A general circuit engine** (`pcbqs.circuit`, model format 2) with ten example circuits, each with a layout;
  KiCad input; the board's copper between the parts' pads in the circuit; the H and E fields over the
  board while the circuit runs (`--fields`).
- **Thermal** (`pcbqs.thermal`): the board's steady temperature on a grid and on the triangle mesh, heat leaving
  by the faces and the edges (within 2 K of Elmer); its options as a table and `--describe --json`.
- **PDN:** the DC solve written where kicad2ems's viewer reads it (the board, the mesh, each layer's drop and
  current). `power_circuit --template simple`.

### Changed
- **The bridge's precharge goes by connectivity (C7).** The starting state was keyed on wpthb's references
  (C25/C27/C30, C19-C24, CA*, L3) and resistances, so another model started with empty capacitors and an inrush
  the power search could not settle. Parts are now set by what they join (the tank's capacitors, the rails'
  capacitors, the series inductors, the rails' drops). wpthb's own branches are kept: its deck is byte-identical.

### Fixed
- ngspice measures are read with every digit kept. A probe has to be on a node some part is on.

## 2026-11-10

### Added
- **PDN loads and targets (C9, C10, I6).** `--load REF=A@RETURN` (a load returning through another part's ground
  pads: its ground rise modelled), `--load-target REF=OHMS`, `--no-target`; the impedance plot's axis is set by
  the loaded curves (the bare planes run off the top).
- **Converter candidates (I5).** Find converters lists each converter and input net (`pcbqs.mqs
  --list-converters`) and fills converter, input and capacitors from the pick; a run that fails on an ambiguous
  net points to it.
- **Parts on pads (C13).** A two-pad chip part attaches where its end cap and solder fillet are, sized from the
  footprint name (`pcbqs.contacts`, `pad_contact`), in the converter loop, circuit and PEEC PDN tools. On KiCad
  footprints the contact is 83-95% of the pad: CM5 loop L -2%, wpthb VHB port +0.5%, settled ripple +2.9%.
- **Real-board regressions** (`tests/test_regression_boards.py`): CM5 U702's near field, wpthb's DC drop and PDN,
  and a wpthb circuit case (90 kHz, 10 W, with copper) against stored results; skipped where the board is not
  installed.
- **Shared board import (I2).** `pcbqs._cli.load_board` caches each KiCad import by content hash and net subset
  (`~/.cache/pcbqs/boards`, `PCBQS_CACHE_DIR`, `PCBQS_NO_CACHE=1`): Jetson +1V15 11.8 s, then 0.5 s.
- **Thick copper.** A finite-thickness kernel for close same-layer pairs (`thick_kernel`, `thick.py`: a bar's DC L
  to 0.01% of FastHenry; one sparse matrix added to either backend). A reference mode of graded stacked sheets
  (`thickness_sheets=N, conductor_loss='dc'`: within 5% / 1.3% in R / L of FastHenry near a plane).
- **Capacitance with thickness.** Each face copy carries its own charge (`charge_of`); `side_charge=True` puts
  charged panels on the copper's sides (`side_charge.py`). Against FasterCap's copper boxes in air: plates -0.3%,
  microstrip -2.5% (two faces alone -7.5%), two traces 0.1 mm apart self -5.5..-7.5%, mutual -10..-13%.
  `dielectric='stack'` with face sheets (`DielectricStack.from_board_thick`): FR-4 microstrip -5.1% against
  Hammerstad-Jensen with thickness at dmax 0.25.
- **One model for R, L and C.** The AC solve with face sheets: a node's copies hold one potential and share its
  charge (`peec.node_potential`; compressed: the redistribution as extra unknowns, 789 -> 52 iterations on a
  1 kHz microstrip). Both backends agree to 1e-4; the 1 kHz capacitance equals `capacitance_matrix`'s to 1e-6.
- **Copper ports on the compressed copper.** `mqs_iterative.port_matrix` (all ports in lockstep GMRES, warm
  started across frequencies); `port_method` picks a direct solve on dense loop matrices when the loops are few,
  and for two faces a two-level preconditioner (`mqs_precond.FaceLoops`). wpthb extraction: one sheet 91.6 ->
  11.6 s; two faces 342 -> 115 s; a whole circuit case 151 -> 72 s (one sheet), 420 -> 185 s (two faces).
- The window plots the solver's own steps on request (`steps.*` in transient.npz). The bench measurement plan
  (`validation/measurements.md`) and the validation board's structures (`validation/test-board.md`).

### Changed
- **Ripple is reported twice (I1).** `bus_ripple_V` is the settled ripple (the final cycle sampled at T/256) and
  `bus_spike_pp_V` the peak to peak on the solver's own steps. Earlier results report the spike as the ripple.
- **`hierarchical` is the circuit's default copper backend** (the same time as dense, half the memory);
  `mesh.matrix_backend: dense` restores the old route.
- **PDN and DC meshing.** Vias drilled into one another are meshed as one ring (`pdn.mesh.overlapping_vias`); the
  quality mesher has a Steiner-point cap; DC links are assembled as arrays; a rail with no load skips its solve.
  wpthb PDN at dmax 0.4: 28.5 s / 4.3 GB -> 3.9 s / 0.24 GB.

### Fixed
- **Eddy currents in terminals.** A terminal's inner sheet edges formed loops of their own, which the field drove
  and the sheet resistance dissipated: an isolated 20 mm strip's terminals took 15% of its R at 1 MHz. No loop
  now (`ideal_terminals`): one sheet converges to a 2D cross-section PEEC within 0.4 / 0.9 / 2% (1 / 10 / 100
  MHz). The CM5 regression moved 0.4-1.2%.
- **Thick copper keeps the dielectric gaps.** Its thickness goes into the stack (`pdn.mesh.copper_centres`), so
  every dielectric keeps KiCad's face-to-face thickness. Centring on z had shrunk a 0.1 mm dielectric to 0.065
  mm; FastHenry's L at 100 MHz rose 36% with the fix and all models now agree with it within a few percent.

### Decided
- **`conductor_loss='slab'` is two face sheets by default, with no face correction.** The per-pair and overlap
  corrections are removed (overlap was +254% in R at 100 MHz with a plane 0.1 mm under the trace). Stacked
  sheets with `thick_kernel` stay the reference. On CM5-lite at dmax 0.35 against FastHenry: stacked R
  -3.6/-5.9/-3.9%, L within 2.8% (319 s); two faces R -3.6/-6.8/+3.3%, L within 3.9% (12 s); one sheet L +8-12%.
- **The circuit tool keeps one sheet.** Two faces move wpthb's settled ripple by 6% and nothing else above 0.4%,
  at 2.6 times the time; `mesh.thickness_sheets: 2` opts in.
- **No side walls for the current.** Rooftop panels on the copper's sides gave +49% R at 10 MHz on an isolated
  strip and were not needed once the terminal loops were gone. `panels3d.py` (1/R moments between triangles of
  any orientation) is kept.
- **FastHenry finer than 0.1 mm is not a reference for CM5-lite.** Its 0.05 mm grid (82k segments) did not get
  past the preconditioner in an hour; its R rose 3-9% from 0.2 to 0.1 mm, so the residual is mostly its own.

## 2026-11-09

### Added
- **The validation suite** (`validation/`): DC drop against Elmer 3D (strip, vias, wpthb, Jetson: 0.2-1.5% on
  boards); loop inductance and |H| against Palace and FastHenry (0.3-2%, about 5%); plane pairs against Palace
  and the modal sum (0.8% at most); AC resistance against FastHenry (good to about 1 MHz, then 11-34% low: C14);
  capacitance against FasterCap with zero-thickness copper in air (plates under 0.1%, microstrip -2.4%, two
  traces 0.1 mm apart -4% self / -11% mutual), on FR-4 against Hammerstad-Jensen (1% in the effective
  permittivity). C11 and C12 were found by it and fixed.
- **Two face sheets** (`MQS(thickness_sheets=2)`), coupled by the slab two-port: the strip loop's 10 / 100 MHz
  resistance within -1 / +3.5% of FastHenry (one sheet -9%). The compressed backend without capacitance
  (`mqs_iterative.py`: the dense solve's answers, 48k triangles in 1.7 GB).
- **PDN:** series filter inductors keep their L and DCR (`--through`, `--dcr`); CA / CE / CP capacitors are decaps
  with electrolytic defaults (`--cap`); `--plane-min`.
- **Circuits:** a relaxed-tolerance phase search with warm starts and a strict final run (wpthb per case 478 ->
  56 s, results within 0.27%); gate pulses start in their steady-state position; a robust abstol and convergence
  retries; `tools/benchmark_circuit_solver.py`, `tools/benchmark_circuit_options.py`.
- The window's Circuit results show input power, the VHB spike and the run time; the PDN summary shows series
  inductance; waveform views tolerate boards without L3 / R9. The power tools review and the workflow guide.

### Changed
- Dead code removed; the cavity physics, the options and board loader (`pcbqs._cli`) and the window's tool
  processes shared instead of duplicated; the `power_*` modules reformatted.

### Fixed
- DC: pads are equipotential and the mesh is graded at vias (found against Elmer).

## 2026-11-08

### Changed
- Mesh simplification by default: lone pads dropped, regular pads and holes, polygonal vias (three times faster
  on the Jetson). H-matrices are built and applied in the compiled core.

### Fixed
- The mutual between segments of one via uses the barrel's radius.

## 2026-11-07

### Added
- **pcbqs, an independent package:** the rail tools (`pcbqs.pdn`), the converter near field (`pcbqs.mqs`) and the
  switching circuits (`pcbqs.power_circuit`), taken out of mom25d with their own copies of the numerics and their
  own native backend. No runtime dependency on mom25d. What it came with:
- **The quasi-static solver, faster.** Reciprocal near / far integration, refined close terms, a sparse
  resistance and loop projection, copper matrices reused: a 2,474-triangle sheet fills 3.1 times faster, the
  inductance within 0.005% of the refined analytic one. Resistance stays sparse, the dense branch L is stored
  once, PEEC preparation and bounded LU factors are reused, capacitance-only extraction skips the magnetic
  matrices: the demo's good layout takes 1.3 s for its planes and volume, 18.5 s for nine harmonics.
- **Fields on a grid by FFT.** `MQS.h_grid`: conservative current deposition and an open-boundary convolution,
  35 times faster than direct fields on a 974-triangle case at 0.38% peak error. Biot-Savart stays the reference
  for single points.
- **Matrix-fill backends:** OpenMP, a threaded NumPy fallback, a serial reference and an optional CuPy
  integrator, with bounded pair tiles and thread / device controls, shared by the magnetic and capacitance fills.
  The CM5 case: 66.8 -> 22.4 s. A fused GPU pair kernel fills the fine CM5 ground plane in 1.55 s (3.06 before),
  still behind eight OpenMP workers at 0.75 s (docs/static-backends.md).
- **The converter workflow.** `python -m pcbqs.mqs` reads KiCad or board JSON, describes its options as JSON,
  finds converter candidates, takes part, net, operating-point and scan overrides, and writes summaries, figures
  and H dumps; `--preview` shows the layer meshes without solving (docs/quasistatic.md).
- **Capacitance and E.** `MQS(capacitance=True, er=...)`: surface charge, displacement current and the Maxwell
  capacitance by a scaled loop / tree PEEC solve; `dielectric='stack'` for planar interfaces and dielectric loss;
  `conductor_loss='slab'` for skin impedance. `MQS.e_field`: the Coulomb E of the surface charges, layered images
  included.
- **Real boards.** CM5 U702's input loop as a benchmark (`tools/benchmark_converter.py`: 9,736 triangles on six
  layers, 85 via segments, 22.4 s, 1.10 GB; two meshes within 3.8% in L). Jetson AGX Thor: all eight regulators
  found, their cropped input loops solved (3,720-10,045 triangles on ten layers at 500 kHz).
