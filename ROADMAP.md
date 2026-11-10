# pcbqs roadmap: power tools

Covers the converter near field (`pcbqs.mqs`), rail PDN (`pcbqs.pdn`), switching
circuits (`pcbqs.power_circuit`) and their kicad2ems window pages. Findings
referenced as C/I/U numbers are in [docs/power-tools-review.md](docs/power-tools-review.md).
Estimates are working days. Last updated 2026-11-10 (C13, measurement plan, R/L/C face model, PDN C9/C10/I6, I5, raw solver steps, port extraction speed, overlapping vias).

## Done recently

- Thick copper validated against FastHenry and a 2D cross-section PEEC: a terminal eddy-current bug fixed
  (`ideal_terminals`), thick copper placed with the dielectric gaps kept (`copper_centres`), finite-thickness kernels
  (`thick_kernel`), stacked sheets as the reference; `conductor_loss='slab'` now defaults to two face sheets.
- PDN/DC meshing: vias drilled into one another are meshed as one ring (`pdn.mesh.overlapping_vias`, reported in
  the notes), and the quality mesher has a Steiner-point cap with a warning. wpthb PDN at dmax 0.4: 28.5 s / 4.3 GB
  -> 3.9 s / 0.24 GB with the same drop and loss (two overlapping ground vias had made 4.4 million triangles).
  DC links are assembled as arrays and a rail with no load skips its solve.
- Real-board regressions (`tests/test_regression_boards.py`): CM5 U702's near field and wpthb's DC drop and PDN at
  coarse settings against stored results (`tests/data/regression_boards.json`; `--update` after a deliberate
  change), skipped where the board is not installed. About 37 s with the board cache.
- Ripple reported twice (I1): `bus_ripple_V` is now the settled ripple (the final cycle sampled at T/256, as the
  saved waveforms are) and `bus_spike_pp_V` the peak to peak on the solver's own steps, edge spikes included; the
  window's Circuit table shows both. Results from before this report the spike value as `bus_ripple_V`.
- Shared board import (I2): `pcbqs._cli.load_board` caches each KiCad import by content hash and net subset in
  `~/.cache/pcbqs/boards` (`PCBQS_CACHE_DIR`, `PCBQS_NO_CACHE=1`), so the three tools and the window import a board
  once (Jetson +1V15: 11.8 s, then 0.5 s).
- PDN: series filter inductors keep L and DCR (`--through`, `--dcr`); CA/CE/CP
  capacitors are decaps with electrolytic defaults (`--cap`); `--plane-min`.
- Circuits: relaxed-tolerance phase search with warm starts and a strict final
  run (wpthb per case 478 → 56 s, results within 0.27%); gate pulses start in
  their steady-state position; robust abstol and convergence retries;
  `tools/benchmark_circuit_solver.py`, `tools/benchmark_circuit_options.py`.
- Window: Circuit results show input power, VHB spike and run time; the PDN
  summary shows series inductance; waveform views tolerate boards without L3/R9.
- Mesh simplification (lone pads, regular pads/holes, polygonal vias) and the
  coaxial via-segment mutual fix; native H-matrix construction and products.
- Code cleanup: dead code removed; the cavity physics, the CLI options/board
  loader (`pcbqs._cli`) and the window's tool processes (`toolprocess.py`)
  shared instead of duplicated; `power_*` modules reformatted (ruff, 120
  columns, quotes preserved).

## Next: correctness and reporting (≈ 4 days)

1. **Template-independent bridge model (C7, U1; 1 day).** Precharge from a DC
   operating point of the actual input network instead of the template's part
   names and drop resistances; move wpthb specifics into the template data.
2. **Settling that sees slow filter modes (C8; 1 day).** Track cycle-mean drift
   of supply current; average input power over whole slow-mode periods, or
   extrapolate the decaying mode (vector extrapolation of the per-cycle state)
   instead of simulating it out.
3. ~~Ripple reported twice (I1)~~: done, and the window plots the solver's own steps on request (Solver steps:
   `steps.*` in transient.npz, the plotted signals' final two periods before ngspice's linearize).
4. ~~PDN loads and targets (C9, C10, I6)~~, done 2026-11-10: `--load REF=A@RETURN` (a load returning through
   another part's ground pads: its ground rise modelled), `--load-target REF=OHMS` and `--no-target`, and the
   impedance plot's axis set by the loaded curves (the bare-planes curve runs off the top; the window too).
5. ~~Real-board regressions~~: done for CM5 (near field) and wpthb (PDN); a wpthb circuit case (56 s) is
   still open.

## Then: workflow (≈ 5 days)

6. **Circuit page from `--describe` (I4, U2, U5; 1.5 days).** Generate the form
   from the tool, as Power Integrity and Near field do; per-group parasitic
   overrides; default to the template's switch model and warn on `switch` with ngspice.
7. ~~Shared board import (I2)~~: done (see above).
8. ~~Converter candidates (I5)~~, done 2026-11-10: the Near field page's Find converters lists each converter and
   input net (`pcbqs.mqs --list-converters`) and fills converter, input and capacitors from the pick; a run
   that fails on an ambiguous net points to it.
9. **Cross-tool flows (U4, I3; 2.5 days).** Near field converter → Circuit
   template (parts, loop copper); Circuit input-current spectrum → PDN loads;
   compare `pcbqs.mqs` assumed-pulse fields with `power_circuit --fields`.

## Simulation speed (measured, see power-circuit.md)

Today a 30 W wpthb case is ~56 s: about two strict-equivalent transients of a
369-equation circuit whose matrix refactorisation is 81% of ngspice time.

| Option | Expected gain | Effort | Notes |
|---|---|---|---|
| Copper-model order (41 modes kept at 3% fit tolerance) | 1.2–2× | 0.5 day | measure error against tolerance; fewer dense F/G couplings |
| Slow-mode extrapolation (item 2) | 1.5–3× on lightly damped filters | 1 day | as-built wpthb settles slowest |
| Xyce backend | robustness, not speed | 2 days | ran behavioural switches that ngspice could not; was 2–7× slower on the benchmark converter; harmonic balance could suit resonant tanks |
| Piecewise-linear switched state-space engine | 10–100× for ideal-switch studies | 1–2 weeks | the copper model and passives are linear; exact matrix exponentials per switch state; device-level effects need ngspice |
| GPU | none for one transient | — | 369 equations: launch latency dominates. Useful only for batched sweeps on the piecewise-linear engine |

## Validation and accuracy (next)

Done (`validation/`, 2026-11-09):
- DC drop against Elmer 3D (strip, vias, wpthb, Jetson; 0.2–1.5% on boards).
- Loop inductance and |H| against Palace and FastHenry (0.3–2%, ~5%).
- Plane pairs against Palace and the modal sum (≤ 0.8%).
- AC resistance against FastHenry (good to ~1 MHz, then 11–34% low: C14).
- Found and fixed C11 and C12.

In order:

1. **Thick copper and AC resistance (C14), in progress.** Done:
   - `MQS(thickness_sheets=2)`: face sheets coupled by the slab two-port, with the partial inductance's own
     between-the-faces share removed per edge pair.
   - The compressed backend without capacitance (`mqs_iterative.py`, same answers as dense, no dense loop
     matrices: 48k triangles in 1.7 GB).

   Against FastHenry the strip loop's 10/100 MHz resistance is within −1/+3.5% (one sheet: −9%). The trace over a
   plane is −12.5% (one sheet: −19/−30%). Next, about 1–2 weeks:
   - Side walls: vertical strips joining the faces along copper edges. These need vertical-panel kernels,
     because today's integrals assume horizontal triangles. That covers the plane case's remaining current
     and the thin-trace capacitance (+12% missing).
   - Charge on the face sheets (capacitance with thickness), then `dielectric='stack'` with faces. Done for the
     homogeneous dielectric (2026-11-10): each face copy carries its own charge (`charge_of`), and
     `side_charge=True` puts charged panels on the copper's sides (`side_charge.py`, dense). Against FasterCap's
     copper boxes in air: plates -0.3%, microstrip -2.5% (two faces alone -7.5%, one sheet against thick +17% low),
     two traces 0.1 mm apart self -5.5..-7.5%, mutual -10..-13% (faces alone -30%; the rest is the narrow-gap
     edge-charge resolution, as one sheet's against thin copper).
     `dielectric='stack'` with face sheets: the stack expanded by the copper's thickness
     (`DielectricStack.from_board_thick`: faces on interfaces; beside the copper air outside, resin (the mean of the
     two neighbours) inside). FR-4 microstrip: faces add 2.3% over one sheet (Hammerstad-Jensen's thickness
     correction: 3.3%); -5.1% against it at dmax 0.25 (one sheet -4.1% against the zero-thickness formula). Side
     charge is homogeneous-only (vertical panels are off the interfaces the images need).
   - AC solve with face sheets, done 2026-11-10 (one model for R, L and C): a node's copies (both faces, its side
     panels) hold one potential and share its charge freely. Dense: the node-level P_eff = (Bn^T Pc^-1 Bn)^-1
     (`peec.node_potential`). Compressed: the redistribution as extra unknowns held at equal potential, with the
     face pairs' 2x2 impedance blocks and the potential's near-field leaves in the preconditioners (1 kHz
     microstrip: 789 -> 52 iterations, fewer than one sheet's 65). Side charge on the compressed backend: the panel
     triangles join its compressed potential. Both backends agree to 1e-4; the 1 kHz AC capacitance equals
     capacitance_matrix's to 1e-6 (homogeneous and lossy stack). `conductor_loss='slab'` now defaults to two face
     sheets with capacitance too (the PEEC PDN trial included); the circuit tool keeps one. A terminal's inner
     edges carry no eddy current with capacitance now as well (`ideal_terminals`: the 20 mm strip's 100 MHz R was
     18% high with capacitance on).
   - Still open here: side charge in a layered dielectric, the narrow-gap edge-charge resolution, and the tree
     preconditioner (a fallback) with face sheets.
   - Circuit tool with two face sheets (in progress 2026-11-10): `mesh.matrix_backend: hierarchical` extracts the
     copper ports by GMRES on the compressed copper (`mqs_iterative.port_matrix`: all ports in lockstep, one
     compressed product per step for every port, warm-started across frequencies), no dense loop matrices.
     wpthb (24 ports, 11 frequencies): one sheet dense 69 s / 5.4 GB; one sheet compressed 545 s per-port GMRES ->
     151 s lockstep / 1.4 GB, results within 0.1% of dense; two faces dense 266 s / 25 GB, compressed per-port
     2334 s / 2.1 GB, lockstep 420 s / 4.0 GB (settled ripple -5.7%, spike +0.4%, efficiency and input power
     0.05%).
   - Port extraction speed (2026-11-10, `tools/benchmark_ports.py`, `tools/perf/`): `port_method` now picks a
     direct solve on dense loop matrices built from the compressed inductance when the loops are few
     (`loop_matrices`), and for two faces a two-level preconditioner (`mqs_precond.FaceLoops`: exact common-mode
     solve, sparse solve on the difference loops). wpthb extraction: one sheet 91.6 -> 11.6 s (2.5 GB); two faces
     342 -> 115 s (21,790 -> 4,171 iterations, 3.8 -> 5.5 GB), both equal to the old route within 1e-7. Whole
     circuit case: one sheet compressed 151 -> 72 s / 2.6 GB (dense 69 s / 5.4 GB); two faces 420 -> 185 s /
     5.7 GB, answers unchanged (ripple -5.6% against one sheet, efficiency 0.05%).
   - Circuit default: stays one sheet. Two faces now costs 2.6x the time for a 6% ripple change and nothing else
     above 0.4%; revisit with a measurement that isolates the copper (item 5's jumper variant, or a
     validation board); the powered ripple cannot resolve 6% until the parts are vendor models. `hierarchical` is now the
     circuit's default backend (same time as dense, half the memory); `mesh.matrix_backend: dense` restores the
     old route.
   - Next for two faces: iterations still grow with frequency (217 at DC, 575 at 20 MHz) and the dense
     common-mode setup is 19 s; a local loop basis (tile-wise projection) would address both.
   - The 2–3% inductance drop with face sheets.
   - Default: once the CM5 loop agrees, two face sheets plus the compressed backend for `conductor_loss='slab'`.

   Started (2026-11-09):
   - `panels3d.py`: the 1/R moments between triangles of any orientation (kept; the walls built on it were removed,
     below).
   - `face_correction='overlap'` made two faces converge on CM5 but was wrong near planes; removed (below).
   - Validation against FastHenry (`validation/face_study.py`, CM5-lite) found two faces wrong near a plane:
     +254% R at 100 MHz with a 35 um plane 0.1 mm under the trace (+0.1% with it 1.6 mm away). The slab two-port is
     driven by the faces' own currents; it has to be driven by the tangential field at the faces, from all currents,
     with the gap's vacuum field (from all currents, not only the pair's own) removed. Next: that formulation in the
     dense solver, then the compressed backend (a field-kernel H-matrix), checked against the reference below.
   - Reference mode: `thickness_sheets=N, conductor_loss='dc'` (graded stacked sheets, no slab formula), within
     5% / 1.3% (R / L) of FastHenry near a plane.
   - Copper placement (2026-11-10): thick copper (two faces, stacked sheets) is placed with its thickness put into
     the stack (`pdn.mesh.copper_centres`), so every dielectric keeps KiCad's face-to-face thickness; one sheet
     stays at the interfaces. Centring on z had shrunk a 0.1 mm dielectric to 0.065 mm: FastHenry's L at 100 MHz
     rose 36% with the fix, and all models now agree with it on L within a few percent.
   - Walls (copper's sides as rooftop panels) were built and removed (2026-11-10): +49% R at 10 MHz on an isolated
     strip, and unnecessary once the terminal eddy bug was fixed and close pairs used the thickness-averaged kernel.
     `panels3d.py` (1/R moments between triangles of any orientation, tested) is kept for vertical structures.
   - FIXED (2026-11-10), the real cause of the isolated-trace errors: a terminal's inner sheet edges (both sides
     in the terminal's one node) formed loops of their own, which carried eddy current that the field drove and the
     sheet resistance dissipated. An isolated 20 mm strip's two terminals took 15% of its R at 1 MHz. Now no loop
     (`ideal_terminals`, default; False for parity with the capacitance path). One sheet then converges to an
     independent 2D cross-section PEEC within 0.4 / 0.9 / 2% (1 / 10 / 100 MHz). The CM5 regression moved 0.4-1.2%.
   - `thick_kernel=True` (dense): close same-layer pairs through the 1/R kernel averaged over both slabs
     (`thick.py`); a bar's DC L to 0.01% of FastHenry. Isolated strip, current uniform through the thickness: within
     1-2% of the 2D PEEC at 1-100 MHz. Stacked sheets + thick kernel against FastHenry with full skin: -0.7% /
     -1.3% (10 / 100 MHz) on the strip; trace 0.1 mm over a plane: R +1/-3/-5%, L within 2.3%. With the plane 1.6
     mm away: converging with the mesh (-27 / -18 / -8.8% at 100 MHz as dmax 0.5 / 0.35 / 0.25). The correction is
     one sparse matrix added to either backend (`thick.delta_inductance`), so the compressed backend runs it.
   - CM5-lite (real copper, FastHenry 0.1 mm grid, correct placement), dmax 0.35: stacked 4 + thick kernel R
     -3.6/-5.9/-3.9%, L -0.4/+2.1/+2.8% (319 s); two faces without correction R -3.6/-6.8/+3.3%, L +0.5/+3.5/+3.9%
     (12 s); one sheet L +8-12%. At dmax 0.25: stacked R -8.1/-8.5/-6.4%, L -3.5/-2.0/-1.6% (655 s); two faces R
     -8.1/-10.0/+0.8%, L -2.7/-0.6/-0.4% (32 s). pcbqs R falls with refinement while FastHenry's rose 3-9% from its
     0.2 to 0.1 mm grid (its staircase of CM5's 0.2-0.3 mm features), so the residual is mostly the reference's; its
     0.05 mm grid (82k segments) did not get past the preconditioner in an hour even with -p diag.
   - Decided (2026-11-10): `conductor_loss='slab'` defaults to two face sheets (inductance only, homogeneous
     dielectric), with no face correction: the per-pair and overlap corrections are removed. Stacked sheets +
     `thick_kernel` stay the reference. The circuit tool's copper model (`power_bridge`) keeps one sheet: two faces
     moved wpthb's settled bus ripple 6% and nothing else beyond 0.03%, at 4.7x the time and 25 GB (copper ports need
     the dense loop matrices); `mesh.thickness_sheets: 2` in the circuit config opts in.
     One sheet + slab loss + thick kernel (placed in its real slab): strip -8 / +6%: the cheap option.
2. **Capacitance (0.5 day left).** Against FasterCap with zero-thickness copper, in air: plates < 0.1%,
   microstrip −2.4%, two traces 0.1 mm apart −4% self / −11% mutual (FasterCap ±3% there). On FR-4 the effective
   permittivity is within 1% of Hammerstad–Jensen; FasterCap gave non-physical FR-4 results, so FR-4 relies on the
   closed form. Copper thickness adds +12% to a 0.3 mm microstrip 0.2 mm over its plane, and pcbqs omits it.
   - Narrow gaps: edge-charge resolution within memory (compressed backend, or graded edge rings).
   - Thickness: side-wall and both-face charge once face sheets carry charge (with item 1).

3. **Real converter loop, in progress.** CM5 U702 (`validation/mqs_cm5.py`): pcbqs with one sheet is within 4%
   of Palace's PEC sheets at the finest meshes, both still converging. FastHenry (R, and L with thickness) is
   running; it decides between one and two face sheets.
4. **~~Parts on pads (C13)~~, done 2026-11-10 bar validation.** A two-pad chip part attaches where its end cap
   and solder fillet are: each pad from (body/2 − cap band) outward, sized from the footprint name
   (`pcbqs.contacts`, `pad_contact`: termination by default, pad, edge as a bound). This is in the converter loop,
   circuit and PEEC PDN tools. On KiCad footprints the contact is 83–95% of the pad, so whole pads were nearly
   right: CM5 loop L −2%, wpthb VHB port +0.5%, circuit settled ripple +2.9%. An inner-edge contact would read
   2.7× on wpthb's port, so the measurement in 5 checks the model's sensitivity directly.
5. **Measurements (plan ready).** `validation/measurements.md`: VNA shunt-through at TP7/TP6 unpowered (as built,
   and optionally one 1206 jumper in place of the ceramics), VHB ripple at the regression operating point, near
   field at 1–3 mm. Predictions: `validation/wpthb_vhb_port.py`.

## Models and validation

- Vendor or fitted MOSFET models (MCG8D2N06YL), datasheet capacitor ESR/ESL
  and measured coil Q; switch internal capacitances as explicit elements so a
  warm start can initialise them.
- Mesh/crop convergence studies per board; VNA PDN and near-field probe
  measurements to calibrate.

