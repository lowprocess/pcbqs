# pcbqs roadmap: power tools

Covers the converter near field (`pcbqs.mqs`), rail PDN (`pcbqs.pdn`), switching
circuits (`pcbqs.power_circuit`) and their kicad2ems window pages. Findings
referenced as C/I/U numbers are in [docs/power-tools-review.md](docs/power-tools-review.md).
Estimates are working days. Last updated 2026-11-09 (evening: validation, C14 started, FasterCap).

## Done recently

- Thick copper validated against FastHenry and a 2D cross-section PEEC: a terminal eddy-current bug fixed
  (`ideal_terminals`), thick copper placed with the dielectric gaps kept (`copper_centres`), finite-thickness kernels
  (`thick_kernel`), stacked sheets as the reference; `conductor_loss='slab'` now defaults to two face sheets.
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
3. ~~Ripple reported twice (I1)~~: done (see above); still open, the window plotting raw solver points.
4. **PDN loads and targets (C9, C10, I6; 1 day).** `--load REF=A@RETURN` for loads
   whose return is another part (a bridge's low side); per-load targets or none;
   clip the bare-planes curve.
5. ~~Real-board regressions~~: done for CM5 (near field) and wpthb (PDN); a wpthb circuit case (56 s) is
   still open.

## Then: workflow (≈ 5 days)

6. **Circuit page from `--describe` (I4, U2, U5; 1.5 days).** Generate the form
   from the tool, as Power Integrity and Near field do; per-group parasitic
   overrides; default to the template's switch model and warn on `switch` with ngspice.
7. ~~Shared board import (I2)~~: done (see above).
8. **Converter candidates (I5; 0.5 day).** When the input net is ambiguous, offer
   the candidates in the Near field page instead of failing.
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
   - Charge on the face sheets (capacitance with thickness), then `dielectric='stack'` with faces.
   - The 2–3% inductance drop with face sheets.
   - Default: once the CM5 loop agrees, two face sheets plus the compressed backend for `conductor_loss='slab'`.

   Started (2026-11-09):
   - `panels3d.py`: the 1/R moments between triangles of any orientation, which side walls need. It reproduces the
     horizontal kernel exactly, and is checked against rotations and brute-force quadrature (`test_panels3d.py`).
     `walls.py`: each layer's boundary loops as wall panels of the copper's thickness, owned by the boundary
     triangle beneath (its circuit node, which both faces already share), carrying rooftop currents along the wall;
     their inductance with each other and with the sheets' RWG functions (`test_walls.py`: a straight wall and a
     rectangle of wall against Grover within 0.1–1%, symmetric positive definite, reciprocal on a sheet's edge).
     Still to do, in order:
     1. The walls' impedance. A skin-only surface impedance vanishes at DC (the walls would short the copper),
        and a fixed slice adds cross-section the faces already carry. Proposed: each wall owns the copper within
        t/2 of its edge (the slab two-port across that depth) and the boundary triangles' faces give that strip up,
        so DC is unchanged and the wall takes the side's skin current above ~3 MHz.
     2. Rooftops as branches in MQS (between consecutive boundary triangles' nodes), with their R, L and the
        mutual inductance to the sheets and vias; dense first, then the compressed backend's entries (native).
     3. Check: DC unchanged; the strip loop, the plane return and CM5 against FastHenry, for five models: one
        sheet, one sheet + walls, two faces, two faces + walls (and FastHenry's volume as the reference). One sheet
        + walls (boundary_walls already centres the panels on the layer) is the cheap candidate: thickness in L
        (the edge current spread up the side) and C (side charge) at about half the unknowns of two faces, but
        no face preference, so R near a return plane should stay low.
   - `face_correction='overlap'`: two face sheets now converge on a real six-layer board (CM5); see
     validation/README.md.
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
     `thick_kernel` stay the reference. The circuit tool's copper model (`power_bridge`) keeps one sheet for now.
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
4. **Parts on pads (C13, 1 day).** Attach parts along their pad edges, not at one node.
5. **Measurements.** VNA shunt-through on the wpthb DC link, and a near-field probe at 1–3 mm.

## Models and validation

- Vendor or fitted MOSFET models (MCG8D2N06YL), datasheet capacitor ESR/ESL
  and measured coil Q; switch internal capacitances as explicit elements so a
  warm start can initialise them.
- Mesh/crop convergence studies per board; VNA PDN and near-field probe
  measurements to calibrate.

