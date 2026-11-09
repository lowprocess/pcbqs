# pcbqs roadmap: power tools

Covers the converter near field (`pcbqs.mqs`), rail PDN (`pcbqs.pdn`), switching
circuits (`pcbqs.power_circuit`) and their kicad2ems window pages. Findings
referenced as C/I/U numbers are in [docs/power-tools-review.md](docs/power-tools-review.md).
Estimates are working days. Last updated 2026-11-09 (evening: validation, C14 started, FasterCap).

## Done recently

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
3. **Ripple reported twice (I1; 0.5 day).** Settled (sampled) ripple and
   edge-spike p-p as separate results; the window plots raw solver points.
4. **PDN loads and targets (C9, C10, I6; 1 day).** `--load REF=A@RETURN` for loads
   whose return is another part (a bridge's low side); per-load targets or none;
   clip the bare-planes curve.
5. **Real-board regressions (0.5 day).** One coarse cached case each for wpthb
   (PDN and circuit) and CM5 (near field), checked against stored results.

## Then: workflow (≈ 5 days)

6. **Circuit page from `--describe` (I4, U2, U5; 1.5 days).** Generate the form
   from the tool, as Power Integrity and Near field do; per-group parasitic
   overrides; default to the template's switch model and warn on `switch` with ngspice.
7. **Shared board import (I2; 0.5 day).** Cache board JSON by file hash for all
   three tools and the window (Jetson import is 25 s per run today).
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

1. **AC resistance (C14), started (2 days).** `MQS(thickness_sheets=2)`: face sheets coupled by the slab
   two-port; the strip loop's 10–100 MHz error against FastHenry falls from −19/−20% to −12%.
   - Edge refinement: current crowds to strip edges; FastHenry uses 31 filaments across a 1 mm strip.
   - Memory: twice the unknowns. The compressed backend needs to work without `capacitance=True`.
   - `dielectric='stack'` with face sheets.
   - Validate on the CM5 loop, then make it the slab default.
2. **Capacitance against FasterCap, in progress (1 day).** FasterCap 5 is built at
   `~/opt/src/FasterCap-build`; `validation/cap_fastercap.py` (work in progress). Plates in air agree to ~1.5%.
   - Traces over a plane in air: pcbqs is 20% low on self capacitance and 45% low on trace-to-trace.
     Run FasterCap with zero-thickness copper to separate the 35 µm side walls from pcbqs error.
   - FasterCap's FR-4 input is wrong (non-physical matrix): fix the dielectric interface where copper sits on it.
   - Terminals: a pcbqs terminal has one charge density, so it must stay small. Note this in `capacitance_matrix`.
   - Then pcbqs's layered-stack images on FR-4, and side-wall capacitance once face sheets carry charge.
3. **Real converter loop (1 day).** CM5 U702 against Palace and FastHenry, with regulator and capacitor pads
   modelled alike (C13: a pcbqs terminal is one node, so large pads shorten a small loop).
4. **Parts on pads (C13, 1 day).** Attach parts along their pad edges, not at one node.
5. **Measurements.** VNA shunt-through on the wpthb DC link, and a near-field probe at 1–3 mm.

## Models and validation

- Vendor or fitted MOSFET models (MCG8D2N06YL), datasheet capacitor ESR/ESL
  and measured coil Q; switch internal capacitances as explicit elements so a
  warm start can initialise them.
- Mesh/crop convergence studies per board; VNA PDN and near-field probe
  measurements to calibrate.

