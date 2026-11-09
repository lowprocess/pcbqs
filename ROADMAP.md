# pcbqs roadmap: power tools

Covers the converter near field (`pcbqs.mqs`), rail PDN (`pcbqs.pdn`), switching
circuits (`pcbqs.power_circuit`) and their kicad2ems window pages. Findings
referenced as C/I/U numbers are in [docs/power-tools-review.md](docs/power-tools-review.md).
Estimates are working days. Last updated 2026-11-09.

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

## Models and validation

- Vendor or fitted MOSFET models (MCG8D2N06YL), datasheet capacitor ESR/ESL
  and measured coil Q; switch internal capacitances as explicit elements so a
  warm start can initialise them.
- Mesh/crop convergence studies per board; VNA PDN and near-field probe
  measurements to calibrate.
- Reformat the `power_*` modules (one statement per line) before larger changes.
