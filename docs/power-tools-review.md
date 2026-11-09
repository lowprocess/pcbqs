# Review: power tools and their GUI integration (2026-11-09)

Scope: `pcbqs.pdn` (rail DC drop and PDN impedance), `pcbqs.mqs` CLI (converter
input-loop near field), `pcbqs.power_circuit` / `power_bridge` / `power_spice`
(switching circuits with PCB copper), and the kicad2ems window's Power
Integrity, Near field and Circuit pages. Exercised on Jetson AGX Thor, CM5
Minima and the wpthb wireless-power bridge; see
[power-workflow.md](power-workflow.md) and `examples/power-workflow/`.

**Fixed on `claude/circuit-sim-speed` and its base** marks findings found and
fixed while running these boards; the rest are in [ROADMAP.md](../ROADMAP.md).

## Correctness

| # | Finding | Where | Status |
|---|---|---|---|
| C1 | The PDN tool shorted series filter inductors: `--through L3` became a 10 µΩ link, losing L3's 5.6 µH and DCR (wpthb DC drop 9.7 instead of 49.9 mV; the 11 kHz filter resonance invisible). | `pdn/rail.py` | Fixed: followed inductors keep L and DCR; `--dcr` |
| C2 | Capacitors named CA1/CE1/CP1 were silently not decaps (reference regex `C\d`); electrolytics would have had ceramic ESR/ESL. | `pdn/rail.py`, `pdn/stack.py` | Fixed: prefixed refs, package defaults, `--cap` |
| C3 | Small rails failed ("no plane pair over 1790 mm²") with no way to set the plane threshold from the CLI/GUI. | `pdn/stack.py:90` | Fixed: `--plane-min` |
| C4 | Gate pulses with an on-interval spanning t = 0 started off: every transient began with a wrong bridge state for part of a cycle, re-exciting the input filter; at 180° a low-side switch skipped a whole period. | `power_spice.py` `switch()` | Fixed |
| C5 | abstol 1e-9 A made strict runs fail ("timestep too small") at hard-switched edges; no retry existed. | `power_spice.py` | Fixed: 1e-6 A default, trap/cold retries, failed logs kept |
| C6 | Coaxial via segments used 1e-7 m separation for their mutual inductance; vias through inner copper were over-inductive (Jetson X −1.6% after fixing). | `mqs.py` `_via_matrices` | Fixed (earlier this branch's base) |
| C7 | Precharge hard-codes the template's references and drops (C19–C24, CA*, L3, C25/C27/C30; 0.055/0.085 Ω). Another board, or a renamed or replaced L3, starts far from steady state. Warm starts now replace it after the first (pilot) run. | `power_bridge.py:98-101,150` | Partly: warm starts; generic DC precharge on ROADMAP |
| C8 | The periodicity check ignores supply current, and input power is averaged over one switching cycle, so a slowly decaying input-filter mode biases input power and efficiency (≤ 0.66% on wpthb in both old and new code). | `power_circuit.py:160,181` | ROADMAP |
| C9 | High-side FET loads have no ground pad, so a bridge's DC return (through the low-side FET) is not modelled; the tool only notes it. | `pdn/dc.py:250` | ROADMAP: `--load REF=A@RETURN` |
| C10 | The PDN target defaults to V × 5% ÷ (half the loads' current) even for a switching bridge, where it has little meaning. | `pdn/cli.py:196-197` | ROADMAP: per-load/no target |

## Integration and results

| # | Finding | Where | Status |
|---|---|---|---|
| I1 | `bus_ripple_V` is measured at every solver timestep (edge spikes), while saved waveforms are resampled at T/256 and hide them; the GUI plots the resampled data. Easy to misread (it was misread during this review). | `power_bridge.py:153` | ROADMAP: report settled ripple and spike p-p separately; plot raw points |
| I2 | Each tool re-imports the KiCad board (25 s for Jetson) on every run; nothing is shared between the three tools or the GUI pages. | CLIs | ROADMAP: shared board-JSON cache by hash |
| I3 | Two near-field paths: `pcbqs.mqs` (assumed pulse spectrum) and `power_circuit --fields` (simulated currents); results are not cross-referenced. | `mqs_cli.py`, `power_circuit.py` | ROADMAP |
| I4 | PDN and Near field forms are generated from `--describe` (good: new options appear without GUI changes). The Circuit page is hand-built around the wpthb template instead. | `powercircuittab.py` | ROADMAP: `--describe` for power_circuit |
| I5 | Converter discovery stops at ambiguous input nets (VCC/VDD/VDRV on five Jetson bucks); the user must know the right net. | `mqs_cli.py:125` | ROADMAP: offer candidates in the GUI |
| I6 | The PDN impedance plot's bare-planes curve can reach 10¹¹ Ω on boards with a tiny cavity, flattening the useful curves. | `pdn/cli.py` figure | ROADMAP: clip or drop it |

## UI flow (kicad2ems window)

| # | Finding | Where | Status |
|---|---|---|---|
| U1 | Circuit page overrides "L3 DCR" and the waveform view "Input filter currents" (`L3.I`, `R9.I`) are wpthb-specific. | `powercircuittab.py:32,182` | ROADMAP |
| U2 | "All-capacitor ESR/ESL override" also changes the resonant bank and polymers. | `powercircuittab.py:96-106` | ROADMAP: per-group overrides |
| U3 | Circuit results table lacks input power, VHB spike, supply ripple, settling and run time; the Power Integrity summary does not show a series inductor's inductance. | `powercircuittab.py`, `pdntab.py:112` | Fixed in kicad2ems-gui branch (table columns, L in the tree) |
| U4 | No path from a converter on Near field to its switching circuit, or from Circuit's input current to a PDN load. The PDN↔Near field transfer exists. | pages | ROADMAP |
| U5 | The Circuit page's switch list offers the behavioural `switch` model first; ngspice did not converge it on the benchmark converter at reltol 1e-3/1e-4 (Xyce did). | `powercircuittab.py:24` | ROADMAP: default to the template's VDMOS, warn |
| U6 | The Power Integrity page split list options at every comma, so `--cap CA1=47e-6,0.02,1e-9` reached the tool as three broken arguments. | `pdntab.py` `pdn_argv` | Fixed: only a comma before a new `REF=` separates items |

## Maintainability

- The `power_*` modules were written as dense one-line statements, which made review and diffs harder than the rest of pcbqs. Reformatted with ruff (120 columns, quotes preserved; `[tool.ruff]` in pyproject.toml).
- Duplication removed: plane-pair physics (`pdn/cavity.py` functions now used by `RailPDN`), CLI options/`--describe`/board loading (`pcbqs/_cli.py`), series-part pad selection (`Series.pads`, `Rail.pads_of`), and the window's describe/process code (`toolprocess.py`). Dead code removed (unused end-state nodes, metric fallbacks, `dc.supply`, unused imports and arrays).
- Tests cover circuit physics with small netlists and now warm starts; there is no end-to-end regression of a real board's results (wpthb or CM5) to catch drift such as C4–C6. Adding one coarse, cached case is cheap.
- Template-specific logic (C7, U1) and the generic engine are mixed in `power_bridge.py`.
