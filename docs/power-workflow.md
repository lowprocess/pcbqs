# Power checks during layout

Three tools, one question each. All read a `.kicad_pcb` (or a kicad2ems board
JSON) and run from the command line or from the kicad2ems window. Runnable
examples with their images are in
[`examples/power-workflow/`](../examples/power-workflow/README.md).

| Stage | Question | Tool | Window page |
|---|---|---|---|
| Converter placed | Is the switching loop small, and where is its field? | `python -m pcbqs.mqs` | Near field |
| Planes poured | Does each rail hold its voltage, at DC and across frequency? | `python -m pcbqs.pdn` | Power Integrity |
| Power stage routed | How do the switches, filter and load behave with this copper? | `python -m pcbqs.power_circuit` | Circuit |

## 1. Converter input loop (`pcbqs.mqs`)

Models the input loop only (input capacitors → switch → ground return), driven
by an assumed trapezoidal current pulse. Set `--converter` (and `--vin` when the
regulator has several supply pins), the operating point (`--fsw`, `--vin-volts`,
`--vout`, `--current`) and use `--preview` to check the meshed copper first.

![CM5 U702 near field](../examples/power-workflow/images/01-cm5-u702-H_top_1mm.png)

- **Copper L**: the loop inductance; it sets switch-node overshoot (V ≈ L·di/dt).
  Under ~1 nH is good (CM5 U702: 1.2 nH; Jetson bucks: 0.4 nH).
- **|H| map** at 1 and 3 mm: keep sensitive traces out of the bright area.
- Use it to **compare layouts**; the pulse spectrum and ESR/ESL are assumptions,
  and the package's own inductance is excluded.

## 2. Rail DC drop and impedance (`pcbqs.pdn`)

Grows the rail from its net through series parts (sense resistors, beads, 0 Ω,
and anything in `--through`), solves DC drop on the real copper and the PDN
impedance seen by each load.

![Jetson +1V15 impedance](../examples/power-workflow/images/02-jetson-1v15-impedance.png)

- **Drop at each load** against the tolerance budget; **hottest vias** and the
  current maps show crowding.
- **Impedance** below the target (V × ripple ÷ load step) holds the rail through
  that step. Peaks are resonances between stages; above the crossing frequency
  only the package and die help. The parts table's **mount (nH)** column shows
  which capacitors are poorly connected.
- Small rails: lower `--plane-min` (mm²). Input filters: `--through L3 --dcr L3=0.03`
  keeps the inductor's L and DCR. Bulk polymers/electrolytics: `--cap CA1=47e-6,0.02,1e-9`.
- **Loads kept with the design.** A part's `pdn.load` field (a symbol field in KiCad, which
  the board carries) is the current it draws, so a rail runs with no `--load` typed:
  `0.5`, `500m` or `2 A` from the rail the part is on; `+3V3=0.4, +1V8=100m` for a part on
  several rails; `1.5@Q2` for a load that returns through another part's ground pads.
  `--load REF=A` overrides that part's field, `--no-part-loads` ignores the fields, and the
  report names the loads that came from them.

![wpthb DC-link variants](../examples/power-workflow/images/03-wpthb-dc-link-variants.png)

On wpthb the input filter L3 resonates with the bridge's ceramics at 11 kHz
(1.8 Ω). A 47 µF polymer with ~0.5 Ω in series on VHB (≈ √(L/C) for that
filter) damps it to 0.5 Ω; replacing L3 with a resistor removes it but lets
the bridge's ripple current reach the supply.

## 3. Switching circuit with copper (`pcbqs.power_circuit`)

Simulates the power stage in ngspice with the copper as an extracted
multi-port model, sweeping frequency and target power. Start from
`--template` and edit components, switches and the coil/receiver.

![wpthb circuit builds](../examples/power-workflow/images/04-wpthb-circuit-builds.png)

- **Efficiency, coil currents, input power** per operating point.
- **VHB p-p** is measured at every solver step and includes switching-edge
  spikes; the plotted waveforms are sampled at T/256 and do not show them.
- **Supply current**: the ripple an input filter keeps off the supply cable
  (wpthb: 0.2–1 mA rms with L3, 35–47 mA without).
- Switch, package and capacitor parasitics in the template are placeholders;
  use datasheet or vendor models before trusting overshoot or loss.

Run time: phase-search trials use relaxed tolerances and continue from each
other; the reported run is strict. See
[power-circuit.md](power-circuit.md#run-time-and-solver-settings).

## Trusting the numbers

- **Mesh**: rerun with smaller `--fine`/`--dmax`. The CM5 loop moved 3–16%
  between 0.6 and 0.25 mm; refine until your margin stops moving.
- **Assumptions**: capacitor ESR/ESL, regulator R/L, switch models and operating
  points are inputs. Replace them where they decide the answer.
- **Measure** when margins are thin: VNA shunt-thru for PDN impedance, a
  near-field probe at 1–3 mm for the |H| maps.
- **Independent solvers**: [validation/](../validation/README.md) compares DC drop
  with Elmer (within 0.2–1.5% on wpthb and Jetson), loop inductance with Palace and
  FastHenry (0.3–2%), |H| with Palace (~5%) and plane pairs with Palace (≤ 0.8%).
  Copper AC resistance above ~3 MHz reads 11–34% low; treat loss figures there as
  lower bounds.

Known gaps and planned work: [power-tools-review.md](power-tools-review.md) and the
[roadmap](https://github.com/lowprocess/kicad2ems/blob/main/ROADMAP.md#power-tools-pcbqs); what was done is in [CHANGELOG.md](../CHANGELOG.md).
