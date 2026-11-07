# Switching circuits and coupled wireless-power loads

`python -m pcbqs.power_circuit` adds an electrical transient circuit workflow to
MQS. ngspice is an external executable dependency. It solves the nonlinear circuit;
MQS supplies the PCB copper terminal model and subsequently infers magnetic fields
from the solved terminal-current spectra. The existing normalized buck-loop CLI
remains available.

```sh
python -m pcbqs.power_circuit --template > wpthb-model.json
python -m pcbqs.power_circuit wpthb-model.json --board wpthb.kicad_pcb \
    --out wpthb-results --copper --fields
```

Without `--copper`, the electrical circuit uses ideal interconnects. `--prepare`
extracts only the PCB equivalent. `--frequencies 90000,110000` and
`--powers 5,10,20,30` override the JSON sweep; `--switch-type` selects the default
switch family, while individual switch entries can override it. All quantities in
component JSON use SI units; PCB geometry and mesh settings use mm.

## Explicit component models

The template describes the inspected wpthb bridge, including its populated local
DC link, the 99 nF primary series bank, R9, L3 and CA1–CA4. It does not import
semiconductor behavior or parasitics automatically from the schematic.

Each entry in `components` has `ref`, `a`, `b`, `type` and `value`. Types are
`resistor`, `capacitor` and `inductor`; replacing a type changes the circuit.
`enabled: false` omits a part. Capacitors accept `ESR_ohm`, `ESL_H`,
`leakage_ohm`, and `bias_factor` (multiplying effective capacitance). Inductors
accept `DCR_ohm` and additional series `ESL_H`. An omitted ideal part can disconnect
a circuit: bypassing a filter inductor requires a resistor/short replacement,
not simply disabling the only feed path.

Switch types:

- `ideal`: finite low Ron, no explicit capacitances; antiparallel diode optional.
- `switch`: bidirectional channel with a smooth gate-dependent conductance,
  on/off resistances, explicit Cds/Cgd/Cgs and optional body diode with transit time.
- `mosfet`: generic ngspice level-1 NMOS, with explicit capacitances.
- `vdmos`: ngspice power MOSFET, including voltage-dependent junction/Miller
  capacitance and body-diode charge behavior. `vdmos_parameters` supplies model
  parameters such as `CGDMIN`, `CGDMAX`, `CJO`, `KP`, `RD`, `RS`, `TT` and `VTO`.
- `subcircuit`: an ngspice-compatible vendor library, selected using `library`,
  `subcircuit`, and `pin_order` (a permutation of D/G/S). External capacitances
  are disabled unless `external_capacitance` is true, to avoid automatically
  duplicating the library's capacitances. Review package parasitics separately.

Switches accept gate rise/fall times, dead time, gate voltage/resistance, and
series drain/source package inductance. Each leg has a nominal 50% duty before
dead time. High/low pulses are complementary; phase zero cancels differential
command, and 180 degrees produces full bipolar drive. Finite edges and gate
loading mean actual conduction timing differs from the command.

The template's semiconductor and passive parasitics are **engineering
placeholders**, not a model of MCG8D2N06YL-TP fitted to measurements. A generic
transistor cannot establish device-specific switching loss, ZVS, reverse recovery
or EMI accuracy. Use measured parameters or a validated vendor model for those.

## Coil loss and matching

Each winding accepts `L_H`, `R_ohm`, `Cp_F` or `SRF_Hz`, and a loss model:

- `dc_only`: series DCR with inductance, shunted by the SRF capacitance.
- `constant_series`: total resistance is max(DCR, `Rac_ref_ohm`). The Q-derived
  total AC resistance is not added again on top of DCR.
- `parallel_core`: series DCR plus a resistor across the inductive winding,
  fitted so the isolated coil's real terminal impedance at `loss_reference_Hz`
  equals `Rac_ref_ohm`. `core_R_ohm` can instead be supplied explicitly.

The template uses 24 µH, 110 mΩ DCR, SRF 7.7 MHz (17.801 pF) and Q=155 at
125 kHz, giving about 30.77 kΩ fitted core resistance. Constant core conductance
is a passive extrapolation from one small-signal datum, not broadband ferrite,
current-dependent or temperature-dependent material characterization. It damps
the unrealistic very-high-Q SRF of the DCR-only lumped model. Coil self-capacitance
is across the winding terminals; it does not model coupling to chassis/earth.

The secondary is assumed to be another identical coil, with explicit k=0.4.
Its series capacitor is tuned at each operating frequency. The load is an
**AC-equivalent resistor selected for maximum sinusoidal link efficiency**,
including winding and specified compensation-capacitor loss, excluding the
bridge/filter. The primary stays at 99 nF. The 5/10/20/30 W operating points use
phase adjustment against the actual transient average load power. They are not
four imposed constant-power sinks and are not a model of a particular rectifier
or DC output voltage.

Frequency-response sweeps hold each selected receiver capacitor and resistor
fixed while sweeping frequency. They report loaded transfer peaks and input
reactance zeros, which are distinct quantities in a coupled resonant system.
Changing the drive frequency alone does not move the natural modes of an
unchanged linear circuit; retuning the receiver changes the circuit.

## PCB reduction and field inference

Pads are mapped explicitly to terminals. Independent current ports connect pads
to a selected reference pad on each connected net; fictitious port filaments
never contribute to fields. MQS extracts the full reciprocal terminal impedance
and copper-current response, including skin effect and return-current
redistribution. The passive matrix Foster equivalent has the form

`Z(s) = Rdc + s Linf + sum(s/(s+p) Rmode)`.

Rdc, Linf and every residue matrix are positive semidefinite, and p is positive.
Mode removal preserves passivity and is limited by the configured
`mesh.circuit_fit_tolerance` (default 3% at the fit frequencies); lower this
value to retain more states at increased circuit-simulation cost.
The PSD-constrained fit is exported as a real passive RL network with ideal
linear modal transformations implemented with native controlled sources. The model's frequency errors are recorded;
a good training fit is not a mesh or independent-frequency convergence result.
Field inference checks the equivalent against additional retained harmonics.
Cache fingerprints include board geometry, terminals, mesh and solver sources;
operating-case caches also include circuit physics, parasitics and device models.

ngspice integrates the circuit with adaptive time steps. Power and RMS measures
are taken on that adaptive grid, before uniform output interpolation, to retain
narrow commutation pulses. Final-cycle repeatability is checked; unsettled runs
extend their duration or fail. Electrical terminal power balance is reported.
`ramp` initialization supports electrical startup; `precharged` initializes the
wpthb supply/filter and tank from a harmonic estimate, then still requires
periodic settling. Precharged initialization is not a startup prediction.

The field step uses the final periodic terminal-current Fourier coefficients,
then the original MQS copper response at every retained harmonic, including DC.
Fourier coefficients are integrated directly on adaptive-step current waveforms
with four-point interval quadrature, before uniform display resampling, avoiding
aliasing of narrow switching pulses. Mean-square current is integrated separately
to measure the spectral energy omitted from the field band.
Its 64 exported VTK frames are actual real-vector time samples; the GUI's time
player reads them directly. Default inference retains 16 harmonics, configurable
up to 31 for the 64-frame export. Discarded current spectral energy and equivalent
errors are reported. Uniform circuit output sampling and retained field bandwidth
need convergence checks, especially around fast edges.

Fields currently include **copper and vias in the local bridge crop**. The input
filter is electrically present; upstream filter copper, external coils/receiver,
lead routing, component-body filaments, gate circuitry and environment coupling
are not field-modeled. Broadband feedback is a reduced model; fields are one-way
inference from its solved currents. Full startup fields, dynamic distributed
capacitance, thermal effects, ferrite field solutions and emissions compliance
predictions remain outside this workflow.

## GUI

The Circuit page loads/saves the same component JSON, selects the switch family,
frequencies, receiver powers and k, runs the CLI, plots selectable winding/bridge/filter waveforms, and opens a selected case's
fields in the existing Viewer. Top controls override the corresponding JSON
settings. Per-part parameters and library paths remain explicit in the editor.
Results include every netlist, model, adaptive-grid power measures, compact
waveforms, matching conditions, convergence history and field assumptions.

## Fixed compensation and frequency tuning

Set `secondary_C_F` to hold the receiver capacitor fixed, and `load_ohm` to
hold its AC-equivalent resistance fixed. Omitting either independently enables
its previous per-frequency tuning/matching behavior. For example, both coils
with 100 nF compensation and a once-selected 6.45 Ω receiver:

```json
{"primary_C_F": 1e-7, "secondary_C_F": 1e-7, "load_ohm": 6.45}
```

These keys supplement the full circuit configuration. Update the actual primary
capacitor bank entries as well: `primary_C_F` is the link calculation's reference,
not an automatic replacement of the `components` list. The fixed-100 nF example
uses three hypothetical 33.333 nF primary capacitors instead of the PCB's 33 nF
parts; original KiCad design files are unchanged.

The GUI exposes fixed receiver capacitance/load controls; blank values retain
automatic per-frequency tuning/matching. Frequency sweeps with both values fixed
now describe one electrical network, without silently retuning its receiver.
Coupled power-transfer peaks need not coincide with the uncoupled LC resonance,
maximum efficiency, or zero input reactance. Frequency-only regulation should
select one monotonic branch rather than searching across multiple peaks.

`transient.linear_solver` selects `klu` (default) or `sparse`, and
`transient.method` selects `gear` (default) or `trap`. These numerical options
can help difficult switching points; power targeting, settling and resolution
checks remain necessary. Changing the startup ramp changes initialization,
not the intended final periodic operating point.

## Runtime and parallel cases

Use `--jobs 4` to run four independent frequency/power cases concurrently. The
PCB extraction is shared and completes before the circuit workers start; field
transfer responses are prefetched before workers start, allowing dense extraction
matrices to be released. Field reconstruction follows the completed cases. Each worker writes its own case
folder. Results retain the requested frequency/power order. Duplicate output
folders are rejected. Stopping the CLI or GUI terminates all active ngspice
children and cancels queued cases.

The CLI defaults to one worker; the GUI's **Parallel circuit cases** control
defaults to four, capped by available logical CPUs. This reduces sweep wall time,
not the work required for an individual transient. Device-model threading is a
separate setting, `transient.ngspice_threads` (default one). It does not parallelize
ngspice's sparse matrix factorization; supported BSIM/OSDI model evaluation may
benefit, while the example's switch/level-1 models do not.

Each transient writes `performance.json` with child wall time, Python output
processing time, and available ngspice iteration, matrix-load, factorization and
solve statistics. `case.json` records total case `runtime_s` (including phase
pilots/retries) and per-trial performance. Field inference and shared extraction
are outside that per-case runtime.

The PCB extraction's BLAS threading is independent of ngspice. Set
`OPENBLAS_NUM_THREADS` before starting the CLI when benchmarking it. More threads
can slow small matrices; measure on the target machine. GPU matrix-fill support
does not make the ngspice transient a GPU simulation. The checked mixed-precision
GPU port-solve experiment is currently a benchmark prototype, not an enabled
circuit backend.
