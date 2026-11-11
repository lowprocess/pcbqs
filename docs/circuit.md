# Circuits (`pcbqs.circuit`)

A general circuit simulator for power switching circuits: the model lists parts, nodes and drives, and the
engine builds an ngspice deck from exactly that, runs it until it repeats, adjusts whatever the model says may be
adjusted until its targets are met, and reports what every part does. Nothing in it knows what kind of converter
it is running.

    python -m pcbqs.circuit --templates                 # the example circuits
    python -m pcbqs.circuit --template boost > boost.json
    python -m pcbqs.circuit boost.json --out result --jobs 4
    python -m pcbqs.circuit boost.json --set parameters.vin=9 --set R1.resistance_ohm=24

(`python -m pcbqs.power_circuit` is the older, format 1 tool for one circuit: a phase-shifted full bridge into a
coupled receiver, with the board's copper. It is unchanged; its model will move to this format.)

## The model (format 2)

```json
{"format": 2, "name": "Boost converter",
 "parameters": {"fsw": 200e3, "vin": 12, "vout": 24, "pout": 12},
 "parts": [
  {"ref": "V1", "type": "source",    "nodes": ["in", "0"],   "dc_V": "vin", "R_ohm": 0.05},
  {"ref": "L1", "type": "inductor",  "nodes": ["in", "sw"],  "value": "47u", "DCR_ohm": 0.05},
  {"ref": "Q1", "type": "switch",    "nodes": ["sw", "0"],   "drive": "pwm", "model": {"Ron_ohm": 0.03}},
  {"ref": "D1", "type": "diode",     "nodes": ["sw", "out"], "model": {"Vf_V": 0.45, "Cj_F": "100p"}},
  {"ref": "C2", "type": "capacitor", "nodes": ["out", "0"],  "value": "22u", "ESR_ohm": 0.01},
  {"ref": "R1", "type": "load",      "nodes": ["out", "0"],  "resistance_ohm": "vout*vout/pout"}],
 "drives":   {"pwm": {"frequency_Hz": "fsw", "duty": 0.5}},
 "probes":   {"vout": {"voltage": ["out", "0"]}, "il": {"current": "L1"}},
 "measures": {"vout_avg": {"avg": "vout"}, "il_ripple": {"pp": "il"}},
 "targets":  [{"adjust": "drives.pwm.duty", "range": [0.05, 0.9], "measure": "vout_avg", "value": "vout"}],
 "sweep":    {"parameters.vin": [9, 12, 15]}}
```

A value is a number, engineering notation (`22u`, `4.7k`, `100n`) or an expression over the parameters
(`1/((2*pi*fsw)**2*l)`; `+ - * / **`, `pi`, `sqrt abs min max exp log`). A path names a number anywhere in the
model: `parameters.vin`, `drives.pwm.duty`, `R1.resistance_ohm`, `Q1.model.Ron_ohm`, `couplings.0.k`.

| part | nodes | fields |
|---|---|---|
| `resistor` | a, b | `value` |
| `capacitor` | a, b | `value`, `ESR_ohm`, `ESL_H`, `leakage_ohm`, `bias_factor` |
| `inductor` | a, b | `value`, `DCR_ohm`; with `SRF_Hz` / `Cp_F` / `core_R_ohm`, a winding with self-capacitance and core loss |
| `diode` | anode, cathode | `model`: `Vf_V` at `If_A` (0.7 V at 1 A), `n`, `R_ohm`, `Cj_F`, `TT_s` |
| `switch` | drain, source | `drive`, `invert`; `model`: `type` (`switch`, `ideal`, `mosfet`, `vdmos`, `subcircuit`), `Ron_ohm`, `Cds_F`, `body_diode`, `diode_TT_s`, ... |
| `source` | +, − | `dc_V`, or `wave` (`pulse`, `sine`, `pwl`), or `dc_A`; `R_ohm`, `L_H` in series |
| `load` | a, b | `resistance_ohm`, or `power_W`, or `current_A` (a resistor below `min_V`, so it starts) |

`0` or `GND` is ground. A part's current is taken as entering its first node, so its voltage times its current is
the power it absorbs.

- **couplings**: `{"inductors": ["LP", "LS"], "k": 0.98}`: a transformer is two inductors and a coupling. The dot
  is at each inductor's first node.
- **drives**: `frequency_Hz`, `duty`, `phase_deg`, `dead_time_s` (between the middles of one switch's fall and
  its complement's rise), `gate_V`, `rise_s`, `fall_s`. A switch follows its drive; with `invert` it is the
  drive's complement. Any number of drives, at frequencies in a whole-number ratio.
- **probes**: a voltage between two nodes, or a part's current, under a name.
- **measures**: `avg`, `rms`, `pp`, `max` or `min` of an expression of probes and parts' signals
  (`{"avg": "vout*il"}`, `{"max": "Q1.V"}`), over the last period.
- **targets**: adjust one number of the model, inside a range, until a measure (or `load_power_W`,
  `source_power_W`, `efficiency`, `loss_W`) has a value. Several targets, each with its own knob.
- **sweep**: paths and the values each takes; every combination is a point, in its own folder.
- **analysis**: `cycles` (200 periods a run), `periodic_tolerance` (0.003), `energy_tolerance` (0.002),
  `target_tolerance` (0.005), `max_cycles`, `samples_per_cycle`, `period_s`, `ignore`, `solver` (ngspice's
  `reltol`, `method`, ...).

## A run

1. The first run starts from ngspice's DC solution at time zero, where every gate is still low: the rails are
   charged through the diodes and the inductors carry their DC current, whatever the circuit.
2. Each run starts where the last one stopped: every capacitor's voltage, every inductor's current, every
   switch's and diode's voltage.
3. It is settled when the state repeats (the last period against every period of the half-run before it) and
   the energy the circuit stores has stopped changing (by less than 0.2 % of the power it handles: that change
   is the error of an efficiency taken over the period). A run that has not settled is followed by one twice
   as long, started from the middle of whatever was still ringing.
4. Settled, each target is compared with its measure and its knob moved by a secant step. A target that cannot be
   reached inside its range ends the point saying which end it stopped at and what it gave there.
5. The reported run repeats a few periods at the strict tolerance (`reltol` 1e-4; the search runs at 1e-3) with
   steps four times finer. If ngspice cannot hold that, the search's result is reported and a note says so.

## Results

`<out>/circuit.json`: `{format, name, complete, sweep, cases, failed}`. Each case, also in `<out>/<point>/case.json`
beside `circuit.cir`, `transient.npz` (every signal, uniformly sampled), `model.json` (the numbers it ran with)
and `waveforms.png`:

- `knobs`, `targets` (what each reached), `measures` (the model's own)
- `source_power_W`, `load_power_W`, `loss_W`, `efficiency`, `gate_drive_power_W`
- `parts`: for every part its `power_W`; a switch and a diode: `avg_A`, `rms_A`, `peak_A`, `peak_V`; a capacitor:
  `avg_V`, `ripple_V`, `rms_A`; an inductor: `avg_A`, `ripple_A`, `peak_A`, `rms_A`; a source and a load: `avg_V`,
  `avg_A`
- `coupled`: each set of coupled inductors with the power the set loses (one winding alone passes power to the
  other, so its own `power_W` is not a loss)
- `periodic_error`, `energy_drift_relative`, `reltol`, `runs`, `periods`, `history` (every run of the search)
- `notes`: anything the result should be read with, such as a voltage no circuit gives

## What to give the parts

A switch or a diode with no capacitance, cutting off the current of an inductance that has nowhere else to go,
has no answer: ngspice either stops ("timestep too small") or returns a spike of thousands of volts, and the
case's notes say so. Give the switches `Cds_F` and the diodes `Cj_F` (a few hundred pF), as real ones have. A
bridge of diodes with none leaves its AC side floating while all four are off, and fails the same way.

A switch's body diode has no reverse recovery unless its model gives `diode_TT_s`; with it, a hard-switched leg
with no loop inductance shows recovery currents of tens of amps, which is the model's answer and not a fault.

## Examples

`rc_filter`, `half_wave_rectifier`, `bridge_rectifier` (constant-power load), `buck`, `buck_synchronous` (dead
time, a load sweep), `boost`, `buck_boost`, `flyback` (coupled inductors, an RCD clamp), `half_bridge_resonant` (a
frequency sweep), `full_bridge_phase_shift` (two drives; the phase between them is the knob). Each is checked in
`tests/test_circuit_examples.py` against its closed form: ripple and duty for the converters, harmonic by
harmonic for the resonant tank, charge and power balance for the rest.

## Not here yet

The board's copper (`--copper`, the PCB terminals and their feedback into the circuit), the temperature sweep and
the report pages of `pcbqs.power_circuit`. They go by net names, not by what the circuit is, and will move over
with the full-bridge model.
