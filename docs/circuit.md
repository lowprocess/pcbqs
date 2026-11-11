# Circuits (`pcbqs.circuit`)

A general circuit simulator for power switching circuits: the model lists parts, nodes and drives, and the
engine builds an ngspice deck from exactly that, runs it until it repeats, adjusts whatever the model says may be
adjusted until its targets are met, and reports what every part does. Nothing in it knows what kind of converter
it is running.

    python -m pcbqs.circuit --templates                 # the example circuits
    python -m pcbqs.circuit --template boost > boost.json
    python -m pcbqs.circuit boost.json --out result --jobs 4
    python -m pcbqs.circuit boost.json --set parameters.vin=9 --set R1.resistance_ohm=24
    python -m pcbqs.circuit boost.json --out result --copper          # with its board's copper
    python -m pcbqs.circuit --from-kicad board.kicad_pcb > model.json  # the parts of a KiCad design

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
- `copper` (with `--copper`): `board`, `loss_W`, `fit_error`, `ports` (pad, its net's reference pad, `rms_A`)
- `stored_energy_change_W`: what went into the capacitors and inductors over the reported period, counted out of
  the parts' powers, the loss and the efficiency
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

## On a board

```json
"board": {"file": "06_boost.kicad_pcb", "parts": {"V1": "J1", "R1": "J2"}}
```

With `--copper` the board's copper is part of the circuit. A part's pin is the pads of its footprint on its
node's net (`0` is the board's GND), so nothing is said twice: the model says which part is on which node, the
board which pad is on which net. `board.parts` names a part's footprint when its reference differs (a source is a
connector), `board.nets` a node's net when its name differs.

Without copper every pin on a node is one point. With it, one pad of each net is the node itself (a source's
pad, else a load's, else the first; `board.reference` chooses another) and every other pad reaches it through a
port of a passive R-L model of the copper: all the nets' ports in one model, so the coupling between them is in
it. The model is extracted by the quasi-static solver (`pcbqs.mqs`) from DC to 20 MHz and at every switching
frequency and its tenth harmonic, fitted, and kept in `<out>/copper-model.json`. The examples' boards take from
a second to under a minute.

The copper costs ngspice about ten times the time, and moves a converter's operating point very little. So the
circuit alone is settled first and its targets met; the copper is put in where that ended, and the runs go on
from there. A case reports `copper`: the board, the fit's error, the copper's loss, and each port's RMS
current. A part with no footprint on the board is at its net's node, and the notes say so.

What the copper shows is mostly at the edges. The loop a switch and a diode commutate around has a few
nanohenries, and against the diode's and the switch's capacitance that rings: the boost's diode sees 57 V on its
board where the circuit alone gives 24 V. Nothing in the examples damps that ring but the parts' series
resistance, so it is larger than on a bench; give the parts their ESL and their losses, or a snubber, to see
what a real one does.

## From KiCad

`--from-kicad` reads a board (`.kicad_pcb`), a schematic (`.kicad_sch`, through `kicad-cli`) or a KiCad XML
netlist and prints a model of what it can tell, each part between the two nets its pins are on:

- resistors, capacitors and inductors, by symbol, reference or the footprint's library, with the value as KiCad
  writes it (`4k7`, `22u`, `0R01`) and `ESR`, `ESL`, `DCR` fields;
- diodes, anode to cathode by pin names (`A`, `K`), else pad 2 to pad 1;
- MOSFETs with pins `D`, `G`, `S`, as switches from drain to source.

A node is the net's name. A board that was read becomes the model's `board`. Left to add, since KiCad does not
say: which connector is the source and which the load, the drives, and what to measure. `pcbqs.circuit.kicad`
has the same for a model that exists (`apply`: add parts, or bring their values and nets up to date).

## Example boards

![The ten example boards](images/circuit-boards.png)

Each example has a layout beside it (`examples/*.kicad_pcb`, KiCad 8), written by `python -m
pcbqs.circuit.boards`: an island of copper for each net on the top layer, ground strips stitched to a ground plane
on the bottom, and every part across the gap between its two nets' islands. Two layers, 1.5 mm, 35 um copper. The
generator checks each (every pad on its net, each net in one piece, 0.4 mm between nets), and the tests read every
file back through kicad2ems's KiCad importer and get the example's netlist.

All ten run with their copper (ngspice 45.2):

| Example | Ports | Efficiency alone | on its board | Copper loss | Highest part voltage, alone to on its board |
|---|---|---|---|---|---|
| `rc_filter` | 3 | | | 0 mW | |
| `half_wave_rectifier` | 5 | 94.29 % | 94.28 % | 0.2 mW | D1 31.5 to 31.5 V |
| `bridge_rectifier` | 10 | 94.08 % | 94.06 % | 5.0 mW | D2 32.3 to 32.3 V |
| `buck` | 10 | 93.61 % | 93.57 % | 7.5 mW | D1 11.9 to 30.1 V |
| `buck_synchronous`, 4 A | 10 | 94.25 % | 94.12 % | 28.8 mW | Q2 11.9 to 24.0 V |
| `boost` | 10 | 97.49 % | 97.50 % | 2.3 mW | D1 24.0 to 57.0 V |
| `buck_boost` | 10 | 95.72 % | 95.75 % | 2.3 mW | D1 27.0 to 65.8 V |
| `flyback` | 16 | 82.18 % | 82.04 % | 4.3 mW | D2 64.2 to 81.4 V |
| `half_bridge_resonant`, 130 kHz | 9 | 98.14 % | 98.08 % | 6.1 mW | Q2 24.8 to 27.0 V |
| `full_bridge_phase_shift` | 15 | 94.51 % | 94.45 % | 14.3 mW | Q1 24.8 to 46.2 V |

The efficiencies are within a few hundredths of a per cent of each other either way, which is what the settling
tolerance allows; the copper's own loss is the column beside them. The soft-switched tank hardly notices its
board; the hard-switched ones ring to twice their bus and more.

## Not here yet

Field export from the copper's currents, the temperature sweep and the report pages of `pcbqs.power_circuit`.
