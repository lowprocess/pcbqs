# Changelog

Notable changes to pcbqs, by day, in the form of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), with
*Decided* for what was looked at and not done. It was started on 2026-11-12 from the finished items of the
roadmap; the git log has every commit, and kicad2ems's CHANGELOG.md has the time these tools lived in mom25d and
kicad2ems (before 2026-11-07). What is planned is in kicad2ems's
[ROADMAP.md](https://github.com/lowprocess/kicad2ems/blob/main/ROADMAP.md#power-tools-pcbqs). C, I and U numbers
are findings of [docs/power-tools-review.md](docs/power-tools-review.md).

## Unreleased

### Changed
- **The charge is solved on graded edge cells** (`pcbqs.sparams`, whenever it solves the copper's charge:
  `EDGE_RINGS` = 6 across half the narrowest port's width; `edge_rings=1` for plain cells). A constant charge per
  triangle does not resolve the edge charge on plain cells, so a trace's capacitance was low: the thru's |S11|
  with the charge was 8.8 % above the field solver's line and is within 0.1 %; demo_transition against mom25d
  0.040 in |S| at 1 GHz, now 0.019; 1.4 times the triangles. The same rings on `MQS(capacitance=True)` (an
  option there, as it was): two traces 0.1 mm apart −2.5 % self and −3.6 % mutual against FasterCap at 0.5 mm
  cells (−10 % and −23 % on plain cells, −7 % and −16 % at 0.25 mm), a microstrip +0.8 % (−2.4 %):
  `validation/README.md`. `--thickness` (the copper's two faces) has the rings too: with the skin effect the
  current crowds at a trace's edges as the charge does. A 1 mm trace 1.6 mm over a plane against FastHenry:
  −9 % at 10 MHz and −10 % at 100 MHz, from −26 % and −33 %; the thru's resistance at 100 MHz 69 mΩ for 46 mΩ,
  and its inductance there the field solver's line's within 0.5 %.

### Added
- **A piecewise-linear engine for the circuit tool** (`circuit/pwl.py`; `python -m pcbqs.circuit --engine pwl`,
  `analysis.engine` in a model, *Engine* in the window). No transient: between switching instants the circuit
  is linear and solved exactly (a matrix exponential), and the periodic steady state is found by Newton's
  method on the period's map in a few periods. Diodes change state when the circuit says so (discontinuous
  conduction, commutation, the dead time through a body diode). Coupled inductors, several drives, constant
  power and constant current loads, targets and sweeps as before; the result is the same signals and figures,
  so the report, the supply currents and the window read it as any run. The seven converter examples agree
  with ngspice (duty within 0.001, ripples within 0.2 %, efficiency within 0.7 point) in 0.3 to 0.8 s where
  ngspice takes 4 to 40 s. Ideal switches and diodes only: no capacitance, no edges, no ESL; MOSFET models, a
  winding's parasitics and the board's copper are still ngspice's, and it says so (docs/circuit.md). Sine
  and pulse sources are in it (a sine: two states that turn at its frequency), so all ten example circuits
  run: the bridge rectifier's DC output 30.27 V for ngspice's 30.35 V.
- **A boost found on a board is made into a circuit** (`pcbqs.mqs --circuit`, *Open ... as a circuit* on a near
  field's result; `circuit/converter.py: boost_of`). Only a buck was: a cell with a diode up from its switch node
  stopped with "only a buck is made into a circuit". The near field's capacitors are the boost's output's, the
  net on the inductor's far side its input (a supply on its connector, the capacitors there), the lower switch
  and the diode as the board names them (two switches: a synchronous boost, where `--vout` is above
  `--vin-volts`), and `--current` is the inductor's current. From the boost example's board alone it settles
  at the example's duty (0.516 for 0.514) and inductor ripple. The output capacitors are given their ESR only:
  the model has no stray inductance to slow the commutation into them, and their ESL rang 27 V on a 24 V output.
- **The low-frequency S-parameters take the board's lumped parts** (`pcbqs.sparams`, `--solvers pcbqs`): each a
  series R, L and C between patches of the copper at its two pads, so a filter of parts or a trace with a
  series capacitor has its low end (they were left out: a series part's gap was an open circuit). The fine
  mesh follows the signal through the series parts. `demo_lowpass` against mom25d solved point by point: |S21|
  0.9910 for 0.9912 at 117 MHz, 0.9886 for 0.9882 at 274 MHz. A part with no value or a pad on no copper is
  left out and named.
- **The settings that name a net or a part say so** (an option's `pick`, kicad2ems API 2.11): the rail, its
  ground, source, ground source, `through` and `stop`; a converter, its inductor, input net and capacitors.
  kicad2ems's window puts a pick button beside each, and a click on the board fills it in.
- **The full bridge takes its coils' inductances as options** (`--primary-l`, `--secondary-l`, uH, beside
  `--coupling`): what a field solution of the coils gives. kicad2ems-fem's *Coils and ferrite* (Elmer) proposes
  the three to the circuit from its result, so a wireless-power bridge runs with the coils as solved beside their
  ferrite, not a datasheet's figure.
- **The rail tool names what differs by rail** (`several_keys`, kicad2ems API 2.10): the rail, its volts, source,
  ground source, loads and load spectra, which kicad2ems's window shows as a table of a board's rails.
- **The near field of a converter built from discrete parts.** The tool knew a converter only as one IC with pads
  on the input, the switch node and ground, so none of the circuit examples (all discrete power stages) had one.
  It now also finds a switching cell: a transistor or diode up from a switch node to a decoupled net and one down
  to ground, at least one a transistor. Its input loop runs from the capacitors through the upper part, the switch
  node's copper and the lower part (a 10 mOhm conductor) back to ground. On the examples: the buck (Q1 and D1,
  1.8 nH), the synchronous buck (Q1 and Q2, 1.5 nH), the boost (D1 and Q1, the output's loop, 2.3 nH), the half
  bridge, and each leg of the full bridge (`--converter Q1` or `Q3`; 5.5 nH for the leg furthest from its
  capacitor). A cell's capacitors out of reach are taken by connection. A discrete buck becomes a circuit with
  its own parts (QH on Q1, D1 or QL on the lower one), and the converter's model now puts the supply and the load
  on the board's connectors where it has one on each net (J1, J2), so their copper is in the circuit. The boost
  and the bridges are not made into circuits (their examples have their own models), and say so. Not found: the
  flyback (its loop runs through the transformer), the inverting buck-boost (through both capacitors).
- **The full bridge says what it draws from its rail.** A run of `pcbqs.power_circuit` has `supply` in each
  point, as a general circuit's has (`circuit/supply.py`, `bridge_currents`): each high-side switch's average
  current and 40 harmonics (`supply-<switch>.csv`), the rail named from the supply's side with the filter's
  inductors on the way (`through`, `dcr`), and its result proposes that rail to the power integrity. On wpthb
  at 90 kHz and 30 W: Q1 and Q2 draw 1.11 and 0.20 A, their fundamentals cancel in the rail, and the rail's
  model gives 36.6 mV at 180 kHz on the bridge rail against the 34.7 mV the circuit's own bus voltage has (20.8
  and 19.7 mΩ of the same capacitors, by two models).
- **Several rails, or converters, in one study (kicad2ems API 2.10).** The power-integrity and near-field
  plugins say that a study may hold several settings of them (`several`: `'rail'`, `'converter'`): a config's
  `[[analysis.pdn]]` tables are a run each of `kicad2ems study`, into folders named by the rail.
- **From one tool to the next: a converter as a circuit, a circuit's current as a rail's load and as the near
  field's.** (Review U4, I3.) *A converter as a circuit*: `python -m pcbqs.mqs ... --circuit` prints what the
  near-field tool selects as a model for `pcbqs.circuit` (`circuit/converter.py`): a supply on the input net,
  the selected input capacitors, two switches on the regulator's own footprint (a diode where the board has one
  from ground to the switch node), the inductor, the output capacitors within reach and a load, the duty
  adjusted until the output is `vout`; what the board cannot say is assumed and listed in its description. A
  near-field run writes it as `circuit-model.json`. *What a circuit draws* (`circuit/supply.py`): each point of
  `circuit.json` has `supply`, for each supply every part on it but its capacitors with its average current and
  40 harmonics, written as `supply-<part>.csv`. *A rail's ripple*: `python -m pcbqs.pdn --load-spectrum
  REF=FILE` takes such a file as a load's AC current, and the result has the ripple the spectra make where each
  load meets the planes (`pdn.ripple`: every harmonic's current times the transfer impedance,
  `RailPDN.transfer`, put back together over the period; peak to peak, RMS, the largest harmonic, against volts
  x ripple), in the report, the headline and a curve set. It is the rail's voltage: a load's own pads add the
  L di/dt of its own connection, which is its pin's. *The near field of a simulated current*: `python -m
  pcbqs.mqs --spectrum FILE` takes the harmonics from such a file in place of the assumed pulse, and the report
  lists the pulse beside them. *What a result proposes* (`suggest`): a near field's, its converter as a circuit;
  a circuit's, its supply rail with its loads and spectra for the power integrity, and a part's input current
  for the near field. On CM5 Minima's U702 (5 V to 3.3 V, 1 MHz, 2 A): the circuit settles at a duty of 0.72
  and 94 %; its 1.40 A from +5V makes 140 mV peak to peak on the rail at U702 and 33 mV at U405 (250 mV
  allowed); the assumed pulse and the simulated current agree at 1 MHz (1.12 and 1.03 A) and differ fivefold at
  the pulse's nulls. The two tools' fields of that current agree within 6 % at 3 mm; at 1 mm the answer is
  what a part's body is taken to be (218 A/m with the parts' currents 0.5 mm up, 76 in the copper's plane, 48
  for the copper alone), and at 10 mm the circuit's is 2.7 times larger, having the output loop too
  (`docs/power-workflow.md`).
- **A load with no pad on ground returns through the part the board shows (kicad2ems API 2.9).** A rail's load
  that is on the rail and not on ground (a bridge's high-side switch: wpthb's Q1 and Q2) had its return left out
  of the drop, with a note in the result that is easy to miss. The rail now takes the return itself where the
  board leaves no choice (`pdn.rail.return_part`): the one part of the load's kind (Q for Q) that is on one of
  its nets that is no rail and on ground, a half bridge's low-side switch, and not the driver that is on that
  node as well. The result names it (`returns`, a note, the report's first section: "Returning through another
  part's ground pads: Q1 through Q3"), and `REF=AMPS@PART` still names another. On wpthb the drop at Q1 goes
  from 49.66 to 50.17 mV, the 0.51 mV its return raises the ground by, and the loss from 66.4 to 67.1 mW.
  Where the board has no such part, or several, the return is still left out, and the result now says so where
  it is read: `no_return` lists the loads, and the report's headline ends "without the return of Q1 (no pad on
  ground: REF=AMPS@PART names the part it returns through)". In kicad2ems's window that case is something the
  run still needs (`PowerIntegrity.missing`), listed under the Rail drop question's settings, and Run goes to
  it instead of starting. Also for that API: the temperature says it can start from a circuit's result
  (`uses`), and the low-frequency solver names its cell size as the setting a run is set up with (`basic`).
- **The full bridge's run is a circuit result.** A `power-circuit.json` is the circuit analysis's as
  `circuit.json` is: its operating points, notes and assumptions as a report and table, each point's waveforms
  as curve sets (the windings, the bridge and coils, the series parts between the supply's rails found by what
  they join, the bus; again at the solver's own timesteps where the run saved them). The window shows them on
  its Results page, as it does earlier runs.
- **`--describe` for the two circuit tools**, `pcbqs.circuit` and `pcbqs.power_circuit`: their options as tables
  (`_cli`; an option may list its choices), which the circuit plugin gives a front end by the kind of model
  (`only`: general, bridge or either) and passes to the tool that runs it. The bridge tool gains `--coupling`,
  `--receiver-c`, `--load` and `--override GROUP.KEY=VALUE`: a value for every part of a group (U2), the groups
  by what the parts join (`power_bridge.groups`: tank, bridge-rail, input-rail, filter; and switch), so an ESR
  for the resonant bank leaves the rails' capacitors alone (docs/power-circuit.md).
- **Low-frequency S-parameters** (`pcbqs.sparams`, docs/sparams.md). A board's ports from the quasi-static
  currents of its copper, DC to about a tenth of a wavelength: each port a 1 A source and its reference impedance
  between the trace's end and the copper under it, one solve per driven port, S from the currents in the
  terminations. Resistance and inductance alone while the copper's capacitance carries next to nothing; the
  charge solved with the currents above that (`--capacitance auto`), with both limits estimated and written
  (`valid_to_Hz`). It writes Touchstone, the path between each two ports as a series R and L per frequency (two
  ports whose return is not joined are said to be so, with the capacitance that closes the path), and each
  layer's current with a port driven, for kicad2ems's Viewer. `--thickness` models the copper's two faces: the
  skin effect, and a narrow trace's inductance. With kicad2ems it is a solver, `--solvers pcbqs` (`qs`,
  `quasistatic`), run only when named. `examples/return-path/return_path.py`: one ground via beside a layer
  change, 8 mm away, at the board's corner (+1 nH of loop a step), or missing (open at DC; the planes' 29 pF
  closes the path, −35 dB at 1 MHz). Against gfdtd on kicad2ems's `demo_transition` and `rf_diffvia`
  from 0.05 to 1 GHz it is within 0.008 rms |ΔS| up to 200 MHz and 0.03 up to 1 GHz (0.010 with `--thickness`
  and a cell at the trace's width); with `rf_diffvia`'s return vias taken out, the pair's common mode falls to
  −11.0 dB at 50 MHz, gfdtd's −11.1 dB, which the line model does not see.
- **The figures of an answer, by name** (`figures(folder)` of the rail and thermal plugins, kicad2ems API 2.8):
  each load's drop in the rail's own copper, and the hottest point and the board under each part, named as any
  other engine of that question names them, so kicad2ems lays pcbqs's answer beside Elmer's; the drop with the
  ground's rise and the loss of both are the rail tool's own figures.
- **Several drives in one solve.** `MQS.solve(freqs, amps=[{...}, ...])` gives a Result per drive with each
  frequency's system factored once for them all (the dense solves, with and without the charge).
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

### Fixed
- **A circuit's fields in the Viewer, not as a picture in its report.** `python -m pcbqs.circuit --fields` wrote its
  H and E only as `fields.npz` and `fields.png`, so a front end showed them as an image in the report and the
  Viewer had nothing. Each point now also writes them as time-domain field dumps, H and E on each plane over the
  period (64 frames), in `<run>/<point>/exc_fields/` (`fields.write_dumps`), where kicad2ems's Viewer finds a run's
  dumps and plays them; the report keeps its table and leaves the picture out. The full bridge's (`power_circuit
  --fields`) were dumps already, one folder too deep for the Viewer (`<point>/mqs/exc_bridge`): now
  `<point>/exc_bridge`.
- **A slow mode of the input filter no longer shifts the bridge's input power (C8).** The inductors in series
  between the supply's rails ring against the rails' capacitors (wpthb: 10.4 kHz, eight to eleven switching
  cycles a period) long after the rest of the circuit repeats, and the powers were averaged over the last
  cycle: the efficiency was off by wherever the ringing was. They are averaged over whole periods of that
  mode (`power_bridge.slow_mode`, `average_cycles`: 9 cycles at 90 kHz on wpthb). With the mode excited on
  purpose, against a settled run of 1,500 cycles: +0.69 points of efficiency before, −0.04 now, in the same
  time (docs/power-circuit.md#slow-mode).

### Changed
- **No part of wpthb is named in the bridge's code (C7, U1).** A precharged start went by wpthb's references
  (C25/C27/C30, C19 to C24, CA*, L3) and its two supply resistances, with a second branch by connectivity for
  other models. It goes by what a part joins for every model, and wpthb's nine decks are the same line for line
  (its starting values are pinned in the tests). The waveforms' "input filter currents" were L3's and R9's by
  name: they are the series parts between the supply's rails.
- **The tools are engines of kicad2ems's questions (API 2.6).** The rail tool answers a rail's drop and its
  impedance from one run (the drop alone, `--no-pdn`, when only it is asked); near field, circuit and thermal
  one question each. Lists show them as "pcbqs". The circuit says its `fields` option adds field dumps; a
  `thermal.json` that names another tool (Elmer's) is not taken for pcbqs's. Needs kicad2ems 261112.post1.
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
