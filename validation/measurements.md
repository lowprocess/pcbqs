# Bench measurements on wpthb (the roadmap's Measurements item)

All pcbqs validation so far is solver against solver. These measurements check the model against the as-built
board (`~/Desktop/wpthb`). They are ordered by the equipment they need and by what they test. Predictions to
compare against are in `validation/results/` (scripts named below).

## 1. VHB impedance, unpowered (VNA or impedance analyser, shunt-through)

- **Port:** TP7 (VHB, 1 mm pad) to TP6 (VHB_GND, 1 mm pad). The six VHB ceramics (C19–C24) sit in a column
  between them.
- **Fixture:** two semi-rigid coax pigtails. Port 1's centre goes to TP7 and port 2's to the same pad; both
  shields go to TP6. Keep the joints under 1 mm. Two-port shunt-through gives Z = 25 Ω · S21 / (2 (1 − S21)),
  which drops the pigtails' series contact from the reading.
- **Ground loop:** the VNA's two shields join through the board, so below ~100 kHz a common-mode choke or ferrite
  sleeve on the port-2 cable is needed (the usual shunt-through cable-braid error).
- **Board state:** unpowered, input open. Note whether the coil is connected.
- **Sweep:** 100 Hz to 100 MHz, IF bandwidth ≤ 100 Hz, port power around 0 dBm.

What each band tests:

| Band | Expected | What it tests |
|---|---|---|
| 5–20 kHz | Peak, 1.6–1.8 Ω near 10.5 kHz (`python -m pcbqs.pdn`; regression `wpthb_pdn`) | L3 against the VHB ceramics' *unbiased* C; the CA polymer caps' ESR as damping |
| 1–3 MHz | Minimum of a few mΩ | The ceramics' ESR (vendor data) |
| 3–100 MHz | Inductive: 2.9 mΩ near 7 MHz, 30 mΩ at 35 MHz, 87 mΩ at 59 MHz, 147 mΩ at 100 MHz (dmax 0.3); about 230 pH in all, 150 pH ± 7% of it the board | The parts' ESL plus the board's mounting and spreading inductance, the pcbqs copper model, and part contacts (C13) |

The unpowered measurement sees the caps' unbiased capacitance. At 24 V an X5R 10 µF 50 V 1206 keeps perhaps
40–60% of it, so the 10.5 kHz peak moves up by about 1.3–1.6× in operation. Compare unbiased with unbiased.

To separate the board from the parts in the 3–100 MHz band:
- (a) As built.
- (b) Optionally, all ceramics but C21 removed, and C21 replaced by a 1206 zero-ohm jumper or a soldered copper
  strap of known size.

With (b), the reading is the board's loop plus the jumper's own inductance. pcbqs gives the board's part:
214 pH at 100 MHz (dmax 0.3), and about 0.5 mΩ of copper resistance at 0.3 MHz. Subtract the jumper's
inductance: about 0.4–0.5 nH for a 1206 zero-ohm link, or better, use a copper strap whose inductance can be
computed. With C21 left as its 10 µF cap, the same run gives 52 mΩ at 0.3 MHz and 386 mΩ at 100 MHz, using the
placeholder ESL.

The contact model hardly matters here: a whole pad and the cap-and-fillet contact differ by 0.5%. An inner-edge
contact would read 2.7× higher, so the board's copper model is what this checks. That includes the slow mesh
convergence (115 pH at dmax 0.8, about 150 pH ± 7% at 0.25–0.6).

## 2. VHB ripple, powered (oscilloscope)

- **Operating point:** the regression case, 90 kHz, 10 W, or the setup actually run.
- **Probe:** the scope across TP7/TP6, with a tip-and-barrel probe or a 50 Ω coax pigtail into the scope's
  50 Ω input, AC-coupled. A ground-lead probe adds tens of nH and swamps the edges.
- **Readings:** the settled ripple, sampled or low-pass filtered to about 20× fsw, against `bus_ripple_V`, and
  the switching-edge spikes against `bus_spike_pp_V`.
- **Dependence on parasitics:** the ripple depends on the placeholder MOSFET and capacitor parasitics, so a
  mismatch here is not yet a copper-model verdict. The 6% one-sheet against two-faces difference is below what
  this reading can resolve until the parts are vendor models.

## 3. Near field, powered (H-field probe, 1–3 mm above the bridge)

- **Probe and scan:** a calibrated loop probe (or a probe with a known antenna factor), scanned on a 1 mm grid at
  1 and 3 mm above F.Cu over the bridge and the VHB caps, at fsw harmonics.
- **Comparison:** `python -m pcbqs.mqs` for wpthb (converter selection: the bridge), as CM5's regression does.
  Compare the shape of |H| first, and the absolute level only with probe calibration.

## Status

- A board made for validation is planned: [test-board.md](test-board.md) lists what to put on it.

- Predictions for test 1: `validation/wpthb_vhb_port.py` (results in `validation/results/wpthb_vhb_port.json`),
  with the part values as placeholders until the vendor models (TDK CGA5L3X5R1H106M160AB, Murata
  GRM219R61H105KA73J) are in.
- Tests 2 and 3 need the bridge running. Test 3 also needs the bridge set up as an MQS converter case.
