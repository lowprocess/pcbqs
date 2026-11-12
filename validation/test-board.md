# Validation test board: what to put on it

A list of structures for a board made only to check pcbqs against measurement. Everything in `validation/` so
far is solver against solver; [measurements.md](measurements.md) covers what can be measured on wpthb as built.
The items are ordered by how much each decides something open in the model (the roadmap's Power tools, "Accuracy and validation").
Items 1–4 are where the solvers disagree with each other or with the references. No layout or dimensions are
fixed yet, so there are no predictions per structure: add them (a script per structure in `validation/`) once
the layout exists.

| # | Item | Structure on the board | What it settles | Instrument |
|---|---|---|---|---|
| 1 | AC resistance of thick copper | Straight strips and a shorted loop, in 1 oz and 2 oz, a few widths | One sheet reads 9% low at 10–100 MHz against FastHenry, two faces within −1/+3.5%. Decides the default thickness model (`thickness_sheets`). | VNA or impedance analyser |
| 2 | Trace over a close plane | The same trace 0.1 mm and 1.6 mm above a plane, shorted at the far end | The largest open error: two faces was wrong near a plane (+254% R at 100 MHz in `face_study.py`), one sheet is 19–30% low. | VNA |
| 3 | Loop inductance of a converter-style loop | A copy of a hot loop (pads, pour, vias) closed by a copper strap in place of the capacitor | Board inductance without part ESL, and the slow mesh convergence (wpthb's VHB port: 115 pH at dmax 0.8, 150 pH ± 7% at 0.25–0.6). | VNA shunt-through |
| 4 | Part contact model (C13) | The same loop closed by 0603, 0805 and 1206 zero-ohm links, on pads of different lengths | Where current enters the pad (`pcbqs.contacts`): an inner-edge contact reads 2.7× the end-cap one on wpthb. | VNA |
| 5 | Via inductance and via arrays | Loops closed through 1, 2, 4 and 9 vias at two pitches, plus a pair of overlapping vias | The parallel-filament via model, the polygonal-via simplification and the overlapping-via merge (`pdn.mesh.overlapping_vias`). | VNA; four-wire DC for resistance |
| 6 | DC drop and current sharing | A necked pour and a via-stitched layer change, with Kelvin sense pads | The DC model against copper as fabricated (plating, etch tolerance); so far only against Elmer. | Bench supply and DMM |
| 7 | Capacitance | A plate pair, a microstrip, and two traces 0.1 mm apart, on FR-4 | Thickness adds about 12% that one sheet omits; narrow-gap mutual is 10–13% low; FasterCap's FR-4 results were non-physical, so FR-4 has no numerical reference. | LCR meter |
| 8 | Plane-pair impedance | A bare rectangular power/ground pair with two or three probe points, then with one decap fitted | Cavity resonances and loss; checked only against Palace and the modal sum. | VNA |
| 9 | PDN with a series inductor | A small rail: an inductor, a known ceramic bank, test points across it | The L-against-C peak (wpthb: 1.6–1.8 Ω near 10.5 kHz) and the ceramics' loss of capacitance under DC bias. | VNA with a bias tee, or an injection transformer and a scope |
| 10 | Near field | A simple loop driven from an SMA at a known current | \|H\| shape and level at 1–3 mm without a switching converter's uncertainty. | Loop probe with a spectrum analyser or scope |

## Design in from the start

- **Calibration structures.** Open, short and through at the same connector footprint as the structures, so the
  fixture's inductance can be removed. Without them the sub-nanohenry items (3–5) cannot be read.
- **One variable at a time.** Width, height over the plane, via count: a trend can be compared where an absolute
  value cannot.
- **Known materials.** The fab's measured stack-up and copper thickness, and a long serpentine with Kelvin pads
  for a four-wire reading of the copper's resistivity.
- **Probing.** The same two-port shunt-through landing everywhere (see measurements.md for the method and the
  cable-braid error below 100 kHz).

## Without a VNA

Items 6 and 7 and the low-frequency end of item 1 need only a supply, a DMM and an LCR meter. Items 1–5, 8 and 9
need a VNA or an impedance analyser.
