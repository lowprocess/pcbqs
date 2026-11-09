# Validation against independent solvers

pcbqs's power tools checked against solvers built on other methods, on cases with closed forms and on real
boards. Each script runs pcbqs and the reference over a ladder of mesh sizes and draws both converging.
Last run 2026-11-09. Reference tools: Elmer 26.2, Palace 0.18.1, FastHenry 3.0.1.

| Feature | pcbqs | Reference | Script |
|---|---|---|---|
| DC drop (`pdn` DC) | 2D copper sheets + via barrels (`pdn.dc.Network`) | Elmer 26.2 `StatCurrentSolver`: 3D copper slabs and barrels, direct solve | `dc_elmer.py` |
| Loop inductance, near field (`mqs`) | PEEC on the copper sheets (`MQS`) | Palace 0.18 driven FEM: PEC copper sheets, PEC via rods, air box, lumped port | `mqs_palace.py` |
| Copper R(f), L(f) (`mqs`) | the same, slab loss model | FastHenry 3.0: volume filaments across width and thickness (skin and proximity effect) | `mqs_fasthenry.py` |
| Plane-pair impedance (`pdn` PDN) | 2D cavity (`pdn.cavity.PlanePair`) | Palace driven FEM: dielectric between PEC planes, open sides, lumped ports; analytic modal sum | `pdn_palace.py` |

## Results (finest meshes)

| Case | pcbqs | Reference | Difference |
|---|---|---|---|
| DC: 50 × 5 mm strip | 3.9409 mV | Elmer 3.9406 mV; closed form 3.9409 mV | +0.01% |
| DC: two plates, one via (r 0.15 mm) | 2.006 mV | Elmer 2.097 mV | −4.4% |
| DC: strip, 2 × 2 vias, strip | 4.256 mV | Elmer 4.276 mV | −0.45% |
| DC: wpthb VHB, L3 → Q1 / Q2 | 0.3828 / 0.6168 mV | Elmer 0.3835 / 0.6179 mV | −0.2% |
| DC: Jetson +1V15, R7 → U21 (2 A) | 7.620 mV | Elmer 7.733 mV (still falling with h) | −1.5% |
| L: 20 × 10 mm strip loop, 1 / 10 / 100 MHz | 38.54 / 37.64 / 37.48 nH | FastHenry 38.69 / 37.87 / 37.59 nH; Palace (zero-thickness PEC) 38.14 | −0.4 / −0.6 / −0.3% |
| L: trace over plane, via return | 12.25 / 12.06 / 12.01 nH | FastHenry 12.21 / 11.91 / 11.76 nH; Palace 12.16 | +0.3 / +1.3 / +2.1% |
| R: strip loop, 1 / 10 / 100 MHz | 32.9 / 43.6 / 114 mΩ | FastHenry 32.8 / 49.1 / 130 mΩ | +0.4 / −11 / −12% |
| R: trace over plane | 15.5 / 17.8 / 41.7 mΩ | FastHenry 15.8 / 23.2 / 63.6 mΩ | −1.5 / −23 / −34% |
| \|H\| 1 mm above loop / trace | centre 66.3 / 103.7 A/m per A | Palace 69.9 / 99.6 | −5% / +4% |
| Plane pair 40 × 30 mm, 10 MHz–3 GHz | \|Z21\| | Palace; modal sum | ≤ 0.4% |
| L-shaped plane pair | \|Z21\| | Palace | ≤ 0.8% |
| Jetson +1V15, In4.Cu over In3.Cu | \|Z21\| | Palace | ≤ 0.25% |

![DC, Jetson +1V15](images/dc_jetson_1v15.png) ![DC, wpthb VHB](images/dc_wpthb_vhb.png)
![Strip loop](images/mqs_loop.png)
![Trace over plane](images/mqs_plane_return.png)
![R and L against FastHenry](images/mqs_fasthenry.png)
![Plane pairs](images/pdn_cavities.png)

## What it found and changed in pcbqs

- **DC terminals were not equipotential** (`pdn/dc.py`). The ties from a pad to its mesh vertices shared
  1e6 S in total, but current enters a pad through the few vertices on its edge. Pads sat ~20 µV per amp off
  equipotential: +1% on the strip, growing as the mesh was refined. Ties are now 1e6 S per vertex, and the
  strip matches its closed form to 0.003% at every mesh.
- **DC meshes had slivers at vias** (`_numerics/geo.py`, `pdn/dc.py`). Triangle ran with an area limit but
  no minimum angle, so a via's octagon fanned out to vertices 2 mm away whatever `dmax` was. The linear
  elements then underestimated each via junction's spreading resistance by ~0.1 mΩ, which refinement could
  not fix. DC meshes now use a 25° minimum angle; the RF and near-field meshes are unchanged.
  - Jetson +1V15 drop: 5.78 → 6.18 mV (+7%).
  - Its PDN impedance peak, which uses the same pad-to-via conduction for mounting inductance: +5.7%.
  - wpthb: −0.05%, because its series parts dominate.
- DC loss is now the terminals' input power, not V·G·V, which cancelled across the stiffer ties.

## What remains different, and why

- **Single small via, −4.4%.** The plates agree with Elmer when the via is replaced by a round pad of the same
  radius. The rest is the junction: Elmer's barrel meets the copper slab through its end face and wall, while
  pcbqs ties it to the ring of a 2D sheet. On real boards, where vias are a fraction of the path, this is
  0.2–1.5%.
- **Inductance agrees, |H| within ~5%.** Against FastHenry, which resolves the copper's thickness, pcbqs is
  within 0.6% on the loop and 2% over the plane. Palace's zero-thickness PEC sheets sit ~1% above both, so the
  gap to Palace is the thickness, not a pcbqs bias.
- **AC resistance is low once the skin depth is below the copper thickness** (above ~3 MHz for 35 µm). The
  slab model spreads the current over one or both faces (`conductor_faces`) of every sheet alike. An isolated
  strip carries it on both faces and its edges, while a trace over a plane and the plane under it crowd it onto
  the facing surfaces. The result is 11–12% low on the loop and 23–34% low over the plane at 10–100 MHz, and
  `conductor_faces=1` overshoots instead (+19 / +70% and +8 / +31%). Below ~1 MHz pcbqs agrees to 1.5%.
- **A pcbqs terminal is one node.** A part attaches at its pads as a whole, so current crosses a pad
  instantly. In a small loop with large pads this lowers inductance: the 20 × 10 mm loop read 37.4 nH with
  1 mm pads against 38.8 nH with 0.25 mm ones. Here terminals are 0.1 mm slivers at the gap, matching
  Palace's gap port. On real converter loops the part-to-pad model (filament at `part_h`) matters at the
  same scale.
- **Plane-pair Z11 differs from Palace by a constant 13 pH (0.1 mm) or 33 pH (0.2 mm).** This is a port-shape
  effect: Palace's flat strip port against pcbqs's disc. Z21, the cavity itself, agrees.

## In progress

- **Capacitance against FasterCap** (`cap_fastercap.py`; FasterCap 5 built headless; wxWidgets 3.2 needs the
  version in its CMakeLists changed). Plates in air agree to ~1.5%. On traces over a plane pcbqs is 20–45% low,
  probably partly FasterCap's 35 µm side walls; a zero-thickness run will tell. The FR-4 input still gives a
  non-physical matrix. Give pcbqs small terminals: a terminal has one charge density, and a whole-conductor
  terminal removed half of a narrow trace's capacitance.
- **Two face sheets for AC resistance** (`MQS(thickness_sheets=2)`, C14): the strip loop's 10–100 MHz error
  falls from −20% to −12%. Mesh limits remain.

## Not done yet

- **A real converter loop (CM5 U702) against Palace.** It needs the regulator's VIN and GND pads and the
  capacitors described the same way in both tools, with the terminal-pad effect above taken into account.

- **Measurement.** VNA shunt-through on a PDN, and a near-field probe at 1–3 mm, on wpthb.

## Running

```sh
export PYTHONPATH=src:../kicad2ems/src
python validation/dc_elmer.py [strip via_single via_hop wpthb_vhb jetson_1v15]    # ElmerSolver, ElmerGrid, gmsh
python validation/pdn_palace.py [rect lshape jetson_1v15]                          # Palace (podman image or native)
python validation/mqs_palace.py [loop plane_return]
python validation/mqs_fasthenry.py [loop plane_return]                             # after mqs_palace.py
python validation/dc_elmer.py --plot        # redraw from saved results (also mqs_palace.py --plot)
```

FastHenry 3.0 (MIT, FastFieldSolvers' Linux fixes) is K&R C that GCC 15+ rejects by default (C23: `f()` means no
arguments; implicit declarations are errors). It builds as the C89 dialect:

```sh
git clone --depth 1 https://github.com/ediloren/FastHenry2.git ~/opt/src/FastHenry2
cd ~/opt/src/FastHenry2/src/fasthenry
make fasthenry CC=gcc CFLAGS="-O2 -std=gnu89 -fcommon -w -DFOUR"      # -> ../../bin/fasthenry
```

Its input's first line is a title (a `.units` there is silently ignored), node names start with N, and it exits
non-zero after a successful run; `mqs_fasthenry.py` handles all three.

Run data go to `PCBQS_VALIDATION_OUT` (default `~/pcbqs-bench/validation`); figures to `images/`. Palace is found
as `K2E_PALACE` (an executable) or `K2E_PALACE_ROOTFS` (an unpacked image, default `~/opt/palace/rootfs`), as in
kicad2ems-palace. The biggest runs use about 1.1 M tetrahedra (Elmer, Jetson) and 0.5 M (Palace, 2 MPI ranks) in
under 32 GB; pcbqs's dense PEEC limits the MQS cases to ~12 k triangles.
