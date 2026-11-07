# Validation at converter frequencies

A full-wave time-domain board solve at the same low frequencies is not the
primary reference for the quasi-static tool. Validate the individual physical
terms and the observables, with independent static solutions and measurements.
Agreement between CPU and GPU checks implementation consistency; it does not
establish that the board, current source or package model is physically right.

## Numerical convergence on the actual crop

Keep terminals, source amplitudes, material parameters and observation points
fixed. Change one discretisation control at a time:

1. Reduce local `--fine` spacing, then coarse `--dmax` spacing. Use **Preview
   mesh** to inspect copper coverage, pads, via fans, long edges and thin
   triangles on each layer before filling matrices. Compare source R/L, complex
   dipole moment, and complex H at the same physical points. A peak over a
   changing scan grid is not a fixed-point comparison.
2. Increase `--margin` without changing mesh spacing or source definitions.
   Check whether the return current reaches the crop boundary and whether
   impedances and fields stabilise. A ground plane can need refinement where
   the return bends or crowds; uniformly refining remote copper is expensive.
3. At fixed geometry, tighten the close-pair quadrature (`outer_levels`) and
   compare with the all-analytic-inner-integration reference (`near=inf`) on
   affordable subproblems. The outer integral is still numerical. These are API
   controls; the converter CLI currently exposes mesh/crop controls.
4. Refine FFT deposition independently and compare sampled points with direct
   Biot–Savart evaluation. `--field-tol` checks the deposition grid at each
   requested height/frequency, not copper mesh, source geometry or material error.

The existing CM5 0.8-to-0.6 mm local-mesh comparison changes R/L by about 2%;
a 3-to-4 mm crop-margin change shifts them by about 0.5%. Those two comparisons
are sensitivity evidence, not a demonstration of asymptotic convergence.
Report tolerances against the intended engineering observable rather than only
matrix-norm differences.

## Independent low-frequency references

- **Resistance:** compare simple conductors with length/(conductivity × area),
  then compare the low-frequency magnetic solve with a separate nodal DC/FE
  solution using identical pads, explicit return copper and excitation.
  Sharing the layout with the PDN page helps construct that comparison; the
  switching pulse amplitude is not automatically a DC load current.
- **Inductance and magnetic fields:** retain straight-strip/plane image-theory,
  filament mutual-inductance, closed-loop and dipole-limit cases. Vary length,
  width, spacing, height and return-path geometry, not just one mesh.
- **Capacitance/E:** parallel plates, dielectric scaling and layered-interface
  continuity provide references for the existing electrostatic/PEEC API.
  Converter CLI field plots currently use the magnetic input-loop model.
- **Structure:** check charge/current continuity, reciprocal moment matrices,
  positive resistive loss and magnetic energy. Check excitation linearity and
  the PEEC low-frequency limit separately from board-level model assumptions.

Small cropped full-wave comparisons remain useful where their operating band
and runtime are practical, but need not gate every converter-frequency check.

## Physical correlation and model capability

Start with a controlled coupon/loop and fixture-corrected complex impedance,
then a converter crop. Compare both magnitude and phase where available. Use
calibrated near-field probe response, known height/orientation and known source
current before interpreting absolute field levels. Source waveform, return
connections, capacitor ESR/ESL and package geometry must be recorded.

Current limitations to keep separate from numerical error include the assumed
input-current pulse, DC sheet loss in the CLI, simplified part filaments,
omitted part/copper mutuals and output-inductor leakage. The API already has
slab conductor loss and layered PEEC electrostatics; exposing those in the
converter workflow adds capability but also needs the corresponding reference
and convergence checks. Imported display-only component silhouettes do not add
electromagnetic package bodies to the model.

## Integration with power integrity

Use a shared board, copper/net selection, component catalogue, crop/mesh preview
and result viewer. Present DC drop, PDN impedance and switching-current/near-field
results as related analyses with explicit excitations and frequency ranges.
The GUI now transfers board, rail and ground reference from Near field to Power
Integrity, without launching a DC solve or inventing loads. A future common
workspace should preserve per-analysis source/load settings and results instead
of silently treating a PDN plane-pair approximation as an MQS copper model.
