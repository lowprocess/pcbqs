# PEEC PDN trial

See the [accuracy report](REPORT.md) for measured results. The trial does not
establish a general accuracy improvement and is not recommended as the default.

This experiment routes small-signal PDN impedance through the existing PEEC and
finite-thickness slab implementation. It does not change the default PDN CLI.

`pcbqs.pdn.peec_trial.PEECPDN` takes a `RailPDN` instance to reuse its rail and
component assumptions. It replaces the cavity and heuristic copper/mounting
paths with explicit power/ground copper, pad terminals, PEEC charge continuity,
and skin impedance. VRM R/L and capacitor C/ESR/ESL remain explicit lumped
elements. Each load is excited separately with 1 A small signal; its operating
DC current does not become a shunt resistor.

The initial comparison used one rail net, one ground net, one source, and no
series rail components. The subsequent Jetson trial adds resistive series links
and a mesh-size guard; beads remain unsupported. Unsupported rail trees are
rejected explicitly. The earlier results retain their original source hashes.

## Reproduce

From a pcbqs checkout with dependencies installed:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python tools/trial_pdn_peec.py --workers 2
```

The default body spacings are 2.5, 1.75 and 1.25 mm. PEEC contact patches are
refined to one twelfth of each body spacing. The current PDN path uses its usual
port meshing. The experiment includes 70 µm copper, εr = 4.4, tan δ = 0.002,
a 2 mΩ + 20 nH small-signal VRM, and a 100 nF / 5 mΩ ESR / 0.5 nH ESL decap.
It sweeps 1 kHz–30 MHz and locates the high-frequency negative-to-positive
reactance crossing directly with a root solver.

The three boards are a 20 × 12 mm pair with a 0.3 mm dielectric, the same pair
with a 1 mm dielectric, and a 0.3 mm pair whose ground has an edge-reaching slot.
The slot case places all contacts below the slot. Each board's JSON is included
in `results/`.

## What the comparisons mean

- **Independent DC reference:** solve the power and ground separately with P1
  conduction FEM at 0.25 and 0.125 mm body spacings, then add the known 2 mΩ VRM
  resistance. Compare with the real part of each AC model at 1 kHz. This checks
  copper/contact resistance against a separate discretization of the DC limit.
- **Independent analytical AC reference:** a rectangular-plane modal expansion
  implemented in the trial script without calling pcbqs's conductor or matrix
  kernels. It uses the exact complex slab impedance and disk-averaged ports,
  with a local equipotential-contact self-term correction. Truncations of 320
  and 480 modes per dimension are compared.
- **Current PDN (stock):** its normal cavity, surface-resistance approximation
  and heuristic mounting inductances.
- **Current PDN (mount=0):** a diagnostic removing heuristic mounts, because
  these test contacts sit directly on the planes. This does not alter the
  production PDN implementation.
- **PEEC one-face / two-face / DC copper:** the principal new model plus two
  loss sensitivities. Opposing plane currents motivate the one-face assumption;
  these cases do not bound arbitrary thickness proximity effects.

The analytical AC model neglects edge fringing and approximates the local port
field. PEEC's open-edge fields and the wide-gap board therefore need not agree
with it. Its percentages are **reference-model agreement, not demonstrated
physical accuracy**. There is no analytical AC reference for the slotted board
in this trial; its AC results report mesh sensitivity and solver disagreement.
Its DC resistance is still compared with the independent conduction solve.
No hardware measurements or independent full-wave solves were performed.

`results/summary.json` contains complex-impedance differences, mesh changes,
DC reference errors, modal truncation changes, directly located resonance
frequencies/resistances, and timings. `results/impedance.csv` contains the finest
mesh curves; `results/comparison.png` plots their magnitude and real part.

The separately documented [PEEC / skin coupons](../peec-skin/README.md) check the
bare capacitor and uniform-strip skin limits. Their good agreement does not
establish the same accuracy for a loaded PDN with small contacts and resonances.
