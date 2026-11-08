# Accuracy report: experimental PEEC PDN path

The trial runs successfully, but these results do **not establish a general accuracy improvement over the existing PDN path**. Keep the current backend as the default. The new adapter is useful for further validation, particularly where plane overlap or mounting approximations are questionable.

Three synthetic loaded PDNs were evaluated on three meshes, using 48 frequency samples from 1 kHz to 30 MHz. The principal trial uses layered-dielectric PEEC and a one-face finite-thickness skin slab. A two-face slab and DC-copper PEEC provide loss sensitivities.

## Independent DC-reference error

The reference is a separate sparse P1 conduction solve of power and ground, with the known 2 mΩ VRM resistance added. The 0.25 → 0.125 mm refinement changed total resistance by less than 0.10%. Errors below compare the AC real part at 1 kHz with that DC reference.

| Board | Current PDN error | PEEC + skin error | PEEC copper-only error |
|---|---:|---:|---:|
| tight_planes | 4.31% | 5.12% | 14.06% |
| wide_gap | 4.31% | 5.12% | 14.06% |
| slotted_return | 3.80% | 5.02% | 11.90% |

For the rectangular boards the reference is 3.1445 mΩ including the VRM, or 1.1445 mΩ of copper. PEEC gives 3.3053 mΩ; current PDN gives 3.0088 mΩ. The smaller total-port percentage must not be mistaken for equally accurate copper extraction.

## AC agreement with the analytical reference

The independent modal calculation uses the full complex slab impedance, disk-averaged ports, and an approximate equipotential-port correction. It neglects open-edge fringing. Its 320 → 480 mode refinement changed the sampled complex impedance by at most 0.17% on the tight pair and 0.55% on the wide pair. This establishes numerical stability of that reference model, not its physical completeness.

The resonance below is the directly solved high-frequency negative-to-positive reactance crossing. The stock PDN model includes heuristic mounting terms. A separate mount=0 comparison removes those terms for the direct plane-contact fixture.

| Board | Reference resonance | Current PDN (stock) | Current PDN (mount=0) | PEEC + one-face skin |
|---|---:|---:|---:|---:|
| tight_planes | 16.911 MHz | 17.146 MHz (1.39%) | 18.345 MHz (8.48%) | 17.911 MHz (5.91%) |
| wide_gap | 12.110 MHz | 12.243 MHz (1.10%) | 13.783 MHz (13.82%) | 15.756 MHz (30.11%) |

Parenthesized percentages are reference disagreement, not proven physical frequency error. In particular, the wide-gap result requires a reference that includes fringing before either path can be declared more accurate. The stock mounting correction can partly cancel another model discrepancy; close agreement alone is insufficient validation.

| Board | PEEC median complex-Z disagreement | PEEC maximum complex-Z disagreement | PEEC resistance disagreement at its own resonance |
|---|---:|---:|---:|
| tight_planes | 2.65% | 104.4% | 11.68% |
| wide_gap | 5.27% | 791.7% | 12.72% |

The median is over the stated 48 samples, including extra samples from 5–9 MHz. Large maxima occur near low-impedance troughs, where a frequency shift causes a large pointwise relative difference. They must not be hidden behind the median. No independent AC accuracy number is available for the slotted board.

## Mesh sensitivity and cost

| Board | Last PEEC refinement: median complex-Z change | Maximum change | Primary PEEC run | Current PDN diagnostic run |
|---|---:|---:|---:|---:|
| tight_planes | 0.43% | 5.74% | 41.00 s | 0.217 s |
| wide_gap | 0.48% | 8.49% | 36.98 s | 0.132 s |
| slotted_return | 0.49% | 7.99% | 37.15 s | 0.121 s |

These are recorded end-to-end trial timings, not a controlled kernel benchmark: the cavity timing includes stock, mount=0 and bare diagnostics, while the primary PEEC timing includes geometry, extraction, sweep and resonance location. PEEC is substantially more expensive even on these small boards. Each PEEC fine mesh has about 1,400 triangles. The final 1.75 → 1.25 mm refinement still changes some impedances by 5.7–8.5%, so a blanket 5% AC-accuracy claim is unsupported.

## Validation and limitations

- 15 focused tests passed, including separate load excitation, bare/loaded model reuse, exact lumped termination on ideal plate contacts, and existing PDN regressions.
- A loaded tight-plane conservation check at 1 kHz, 1 MHz, 18 MHz and 30 MHz gave maximum real-power imbalance below 4 × 10⁻¹² relative and charge-continuity residual below 1.3 × 10⁻¹⁵ A for a 1 A peak source. See `results/validation.json`.
- These checks establish solver and adapter consistency, not physical accuracy.
- No hardware or independent full-wave validation was performed. The slotted geometry has only DC reference error, AC mesh sensitivity and solver disagreement in this report.
- The trial supports one rail, ground and source with no series rail parts. Component ESR/ESL/VRM assumptions remain inputs. Via skin effect, arbitrary thickness proximity and part-to-copper magnetic mutuals are excluded.

Before integration, prioritize converged small-contact resistance, a matched finite-plane AC reference or independent full-wave comparison, and mounting/port definitions that can be compared without compensating approximations.

[Method and reproduction](README.md) · [Plots](results/comparison.png) · [Raw results](results/summary.json) · [CSV](results/impedance.csv)
