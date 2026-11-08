# Reducing full-rail PEEC memory

Follow-up: the opt-in hierarchical backend and grading controls are now
implemented. See [usage and limits](../../docs/peec-memory.md) and
[measured backend comparisons](IMPLEMENTATION.md). The investigation below
records the measurements and recommendations made before that implementation.

## Finding

Use graded meshing **and** an end-to-end hierarchical operator. Grading is a
useful first step on simple planes, but it barely changes the Jetson +1V15 rail
mesh. Compressing the initial triangle matrices alone will also be insufficient:
the current PEEC implementation subsequently constructs several dense matrices
and a dense frequency factorization.

This investigation adds reproducible experiments and recommendations. It does
not implement a compressed production solver or claim a completed full-rail
Jetson PEEC result. All GB figures below are decimal GB of array payload, not RSS.

## What was measured

The Jetson case uses the same board, source L5, ground source U69, load U21,
32 terminal groups and 25.005 x 19.45 mm ten-layer crop as the earlier feasibility
study. See [provenance and previous results](../jetson-pdn/REPORT.md).
No interaction matrices were allocated for these full-rail mesh experiments.

| Mesh setting, base spacing 3 mm | Triangles | Four moment matrices |
| --- | ---: | ---: |
| Existing mesh | 25,529 | 20.86 GB |
| Bulk growth cap 2x, 4x or 8x | 25,370 | 20.60 GB |
| 4x growth + 5 micrometre outline simplification | 25,104 | 20.17 GB |
| 4x growth + 10 micrometre outline simplification | 21,996 | 15.48 GB |
| 4x growth + 25 micrometre outline simplification | 15,406 | 7.60 GB |

The baseline has 30,372 RWG edges and 20,952 charge nodes. One dense real
RWG inductance matrix alone is another 7.38 GB; one complex charge-potential
matrix is 7.02 GB. They are followed by projected matrices and frequency solves.
Thus even the most simplified mesh above cannot fit the present algorithm in
the available 8 GiB process environment.

All mesh variants retained 459 via segments and 32 terminal groups. That count
check is not proof of equivalent connectivity, current distribution or impedance.
The simplification experiments use Shapely's topology-preserving simplifier, but
still move conductor edges and change local widths, clearances and capacitance.
They are sizing experiments, **not approved accuracy-preserving replacements**.

For contrast, two 20 x 12 mm two-plane fixtures were actually solved with PEEC,
the layered dielectric kernel, and finite-thickness slab skin impedance. Base
spacing was 1.25 mm, with 0.1042 mm contact regions, eight frequencies spanning
100 kHz to 30 MHz. Changing only the bulk growth cap to 4x gave:

| Fixture | Triangles before / after | Solve study time before / after | Largest relative complex Z change |
| --- | ---: | ---: | ---: |
| Rectangular planes | 1,436 / 552 | 15.54 / 4.79 s | 4.97% |
| Slotted return | 1,391 / 540 | 13.64 / 4.08 s | 5.87% |

Times include meshing, fill and eight frequency solves on two fill workers,
with BLAS restricted to one thread. They are single runs, not timing statistics.
The changes are relative to the original mesh, not physical error bounds or a
converged reference. Eight samples do not establish resonance-location accuracy.
The initial moment storage falls about 85% on these fixtures, but that reduction
does not describe all process memory. Both variants had positive port resistance
at the sampled frequencies; this alone is not a passivity proof.

## Meshing strategy

The private numerical mesher already includes the relevant `mom25d` facilities:
`coarsen`, `grow`, `fine`, `edge_zone`, edge rings, `gap_rings` and structured
trace runs. `pdn.mesh.mesh_sheet` exposes bulk coarsening but uses `ncross=1`
(no inward edge rings). `MQS` currently does not pass a coarsening option.

The current spacing field is `min(coarsen*dmax, dmax + grow*distance_to_feature)`.
Corners, outline bends, pad constraints and vias are features. Original CAD
vertices remain in the mesh, and each via has a constrained fan on each layer.
On this densely detailed Jetson crop there is almost no feature-free interior
left for larger cells. Raising the growth cap beyond 2x consequently does nothing.

A useful next mesh interface would separate bulk spacing, contact spacing,
edge-ring width/count, growth rate and a spatial refinement region. Preserve
detail at source/load/decap contacts, via entries, necks, slots and the return
paths beneath current-carrying copper. Refine those paths using current/charge
variation or a port-impedance error estimator across representative frequencies.
Refining every plane perimeter indiscriminately can increase the unknown count.
Edge detail alone cannot resolve a broad return-current pattern inside a plane.

CAD cleanup needs electrical safeguards: preserve holes, nets, terminal/via
incidences and minimum neck/clearance dimensions. Check DC resistance, graph
connectivity, AC port impedance and resonance shifts before accepting a tolerance.
For this case, cleanup complements compression rather than replacing it.

Skin depth is through the copper thickness; the analytic slab model already
handles that dimension. It does not require globally shrinking in-plane cells to
skin depth. In-plane edge and proximity current variation still needs resolution.

## Compression evidence and choice

`memory_study.py` evaluates sampled 384 x 384 blocks directly with the current
triangle-pair `FillEngine`, and measures best truncated-SVD ranks. It never
forms a full rail kernel. See the accompanying JSON for all moment results.
Area-normalized scalar blocks remove panel-area weighting so small panels are
not hidden simply by having small integration weights. The reference error is
relative Frobenius error against the current numerical integration, not exact
analytic physics. These are **free-space** kernels, not the complete layered
dielectric kernel or the final PEEC system.

Well-separated panels are strongly compressible. Adjacent/overlapping plane
regions need much higher ranks; low-rank factors can cost more than a dense
block. Use a geometric admissibility test based on full panel/support bounding
boxes in three dimensions, and subdivide near interactions until they become
small dense blocks. Keep self, shared-edge, via/contact and close plane
interactions accurate. A layer-only partition is insufficient because neighboring
planes overlap over most of their area.

| Sampled block | Area-normalized M0 rank at 1e-6 relative error |
| --- | ---: |
| Separated regions on F.Cu | 9 |
| Touching regions on F.Cu | 47 |
| Overlapping F.Cu / In1.Cu, 0.12 mm separation | 317 |

For the separated block, rank-9 factors use 21.3x less array storage than the
384 x 384 dense block. Rank-317 factors for the overlapping block would instead
use 1.65x more storage, so that block should be subdivided or retained dense.
This is a block experiment, not a whole-matrix compression ratio. The four raw
moment blocks had ranks 8-9 in the separated case; centering and RWG projection
can change approximation errors and need their own checks.

Recommended implementation: an H-matrix with adaptive cross approximation
(ACA) as the first scalable backend, retaining dense near blocks and low-rank
far blocks. ACA requests selected rows/columns instead of forming a whole block.
The existing entry evaluator is a useful starting point. Use paired reciprocal
blocks and verify approximation residuals; an unchecked ACA stopping criterion
can miss localized interactions. Evaluate the actual layered potential kernel
at its requested entries, including its image expansion.

H-matrix storage and application can approach O(k N log N) for favorable
geometry/ranks. H2 or Laplace FMM can improve this further using shared cluster
bases, but layered images and preconditioning add implementation work. These
are algorithmic expectations, not a measured full-rail memory or runtime promise.

`mom25d` already has entry-based HODLR and block-low-rank preconditioning:

- [HODLR, inspected revision](https://github.com/lowprocess/mom25d/blob/77509a598561e79c292e65369cec10c1816fc25a/src/mom25d/hodlr.py)
- [BLR, inspected revision](https://github.com/lowprocess/mom25d/blob/77509a598561e79c292e65369cec10c1816fc25a/src/mom25d/blr.py)

HODLR is a useful prototype, but its sibling blocks can contain touching
surfaces and large overlapping regions, so ranks can grow considerably. A
geometrically admissible H-matrix is the better default for this layered case.
BLR also offers a useful pattern: approximate compressed factorization as a
GMRES preconditioner, with residuals evaluated against a more accurate operator.
Neither module is a drop-in replacement for the present PEEC equations.

## Changes needed through the solve

1. **Stop materializing four global moment matrices.** Generate bounded blocks
   directly into compressed current/potential operators, or apply moments as
   operators through sparse RWG transforms. Center local coordinates to limit
   cancellation. Free temporary blocks promptly. This also reduces dense peaks
   on small cases, but by itself does not solve full-rail scaling.
2. **Keep compression through PEEC assembly.** `mqs._edge_matrices` builds dense
   L, `_set_potential` builds dense P, and `peec.prepare` constructs dense
   projected R, L, electrostatic and PT arrays. `peec.solve` builds and factors
   another dense complex system. Every stage needs an operator/block interface.
   Keep conductor and skin resistance sparse; retain exact sparse incidence.
3. **Avoid explicit dense fundamental-cycle projections.** Their long support
   obscures geometric locality and can make even sparse resistance expensive.
   Prototype a scaled mixed branch-current/charge formulation, or apply the
   existing loop/tree transforms implicitly. Preserve its low-frequency charge
   equilibrium treatment, floating-conductor constraints and voltage reference.
4. **Use preconditioned GMRES or a hierarchical factorization.** Start with an
   accurate sparse/dense near-field block preconditioner and reuse geometry
   across frequencies. Preconditioning must handle tightly coupled planes and
   low-frequency scaling. Matrix-free evaluation alone removes storage but
   still costs O(N^2) per product unless coupled to compression/FMM.
5. **Validate before advertising full-rail accuracy.** Compare dense small-board
   solutions, layered capacitance, DC limits, charge/current continuity,
   reciprocity, energy/passivity and port impedances. Independently tighten
   mesh, kernel/compression and iterative tolerances, especially near resonances.
   A 1e-6 block tolerance is not a 1e-6 impedance-error guarantee.

Lower precision and memory-mapped dense arrays offer only constant-factor
relief and leave dense factorization cost. Mixed precision may help a
preconditioner with double-precision residual correction. Dropping weak mutuals
without an error-controlled approximation can damage cancellation, reciprocity
and passivity. Port/model-order reduction helps repeated sweeps after a model
is built; it does not remove the initial full-order memory barrier. A hybrid
local-PEEC/plane solver is another practical option, but is a different model
from full-rail PEEC.

## Reproduce

From the pcbqs checkout, with dependencies installed:

```sh
PYTHONPATH=src:../kicad2ems/src OPENBLAS_NUM_THREADS=1 python examples/peec-memory/graded_accuracy.py
JETSON_PDN_DIR=/workspace/jetson-pdn PYTHONPATH=src:../kicad2ems/src OPENBLAS_NUM_THREADS=1 python examples/peec-memory/memory_study.py
```

The second command requires the imported Jetson `board.json` described in the
previous study. Meshing overrides are process-local experiments; no production
solver defaults or external dependencies were changed.
