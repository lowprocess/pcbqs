# Experimental compressed PEEC

The default solver remains dense. The Python API now offers an experimental
hierarchical backend for `MQS(..., capacitance=True)` and the PDN trial adapter:

```python
from pcbqs.pdn.peec_trial import PEECPDN

trial = PEECPDN(legacy_rail_pdn, dmax=1.25, contact_spacing=.1,
                coarsen=4, grow=.5,
                matrix_backend='hierarchical',
                compression_tol=1e-6, compression_leaf=64,
                solver_tol=1e-8, solver_maxiter=2000)
z = trial.z([1e5, 1e6, 30e6])
model = trial._model(False)  # experimental adapter's loaded model
print(model.fill_info, model.solve_info)
```

Mesh dimensions are mm. `coarsen` caps bulk spacing at a multiple of `dmax`;
`grow` controls its growth with distance from geometric features. `fine` /
`dmax_fine` in MQS or `contact_spacing` in PEECPDN protect local detail.
`edge_rings=1` disables inward rings; larger counts add them using `edge_width`
as the reference width. `edge_zone`, a Shapely region, restricts those rings.
These controls preserve the imported outlines, pad constraints and via fans.
They do not automatically provide a port-error estimate.

## Mesh simplification (MQS defaults)

| Option | Default | Effect |
|---|---|---|
| `drop_lone_pads` | `True` | copper pieces under `pad_area` (1 mm²) that touch no terminal and hold at most one via are dropped; the via spans to the next copper layer |
| `pad_sides` | `8` | other small round pads and clearance holes become equal-area regular polygons |
| `via_sides` | `6` | via rings are regular polygons with the via circle's perimeter |

`False`/`None` restores the unsimplified mesh; `q.mesh_info` records counts.
Lone pads carry no lateral current, but in PEEC they carry charge, so set
`drop_lone_pads=False` where small pad capacitance matters.

Measured with 16 threads (i7-13700K):

| Case | Unsimplified | Simplified |
|---|---|---|
| Jetson +1V15 full rail, cold (compressed PEEC) | 25,370 triangles, 186 s, 632 GMRES iterations, 3.4 GiB | 9,879 triangles, **61 s**, 267 iterations, 1.4 GiB |
| Jetson Z(U21) at 1 MHz | 8.8698 + j5.0532 mΩ | +0.26% R, +1.3% X |
| CM5 U702 loop, fine 0.6 mm (dense MQS) | 9,736 triangles, 4.0 s | 7,361 triangles, **1.5 s** |
| CM5 at fine 0.25 mm, R/L at 1 kHz, 1 MHz, 10 MHz | 19,647 triangles, 56 s | 17,876–17,636, 42–45 s; within 0.1–1.2% |

Lone pads were half of the Jetson mesh. Square via rings (`via_sides=4`) are
faster still but bias R/L up by ~1–2% (smaller contact); a bare via vertex is
not offered, as it changed CM5 R/L by 6–18%. The meshes above are far from
converged (CM5 at 0.6 mm is 3–16% from 0.25 mm), so the saved elements are
best spent on a finer `dmax_fine`: simplified at 0.4 mm costs about what the
unsimplified 0.6 mm mesh did and is about twice as accurate.

Via segments of one barrel now use the barrel radius for their mutual
inductance (previously 0.1 µm), so stacked segments add up to the whole
barrel. Vias crossing inner copper were over-inductive before: Jetson X fell
1.6% and the CM5 loop L 0.5% with this fix alone.

The H-matrix partitions full basis-support boxes in 3D. Separated blocks use
checked adaptive cross approximation; nearby blocks remain dense. Failed or
uneconomical compression is subdivided instead of silently truncating rank.
Reciprocal blocks share factors and transpose symmetry, including dielectric
loss. Inductance entries are assembled directly from selected RWG-support
triangle moments, in centered coordinates. Potential entries use the existing
homogeneous kernel or the layered image/spline integration. Triangle charges
are aggregated into terminal/via nodes through sparse area weights.

The implementation does not allocate global triangle-square moments, dense
edge/via inductance, dense charge potential, dense loop/tree projections, PT,
or dense frequency LU factors. Resistance and analytic slab skin corrections
remain sparse. Restarted GMRES applies these operators through sparse
loop/tree transforms, retaining the reference solver's equilibrium-charge split
and low-frequency scaling. A sparse approximate impedance projection and ILU
precondition small solves. Before assembling that surrogate, a conservative
nonzero-count estimate is checked against 128 MiB. Larger cases use an ILU
solve on the original branch graph for loop impedance, followed by a stable
local-potential tree solve. This avoids the projected sparse square matrix;
if graph factorization fails, a loop-diagonal fallback remains available.
This bounds that assembly step, not total RSS or all factorization
workspace. The full compressed operator determines the convergence residual.
Nonconvergence raises rather than returning an unchecked impedance.

With the compiled core (`python -m pcbqs.core.build`), H-matrix products run
natively: on first use the blocks are packed into flat arrays (the blocks are
rebound to views of them, so storage is not duplicated), each block writes its
own contribution rows in parallel, and each output row then sums its
contributions in block order. Products are therefore bitwise identical for any
thread count. Threads default to the CPU affinity/cgroup quota;
`PCBQS_HMATRIX_THREADS` or `H.workers` override it, and `PCBQS_NO_CORE` selects
the NumPy block loop. The packed cache is dropped when pickling and rebuilt
after loading. Scratch holds one row per block contribution, typically about
100 per unknown and column, and is reused between products.

On the 30 x 20 mm two-layer layered-dielectric case (5,816 triangles, 8,596
RWG unknowns, 11,079 inductance blocks, 141 MB), one inductance product went
from 59 ms (NumPy loop) to 3.9 ms with 16 threads on an i7-13700K. A
three-frequency GMRES solve fell from 45.7 s to 6.0 s (medians of three), with
identical iteration counts and impedances.

When the fill backend is OpenMP, compressed entry blocks are also assembled
natively: one call per requested block finds the distinct support triangles,
classifies and integrates each canonical triangle pair once, and combines the
RWG (or homogeneous potential) entries. Products are rounded before addition
as in NumPy, and one out-of-line pair integrator serves every kernel, so the
native blocks are bitwise identical to the NumPy assembly and independent of
thread count. Other fill backends (`numpy`, `threaded`, `cupy`) keep the
NumPy assembly. `fill_info['magnetic']['native_entries']` records which ran.

On the same case (medians of three), inductance construction fell from 30.7 s
to 4.9 s, and homogeneous potential construction from 3.8 s to 1.1 s; total
homogeneous assembly from 34.7 s to 6.1 s. With the layered dielectric, total
assembly fell from 41.6 s to 17.0 s; the layered image/spline potential (11.8 s)
still uses the NumPy image loop. Of the remaining inductance time, about 2 s is
native integration and the rest is the Python ACA pivot loop.

The OpenMP backend also builds the H-matrices natively. Compiled entry
providers (RWG inductance, homogeneous potential and the layered image kernel)
are passed to `HMatrix(..., native=provider)`; the tree is traversed one level
at a time, and all admissible blocks of a level run ACA in parallel, followed
by its dense leaves. The ACA rules match the NumPy implementation (rank cap,
16-row check sets, a second independent check before acceptance), but check
rows are drawn from a per-block SplitMix64 stream seeded by the block, so the
factors are independent of thread count and differ from the NumPy build
within the compression tolerance (impedances here moved by 1e-7 to 7e-7 at
`compression_tol=1e-6`). `H.info['native_build']` records the path;
`PCBQS_NO_CORE` or other fill backends keep the NumPy build.

With native ACA (medians of three, 16 threads), total assembly fell from
6.1 s to 0.68 s (homogeneous) and from 17.0 s to 0.92 s (layered); GMRES
iteration counts were unchanged.

`fill_info` reports matrix payload/storage and evaluated entry counts.
`solve_info` records GMRES iterations, relative operator residual and sparse
preconditioner storage at each frequency. Storage figures are not process RSS
or peak construction memory. `factor_cache_bytes` applies to dense factors;
the iterative path rebuilds its sparse preconditioner per frequency.

Experimental iterative controls can also be set on an MQS model before solving:
`q.solver_preconditioner = 'auto'`, `'graph'` or `'tree'`, and
`q.solver_restart = 100` (a positive number of GMRES basis vectors). The default
automatic choice protects projected-surrogate storage. Larger restarts use more
Krylov workspace and can reduce stagnation. These choices do not change kernels
or their checkpoint fingerprints.

This is an initial scalable path, not a guaranteed memory/runtime bound.
Fundamental-cycle transforms can still have substantial support on large
graphs; approximate graph solves and the tree/loop-diagonal fallback may need
more iterations. Floating passive components can also
make the equilibrium solve expensive. Near blocks between overlapping planes
can dominate; entry integration and iterative convergence can dominate time.
Compression and iterative tolerances must be tightened independently from
mesh and dielectric-image tolerances. Check port impedance, resonance shifts,
continuity, reciprocity and passivity against dense small-board references;
a block tolerance or small linear residual alone is not a physical error bound.

Reproduce the separate-process benchmark (same meshes for both backends):

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python tools/benchmark_peec_memory.py
```

See [the investigation](../examples/peec-memory/REPORT.md), accompanying JSON
and scripts for measurements. The production PDN CLI still uses its existing
backend; the new path is opt-in through MQS/PEECPDN.

The [batching and checkpoint follow-up](../examples/peec-memory/OPTIMIZATION.md)
describes faster on-demand entry evaluation and resuming the Jetson benchmark
without rebuilding completed operators. Checkpoints belong to the benchmark
workflow; the API does not implicitly write caches.
