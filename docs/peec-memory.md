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
They do not automatically provide a port-error estimate or simplify CAD.

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
nonzero-count estimate is checked against 128 MiB. Larger cases use a loop
diagonal and local-potential tree solve, avoiding the projected sparse square
matrix. This bounds that assembly step, not total RSS or all factorization
workspace. The full compressed operator determines the convergence residual.
Nonconvergence raises rather than returning an unchecked impedance.

`fill_info` reports matrix payload/storage and evaluated entry counts.
`solve_info` records GMRES iterations, relative operator residual and sparse
preconditioner storage at each frequency. Storage figures are not process RSS
or peak construction memory. `factor_cache_bytes` applies to dense factors;
the iterative path rebuilds its sparse preconditioner per frequency.

This is an initial scalable path, not a guaranteed memory/runtime bound.
Fundamental-cycle transforms can still have substantial support on large
graphs; the tree/loop-diagonal fallback may need more iterations. Floating passive components can also
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
