# Batched entries, checkpoints and full-rail convergence

This continues the [initial implementation](IMPLEMENTATION.md).
See [the profiling breakdown](PROFILING.md) for cold-stage timings and separate
construction/solver sampling profiles.

## Full-rail result

The complete 25,370-triangle cropped Jetson +1V15 PEEC model now constructs both
operators and converges at 1 MHz within the 8 GiB environment. Peak process RSS
was 2.85 GiB. Driving-point impedance at U21 is **8.880 + j 5.137 milliohms**
(magnitude 10.26 milliohms), including the configured VRM and component RLC
assumptions. This is a numerical model result, not hardware validation.

| Cold-run stage | Time | Operator storage |
| --- | ---: | ---: |
| Inductance | 374.7 s | 882.0 MiB |
| Layered potential | 2,537.8 s | 1,126.4 MiB |
| 1 MHz solve and preparation | 764.6 s | 21.8 MiB preconditioner |

Total time from import through the solve was 3,695.5 s (61.6 minutes), with two
fill workers and single-thread BLAS. The first solve used the automatic
projected-ILU preconditioner: 631 iterations, 2.15e-8 true relative residual,
and 5.05e-16 A maximum charge-continuity residual. The potential panel kernel
alone would require 9.59 GiB densely; the compressed aggregated operator uses
about 1.10 GiB. No global dense moments or dense projected PEEC solve were used.

The cold build is still expensive: it evaluated about 1.75 billion image/panel
interactions. Both operators are checkpointed. A subsequent process loaded
both in 18.2 s including board import and meshing, enabling solver refinement
without repeating construction. Timings are single samples with other checks
occasionally sharing the CPU; they are not controlled performance statistics.

Raw completed stages are in [jetson-compressed-complete.json](jetson-compressed-complete.json).

## Cached solver refinement

All runs below use the identical saved operators and 1 MHz excitation.

| Selection | GMRES restart | Requested tolerance | Iterations | True relative residual | Preconditioner | Time after potential available |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Automatic, projected ILU | 100 | 1e-7 | 631 | 2.15e-8 | 21.81 MiB | 764.6 s |
| Graph ILU | 100 | 1e-8 | 660 | 9.02e-9 | 4.09 MiB | 963.8 s |
| Automatic, projected ILU | 300 | 1e-8 | 424 | 6.36e-9 | 21.81 MiB | 544.0 s |

The refined results differ from the first by 1.19e-7 and 1.94e-7 relative complex
impedance respectively. Maximum continuity residual stays below 5.1e-16 A.
For this rail, **automatic preconditioning with restart 300** is the better
measured choice: fewer iterations despite the tighter tolerance. Graph ILU is
a useful lower-storage option but was weaker here. The API defaults remain
automatic selection and restart 100, since the larger basis is not universally
better. Increasing restart from 100 to 300 adds about 94 MiB of Krylov basis
workspace at 30,621 unknowns; restart is capped at the iteration budget.

Both retries loaded the saved operators in 18–23 s including board import and
meshing. The restart-300 run completed in 567.0 s from startup with 2.85 GiB peak
RSS. The first cold-run driver retained additional metadata, so equal measured
RSS does not mean the extra Krylov vectors have no memory cost. The retries
overlapped on the two-core environment; use iteration counts and residuals for
algorithmic comparison, not these single-run timings as isolated benchmarks.

See [solver comparison](jetson-solver-comparison.json),
[graph refinement](jetson-graph-refined.json) and
[restart refinement](jetson-restart-refined.json). These show iterative
convergence for the fixed approximate operators. They do not check mesh,
compression-tolerance or dielectric-image-tolerance convergence.

## Changes

- Batch different dielectric image offsets into each native integration call.
  Smooth images remain in the existing interpolation spline; only image/panel
  combinations requiring analytic corrections are expanded. Temporary arrays
  are processed in bounded tiles, including images exceeding a tiny tile size.
- Reuse previously evaluated ACA reference/pivot rows and their column entries.
  This preserves the existing independent-row acceptance check.
- Deduplicate shared triangle pairs across the four RWG support combinations,
  also in bounded tiles. Centered coordinates and reciprocal ordering remain.
- Make the aggregated potential operator serializable. The Jetson benchmark
  checkpoints inductance/skin terms and potential separately with atomic writes.
  Geometry, materials, tolerances and kernel source hashes invalidate stale data;
  solver-tolerance changes reuse the operators and rebuild current topology.
- Report assembly progress and iterative residuals. Nonconvergence still raises,
  with structured diagnostics retained for inspection.
- Strengthen the large-system preconditioner through a sparse graph Laplacian
  solve for loop impedance, without constructing the projected cycle matrix.
  Apply the charge-tree inverse separately to preserve low-frequency stability.
  The forced large-system test still agrees with dense PEEC from 1 mHz to 30 MHz.

## Measurements

For the same 25,370-triangle, ten-layer Jetson rail crop, inductance construction
fell from 715.8 to 374.7 seconds. Storage and the block partition are unchanged:
882.0 MiB, 32,088 blocks, including 10,436 low-rank blocks. Evaluated edge entries
fell from 215.2 million to 172.9 million; native triangle integrations also avoid
duplicates shared among neighboring basis supports. The saved inductance and
skin-term checkpoint occupies about 851 MiB on disk.

On the 40-triangle Jetson dielectric sample, the batched evaluator matched the
original dense integrator with maximum relative entry error 4.51e-15. It took
0.534 s for 1,600 requested entries, versus 6.462 s for the dense integrator's
reciprocal matrix fill. The former used 63 native calls for 44,661 image/pair
interactions. This is an integration microbenchmark, not an end-to-end speedup
or a physical accuracy bound; timings are single runs with shared CPU activity.
See [raw kernel sample](jetson-kernel-batched.json).

A 40-edge principal block sampled from the completed full-rail inductance
operator agrees with direct dense moment integration to 5.59e-7 relative
Frobenius error. Maximum absolute error is 2.15e-16 H. The largest entry-relative
error is 0.79 for a very weak mutual entry, using a denominator floor of
4.23e-22 H; the aggregate norm must not be read as an entrywise guarantee.
This is operator validation on actual geometry, not a port-error bound.

The corresponding 40-panel potential sample has 7.04e-7 relative Frobenius
error and 9.33e-4 maximum entry-relative error against direct dense integration.
Both samples come from the actual full-rail operators at compression tolerance
1e-5. Neither the samples nor the small iterative residual establish a global
port-error bound. Mesh, compression and dielectric-image convergence studies
remain necessary before claiming full-rail physical accuracy.
The sampled full-operator data are in [jetson-operator-check.json](jetson-operator-check.json).

A 15-second nonblocking sampling profile of potential construction captured
712 main-thread samples (26 sampling errors). About 53% of captured stack time
was at the native integration call site. Spline evaluation and array processing
also contributed. This locates the remaining cost; it is not a controlled CPU
benchmark. The largest grouped contact contains only 162 triangles, so grouped
self terms do not explain the long build on their own.

## Reproduce and resume

```sh
JETSON_PDN_DIR=/workspace/jetson-pdn PYTHONPATH=src:../kicad2ems/src OPENBLAS_NUM_THREADS=1 python -u examples/peec-memory/jetson_compressed.py
```

The imported board is required. Checkpoints default to
`$JETSON_PDN_DIR/operator-cache` (override with `PEEC_CACHE_DIR`), and results to
`$JETSON_PDN_DIR/compressed-rail.json` (override with `PEEC_RESULT`). Only load
these local pickle checkpoints from a trusted folder. They are not exchange
files and are excluded from the repository results.

Use `PEEC_FREQUENCIES=100000,1000000,30000000`, `PEEC_SOLVER_TOL=1e-8` or
`PEEC_SOLVER_MAXITER=2000` to retry solves with the same operators.
`PEEC_PRECONDITIONER=graph` or `PEEC_RESTART=300` also reuses those operators.
Changing `PEEC_DIELECTRIC_TOL` changes the kernel fingerprint and requires rebuilding.
The defaults remain dielectric tolerance 1e-4, compression 1e-5 and iterative
tolerance 1e-7. These relaxed kernel settings need convergence checks before
claiming full-rail physical accuracy.
