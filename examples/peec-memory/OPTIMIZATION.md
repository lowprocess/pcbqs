# Batched entries, checkpoints and full-rail convergence

This continues the [initial implementation](IMPLEMENTATION.md).

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
`PEEC_SOLVER_MAXITER=2000` to retry solves with the same operators. Changing
`PEEC_DIELECTRIC_TOL` changes the kernel fingerprint and requires rebuilding.
The defaults remain dielectric tolerance 1e-4, compression 1e-5 and iterative
tolerance 1e-7. These relaxed kernel settings need convergence checks before
claiming full-rail physical accuracy.
