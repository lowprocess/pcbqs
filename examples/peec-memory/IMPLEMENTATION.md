# Initial implementation and measurements

`matrix_backend='hierarchical'` now provides on-demand compressed PEEC through
the MQS API and experimental PEECPDN adapter. It includes RWG inductance,
layered/homogeneous potential, via inductance, sparse slab skin impedance,
implicit loop/tree products and preconditioned GMRES. Dense solving remains
the default. See [API usage and limitations](../../docs/peec-memory.md).

The grading API now exposes bulk coarsening, growth rate, edge subdivision
count/width and edge-ring refinement region. Source outlines are preserved;
there is no automatic CAD simplification or adaptive error estimator yet.

Separate-process measurements on the same meshes and component assumptions:

| Fixture | Triangles | Dense peak RSS | Compressed peak RSS | Dense / compressed time | Max relative complex Z difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Rectangular two-plane PDN | 1,436 | 676.8 MiB | 129.3 MiB | 16.93 / 55.31 s | 3.07e-8 |
| Slotted return PDN | 1,391 | 661.1 MiB | 113.5 MiB | 15.63 / 30.75 s | 1.01e-7 |

Eight frequencies span 100 kHz to 30 MHz. Both backends use identical 1.25 mm
meshes and 0.1042 mm contact refinement, the layered dielectric model and slab
skin loss. Compression tolerance is 1e-6; GMRES tolerance is 1e-8. The script
restricts BLAS to one thread and fill to two workers. Times include meshing,
matrix construction and frequency solves after legacy PDN metadata creation.
Peak RSS includes imports, that metadata, all operators and solver workspace.
Runs are single samples, not timing statistics or hardware validation.

Memory fell by 5.2x / 5.8x, while time increased by 3.3x / 2.0x on these small
examples. This version targets memory feasibility. Far-entry batching, reuse of
frequency preconditioners and improved plane-coupling preconditioning remain
optimization opportunities. Accuracy here is agreement with the dense numerical
reference at the same mesh, not an estimate of physical/discretization error.

Reproduce with `tools/benchmark_peec_memory.py`. Raw values, matrix storage,
entry counts and per-frequency residuals are in [backend-comparison.json](backend-comparison.json).

Unit tests additionally cover complex reciprocity, low-frequency continuity,
disconnected conductors, capacitance extraction, layered loss, source scaling,
via inductance, failure on nonconvergence and avoidance of the dense moment
and inductance-projection paths.

A projected sparse preconditioner can itself grow quadratically. An upper
nonzero-count estimate now triggers a local-potential/tree and loop-diagonal
fallback before allocating a large projected matrix. A forced-fallback test
checks agreement and low-frequency stability from 1 mHz through 30 MHz.
Convergence on a large rail still needs measurement; this fallback trades
preconditioner strength for storage.

A complete 25k-triangle Jetson rail attempt is supplied in
`jetson_compressed.py`. Its stage results distinguish successful operator
construction from a converged solve; neither alone establishes full-rail
physical accuracy. This attempt uses relaxed dielectric images (1e-4) and
compression (1e-5), which require convergence studies before accuracy claims.

The initial full-rail inductance stage completed for 25,370 triangles and
30,147 RWG edges: 882.0 MiB operator storage versus 6.77 GiB for dense L,
with 1.60 GiB peak process RSS including the imported board and topology.
It took 715.8 s to construct L (731.8 s from import through assembly). This
establishes inductance storage feasibility; it is not a completed PEEC solve.

Validation at the implementation checkpoint: 154 tests passed, 13 skipped.
