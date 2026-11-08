# Where PEEC time goes

There are stage timings for the full Jetson rail run and two short py-spy
sampling profiles: layered-potential construction and a cached iterative solve.
These are main-thread wall-stack samples, not a complete native CPU profile.

## First completed cold run

| Stage | Elapsed time | Share of 61.6-minute run |
| --- | ---: | ---: |
| Import, rail setup and mesh | 15.4 s | 0.4% |
| Inductance, sheet/via terms and checkpoint | 376.1 s | 10.2% |
| Layered potential and checkpoint | 2,539.4 s | 68.7% |
| PEEC preparation, preconditioning and 1 MHz solve | 764.6 s | 20.7% |

The last stage used 631 GMRES iterations. The later restart-300 run used 424
iterations at a tighter tolerance, taking 544.0 s after the potential became
available. With saved operators, import, meshing and loading took 18–23 s.
The first three construction stages are reused across frequencies/excitations.

## Layered-potential construction profile

A 15-second, 50 Hz nonblocking sample captured 712 stacks with 26 sampling
errors. Categorizing valid nonempty stacks by their inclusive call paths gives:

| Category | Sample share |
| --- | ---: |
| Native triangle integration call site | 54.3% |
| Image selection, quadrature corrections and array handling | 39.0% |
| Cubic-spline evaluation | 4.5% |
| ACA/block assembly and other work | 2.3% |

Leaf-frame attribution places 53.2% directly at the native call site; the
inclusive figure also includes descendants of that call. Neither separates
OpenMP worker CPU time from main-thread waiting. The completed construction
evaluated 1.75 billion image/panel interactions through about 926k native calls.
Inductance construction has stage timing but has not been separately sampled.

## Cached-solver profile

A new 15-second, 100 Hz nonblocking sample captured 1,499 stacks with no errors,
during an early portion of a cached projected-ILU solve:

| Category | Sample share |
| --- | ---: |
| Inductance operator application | 61.4% |
| Potential operator application | 37.2% |
| Preconditioner application | 0.9% |
| Other sparse/operator work | 0.3% |
| GMRES and other work | 0.1% |

Almost all sampled work is in `HMatrix._matmat`: traversing blocks, gathering
vectors, multiplying dense or low-rank blocks, and accumulating reciprocal
contributions. There are 32,088 inductance blocks and 29,680 potential blocks.
The profile includes Python/NumPy dispatch and native small-matrix arithmetic;
it does not establish what fraction of those lines is Python overhead alone.

The profiling process deliberately stopped after 30 iterations. Its expected
nonconvergence is unrelated to the validated complete solves. Its restart was
capped at the 30-iteration budget, so this sample represents early matrix
application cost; it does not measure late restart-300 orthogonalization or
preconditioner construction. A complete end-to-end solver profile is still absent.

## Optimization priorities

For cold construction, reduce the cost/count of layered-image integrations and
the array work around them. For cached frequency sweeps, batch or compile the
hierarchical matrix-vector applications, and reduce the number of applications
through better convergence. Making the already-small preconditioner application
faster alone offers little gain in this sample.

Raw Speedscope traces: [potential](profile-potential.json) and
[solver](profile-solver.json). Category totals are in
[profile-summary.json](profile-summary.json). These short windows are indicative,
not statistical confidence bounds or isolated performance benchmarks.
