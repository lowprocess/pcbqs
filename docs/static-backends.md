# Quasi-static matrix-fill backends

The magnetic and homogeneous-capacitance fills use bounded tiles of unordered
triangle pairs. Layered capacitance uses the same pair integrators for direct
and image terms, with independent observation rows. All paths use double
precision, the same near/far criterion, and analytic source integration for
close pairs. The solver equations and reciprocal matrix assembly are common
to CPU and device backends.

```python
q = MQS(board, nets=nets, terminals=terminals,
        fill_backend='openmp', fill_workers=8, fill_chunk=16384)
r = q.solve(freqs)
print(q.fill_info)  # backend, worker counts, precision, timing, pair counts
```

| Backend | Execution |
|---|---|
| `auto` | OpenMP if the updated core is built; otherwise threaded NumPy |
| `openmp` | GIL-free C++ whole-matrix fill: classification, integration and reciprocal writes in one OpenMP team |
| `threaded` | NumPy integration in a bounded `ThreadPoolExecutor`; ufuncs release the GIL |
| `numpy` | Serial NumPy reference; always one worker |
| `cupy` | Optional CuPy GPU integration; one geometry upload per fill, host matrices |

Build the OpenMP backend with:

```sh
pip install '.[core]'
python -m pcbqs.core.build
```

An older core without the static kernels is treated as unavailable for this
fill, while remaining usable by the full-wave solver. Explicit selection of
an unavailable backend raises; execution failures never silently retry on a
different backend. Auto does not probe or initialize GPUs. `PCBQS_NO_CORE`
disables automatic compiled-core selection; explicit `openmp` still requests
it, matching the full-wave solver's explicit compiled selection semantics.

## Threads, memory and determinism

`fill_workers` sets the CPU parallelism. Without it, precedence is
`PCBQS_FILL_THREADS`, `PCBQS_OMP_THREADS`, `OMP_NUM_THREADS` (the first team
size), then CPU affinity constrained by Linux cgroup v2 quotas, or CPU count
on platforms without affinity. Explicit counts can exceed that default.
`PCBQS_FILL_BACKEND` selects a backend when the API requests auto.

```sh
PCBQS_FILL_BACKEND=openmp PCBQS_FILL_THREADS=8 python tools/nearfield_demo.py out/
```

The free-space OpenMP fill is one native call: square tiles of unordered
pairs are OpenMP tasks that classify, integrate and write both reciprocal
entries in place (`fill_info['scheduler'] == 'native'`), so no per-tile host
work remains. Other backends keep the host tile scheduler (`'host'`).
The magnetic RWG projection (`fill_info['magnetic']['combine']`) uses a
native gather when the core is built, otherwise sparse products; it writes
each symmetric entry once, with no temporaries beyond the output.

Layered electrostatics with the OpenMP backend runs one native image kernel
per block of observation rows (`fill_info['capacitance']['native_images']`):
each pair sums the spline of the smooth positive images and integrates every
close image (and the direct d = 0 terms) analytically, correcting the spline
at the same quadrature points. Image order is fixed per pair, so results do
not depend on thread count, and agree with the NumPy image loop to roundoff.
Other backends keep the NumPy loop, parallel over row blocks.
Threaded NumPy parallelizes tiles/rows. No global OpenMP or BLAS thread limits
are changed. BLAS solves and field FFTs retain their separate controls.

`fill_chunk` bounds pairs per host tile (host-scheduled backends), not triangle count. The scheduler keeps
at most one task/result per host worker in flight. Integration scratch is
bounded per worker; refined NumPy/CuPy batches are subdivided further to bound
quadrature temporaries. Dense host matrices still require O(Nt²) storage.

Each pair has exactly one owner. Independent pair reductions retain their
summation order; one host writer fills both reciprocal entries. Tests check
bitwise repeatability across tile sizes and thread counts within a backend,
and numerical agreement between implementations. OpenMP/libm and GPU math
can differ from NumPy at floating-point roundoff, so cross-backend bitwise
identity is not required.

`q.fill_info` distinguishes total requested workers, native `kernel_workers`,
and host `tile_workers`. OpenMP also reports the team size observed in a
runtime probe; OpenMP dynamic limits can constrain teams. Capacitance metadata
notes when magnetic moments were reused. Timings cover fill preparation and
moment assembly, excluding RWG projection and the linear solve.

## Optional GPU fill

Install the CuPy extra matching your driver, then select it explicitly:

```python
q = MQS(board, nets=nets, terminals=terminals,
        fill_backend='cupy', fill_device=0, fill_chunk=16384)
```

The backend uploads geometry once. Free-space moment fills (`static_moments`,
used by MQS/PEEC) run entirely on the device in row strips of at least 4M
pairs: each thread generates its pair, classifies it with the host
near/far rule (no FMA contraction, so quadrature choices match the CPU
backends), and integrates far pairs immediately. Near pairs are compacted
into per-level lists and integrated by a second launch, because mixing the
product rule with the analytic asinh/atan2 loop in one wavefront was measured
at about 8x slower. Every entry is computed from its canonical (min, max)
pair, so reciprocal entries are bitwise identical without host transposes.
Two strip buffers alternate on separate streams, overlapping integration of
one strip with the copy of the previous one into contiguous host rows. Device
memory is bounded by the two strips (about 330 MB); there is no dense GPU matrix.

Explicit pair-list callers (layered capacitance images, compressed PEEC) use
the pair protocol below; the device sorts each batch by kind before one
fused launch. These callers batch at least 1M pairs per call on the device
(`FillEngine.pair_batch`). Per-call latency (about 0.2-0.5 ms launch, synchronize
and copy on ROCm) dominates small requests, so when the compiled core is built
pair batches below `PCBQS_CUPY_HOST_PAIRS` (default 32,768 per host thread,
the measured crossover here) run on the host core instead;
`fill_info[...]['routing']` counts host and device batches, and 0 keeps every
batch on the device. Free-space strip fills always run on the device. For
compressed (hierarchical) PEEC, prefer the OpenMP backend: its native entry
providers and ACA built the 5,816-triangle layered case in 0.9 s, against
39.6 s with CuPy selected (66.8 s without host routing).
CPU worker counts do not create multiple GPU submitters. Initialization,
allocation and kernel errors propagate.

This is an **experimental optional backend**: its formulas are independently
checked on a host array namespace, and hardware tests run when CuPy and a
working device are present. Local validation on an RX 7800 XT with CuPy 14.2.0/ROCm now covers the
real device path, all close-quadrature levels, magnetic/scalar fills and MQS,
PEEC and electrostatic solver tests. Other device/driver combinations and
consumer-device FP64 throughput still need target-hardware measurements.
Selecting a GPU fill does not move RWG projection or the linear solve to GPU.

Measured on an RX 7800 XT (CuPy 14.2.0/ROCm) against OpenMP with 16 threads on
an i7-13700K. "Host tiles" is the pair-protocol path with host classification
and block assembly (already kind-sorted; the unsorted original was ~7% slower),
medians of three runs, rectangular single-layer meshes:

| Triangles | Moments | OpenMP-16 | CuPy host tiles | CuPy device strips |
|---:|---|---:|---:|---:|
| 2,838 | full | 0.25 s | 0.42 s | 0.08 s |
| 7,500 | full | 1.58 s | 2.63 s | 0.25 s |
| 16,472 | full | 7.39 s | 11.3 s | 0.93 s |
| 16,472 | scalar | 6.07 s | 9.56 s | 0.57 s |

A 6,844-triangle two-dielectric layered capacitance fill fell from 152 s to
19.5 s on CuPy and from 65.5 s to 24.0 s (profiled; 17.4 s unprofiled) on OpenMP
by batching observation rows (bitwise-identical results). The native OpenMP
image kernel then takes 1.49 s (medians of three, 16 threads), so the CPU path
is now much faster than the device path for layered stacks.

Dense MQS with the native whole-matrix fill and RWG gather (16 threads):

| Triangles | Fill | RWG projection | Assembly | Peak RSS |
|---:|---:|---:|---:|---:|
| 3,747 | 0.47 -> 0.07 s | 1.07 -> 0.04 s | 1.75 -> 0.28 s | 1.36 -> 0.81 GB |
| 6,630 | 1.37 -> 0.16 s | 3.77 -> 0.16 s | 5.92 -> 0.93 s | 4.01 -> 2.24 GB |

## Add a backend

Register a factory, or expose it through the `pcbqs.static_backends` Python
entry-point group. Providers are imported only when selected. Names cannot
replace built-ins or already registered providers.

```python
from pcbqs.static_backends import register_fill_backend
register_fill_backend('my-device', make_backend)
```

The factory signature is `make_backend(geometry, workers, chunk, device)`.
`geometry` is a `FillGeometry` containing read-only local nodes, topology,
heights, centroids, radii, triangle areas and prepared quadrature/edge frames.
Keep it alive while using its arrays; device providers may upload them once.

The returned object implements:

```python
parallel_tiles = False  # True only for safe concurrent, internally serial calls

def evaluate(p, q, d, kind, scalar_only):
    # p, q: int32 triangle indices; d: vertical offsets (m)
    # kind: -1 = far three-point product rule; 0/1/2 = analytic
    #       source integration, with that observation refinement level
    # return host numpy float64 array, fully synchronized:
    # scalar_only: (npairs, 1), M0
    # otherwise:   (npairs, 6), M0, Mro_x, Mro_y, Mrs_x, Mrs_y, Mrr
    ...
```

Providers own their execution resources and device synchronization. They must
not mutate geometry or launch uncontrolled nested CPU teams. The engine
validates output shape, dtype and finiteness. Backend exceptions propagate;
the scheduler cancels queued work and waits for running tasks before returning
the error. A provider can implement CUDA, ROCm, Vulkan or another execution
system without changing MQS/PEEC's equations or matrix assembly.

## Reproduce scaling and accuracy checks

```sh
OPENBLAS_NUM_THREADS=1 python tools/benchmark_static_fill.py --dmax .5 --repeat 3 --out fill.json
OPENBLAS_NUM_THREADS=2 python tools/benchmark_converter.py CM5_MINIMA_3.kicad_pcb \
  --fill-backend openmp --fill-workers 2 --out cm5-openmp.json
```

The static benchmark reports median/sample times and moment errors against
the serial NumPy reference. `--scalar` benchmarks capacitance-only filling.
Device measurements can use `--backends numpy,cupy --workers 1 --device 0`.
The converter benchmark records fill metadata alongside solve and scan times.

On this two-CPU-quota environment, the 2,838-triangle benchmark gave these
medians over three runs (16,384-pair tiles):

| Backend | CPU workers | Fill time |
|---|---:|---:|
| Serial NumPy | 1 | 3.98 s |
| Threaded NumPy | 2 | 2.33 s |
| OpenMP | 1 | 0.75 s |
| OpenMP | 2 | 0.59 s |

The maximum relative moment-matrix difference from NumPy was 4.5e-16 for
OpenMP, and zero for threaded NumPy. The gain includes compiled integration
and tiling; the pure OpenMP thread-scaling gain here is about 1.27x. Host
classification, matrix writes and preparation remain partly serial, and this
does not predict scaling on larger machines. Raw samples are in
[static-fill.json](benchmarks/static-fill.json).

For the same 9,736-triangle CM5 U702 input loop, OpenMP moment filling took
12.65 s with one thread and 9.56 s with two. The complete assembly/sweep fell
from the previous 66.8 s to 22.4 s, with unchanged geometry and copper-matrix
storage. R/L differed by less than 1.4e-14 relative. These are individual
real-board runs, not controlled medians; see the
[CM5 benchmark report](benchmarks/cm5-u702.md).
