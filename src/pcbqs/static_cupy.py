"""Optional CuPy float64 pair integrator. No GPU import/probe until selected.

Geometry is uploaded once per fill. Only bounded pair batches and their
results cross the device boundary; dense matrices remain on the host.
Small explicit pair batches run on the compiled host core when it is built,
since each device call costs a launch, synchronization and copy.
"""
import os

import numpy as np


def _host_routing():
    """(core, workers, threshold pairs) for small batches, or (None, 0, 0).

    About 32k pairs per host thread matched the device time of one call on an
    RX 7800 XT against an i7-13700K; PCBQS_CUPY_HOST_PAIRS overrides it and 0
    keeps every batch on the device.
    """
    from .static_backends import _compiled, resolve_workers
    core = None if os.environ.get('PCBQS_NO_CORE') else _compiled()
    if core is None:
        return None, 0, 0
    workers = resolve_workers(None)
    env = os.environ.get('PCBQS_CUPY_HOST_PAIRS')
    try:
        threshold = int(env) if env is not None else 32768*workers
    except ValueError as exc:
        raise ValueError('PCBQS_CUPY_HOST_PAIRS must be a nonnegative integer') from exc
    if threshold < 0:
        raise ValueError('PCBQS_CUPY_HOST_PAIRS must be a nonnegative integer')
    return (core, workers, threshold) if threshold else (None, 0, 0)

def _analytic(xp, geometry, p, q, dz, scalar_only):
    V, length, tau, un, area, points, weights = geometry
    r = points[p]
    dv = V[q, None]-r[:, :, None]
    sm = (dv*tau[q, None]).sum(-1)
    sp = sm+length[q, None]
    t0 = (dv*un[q, None]).sum(-1)
    d = dz[:, None, None]
    ad = xp.abs(d)
    r02 = t0*t0+d*d
    r0 = xp.sqrt(r02)
    rm, rp = xp.sqrt(r02+sm*sm), xp.sqrt(r02+sp*sp)
    ok = r0 > 1e-14*length[q, None]
    safe = xp.where(ok, r0, 1.)
    lg = xp.where(ok, xp.arcsinh(sp/safe)-xp.arcsinh(sm/safe), 0.)
    beta = xp.arctan2(t0*sp, r02+ad*rp)-xp.arctan2(t0*sm, r02+ad*rm)
    inner = (t0*lg-ad*beta).sum(-1)
    w = weights[None]*area[p, None]
    m0 = (w*inner).sum(1)
    if scalar_only:
        return m0[:, None]
    edge = .5*(r02*lg+sp*rp-sm*rm)
    ir = (edge[..., None]*un[q, None]).sum(-2)
    source = ir+r*inner[..., None]
    ro = (w[..., None]*r*inner[..., None]).sum(1)
    rs = (w[..., None]*source).sum(1)
    rr = (w*(r*source).sum(-1)).sum(1)
    return xp.column_stack((m0, ro, rs, rr))


class CupyBackend:
    parallel_tiles = False
    # Dense row strips hold at least this many pairs: the device needs millions
    # of threads in flight, while host tiles are sized for cache.
    min_strip_pairs = 1 << 22
    # Explicit pair-list callers batch at least this many pairs per call, so
    # launch and synchronization latency do not dominate (layered images).
    min_batch_pairs = 1 << 20

    def __init__(self, geometry, workers, chunk, device):
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError('CuPy fill unavailable: install a pcbqs GPU CuPy extra matching your driver') from exc
        try:
            self.device = cp.cuda.Device(0 if device is None else device)
            self.device_id = self.device.id
            with self.device:
                cp.zeros(1, dtype=cp.float64).sum().item()
                self.levels = {k: tuple(cp.asarray(a) for a in g) for k, g in geometry.levels.items()}
                self.points = cp.asarray(geometry.points)
                self.z, self.cent, self.radius = (cp.asarray(a) for a in (geometry.z, geometry.cent, geometry.radius))
        except Exception as exc:
            raise RuntimeError(f'CuPy fill could not initialize device {device}: {exc}') from exc
        from .static_cupy_kernels import SOURCE
        module = cp.RawModule(code=SOURCE)
        self.kernel = module.get_function('static_pairs')
        self.rows_far = module.get_function('static_rows_far')
        self.rows_near = module.get_function('static_rows_near')
        levels = [self.levels.get(k, self.levels[0]) for k in range(3)]
        V, length, tau, un, area, _, _ = self.levels[0]
        quadrature = tuple(a for level in levels for a in (level[5], level[6], np.int32(len(level[6]))))
        self.geometry_args = (V, length, tau, un, area, self.points, *quadrature)
        self.cp, self.chunk = cp, chunk
        self.host_core, self.host_workers, self.host_pairs = _host_routing()
        self.routing = dict(host_threshold_pairs=self.host_pairs, host_workers=self.host_workers,
                            host_batches=0, host_pairs=0, device_batches=0, device_pairs=0)
        if self.host_core is not None:
            self.host_data = dict(V=geometry.levels[0][0], L=geometry.levels[0][1], tau=geometry.levels[0][2],
                                  un=geometry.levels[0][3], area=geometry.levels[0][4], P3=geometry.points)
            for level in range(3):
                g = geometry.levels.get(level, geometry.levels[0])
                self.host_data[f'P{level}'], self.host_data[f'W{level}'] = g[5], g[6]

    def evaluate(self, p, q, d, kind, scalar_only):
        cp = self.cp
        if len(p) < self.host_pairs:
            self.routing['host_batches'] += 1; self.routing['host_pairs'] += len(p)
            return self.host_core.static_pairs(self.host_data, p, q, d, kind, scalar_only, self.host_workers)
        self.routing['device_batches'] += 1; self.routing['device_pairs'] += len(p)
        with self.device:
            out = cp.empty((len(p), 1 if scalar_only else 6), dtype=cp.float64)
            if len(p):
                # Upload one bounded pair tile and launch one fused kernel in
                # kind-sorted order, so far and near pairs do not share wavefronts.
                kind = cp.asarray(kind, dtype=cp.int32)
                order = cp.argsort(kind).astype(cp.int32) if len(p) > 1 else None
                inputs = (cp.asarray(p, dtype=cp.int32), cp.asarray(q, dtype=cp.int32),
                          cp.asarray(d, dtype=cp.float64), kind, order)
                self.kernel(((len(p)+127)//128,), (128,),
                            (*self.geometry_args, *inputs, np.int32(len(p)), np.int32(scalar_only), out))
            return cp.asnumpy(out)

    def fill_symmetric(self, matrices, near, outer_levels, scalar_only):
        """Write dense reciprocal moments straight into host `matrices`.

        Row strips are generated, classified and integrated on the device,
        then copied into contiguous host rows, so no pair lists, kinds or
        transposed block writes run on the host. Two strip buffers alternate:
        the device integrates one strip while the host receives the previous
        one. Device memory is bounded by two strips, not by the dense
        matrices. Every entry is computed from its canonical (min, max) pair,
        so reciprocal entries are bitwise identical. Returns the analytic
        count of unordered pairs.
        """
        cp = self.cp
        n = matrices[0].shape[0]
        width = len(matrices)
        rows = max(1, min(n, max(self.chunk, self.min_strip_pairs)//n))
        plane = np.uint64(rows*n)
        all_near = bool(np.isinf(near))
        near = np.float64(0. if all_near else near)
        threads = 256
        with self.device:
            compute, copy = cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=True)
            buffers = [dict(out=cp.empty((width, rows*n), dtype=cp.float64),
                            lists=cp.empty((2, rows*n), dtype=cp.int32),
                            count=cp.empty(3, dtype=cp.uint32), done=cp.cuda.Event(), copied=None)
                       for _ in range(2)]
            with compute:
                analytic = cp.zeros((), dtype=cp.uint64)
                finite = cp.ones((), dtype=cp.bool_)
            near_blocks = 4*self.device.attributes['MultiProcessorCount']*(1024//threads)

            def launch(index, r0):
                b = buffers[index % 2]
                m = min(rows, n-r0)
                with compute:
                    if b['copied'] is not None:
                        compute.wait_event(b['copied'])
                    b['count'].fill(0)
                    self.rows_far((((m*n)+threads-1)//threads,), (threads,),
                                  (*self.geometry_args, self.z, self.cent, self.radius,
                                   np.int32(r0), np.int32(m), np.int32(n), near, np.int32(all_near),
                                   np.int32(outer_levels), np.int32(scalar_only),
                                   b['out'], plane, b['lists'][0], b['lists'][1], b['count']))
                    for level in (0, 1):
                        self.rows_near((near_blocks,), (threads,),
                                       (*self.geometry_args, self.z, np.int32(r0), np.int32(n),
                                        np.int32(outer_levels if level else 0), np.int32(scalar_only),
                                        b['out'], plane, b['lists'][level], b['count'][level:level+1]))
                    analytic[...] += b['count'][2]
                    finite[...] &= cp.isfinite(b['out'][:, :m*n]).all()
                    b['done'].record(compute)

            def receive(index, r0):
                b = buffers[index % 2]
                m = min(rows, n-r0)
                copy.wait_event(b['done'])
                for k, matrix in enumerate(matrices):
                    b['out'][k, :m*n].get(stream=copy, out=matrix[r0:r0+m].reshape(-1))
                b['copied'] = copy.record()
                copy.synchronize()

            starts = list(range(0, n, rows))
            launch(0, starts[0])
            for index, r0 in enumerate(starts):
                if index+1 < len(starts):
                    launch(index+1, starts[index+1])
                receive(index, r0)
            compute.synchronize()
            if not bool(finite.get()):
                raise FloatingPointError('non-finite triangle-pair moments from fill backend')
            return int(analytic.get())
