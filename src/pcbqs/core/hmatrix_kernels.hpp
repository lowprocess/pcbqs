// H-matrix products from packed blocks. GIL-free OpenMP, no global thread
// settings. Phase 1: each block writes its own contribution rows (one owner,
// fixed loop order). Phase 2: each output row sums its contributions in block
// order. Results are therefore independent of thread count and scheduling.
#include <complex>
#include <vector>

using cplx = std::complex<double>;

// acc += a*b without std::complex's NaN/inf recovery (which blocks vectorization).
static inline void madd(double &acc, double a, double b) { acc += a * b; }
static inline void madd(cplx &acc, const cplx &a, const cplx &b) {
    acc = cplx(acc.real() + (a.real() * b.real() - a.imag() * b.imag()),
               acc.imag() + (a.real() * b.imag() + a.imag() * b.real()));
}

// meta row: a_off, la, b_off, lb, u_off, v_off (-1: dense u is la x lb), rank, same,
//           fwd_slot, tr_slot (-1 when same)
template <class T>
py::array_t<T> hmatrix_apply_t(arr<long long> meta, arr<long long> idx, arr<T> data,
                               arr<long long> row_ptr, arr<long long> slots, arr<T> X,
                               py::array_t<T, py::array::c_style> work, long long nslots, int workers) {
    if (workers < 1 || meta.ndim() != 2 || meta.shape(1) != 10 || idx.ndim() != 1 || data.ndim() != 1 ||
        row_ptr.ndim() != 1 || slots.ndim() != 1 || X.ndim() != 2 || nslots < 0 || slots.size() != nslots)
        throw std::invalid_argument("hmatrix_apply: invalid packed operator, vector or worker count");
    const long long nb = meta.shape(0), n = X.shape(0), k = X.shape(1);
    const long long ni = idx.size(), nd = data.size();
    if (work.ndim() != 1 || work.size() < nslots * k)
        throw std::invalid_argument("hmatrix_apply: scratch smaller than contributions x columns");
    if (row_ptr.size() != n + 1)
        throw std::invalid_argument("hmatrix_apply: row pointer does not match the vector length");
    const long long *m = meta.data(), *ix = idx.data(), *rp = row_ptr.data(), *sl = slots.data();
    for (long long i = 0; i < ni; ++i)
        if (ix[i] < 0 || ix[i] >= n) throw std::invalid_argument("hmatrix_apply: index out of range");
    long long rank_max = 0;
    for (long long blk = 0; blk < nb; ++blk) {
        const long long *b = m + 10 * blk;
        long long ao = b[0], la = b[1], bo = b[2], lb = b[3], uo = b[4], vo = b[5], r = b[6], same = b[7];
        long long fs = b[8], ts = b[9];
        bool dense = vo < 0;
        long long usize = dense ? la * lb : la * r;
        if (la < 0 || lb < 0 || ao < 0 || ao + la > ni || bo < 0 || bo + lb > ni || uo < 0 || uo + usize > nd ||
            (!dense && (r < 1 || vo + r * lb > nd)) || (same && (!dense || la != lb)) ||
            fs < 0 || fs + la > nslots || (!same && (ts < 0 || ts + lb > nslots)))
            throw std::invalid_argument("hmatrix_apply: inconsistent block layout");
        rank_max = std::max(rank_max, r);
    }
    if (rp[0] != 0 || rp[n] != nslots) throw std::invalid_argument("hmatrix_apply: invalid row pointer");
    for (long long i = 0; i < n; ++i)
        if (rp[i + 1] < rp[i]) throw std::invalid_argument("hmatrix_apply: invalid row pointer");
    for (long long i = 0; i < nslots; ++i)
        if (sl[i] < 0 || sl[i] >= nslots) throw std::invalid_argument("hmatrix_apply: slot out of range");

    py::array_t<T> out({n, k});
    T *y = out.mutable_data();
    const T *x = X.data(), *d = data.data();
    {
        py::gil_scoped_release release;
        // Uninitialized caller scratch: each block zeroes (first-touches) its
        // own rows in the parallel loop, avoiding a serial fill per product.
        T *s = work.mutable_data();
#pragma omp parallel num_threads(workers)
        {
            std::vector<T> tmp((size_t)(2 * rank_max * k));
#pragma omp for schedule(dynamic, 8)
            for (long long blk = 0; blk < nb; ++blk) {
                const long long *b = m + 10 * blk;
                const long long *ra = ix + b[0], *cb = ix + b[2];
                const long long la = b[1], lb = b[3], r = b[6];
                const bool dense = b[5] < 0, same = b[7] != 0;
                const T *u = d + b[4], *v = dense ? nullptr : d + b[5];
                T *fwd = s + b[8] * k, *tr = same ? nullptr : s + b[9] * k;
                std::fill(fwd, fwd + la * k, T(0));
                if (tr) std::fill(tr, tr + lb * k, T(0));
                if (dense) {
                    // fwd = u x[b]; tr = u^T x[a]
                    for (long long i = 0; i < la; ++i) {
                        const T *ui = u + i * lb, *xa = x + ra[i] * k;
                        T *fi = fwd + i * k;
                        for (long long j = 0; j < lb; ++j) {
                            const T *xb = x + cb[j] * k;
                            for (long long c = 0; c < k; ++c) madd(fi[c], ui[j], xb[c]);
                            if (tr) {
                                T *tj = tr + j * k;
                                for (long long c = 0; c < k; ++c) madd(tj[c], ui[j], xa[c]);
                            }
                        }
                    }
                } else {
                    // t = v x[b]; fwd = u t; q = u^T x[a]; tr = v^T q
                    T *t = tmp.data(), *q = tmp.data() + rank_max * k;
                    std::fill(t, t + r * k, T(0));
                    std::fill(q, q + r * k, T(0));
                    for (long long l = 0; l < r; ++l) {
                        const T *vl = v + l * lb;
                        T *tl = t + l * k;
                        for (long long j = 0; j < lb; ++j) {
                            const T *xb = x + cb[j] * k;
                            for (long long c = 0; c < k; ++c) madd(tl[c], vl[j], xb[c]);
                        }
                    }
                    for (long long i = 0; i < la; ++i) {
                        const T *ui = u + i * r, *xa = x + ra[i] * k;
                        T *fi = fwd + i * k;
                        for (long long l = 0; l < r; ++l) {
                            const T *tl = t + l * k;
                            T *ql = q + l * k;
                            for (long long c = 0; c < k; ++c) {
                                madd(fi[c], ui[l], tl[c]);
                                madd(ql[c], ui[l], xa[c]);
                            }
                        }
                    }
                    for (long long l = 0; l < r; ++l) {
                        const T *vl = v + l * lb, *ql = q + l * k;
                        for (long long j = 0; j < lb; ++j) {
                            T *tj = tr + j * k;
                            for (long long c = 0; c < k; ++c) madd(tj[c], vl[j], ql[c]);
                        }
                    }
                }
            }
            // Implicit barrier: all contributions exist before rows are summed.
#pragma omp for schedule(static)
            for (long long row = 0; row < n; ++row) {
                T *yr = y + row * k;
                std::fill(yr, yr + k, T(0));
                for (long long p = rp[row]; p < rp[row + 1]; ++p) {
                    const T *src = s + sl[p] * k;
                    for (long long c = 0; c < k; ++c) yr[c] += src[c];
                }
            }
        }
    }
    return out;
}

py::array hmatrix_apply(arr<long long> meta, arr<long long> idx, py::array data, arr<long long> row_ptr,
                        arr<long long> slots, py::array X, py::array work, long long nslots, int workers) {
    // Scratch must already have the operator's dtype: a forcecast copy would be discarded.
    if (data.dtype().kind() == 'c') {
        if (!py::isinstance<py::array_t<cplx, py::array::c_style>>(work))
            throw std::invalid_argument("hmatrix_apply: complex128 C-contiguous scratch required");
        return hmatrix_apply_t<cplx>(meta, idx, arr<cplx>(data), row_ptr, slots, arr<cplx>(X),
                                     py::array_t<cplx, py::array::c_style>::ensure(work), nslots, workers);
    }
    if (!py::isinstance<py::array_t<double, py::array::c_style>>(work))
        throw std::invalid_argument("hmatrix_apply: float64 C-contiguous scratch required");
    return hmatrix_apply_t<double>(meta, idx, arr<double>(data), row_ptr, slots, arr<double>(X),
                                   py::array_t<double, py::array::c_style>::ensure(work), nslots, workers);
}
