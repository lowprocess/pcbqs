// H-matrix products over the operator's own block arrays (no packed copy). GIL-free OpenMP, no global thread
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

// meta row: a_off, la, b_off, lb, (unused), dense (-1) or low rank (0), rank, same,
//           fwd_slot, tr_slot (-1 when same)
template <class T>
static void hmatrix_product(const long long *m, long long nb, const long long *ix, const long long *rp,
                            const long long *sl, long long n, long long k, long long rank_max,
                            const T *const *U, const T *const *V, const T *x, T *y, T *s, int workers) {
    {
        // Called without the GIL.
        // Uninitialized caller scratch: each block zeroes (first-touches) its
        // own rows in the parallel loop, avoiding a serial fill per product.
#pragma omp parallel num_threads(workers)
        {
            std::vector<T> tmp((size_t)(2 * rank_max * k));
#pragma omp for schedule(dynamic, 8)
            for (long long blk = 0; blk < nb; ++blk) {
                const long long *b = m + 10 * blk;
                const long long *ra = ix + b[0], *cb = ix + b[2];
                const long long la = b[1], lb = b[3], r = b[6];
                const bool dense = b[5] < 0, same = b[7] != 0;
                const T *u = U[blk], *v = dense ? nullptr : V[blk];
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
}

// Validated block layout and pointers into the H-matrix's block arrays, which
// this object keeps alive. Every pointer's extent is checked once here.
struct HBlocks {
    py::list keep;
    arr<long long> meta, idx, row_ptr, slots;
    std::vector<const void *> U, V;
    long long nb = 0, n = 0, nslots = 0, rank_max = 0;
    bool complex_ = false;

    HBlocks(arr<long long> meta_, arr<long long> idx_, arr<long long> row_ptr_, arr<long long> slots_,
            long long nslots_, long long n_, py::list blocks, bool complex_values)
        : meta(meta_), idx(idx_), row_ptr(row_ptr_), slots(slots_), n(n_), nslots(nslots_), complex_(complex_values) {
        if (meta.ndim() != 2 || meta.shape(1) != 10 || idx.ndim() != 1 || row_ptr.ndim() != 1 ||
            slots.ndim() != 1 || nslots < 0 || slots.size() != nslots || n < 0 || row_ptr.size() != n + 1 ||
            (long long)blocks.size() != meta.shape(0))
            throw std::invalid_argument("HBlocks: invalid layout");
        nb = meta.shape(0);
        const long long ni = idx.size(), *m = meta.data(), *ix = idx.data(), *rp = row_ptr.data(), *sl = slots.data();
        for (long long i = 0; i < ni; ++i)
            if (ix[i] < 0 || ix[i] >= n) throw std::invalid_argument("HBlocks: index out of range");
        auto extent = [&](py::handle h, long long rows, long long cols) -> const void * {
            py::array a = py::reinterpret_borrow<py::array>(h);
            const bool ok = complex_ ? py::isinstance<py::array_t<cplx, py::array::c_style>>(a)
                                     : py::isinstance<py::array_t<double, py::array::c_style>>(a);
            if (!ok || a.ndim() != 2 || a.shape(0) != rows || a.shape(1) != cols)
                throw std::invalid_argument("HBlocks: block arrays must be C-contiguous operator dtype of the layout shape");
            keep.append(a);
            return a.data();
        };
        for (long long blk = 0; blk < nb; ++blk) {
            const long long *b = m + 10 * blk;
            const long long ao = b[0], la = b[1], bo = b[2], lb = b[3], r = b[6], same = b[7], fs = b[8], ts = b[9];
            const bool dense = b[5] < 0;
            if (la < 0 || lb < 0 || ao < 0 || ao + la > ni || bo < 0 || bo + lb > ni || (!dense && r < 1) ||
                (same && (!dense || la != lb)) || fs < 0 || fs + la > nslots || (!same && (ts < 0 || ts + lb > nslots)))
                throw std::invalid_argument("HBlocks: inconsistent block layout");
            auto pair = blocks[blk].cast<py::tuple>();
            if (pair.size() != 2) throw std::invalid_argument("HBlocks: (u, v) per block");
            U.push_back(extent(pair[0], la, dense ? lb : r));
            V.push_back(dense ? nullptr : extent(pair[1], r, lb));
            rank_max = std::max(rank_max, dense ? 0LL : r);
        }
        if (rp[0] != 0 || rp[n] != nslots) throw std::invalid_argument("HBlocks: invalid row pointer");
        for (long long i = 0; i < n; ++i)
            if (rp[i + 1] < rp[i]) throw std::invalid_argument("HBlocks: invalid row pointer");
        for (long long i = 0; i < nslots; ++i)
            if (sl[i] < 0 || sl[i] >= nslots) throw std::invalid_argument("HBlocks: slot out of range");
    }

    template <class T> py::array_t<T> apply_t(py::array X_, py::array work_, int workers) const {
        if (!py::isinstance<py::array_t<T, py::array::c_style>>(work_))
            throw std::invalid_argument("HBlocks.apply: C-contiguous scratch of the operator dtype required");
        arr<T> X(X_);
        auto work = py::array_t<T, py::array::c_style>::ensure(work_);
        if (workers < 1 || X.ndim() != 2 || X.shape(0) != n)
            throw std::invalid_argument("HBlocks.apply: vector rows must match the operator, workers positive");
        const long long k = X.shape(1);
        if (work.ndim() != 1 || work.size() < nslots * k)
            throw std::invalid_argument("HBlocks.apply: scratch smaller than contributions x columns");
        py::array_t<T> out({n, k});
        T *y = out.mutable_data(), *s = work.mutable_data();
        const T *x = X.data();
        std::vector<const T *> u(nb), v(nb);
        for (long long i = 0; i < nb; ++i) { u[i] = (const T *)U[i]; v[i] = (const T *)V[i]; }
        {
            py::gil_scoped_release release;
            hmatrix_product<T>(meta.data(), nb, idx.data(), row_ptr.data(), slots.data(), n, k, rank_max,
                               u.data(), v.data(), x, y, s, workers);
        }
        return out;
    }

    py::array apply(py::array X, py::array work, int workers) const {
        return complex_ ? py::array(apply_t<cplx>(X, work, workers)) : py::array(apply_t<double>(X, work, workers));
    }
};

static void bind_hmatrix(py::module_ &m) {
    py::class_<HBlocks>(m, "HBlocks")
        .def(py::init<arr<long long>, arr<long long>, arr<long long>, arr<long long>, long long, long long,
                      py::list, bool>())
        .def_readonly("nslots", &HBlocks::nslots)
        .def("apply", &HBlocks::apply);
}
