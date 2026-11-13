// Dense systems filled in one pass, and points located in a triangle mesh: what the frequency loops and the
// thermal maps did in numpy, a temporary of the whole matrix per operation. Each row (or point) is one thread's;
// nothing is summed across threads, so results do not depend on their number. pcbqs/kernels.py has the numpy
// twins, which the tests hold these to.
#include <complex>
#include <limits>

using cplx = std::complex<double>;

// A CSR matrix's three arrays, checked against its shape; indices of either integer width.
struct SparseRows {
    arr<long long> indptr, indices;
    arr<cplx> data;
    const long long *ptr, *idx;
    const cplx *val;
    bool any;
    SparseRows(const arr<long long> &p, const arr<long long> &i, const arr<cplx> &d, long long rows, long long cols,
               const char *name)
        : indptr(p), indices(i), data(d), ptr(p.data()), idx(i.data()), val(d.data()), any(d.size() > 0) {
        if (!any) return;
        if (p.ndim() != 1 || p.shape(0) != rows + 1 || i.ndim() != 1 || d.ndim() != 1 || i.shape(0) != d.shape(0)
            || ptr[0] != 0 || ptr[rows] != d.shape(0))
            throw std::invalid_argument(std::string(name) + ": sparse term does not fit the matrix");
        for (long long r = 0; r < rows; ++r)
            if (ptr[r+1] < ptr[r]) throw std::invalid_argument(std::string(name) + ": sparse rows out of order");
        for (long long k = 0; k < d.shape(0); ++k)
            if (idx[k] < 0 || idx[k] >= cols) throw std::invalid_argument(std::string(name) + ": sparse column out of range");
    }
};

// out[a, b] = R[rows[a], cols[b]] + jw L[rows[a], cols[b]] + S[a, b], S sparse (CSR over out's shape, or empty).
// The loop impedance of the copper with what changes with frequency (the parts' own impedances, the conductors'
// resistance) on the unknowns kept: rows and cols pick them out of the whole matrices.
py::array_t<cplx> loop_system(arr<double> R, arr<double> L, cplx jw, arr<long long> rows, arr<long long> cols,
                              arr<long long> indptr, arr<long long> indices, arr<cplx> data, int workers) {
    if (workers < 1) throw std::invalid_argument("workers must be positive");
    if (R.ndim() != 2 || L.ndim() != 2 || R.shape(0) != L.shape(0) || R.shape(1) != L.shape(1)
        || rows.ndim() != 1 || cols.ndim() != 1)
        throw std::invalid_argument("loop_system: R and L must be matrices of one shape, rows and cols lists");
    const long long N0 = R.shape(0), N1 = R.shape(1), nr = rows.shape(0), nc = cols.shape(0);
    const long long *ra = rows.data(), *ca = cols.data();
    for (long long a = 0; a < nr; ++a)
        if (ra[a] < 0 || ra[a] >= N0) throw std::invalid_argument("loop_system: row out of range");
    for (long long b = 0; b < nc; ++b)
        if (ca[b] < 0 || ca[b] >= N1) throw std::invalid_argument("loop_system: column out of range");
    const SparseRows S(indptr, indices, data, nr, nc, "loop_system");
    py::array_t<cplx> out({(py::ssize_t)nr, (py::ssize_t)nc});
    cplx *o = out.mutable_data();
    const double *r = R.data(), *l = L.data();
    const double wr = jw.real(), wi = jw.imag();
    {
        py::gil_scoped_release release;
#pragma omp parallel for num_threads(workers) schedule(static) if(workers > 1 && nr * nc > 65536)
        for (long long a = 0; a < nr; ++a) {
            const double *rr = r + (size_t)ra[a] * N1, *lr = l + (size_t)ra[a] * N1;
            cplx *row = o + (size_t)a * nc;
            for (long long b = 0; b < nc; ++b) {
                const double x = lr[ca[b]];
                row[b] = cplx(rr[ca[b]] + wr * x, wi * x);
            }
            if (S.any)
                for (long long k = S.ptr[a]; k < S.ptr[a+1]; ++k) row[S.idx[k]] += S.val[k];
        }
    }
    return out;
}

// The charge solve's system at one frequency, from Z = R + jw L + S (S sparse: the parts and the resistance's
// change): columns below nl (the loops) are Z / zscale, the others (the tree's) jw Z / pscale, with `electro` added
// to the tree's own block; zscale = max |Z|. Returns (system, zscale). electro is real, or complex in a lossy stack.
template <class E>
static py::tuple peec_system_of(arr<double> R, arr<double> L, cplx jw, double pscale, long long nl,
                                arr<E> electro,
                                arr<long long> indptr, arr<long long> indices, arr<cplx> data, int workers) {
    if (workers < 1) throw std::invalid_argument("workers must be positive");
    if (R.ndim() != 2 || L.ndim() != 2 || R.shape(0) != R.shape(1) || L.shape(0) != R.shape(0) || L.shape(1) != R.shape(0))
        throw std::invalid_argument("peec_system: R and L must be square matrices of one size");
    const long long n = R.shape(0), nt = n - nl;
    if (nl < 0 || nl > n || electro.ndim() != 2 || electro.shape(0) != nt || electro.shape(1) != nt)
        throw std::invalid_argument("peec_system: electro must be the tree's block");
    if (!(pscale > 0)) throw std::invalid_argument("peec_system: pscale must be positive");
    const SparseRows S(indptr, indices, data, n, n, "peec_system");
    py::array_t<cplx> out({(py::ssize_t)n, (py::ssize_t)n});
    cplx *o = out.mutable_data();
    const double *r = R.data(), *l = L.data();
    const E *el = electro.data();
    const double wr = jw.real(), wi = jw.imag();
    std::vector<double> rowmax((size_t)n, 0.);
    double zscale = std::numeric_limits<double>::min();
    {
        py::gil_scoped_release release;
#pragma omp parallel for num_threads(workers) schedule(static) if(workers > 1 && n > 256)
        for (long long i = 0; i < n; ++i) {
            const double *rr = r + (size_t)i * n, *lr = l + (size_t)i * n;
            cplx *row = o + (size_t)i * n;
            for (long long j = 0; j < n; ++j) row[j] = cplx(rr[j] + wr * lr[j], wi * lr[j]);
            if (S.any)
                for (long long k = S.ptr[i]; k < S.ptr[i+1]; ++k) row[S.idx[k]] += S.val[k];
            double big = 0.;
            for (long long j = 0; j < n; ++j) big = std::max(big, std::norm(row[j]));
            rowmax[(size_t)i] = big;
        }
        for (long long i = 0; i < n; ++i) zscale = std::max(zscale, std::sqrt(rowmax[(size_t)i]));
#pragma omp parallel for num_threads(workers) schedule(static) if(workers > 1 && n > 256)
        for (long long i = 0; i < n; ++i) {
            cplx *row = o + (size_t)i * n;
            for (long long j = 0; j < nl; ++j) row[j] /= zscale;
            for (long long j = nl; j < n; ++j) row[j] = (jw * row[j]) / pscale;
            if (i >= nl) {
                const E *er = el + (size_t)(i - nl) * nt;
                for (long long j = 0; j < nt; ++j) row[nl + j] += er[j];
            }
        }
    }
    return py::make_tuple(out, zscale);
}

py::tuple peec_system(arr<double> R, arr<double> L, cplx jw, double pscale, long long nl, py::array electro,
                      arr<long long> indptr, arr<long long> indices, arr<cplx> data, int workers) {
    if (electro.dtype().kind() == 'c')
        return peec_system_of<cplx>(R, L, jw, pscale, nl, arr<cplx>::ensure(electro), indptr, indices, data, workers);
    return peec_system_of<double>(R, L, jw, pscale, nl, arr<double>::ensure(electro), indptr, indices, data, workers);
}

// Which triangle of a mesh holds each point, and the point's weights on its corners: (index (K,), -1 where no
// triangle does; weights (K, 3), which sum to 1). The triangles are binned on a uniform grid over the mesh's box,
// about two to a cell; a point on an edge goes to the first triangle (in the mesh's order) that holds it within
// `tol` of its weights, so two that share the edge give it the same value.
py::tuple tri_locate(arr<double> nodes, arr<long long> tris, arr<double> points, double tol, int workers) {
    if (workers < 1) throw std::invalid_argument("workers must be positive");
    if (nodes.ndim() != 2 || nodes.shape(1) != 2 || tris.ndim() != 2 || tris.shape(1) != 3
        || points.ndim() != 2 || points.shape(1) != 2)
        throw std::invalid_argument("tri_locate: nodes (N, 2), triangles (M, 3), points (K, 2)");
    const long long nn = nodes.shape(0), nt = tris.shape(0), np_ = points.shape(0);
    const double *xy = nodes.data(), *p = points.data();
    const long long *t = tris.data();
    for (long long k = 0; k < 3 * nt; ++k)
        if (t[k] < 0 || t[k] >= nn) throw std::invalid_argument("tri_locate: triangle corner out of range");
    py::array_t<long long> index(np_);
    py::array_t<double> weight({(py::ssize_t)np_, (py::ssize_t)3});
    long long *oi = index.mutable_data();
    double *ow = weight.mutable_data();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (nt == 0) {
        for (long long k = 0; k < np_; ++k) { oi[k] = -1; ow[3*k] = ow[3*k+1] = ow[3*k+2] = nan; }
        return py::make_tuple(index, weight);
    }
    {
        py::gil_scoped_release release;
        double x0 = xy[0], x1 = xy[0], y0 = xy[1], y1 = xy[1];
        for (long long k = 0; k < 3 * nt; ++k) {
            const double x = xy[2*t[k]], y = xy[2*t[k]+1];
            x0 = std::min(x0, x); x1 = std::max(x1, x); y0 = std::min(y0, y); y1 = std::max(y1, y);
        }
        const double w = std::max(x1 - x0, 1e-300), h = std::max(y1 - y0, 1e-300);
        const double cell = std::sqrt(w * h / std::max(0.5 * (double)nt, 1.));
        const long long nx = std::max(1LL, std::min((long long)std::ceil(w / cell), 4096LL));
        const long long ny = std::max(1LL, std::min((long long)std::ceil(h / cell), 4096LL));
        const double sx = (double)nx / w, sy = (double)ny / h;
        auto col = [&](double x) { return std::min(std::max((long long)std::floor((x - x0) * sx), 0LL), nx - 1); };
        auto rowof = [&](double y) { return std::min(std::max((long long)std::floor((y - y0) * sy), 0LL), ny - 1); };
        // the cells each triangle's box touches: counted, then filled (a CSR of triangles per cell, in mesh order)
        std::vector<long long> start((size_t)(nx * ny + 1), 0);
        std::vector<long long> box((size_t)(4 * nt));
        for (long long k = 0; k < nt; ++k) {
            double ax = xy[2*t[3*k]], bx = ax, ay = xy[2*t[3*k]+1], by = ay;
            for (int c = 1; c < 3; ++c) {
                const double x = xy[2*t[3*k+c]], y = xy[2*t[3*k+c]+1];
                ax = std::min(ax, x); bx = std::max(bx, x); ay = std::min(ay, y); by = std::max(by, y);
            }
            const long long c0 = col(ax), c1 = col(bx), r0 = rowof(ay), r1 = rowof(by);
            box[4*k] = c0; box[4*k+1] = c1; box[4*k+2] = r0; box[4*k+3] = r1;
            for (long long rr = r0; rr <= r1; ++rr)
                for (long long cc = c0; cc <= c1; ++cc) ++start[(size_t)(rr * nx + cc + 1)];
        }
        for (long long c = 0; c < nx * ny; ++c) start[(size_t)(c + 1)] += start[(size_t)c];
        std::vector<long long> member((size_t)start[(size_t)(nx * ny)]), fill(start.begin(), start.end() - 1);
        for (long long k = 0; k < nt; ++k)
            for (long long rr = box[4*k+2]; rr <= box[4*k+3]; ++rr)
                for (long long cc = box[4*k]; cc <= box[4*k+1]; ++cc) member[(size_t)fill[(size_t)(rr * nx + cc)]++] = k;
#pragma omp parallel for num_threads(workers) schedule(static) if(workers > 1 && np_ > 4096)
        for (long long k = 0; k < np_; ++k) {
            const double px = p[2*k], py_ = p[2*k+1];
            oi[k] = -1; ow[3*k] = ow[3*k+1] = ow[3*k+2] = nan;
            if (!(px >= x0 - tol * w && px <= x1 + tol * w && py_ >= y0 - tol * h && py_ <= y1 + tol * h)) continue;
            const long long c = rowof(py_) * nx + col(px);
            for (long long m = start[(size_t)c]; m < start[(size_t)(c + 1)]; ++m) {
                const long long *v = t + 3 * member[(size_t)m];
                const double ax = xy[2*v[0]], ay = xy[2*v[0]+1];
                const double e1x = xy[2*v[1]] - ax, e1y = xy[2*v[1]+1] - ay;
                const double e2x = xy[2*v[2]] - ax, e2y = xy[2*v[2]+1] - ay;
                const double det = e1x * e2y - e1y * e2x;
                if (det == 0.) continue;
                const double dx = px - ax, dy = py_ - ay;
                const double w1 = (dx * e2y - dy * e2x) / det, w2 = (e1x * dy - e1y * dx) / det, w0 = 1. - w1 - w2;
                if (w0 >= -tol && w1 >= -tol && w2 >= -tol) {
                    oi[k] = member[(size_t)m]; ow[3*k] = w0; ow[3*k+1] = w1; ow[3*k+2] = w2;
                    break;
                }
            }
        }
    }
    return py::make_tuple(index, weight);
}
