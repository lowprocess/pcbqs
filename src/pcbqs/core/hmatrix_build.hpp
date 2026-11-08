// Native H-matrix construction for the compressed PEEC operators: entry
// providers (RWG inductance, homogeneous potential, layered potential) and
// batched ACA / dense leaf evaluation, parallel over independent blocks.
//
// ACA follows hmatrix.HMatrix: partial pivoting, a 16-row check set drawn
// without replacement, a second independent check set before accepting, and
// rank cap min(128, 0.7 m n / (m + n)). Check rows come from a per-block
// SplitMix64 stream seeded by the block's identity, so factors do not depend
// on thread count or scheduling (they differ from NumPy's Generator draws).
#include <cfloat>
#include <cstdint>
#include <memory>

template <class T> struct Entries {
    long long n = 0;
    virtual ~Entries() = default;
    // out[i*lb + j] = entry(rows[i], cols[j]); indices validated by the caller.
    virtual void block(const long long *rows, long long la, const long long *cols, long long lb,
                       T *out, int workers) const = 0;
};

static double magnitude(double x) { return std::fabs(x); }
static double magnitude(const cplx &x) { return std::abs(x); }
static double magnitude2(double x) { return x*x; }
static double magnitude2(const cplx &x) { return x.real()*x.real()+x.imag()*x.imag(); }

struct RwgEntries : Entries<double> {
    StaticGeometry g; PairRule rule;
    arr<double> z, cent, radius, vertices, alpha; arr<long long> support;
    double scale;
    RwgEntries(py::dict geometry, arr<double> z_, arr<double> cent_, arr<double> radius_, double near, int outer,
               arr<long long> support_, arr<double> vertices_, arr<double> alpha_, double scale_)
        : g(static_geometry(geometry, "RwgEntries")), z(z_), cent(cent_), radius(radius_),
          vertices(vertices_), alpha(alpha_), support(support_), scale(scale_) {
        rule = pair_rule(g, z, cent, radius, near, outer, "RwgEntries");
        check_rwg(g, support, vertices, alpha, "RwgEntries");
        n = support.shape(0);
    }
    void block(const long long *r, long long la, const long long *c, long long lb, double *o, int workers) const override {
        rwg_block(g, rule, support.data(), vertices.data(), alpha.data(), r, la, c, lb, scale, workers, o);
    }
};

// (M0 / er) / (a_p a_q 4 pi eps0), in the NumPy operation order.
struct PotentialEntries : Entries<double> {
    StaticGeometry g; PairRule rule;
    arr<double> z, cent, radius;
    double er, pi, eps0;
    PotentialEntries(py::dict geometry, arr<double> z_, arr<double> cent_, arr<double> radius_, double near,
                     int outer, double er_, double pi_, double eps0_)
        : g(static_geometry(geometry, "PotentialEntries")), z(z_), cent(cent_), radius(radius_),
          er(er_), pi(pi_), eps0(eps0_) {
        rule = pair_rule(g, z, cent, radius, near, outer, "PotentialEntries");
        if (!(er > 0)) throw std::invalid_argument("PotentialEntries: er must be positive");
        n = g.nt;
    }
    void block(const long long *r, long long la, const long long *c, long long lb, double *o, int workers) const override {
        const long long total = la*lb;
#pragma omp parallel for num_threads(workers) schedule(guided) if(workers > 1 && total > 512)
        for (long long i = 0; i < total; ++i) {
            const int p = (int)r[i / lb], q = (int)c[i % lb];
            double m[6];
            canonical_pair(g, rule, p, q, true, m);
            o[i] = (m[0]/er)/((((g.a[p]*g.a[q])*4)*pi)*eps0);
        }
    }
};

// Layered image sums for canonical pairs, per (layer, layer) image set.
struct LayeredEntries : Entries<cplx> {
    StaticGeometry g; PairRule rule;
    arr<double> z, cent, radius; arr<long long> layer;
    std::vector<arr<double>> dist, sx; std::vector<arr<cplx>> coef, sc;
    std::vector<LayeredSpline> splines;
    int nl; double pi, eps0;
    LayeredEntries(py::dict geometry, arr<double> z_, arr<double> cent_, arr<double> radius_, double near,
                   int outer, arr<long long> layer_, py::list images, double pi_, double eps0_)
        : g(static_geometry(geometry, "LayeredEntries")), z(z_), cent(cent_), radius(radius_), layer(layer_),
          pi(pi_), eps0(eps0_) {
        rule = pair_rule(g, z, cent, radius, near, outer, "LayeredEntries");
        if (layer.ndim() != 1 || layer.shape(0) != g.nt)
            throw std::invalid_argument("LayeredEntries: one layer index per triangle required");
        const long long count = (long long)images.size();
        nl = (int)std::lround(std::sqrt((double)count));
        if ((long long)nl*nl != count || nl < 1)
            throw std::invalid_argument("LayeredEntries: images must list every (layer, layer) code");
        for (long long i = 0; i < g.nt; ++i)
            if (layer.data()[i] < 0 || layer.data()[i] >= nl)
                throw std::invalid_argument("LayeredEntries: layer index out of range");
        for (auto item : images) {
            auto t = item.cast<py::tuple>();
            if (t.size() != 4) throw std::invalid_argument("LayeredEntries: (d, c, spline_x, spline_c) per code");
            dist.emplace_back(py::object(t[0])); coef.emplace_back(py::object(t[1]));
            sx.emplace_back(py::object(t[2])); sc.emplace_back(py::object(t[3]));
            const auto &d = dist.back(), &x = sx.back(); const auto &c = coef.back(), &s = sc.back();
            if (d.ndim() != 1 || c.ndim() != 1 || c.size() != d.size() || x.ndim() != 1)
                throw std::invalid_argument("LayeredEntries: invalid image arrays");
            for (long long k = 0; k < d.size(); ++k)
                if (!std::isfinite(d.data()[k]) || d.data()[k] < 0 || (k && d.data()[k] < d.data()[k-1]))
                    throw std::invalid_argument("LayeredEntries: image distances must be ascending and nonnegative");
            LayeredSpline sp{nullptr, nullptr, 0, 0.};
            if (x.size()) {
                const long long m = x.size()-1;
                if (m < 1 || s.ndim() != 2 || s.shape(0) != 4 || s.shape(1) != m || x.data()[0] != 0)
                    throw std::invalid_argument("LayeredEntries: invalid spline");
                for (long long k = 0; k < m; ++k)
                    if (!(x.data()[k+1] > x.data()[k])) throw std::invalid_argument("LayeredEntries: spline breaks must increase");
                sp = LayeredSpline{x.data(), s.data(), m, x.data()[m]};
            }
            splines.push_back(sp);
        }
        n = g.nt;
    }
    void block(const long long *r, long long la, const long long *c, long long lb, cplx *o, int workers) const override {
        const long long total = la*lb, *L = layer.data();
#pragma omp parallel for num_threads(workers) schedule(guided) if(workers > 1 && total > 256)
        for (long long i = 0; i < total; ++i) {
            const long long p = r[i / lb], q = c[i % lb];
            const int a = (int)std::min(p, q), b = (int)std::max(p, q);
            const long long code = L[a]*nl+L[b];
            const LayeredSpline *spline = splines[code].x ? &splines[code] : nullptr;
            const cplx v = layered_pair(g, rule, spline, dist[code].data(), coef[code].data(),
                                        (int)dist[code].size(), a, b);
            const double den = (((g.a[p]*g.a[q])*4)*pi)*eps0;
            o[i] = cplx(v.real()/den, v.imag()/den);
        }
    }
};

template <class T> py::array_t<T> entries_block(const Entries<T> &E, arr<long long> rows, arr<long long> cols, int workers) {
    if (workers < 1 || rows.ndim() != 1 || cols.ndim() != 1)
        throw std::invalid_argument("entries.block: invalid indices or workers");
    check_edges(rows.data(), rows.size(), E.n, "entries.block");
    check_edges(cols.data(), cols.size(), E.n, "entries.block");
    py::array_t<T> out({(long long)rows.size(), (long long)cols.size()});
    {
        py::gil_scoped_release release;
        E.block(rows.data(), rows.size(), cols.data(), cols.size(), out.mutable_data(), workers);
    }
    return out;
}

static inline uint64_t splitmix64(uint64_t &s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// k distinct indices of [0, m), partial Fisher-Yates.
static std::vector<long long> draw(uint64_t &state, long long m, long long k) {
    std::vector<long long> all(m);
    for (long long i = 0; i < m; ++i) all[i] = i;
    for (long long i = 0; i < k; ++i) {
        const long long j = i + (long long)(splitmix64(state) % (uint64_t)(m - i));
        std::swap(all[i], all[j]);
    }
    all.resize(k);
    return all;
}

template <class T> struct AcaResult {
    bool ok = false; long long rank = 0, entries = 0, row_hits = 0;
    std::vector<T> U, V;   // U: rank columns of length m; V: rank rows of length n
};

template <class T> AcaResult<T> aca(const Entries<T> &E, const long long *a, long long m, const long long *b, long long n,
                                    double tol, int workers) {
    AcaResult<T> R;
    const long long cap = std::min(128LL, (long long)((0.7*(double)m)*(double)n/(double)(m+n)));
    if (cap < 1) return R;
    uint64_t state = 0x5DEECE66DULL ^ ((uint64_t)a[0] * 0x100000001B3ULL) ^ ((uint64_t)b[0] << 21) ^
                     ((uint64_t)m << 42) ^ (uint64_t)n;
    std::vector<std::vector<T>> cache(m);
    std::vector<long long> scratch;
    auto rows = [&](const std::vector<long long> &idx) {
        std::vector<long long> missing;
        for (long long k : idx) if (cache[k].empty() && std::find(missing.begin(), missing.end(), k) == missing.end())
            missing.push_back(k);
        R.row_hits += (long long)idx.size() - (long long)missing.size();
        if (missing.empty()) return;
        scratch.resize(missing.size());
        for (size_t t = 0; t < missing.size(); ++t) scratch[t] = a[missing[t]];
        std::vector<T> block(missing.size()*(size_t)n);
        E.block(scratch.data(), (long long)missing.size(), b, n, block.data(), workers);
        R.entries += (long long)missing.size()*n;
        for (size_t t = 0; t < missing.size(); ++t)
            cache[missing[t]].assign(block.begin()+t*n, block.begin()+(t+1)*n);
    };
    auto approx = [&](long long i, long long j) {        // (U V)[i, j]
        T s = T(0);
        for (long long l = 0; l < R.rank; ++l) madd(s, R.U[l*m+i], R.V[l*n+j]);
        return s;
    };
    std::vector<long long> check = draw(state, m, std::min(m, 16LL));
    rows(check);
    std::vector<T> residual;                             // check rows minus U V
    for (long long k : check) residual.insert(residual.end(), cache[k].begin(), cache[k].end());
    double ref2 = 0;
    for (const T &x : residual) ref2 += magnitude2(x);
    std::vector<char> used(m, 0);
    std::vector<T> row(n), col(m);
    long long i = 0;
    for (long long it = 0; it < cap; ++it) {
        rows({i});
        for (long long j = 0; j < n; ++j) row[j] = cache[i][j] - approx(i, j);
        long long jbest = 0; double best = -1;
        for (long long j = 0; j < n; ++j) { const double v = magnitude(row[j]); if (v > best) { best = v; jbest = j; } }
        const T pivot = row[jbest];
        if (magnitude(pivot) > DBL_MIN) {
            std::vector<long long> unknown;
            for (long long k = 0; k < m; ++k) {
                if (!cache[k].empty()) col[k] = cache[k][jbest]; else unknown.push_back(k);
            }
            if (!unknown.empty()) {
                scratch.resize(unknown.size());
                for (size_t t = 0; t < unknown.size(); ++t) scratch[t] = a[unknown[t]];
                std::vector<T> values(unknown.size());
                E.block(scratch.data(), (long long)unknown.size(), b + jbest, 1, values.data(), workers);
                R.entries += (long long)unknown.size();
                for (size_t t = 0; t < unknown.size(); ++t) col[unknown[t]] = values[t];
            }
            for (long long k = 0; k < m; ++k) col[k] -= approx(k, jbest);
            R.U.insert(R.U.end(), col.begin(), col.end());
            for (long long j = 0; j < n; ++j) R.V.push_back(row[j] / pivot);
            ++R.rank;
            const T *u = &R.U[(R.rank-1)*m], *v = &R.V[(R.rank-1)*n];
            for (size_t c = 0; c < check.size(); ++c)
                for (long long j = 0; j < n; ++j) residual[c*n+j] -= u[check[c]] * v[j];
        }
        used[i] = 1;
        double res2 = 0;
        for (const T &x : residual) res2 += magnitude2(x);
        if (std::sqrt(res2) <= tol*std::max(std::sqrt(ref2), DBL_MIN)) {
            // A second independent set is checked before accepting.
            std::vector<long long> extra = draw(state, m, std::min(m, 16LL));
            rows(extra);
            double e2 = 0, f2 = 0;
            std::vector<T> err;
            for (long long k : extra)
                for (long long j = 0; j < n; ++j) {
                    const T x = cache[k][j], e = x - approx(k, j);
                    err.push_back(e); e2 += magnitude2(e); f2 += magnitude2(x);
                }
            if (std::sqrt(e2) <= tol*std::max(std::sqrt(f2), DBL_MIN)) { R.ok = true; return R; }
            check.insert(check.end(), extra.begin(), extra.end());
            residual.insert(residual.end(), err.begin(), err.end());
            ref2 += f2;
        }
        // Next pivot row: the unused check row with the largest residual.
        long long next = -1; double worst = -1;
        for (size_t c = 0; c < check.size(); ++c) {
            if (used[check[c]]) continue;
            double s = 0;
            for (long long j = 0; j < n; ++j) s += magnitude2(residual[c*n+j]);
            if (s > worst) { worst = s; next = check[c]; }
        }
        if (next < 0) {
            double big = -1;
            for (long long k = 0; k < m; ++k) {
                if (used[k]) continue;
                const double v = R.rank ? magnitude(R.U[(R.rank-1)*m+k]) : 0.;
                if (next < 0 || v > big) { big = v; next = k; }
            }
        }
        if (next < 0) break;
        i = next;
    }
    return R;
}

template <class T> static std::vector<std::pair<const long long *, long long>> index_lists(py::list items, long long n,
                                                                                       std::vector<arr<long long>> &keep) {
    std::vector<std::pair<const long long *, long long>> out;
    for (auto item : items) {
        keep.emplace_back(py::reinterpret_borrow<py::object>(item));
        const auto &ids = keep.back();
        if (ids.ndim() != 1 || ids.size() < 1) throw std::invalid_argument("hmatrix blocks: nonempty index vectors required");
        check_edges(ids.data(), ids.size(), n, "hmatrix blocks");
        out.emplace_back(ids.data(), (long long)ids.size());
    }
    return out;
}

// One task per block: parallel over blocks when there are enough of them,
// otherwise one block at a time with threads inside the entry kernels.
template <class F> static void over_blocks(long long count, int workers, F &&task) {
    if (count >= workers) {
#pragma omp parallel for num_threads(workers) schedule(dynamic, 1) if(workers > 1)
        for (long long k = 0; k < count; ++k) task(k, 1);
    } else {
        for (long long k = 0; k < count; ++k) task(k, workers);
    }
}

template <class T> py::tuple aca_blocks(const Entries<T> &E, py::list rows, py::list cols, double tol, int workers) {
    if (workers < 1 || rows.size() != cols.size() || !(tol > 0 && tol < 1))
        throw std::invalid_argument("aca_blocks: matching block lists, tolerance in (0, 1) and workers required");
    std::vector<arr<long long>> keep;
    auto A = index_lists<T>(rows, E.n, keep), B = index_lists<T>(cols, E.n, keep);
    std::vector<AcaResult<T>> results(A.size());
    bool finite = true;
    {
        py::gil_scoped_release release;
        over_blocks((long long)A.size(), workers, [&](long long k, int inner) {
            results[k] = aca(E, A[k].first, A[k].second, B[k].first, B[k].second, tol, inner);
        });
    }
    py::list factors;
    long long entries = 0, hits = 0;
    for (size_t k = 0; k < results.size(); ++k) {
        auto &r = results[k];
        entries += r.entries; hits += r.row_hits;
        if (!r.ok) { factors.append(py::none()); continue; }
        const long long m = A[k].second, n = B[k].second, rank = r.rank;
        py::array_t<T> U({m, rank}), V({rank, n});
        T *u = U.mutable_data(), *v = V.mutable_data();
        for (long long l = 0; l < rank; ++l)
            for (long long i = 0; i < m; ++i) u[i*rank+l] = r.U[l*m+i];
        std::copy(r.V.begin(), r.V.end(), v);
        for (long long i = 0; i < m*rank; ++i) finite = finite && std::isfinite(magnitude(u[i]));
        for (long long i = 0; i < rank*n; ++i) finite = finite && std::isfinite(magnitude(v[i]));
        factors.append(py::make_tuple(U, V));
    }
    if (!finite) throw std::range_error("aca_blocks: non-finite factors");
    return py::make_tuple(factors, entries, hits);
}

template <class T> py::list dense_blocks(const Entries<T> &E, py::list rows, py::list cols, int workers) {
    if (workers < 1 || rows.size() != cols.size())
        throw std::invalid_argument("dense_blocks: matching block lists and workers required");
    std::vector<arr<long long>> keep;
    auto A = index_lists<T>(rows, E.n, keep), B = index_lists<T>(cols, E.n, keep);
    std::vector<py::array_t<T>> out;
    for (size_t k = 0; k < A.size(); ++k) out.emplace_back(std::vector<py::ssize_t>{A[k].second, B[k].second});
    std::vector<T *> ptr;
    for (auto &o : out) ptr.push_back(o.mutable_data());
    bool finite = true;
    {
        py::gil_scoped_release release;
        over_blocks((long long)A.size(), workers, [&](long long k, int inner) {
            E.block(A[k].first, A[k].second, B[k].first, B[k].second, ptr[k], inner);
        });
        for (size_t k = 0; k < A.size(); ++k)
            for (long long i = 0; i < A[k].second*B[k].second; ++i) finite = finite && std::isfinite(magnitude(ptr[k][i]));
    }
    if (!finite) throw std::range_error("dense_blocks: non-finite entries");
    py::list result;
    for (auto &o : out) result.append(o);
    return result;
}

static void bind_hmatrix_build(py::module_ &m) {
    py::class_<Entries<double>>(m, "RealEntries")
        .def_property_readonly("size", [](const Entries<double> &e) { return e.n; })
        .def("block", &entries_block<double>);
    py::class_<Entries<cplx>>(m, "ComplexEntries")
        .def_property_readonly("size", [](const Entries<cplx> &e) { return e.n; })
        .def("block", &entries_block<cplx>);
    py::class_<RwgEntries, Entries<double>>(m, "RwgEntries")
        .def(py::init<py::dict, arr<double>, arr<double>, arr<double>, double, int, arr<long long>,
                      arr<double>, arr<double>, double>());
    py::class_<PotentialEntries, Entries<double>>(m, "PotentialEntries")
        .def(py::init<py::dict, arr<double>, arr<double>, arr<double>, double, int, double, double, double>());
    py::class_<LayeredEntries, Entries<cplx>>(m, "LayeredEntries")
        .def(py::init<py::dict, arr<double>, arr<double>, arr<double>, double, int, arr<long long>, py::list,
                      double, double>());
    m.def("aca_blocks", &aca_blocks<double>);
    m.def("aca_blocks", &aca_blocks<cplx>);
    m.def("dense_blocks", &dense_blocks<double>);
    m.def("dense_blocks", &dense_blocks<cplx>);
}
