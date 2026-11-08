// Entry blocks for the compressed (H-matrix) PEEC operators. One native call
// classifies, integrates and combines a requested block; no host pair lists,
// np.unique deduplication or support broadcasts. Every triangle pair is
// integrated in canonical (min, max) order, so blocks are exactly reciprocal.
#include <algorithm>
#include <vector>

// Canonical moments of the ordered pair (p, q), as compressed._moments.
static inline void canonical_pair(const StaticGeometry &g, const PairRule &r, int p, int q,
                                  bool scalar_only, double *m) {
    const int a = std::min(p, q), b = std::max(p, q);
    const double d = r.z[a]-r.z[b];
    static_pair(g, a, b, d, pair_kind(r, a, b, d), scalar_only, m);
    if (scalar_only) return;
    if (p > q) { std::swap(m[1], m[3]); std::swap(m[2], m[4]); }
    else if (p == q) {
        const double x = (m[1]+m[3])/2, y = (m[2]+m[4])/2;
        m[1] = m[3] = x; m[2] = m[4] = y;
    }
}

// One edge-pair entry, evaluated as the NumPy assembly does (rounded products:
// the terms cancel strongly, and fused rounding would differ by ~1e-13).
static inline double rwg_entry(const double *mm, long long ntb, const int *aloc, const int *bloc,
                                          const double *vx, const double *al, long long ea, long long eb) {
    double total = 0;
    for (int sa = 0; sa < 2; ++sa) {
        const double vpx = vx[4*ea+2*sa], vpy = vx[4*ea+2*sa+1], wa = al[2*ea+sa];
        for (int sb = 0; sb < 2; ++sb) {
            const double *m = mm + 6*((long long)aloc[sa]*ntb + bloc[sb]);
            const double vqx = vx[4*eb+2*sb], vqy = vx[4*eb+2*sb+1];
            const double val = m[5]-(rmul(vqx, m[1])+rmul(vqy, m[2]))-(rmul(vpx, m[3])+rmul(vpy, m[4]))
                               +rmul(rmul(vpx, vqx)+rmul(vpy, vqy), m[0]);
            total += rmul(rmul(wa, al[2*eb+sb]), val);
        }
    }
    return total;
}

static void check_triangles(const long long *t, long long n, int nt, const char *who) {
    for (long long i = 0; i < n; ++i)
        if (t[i] < 0 || t[i] >= nt) throw std::invalid_argument(std::string(who) + ": triangle index out of range");
}

// Dense block of canonical moments: (len(rows), len(cols), 1 or 6).
py::array_t<double> static_block(py::dict geometry, arr<double> z, arr<double> cent, arr<double> radius,
                                 double near, int outer, arr<long long> rows, arr<long long> cols,
                                 bool scalar_only, int workers) {
    const StaticGeometry g = static_geometry(geometry, "static_block");
    const PairRule r = pair_rule(g, z, cent, radius, near, outer, "static_block");
    if (workers < 1 || rows.ndim() != 1 || cols.ndim() != 1)
        throw std::invalid_argument("static_block: invalid indices or worker count");
    const long long na = rows.size(), nb = cols.size(), width = scalar_only ? 1 : 6;
    const long long *ra = rows.data(), *cb = cols.data();
    check_triangles(ra, na, g.nt, "static_block"); check_triangles(cb, nb, g.nt, "static_block");
    py::array_t<double> out({na, nb, width});
    double *o = out.mutable_data();
    {
        py::gil_scoped_release release;
        const long long n = na*nb;
#pragma omp parallel for num_threads(workers) schedule(guided) if(workers > 1 && n > 512)
        for (long long i = 0; i < n; ++i) {
            double m[6];
            canonical_pair(g, r, (int)ra[i / nb], (int)cb[i % nb], scalar_only, m);
            for (long long j = 0; j < width; ++j) o[i*width+j] = m[j];
        }
    }
    return out;
}

// Edge-function inductance block, scaled by `scale` (mu0/4pi):
// sum over support pairs of alpha_a alpha_b int int (r - v_a).(r' - v_b)/R.
// Called without the GIL; indices must already be validated.
static void rwg_block(const StaticGeometry &g, const PairRule &r, const long long *s, const double *vx,
                      const double *al, const long long *ra, long long la, const long long *cb, long long lb,
                      double scale, int workers, double *o) {
    // Neighbouring edges share triangles: integrate each distinct support
    // pair once (about 9x fewer pairs for large blocks, 3x for ACA rows).
    auto distinct = [&](const long long *edges, long long n, std::vector<long long> &tri,
                        std::vector<int> &local) {
        tri.resize(2*n);
        for (long long i = 0; i < n; ++i) { tri[2*i] = s[2*edges[i]]; tri[2*i+1] = s[2*edges[i]+1]; }
        std::vector<long long> sorted(tri);
        std::sort(sorted.begin(), sorted.end());
        sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());
        local.resize(2*n);
        for (long long i = 0; i < 2*n; ++i)
            local[i] = (int)(std::lower_bound(sorted.begin(), sorted.end(), tri[i]) - sorted.begin());
        tri.swap(sorted);
    };
    std::vector<long long> ta, tb; std::vector<int> la_loc, lb_loc;
    distinct(ra, la, ta, la_loc); distinct(cb, lb, tb, lb_loc);
    const long long nta = (long long)ta.size(), ntb = (long long)tb.size(), npair = nta*ntb;
    std::vector<double> M((size_t)(npair*6));
    double *mm = M.data();
#pragma omp parallel num_threads(workers) if(workers > 1 && npair > 512)
    {
#pragma omp for schedule(guided)
        for (long long i = 0; i < npair; ++i)
            canonical_pair(g, r, (int)ta[i / ntb], (int)tb[i % ntb], false, mm + 6*i);
#pragma omp for schedule(static)
        for (long long i = 0; i < la; ++i) {
            const long long ea = ra[i];
            for (long long j = 0; j < lb; ++j)
                o[i*lb+j] = scale*rwg_entry(mm, ntb, &la_loc[2*i], &lb_loc[2*j], vx, al, ea, cb[j]);
        }
    }
}

static void check_edges(const long long *e, long long n, long long ne, const char *who) {
    for (long long i = 0; i < n; ++i)
        if (e[i] < 0 || e[i] >= ne) throw std::invalid_argument(std::string(who) + ": edge index out of range");
}

static void check_rwg(const StaticGeometry &g, const arr<long long> &support, const arr<double> &vertices,
                      const arr<double> &alpha, const char *who) {
    const long long ne = support.ndim() == 2 ? support.shape(0) : -1;
    if (ne < 0 || support.shape(1) != 2 || vertices.ndim() != 3 || vertices.shape(0) != ne ||
        vertices.shape(1) != 2 || vertices.shape(2) != 2 || alpha.ndim() != 2 || alpha.shape(0) != ne ||
        alpha.shape(1) != 2)
        throw std::invalid_argument(std::string(who) + ": invalid supports, vertices or weights");
    check_triangles(support.data(), 2*ne, g.nt, who);
}

py::array_t<double> rwg_inductance(py::dict geometry, arr<double> z, arr<double> cent, arr<double> radius,
                                   double near, int outer, arr<long long> support, arr<double> vertices,
                                   arr<double> alpha, arr<long long> rows, arr<long long> cols,
                                   double scale, int workers) {
    const StaticGeometry g = static_geometry(geometry, "rwg_inductance");
    const PairRule r = pair_rule(g, z, cent, radius, near, outer, "rwg_inductance");
    check_rwg(g, support, vertices, alpha, "rwg_inductance");
    if (workers < 1 || rows.ndim() != 1 || cols.ndim() != 1)
        throw std::invalid_argument("rwg_inductance: invalid indices or workers");
    const long long la = rows.size(), lb = cols.size(), ne = support.shape(0);
    check_edges(rows.data(), la, ne, "rwg_inductance"); check_edges(cols.data(), lb, ne, "rwg_inductance");
    py::array_t<double> out({la, lb});
    double *o = out.mutable_data();
    {
        py::gil_scoped_release release;
        rwg_block(g, r, support.data(), vertices.data(), alpha.data(), rows.data(), la, cols.data(), lb,
                  scale, workers, o);
    }
    return out;
}
