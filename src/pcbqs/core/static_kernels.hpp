// Quasi-static pair batches: GIL-free OpenMP, bounded outputs and no global
// omp_set_num_threads calls. A pair has one owner and a deterministic sum.
#include <string>
#include <utility>
#include <vector>

// Validated geometry views; the owning arrays keep the buffers alive while
// raw pointers are used without the GIL.
struct StaticGeometry {
    std::vector<arr<double>> keep;
    const double *v, *length, *tangent, *normal, *a, *q3;
    const double *qp[3], *qw[3];
    int nq[3], nt;
};

static StaticGeometry static_geometry(py::dict geometry, const char *who) {
    StaticGeometry g;
    arr<double> V(geometry["V"]), L(geometry["L"]), tau(geometry["tau"]), un(geometry["un"]);
    arr<double> area(geometry["area"]), P3(geometry["P3"]);
    const int nt = (int)area.size();
    if (V.ndim() != 3 || V.shape(0) != nt || V.shape(1) != 3 || V.shape(2) != 2 ||
        L.ndim() != 2 || L.shape(0) != nt || L.shape(1) != 3 ||
        tau.ndim() != 3 || tau.shape(0) != nt || tau.shape(1) != 3 || tau.shape(2) != 2 ||
        un.ndim() != 3 || un.shape(0) != nt || un.shape(1) != 3 || un.shape(2) != 2 ||
        P3.ndim() != 3 || P3.shape(0) != nt || P3.shape(1) != 3 || P3.shape(2) != 2)
        throw std::invalid_argument(std::string(who) + ": invalid geometry");
    g.keep = {V, L, tau, un, area, P3};
    for (int level = 0; level < 3; ++level) {
        arr<double> p(geometry[py::str("P" + std::to_string(level))]);
        arr<double> w(geometry[py::str("W" + std::to_string(level))]);
        if (p.ndim() != 3 || p.shape(0) != nt || p.shape(2) != 2 ||
            w.ndim() != 1 || w.shape(0) != p.shape(1))
            throw std::invalid_argument(std::string(who) + ": invalid observation quadrature");
        g.keep.push_back(p); g.keep.push_back(w);
    }
    g.v = V.data(); g.length = L.data(); g.tangent = tau.data(); g.normal = un.data();
    g.a = area.data(); g.q3 = P3.data(); g.nt = nt;
    for (int k = 0; k < 3; ++k) {
        g.qp[k] = g.keep[6 + 2*k].data(); g.qw[k] = g.keep[7 + 2*k].data();
        g.nq[k] = (int)g.keep[7 + 2*k].size();
    }
    return g;
}

// Moments M0, Mro_x, Mro_y, Mrs_x, Mrs_y, Mrr of one ordered pair. One
// out-of-line copy serves every kernel: inlined copies may contract
// multiply-adds differently and disagree at roundoff.
#if defined(__GNUC__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
static void static_pair(const StaticGeometry &g, int p, int q, double d, int k,
                               bool scalar_only, double *m) {
    const double ad = std::fabs(d);
    for (int j = 0; j < 6; ++j) m[j] = 0;
    if (k < 0) {
        const double w = g.a[p] * g.a[q] / 9;
        for (int ip = 0; ip < 3; ++ip) {
            const double x = g.q3[6*(size_t)p+2*ip], y = g.q3[6*(size_t)p+2*ip+1];
            for (int iq = 0; iq < 3; ++iq) {
                const double sx = g.q3[6*(size_t)q+2*iq], sy = g.q3[6*(size_t)q+2*iq+1];
                const double dx = x-sx, dy = y-sy;
                const double gg = w / std::sqrt(dx*dx+dy*dy+d*d);
                m[0] += gg;
                if (!scalar_only) {
                    m[1] += gg*x; m[2] += gg*y; m[3] += gg*sx; m[4] += gg*sy;
                    m[5] += gg*(x*sx+y*sy);
                }
            }
        }
        return;
    }
    const double *qp = g.qp[k], *qw = g.qw[k];
    const int nq = g.nq[k];
    for (int j = 0; j < nq; ++j) {
        const double x = qp[2*((size_t)p*nq+j)], y = qp[2*((size_t)p*nq+j)+1];
        double I1 = 0, IRx = 0, IRy = 0;
        for (int e = 0; e < 3; ++e) {
            const size_t ve = 6*(size_t)q+2*e;
            const double vx = g.v[ve]-x, vy = g.v[ve+1]-y;
            const double sm = vx*g.tangent[ve]+vy*g.tangent[ve+1];
            const double sp = sm+g.length[3*(size_t)q+e];
            const double t0 = vx*g.normal[ve]+vy*g.normal[ve+1];
            const double r02 = t0*t0+d*d, r0 = std::sqrt(r02);
            const double rm = std::sqrt(r02+sm*sm), rp = std::sqrt(r02+sp*sp);
            const double lg = r0 > 1e-14*g.length[3*(size_t)q+e]
                              ? std::asinh(sp/r0)-std::asinh(sm/r0) : 0;
            double inner = t0*lg;
            if (ad > 0)
                inner -= ad*(std::atan2(t0*sp, r02+ad*rp)-std::atan2(t0*sm, r02+ad*rm));
            I1 += inner;
            if (!scalar_only) {
                const double er = .5*(r02*lg+sp*rp-sm*rm);
                IRx += er*g.normal[ve]; IRy += er*g.normal[ve+1];
            }
        }
        const double w = g.a[p]*qw[j];
        m[0] += w*I1;
        if (!scalar_only) {
            const double sx = IRx+x*I1, sy = IRy+y*I1;
            m[1] += w*x*I1; m[2] += w*y*I1;
            m[3] += w*sx; m[4] += w*sy; m[5] += w*(x*sx+y*sy);
        }
    }
}

// A product rounded before use: compilers may not fuse it into an FMA. GCC
// ignores fp-contract attributes/pragmas per function, so an empty asm
// barrier materializes the product (MSVC does not contract by default).
static inline double rmul(double a, double b) {
    double r = a * b;
#if defined(__GNUC__) && defined(__x86_64__)
    __asm__("" : "+x"(r));
#elif defined(__GNUC__) && defined(__aarch64__)
    __asm__("" : "+w"(r));
#elif defined(__GNUC__)
    volatile double v = r; r = v;
#endif
    return r;
}

struct PairRule {
    const double *z, *cent, *radius;
    double near; bool all_near; int outer;
};

// FillEngine.kinds(): far (-1), analytic (0) or refined analytic (outer).
// Rounded products keep quadrature choices identical to the NumPy rule.
static inline int pair_kind(const PairRule &r, int p, int q, double d) {
    const double dx = r.cent[2*p]-r.cent[2*q], dy = r.cent[2*p+1]-r.cent[2*q+1];
    const double distance = std::sqrt((rmul(dx, dx)+rmul(dy, dy))+rmul(d, d));
    const double size = r.radius[p]+r.radius[q];
    if (!r.all_near && !(distance <= rmul(r.near, size))) return -1;
    return distance <= size ? r.outer : 0;
}

static PairRule pair_rule(const StaticGeometry &g, const arr<double> &z, const arr<double> &cent,
                          const arr<double> &radius, double near, int outer, const char *who) {
    if (z.ndim() != 1 || z.shape(0) != g.nt || cent.ndim() != 2 || cent.shape(0) != g.nt ||
        cent.shape(1) != 2 || radius.ndim() != 1 || radius.shape(0) != g.nt ||
        std::isnan(near) || near < 0 || outer < 0 || outer > 2)
        throw std::invalid_argument(std::string(who) + ": invalid heights, centroids, radii or near rule");
    return PairRule{z.data(), cent.data(), radius.data(), near, std::isinf(near), outer};
}

py::array_t<double> static_pairs(py::dict geometry, arr<int> P, arr<int> Q,
                                arr<double> D, arr<int> kind, bool scalar_only, int workers) {
    const StaticGeometry g = static_geometry(geometry, "static_pairs");
    const int nt = g.nt, n = (int)P.size(), width = scalar_only ? 1 : 6;
    if (workers < 1 || P.ndim() != 1 || Q.ndim() != 1 || D.ndim() != 1 || kind.ndim() != 1 ||
        Q.size() != n || D.size() != n || kind.size() != n)
        throw std::invalid_argument("static_pairs: invalid geometry, pair arrays or worker count");
    const int *pp = P.data(), *qq = Q.data(), *kk = kind.data();
    const double *dd = D.data();
    for (int i = 0; i < n; ++i)
        if (pp[i] < 0 || pp[i] >= nt || qq[i] < 0 || qq[i] >= nt ||
            kk[i] < -1 || kk[i] > 2 || !std::isfinite(dd[i]))
            throw std::invalid_argument("static_pairs: invalid triangle index, kind or offset");
    py::array_t<double> out({n, width});
    double *result = out.mutable_data();
    {
        py::gil_scoped_release release;
#pragma omp parallel for num_threads(workers) schedule(guided) if(workers > 1)
        for (int i = 0; i < n; ++i) {
            double m[6];
            static_pair(g, pp[i], qq[i], dd[i], kk[i], scalar_only, m);
            for (int j = 0; j < width; ++j) result[(size_t)i*width+j] = m[j];
        }
    }
    return out;
}

static double *writable_square(py::array &a, py::ssize_t n, const char *what) {
    if (!a.dtype().is(py::dtype::of<double>()) || a.ndim() != 2 || a.shape(0) != n || a.shape(1) != n ||
        !(a.flags() & py::array::c_style) || !a.writeable())
        throw std::invalid_argument(std::string(what) + ": outputs must be writable C-contiguous float64 n x n");
    return static_cast<double *>(a.mutable_data());
}

// The complete reciprocal moment matrices, written in place. Square tiles of
// unordered pairs are independent tasks: each pair has one owner, writes both
// reciprocal entries, and its reduction order does not depend on scheduling.
py::tuple static_fill(py::dict geometry, arr<double> cent, arr<double> radius, arr<double> z,
                      double near, int outer_levels, bool scalar_only, int workers, py::list outputs) {
    const StaticGeometry g = static_geometry(geometry, "static_fill");
    const int nt = g.nt;
    if (workers < 1 || (int)outputs.size() != (scalar_only ? 1 : 4))
        throw std::invalid_argument("static_fill: invalid geometry, options or outputs");
    const PairRule rule = pair_rule(g, z, cent, radius, near, outer_levels, "static_fill");
    std::vector<py::array> arrays;
    double *M[4] = {};
    for (size_t k = 0; k < outputs.size(); ++k) {
        arrays.push_back(outputs[k].cast<py::array>());
        M[k] = writable_square(arrays.back(), nt, "static_fill");
    }
    const double *zz = z.data();
    const int side = 64, blocks = (nt + side - 1) / side;
    std::vector<std::pair<int, int>> tiles;
    for (int bp = 0; bp < blocks; ++bp)
        for (int bq = bp; bq < blocks; ++bq) tiles.emplace_back(bp, bq);
    long long analytic = 0;
    bool finite = true;
    {
        py::gil_scoped_release release;
#pragma omp parallel num_threads(workers) if(workers > 1) reduction(+:analytic) reduction(&&:finite)
        {
            std::vector<double> buffer((size_t)side*side*6);
#pragma omp for schedule(dynamic, 1)
            for (long t = 0; t < (long)tiles.size(); ++t) {
                const int p0 = tiles[t].first*side, q0 = tiles[t].second*side;
                const int p1 = std::min(nt, p0+side), q1 = std::min(nt, q0+side);
                const bool diagonal = p0 == q0;
                for (int p = p0; p < p1; ++p)
                    for (int q = diagonal ? p : q0; q < q1; ++q) {
                        const double d = zz[p]-zz[q];
                        const int k = pair_kind(rule, p, q, d);
                        analytic += k >= 0;
                        double *m = &buffer[6*((size_t)(p-p0)*side+(q-q0))];
                        static_pair(g, p, q, d, k, scalar_only, m);
                        for (int j = 0; j < (scalar_only ? 1 : 6); ++j) finite = finite && std::isfinite(m[j]);
                    }
                // Forward rows, then reverse rows: both contiguous within the tile.
                for (int p = p0; p < p1; ++p)
                    for (int q = diagonal ? p : q0; q < q1; ++q) {
                        const double *m = &buffer[6*((size_t)(p-p0)*side+(q-q0))];
                        const size_t pq = (size_t)p*nt+q;
                        M[0][pq] = m[0];
                        if (!scalar_only) {
                            if (p == q) {
                                M[1][pq] = (m[1]+m[3])/2; M[2][pq] = (m[2]+m[4])/2;
                            } else {
                                M[1][pq] = m[1]; M[2][pq] = m[2];
                            }
                            M[3][pq] = m[5];
                        }
                    }
                for (int q = q0; q < q1; ++q)
                    for (int p = p0; p < std::min(p1, diagonal ? q : p1); ++p) {
                        const double *m = &buffer[6*((size_t)(p-p0)*side+(q-q0))];
                        const size_t qp = (size_t)q*nt+p;
                        M[0][qp] = m[0];
                        if (!scalar_only) {
                            M[1][qp] = m[3]; M[2][qp] = m[4]; M[3][qp] = m[5];
                        }
                    }
            }
        }
    }
    if (!finite) throw std::range_error("non-finite triangle-pair moments from fill backend");
    return py::make_tuple((long long)nt*(nt+1)/2, analytic);
}

// L[m, n] = factor s_m s_n sum_{i,j} a_mi a_nj [Mrr(p,q) - v_nj.Mr(p,q) - v_mi.Mr(q,p)
//           + v_mi.v_nj M0(p,q)], p = t_mi, q = t_nj: the RWG Galerkin combination of
// triangle-pair moments (two triangles per basis function), written symmetrically.
py::array_t<double> static_combine(py::array Mrr, py::array Mrx, py::array Mry, py::array M0,
                                   arr<int> tri, arr<double> alpha, arr<double> vertex, arr<double> scale,
                                   double factor, int workers) {
    const py::ssize_t ne = scale.size();
    if (workers < 1 || scale.ndim() != 1 || tri.ndim() != 2 || tri.shape(0) != 2 || tri.shape(1) != ne ||
        alpha.ndim() != 2 || alpha.shape(0) != 2 || alpha.shape(1) != ne ||
        vertex.ndim() != 3 || vertex.shape(0) != 2 || vertex.shape(1) != ne || vertex.shape(2) != 2)
        throw std::invalid_argument("static_combine: invalid basis arrays");
    const py::ssize_t nt = M0.ndim() == 2 ? M0.shape(0) : -1;
    auto readable = [nt](py::array &a) {
        if (!a.dtype().is(py::dtype::of<double>()) || a.ndim() != 2 || a.shape(0) != nt || a.shape(1) != nt ||
            !(a.flags() & py::array::c_style))
            throw std::invalid_argument("static_combine: moments must be C-contiguous float64 n x n");
        return static_cast<const double *>(a.data());
    };
    const double *rr = readable(Mrr), *mx = readable(Mrx), *my = readable(Mry), *m0 = readable(M0);
    const int *t = tri.data();
    for (py::ssize_t i = 0; i < 2*ne; ++i)
        if (t[i] < 0 || t[i] >= nt) throw std::invalid_argument("static_combine: invalid triangle index");
    const double *al = alpha.data(), *vv = vertex.data(), *s = scale.data();
    py::array_t<double> out({ne, ne});
    double *o = out.mutable_data();
    const py::ssize_t side = 32, blocks = (ne + side - 1) / side;
    {
        py::gil_scoped_release release;
#pragma omp parallel for num_threads(workers) schedule(dynamic, 1) if(workers > 1)
        for (py::ssize_t tile = 0; tile < blocks*(blocks+1)/2; ++tile) {
            py::ssize_t bm = 0, rest = tile;      // tile -> (bm <= bn) in row-major upper order
            while (rest >= blocks - bm) { rest -= blocks - bm; ++bm; }
            const py::ssize_t bn = bm + rest;
            const py::ssize_t m0_ = bm*side, m1 = std::min(ne, m0_+side);
            const py::ssize_t n0 = bn*side, n1 = std::min(ne, n0+side);
            for (py::ssize_t m = m0_; m < m1; ++m)
                for (py::ssize_t n = bm == bn ? m : n0; n < n1; ++n) {
                    double sum = 0;
                    for (int i = 0; i < 2; ++i) {
                        const size_t p = (size_t)t[i*ne+m];
                        const double am = al[i*ne+m], vmx = vv[2*(i*ne+m)], vmy = vv[2*(i*ne+m)+1];
                        for (int j = 0; j < 2; ++j) {
                            const size_t q = (size_t)t[j*ne+n];
                            const double vnx = vv[2*(j*ne+n)], vny = vv[2*(j*ne+n)+1];
                            const size_t pq = p*nt+q, qp = q*nt+p;
                            sum += am*al[j*ne+n]*(rr[pq] - vnx*mx[pq] - vny*my[pq] - vmx*mx[qp] - vmy*my[qp]
                                                  + (vmx*vnx + vmy*vny)*m0[pq]);
                        }
                    }
                    const double value = factor*s[m]*s[n]*sum;
                    o[(size_t)m*ne+n] = value;
                    o[(size_t)n*ne+m] = value;
                }
        }
    }
    return out;
}

int static_thread_count(int workers) {
    if (workers < 1) throw std::invalid_argument("workers must be positive");
    int count = 1;
    py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel num_threads(workers)
    {
#pragma omp single
        count = omp_get_num_threads();
    }
#endif
    return count;
}
