// Quasi-static pair batches: GIL-free OpenMP, bounded outputs and no global
// omp_set_num_threads calls. A pair has one owner and a deterministic sum.
#include <string>
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
