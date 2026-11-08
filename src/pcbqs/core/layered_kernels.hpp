// Layered-dielectric image sums per triangle pair (electrostatics, no PEC).
// For each pair: the smooth positive images through a cubic spline of the
// image kernel sampled on a sqrt-uniform rho grid, then every image that is
// close to the pair (or every image when there is no spline, and the d = 0
// images always) by the same analytic/product rules as the direct term. A
// close positive image replaces its own spline contribution, so the spline is
// corrected by the product rule at the same quadrature points. One thread
// owns one pair; image order is fixed, so results do not depend on threads.
#include <complex>

using cplx = std::complex<double>;

static inline cplx cscale(const cplx &a, double s) { return cplx(a.real()*s, a.imag()*s); }

struct LayeredSpline {
    const double *x; const cplx *c; long long m;   // breaks x[0..m], coefficients c[4][m]
    double rho_max;
};

// scipy PPoly with extrapolation: interval i with x[i] <= r < x[i+1], clamped.
static inline cplx spline_eval(const LayeredSpline &s, double r) {
    const long long last = s.m - 1;
    long long i = (long long)(std::sqrt(std::max(r, 0.) / s.rho_max) * (double)s.m);
    i = std::min(std::max(i, 0LL), last);
    while (i > 0 && r < s.x[i]) --i;
    while (i < last && r >= s.x[i+1]) ++i;
    const double dx = r - s.x[i];
    const cplx *c = s.c;
    cplx v = c[i];
    v = cscale(v, dx) + c[s.m + i];
    v = cscale(v, dx) + c[2*s.m + i];
    v = cscale(v, dx) + c[3*s.m + i];
    return v;
}

static inline cplx layered_pair(const StaticGeometry &g, const PairRule &rule, const LayeredSpline *spline,
                                const double *dist_img, const cplx *coef, int nimg, int p, int q) {
    const double A = g.a[p]*g.a[q];
    double quad[9];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const double dx = g.q3[6*(size_t)p+2*i]-g.q3[6*(size_t)q+2*j];
            const double dy = g.q3[6*(size_t)p+2*i+1]-g.q3[6*(size_t)q+2*j+1];
            quad[3*i+j] = dx*dx+dy*dy;
        }
    cplx value(0., 0.);
    if (spline) {
        cplx s(0., 0.);
        for (int k = 0; k < 9; ++k) s += spline_eval(*spline, std::sqrt(quad[k]));
        value += cscale(s, A/9);
    }
    double m[6];
    for (int k = 0; k < nimg; ++k) {
        const double d = dist_img[k];
        const int kind = pair_kind(rule, p, q, d);
        if (spline && d > 0) {
            // Positive images are sorted: once one is far, all later ones are.
            if (kind < 0) break;
            double approx = 0;
            for (int j = 0; j < 9; ++j) approx += 1/std::sqrt(quad[j]+d*d);
            value -= cscale(coef[k], A*approx/9);
        }
        if (kind >= 0 || d == 0 || !spline) {
            static_pair(g, p, q, d, kind, true, m);
            value += cscale(coef[k], m[0]);
        }
    }
    return value;
}

// values[i] = image sum for pair (P[i], Q[i]). image_d ascending, zero-distance
// images first; spline_x/spline_c empty when there is no smooth spline.
py::array_t<cplx> layered_pairs(py::dict geometry, arr<double> z, arr<double> cent, arr<double> radius,
                                double near, int outer, arr<long long> P, arr<long long> Q,
                                arr<double> image_d, arr<cplx> image_c, arr<double> spline_x,
                                arr<cplx> spline_c, int workers) {
    const StaticGeometry g = static_geometry(geometry, "layered_pairs");
    const PairRule rule = pair_rule(g, z, cent, radius, near, outer, "layered_pairs");
    const long long n = P.size();
    const int nimg = (int)image_d.size();
    if (workers < 1 || P.ndim() != 1 || Q.ndim() != 1 || Q.size() != n || image_d.ndim() != 1 ||
        image_c.ndim() != 1 || image_c.size() != nimg || spline_x.ndim() != 1)
        throw std::invalid_argument("layered_pairs: invalid pairs, images or worker count");
    const double *dist = image_d.data();
    for (int k = 0; k < nimg; ++k)
        if (!std::isfinite(dist[k]) || dist[k] < 0 || (k && dist[k] < dist[k-1]))
            throw std::invalid_argument("layered_pairs: image distances must be finite, nonnegative and ascending");
    LayeredSpline spline{nullptr, nullptr, 0, 0.};
    const bool smooth = spline_x.size() > 0;
    if (smooth) {
        const long long m = spline_x.size() - 1;
        if (m < 1 || spline_c.ndim() != 2 || spline_c.shape(0) != 4 || spline_c.shape(1) != m)
            throw std::invalid_argument("layered_pairs: spline coefficients must be 4 x (breaks - 1)");
        const double *x = spline_x.data();
        for (long long i = 0; i < m; ++i)
            if (!(x[i+1] > x[i])) throw std::invalid_argument("layered_pairs: spline breaks must increase");
        if (x[0] != 0) throw std::invalid_argument("layered_pairs: spline grid must start at rho = 0");
        // The index estimate assumes rho = rho_max (i/m)^2; it is only a
        // starting point, corrected to the bracketing interval.
        spline = LayeredSpline{x, spline_c.data(), m, x[m]};
    }
    const long long *pp = P.data(), *qq = Q.data();
    for (long long i = 0; i < n; ++i)
        if (pp[i] < 0 || pp[i] >= g.nt || qq[i] < 0 || qq[i] >= g.nt)
            throw std::invalid_argument("layered_pairs: triangle index out of range");
    py::array_t<cplx> out(n);
    cplx *o = out.mutable_data();
    const cplx *coef = image_c.data();
    bool finite = true;
    {
        py::gil_scoped_release release;
#pragma omp parallel for num_threads(workers) schedule(guided) if(workers > 1 && n > 256) reduction(&&:finite)
        for (long long i = 0; i < n; ++i) {
            o[i] = layered_pair(g, rule, smooth ? &spline : nullptr, dist, coef, nimg, (int)pp[i], (int)qq[i]);
            finite = finite && std::isfinite(o[i].real()) && std::isfinite(o[i].imag());
        }
    }
    if (!finite) throw std::range_error("non-finite layered image moments");
    return out;
}
