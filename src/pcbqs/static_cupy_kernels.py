"""Fused float64 static pair integration, shared by CUDA and HIP through CuPy.

One thread owns one pair; near observation reductions retain their order.
There are no pair-by-quadrature temporaries. Far product-rule pairs and
analytic near pairs run in separate launches (or in kind-sorted order), since
mixing them in one wavefront serializes the cheap far path behind the near
path's asinh/atan2 loop, which was measured at about 8x on an RX 7800 XT.
"""
SOURCE = r'''
struct Geometry {
    const double* V; const double* L; const double* tau; const double* un;
    const double* area; const double* P3;
    const double* P0; const double* W0; int n0;
    const double* P1; const double* W1; int n1;
    const double* P2; const double* W2; int n2;
};

__device__ __forceinline__ void pair_moments(const Geometry& g, int p, int q, double d, int k,
                                             int scalar, double* m) {
    double ad = fabs(d);
    for (int j=0; j<6; ++j) m[j] = 0.;
    if (k < 0) {
        double w = g.area[p] * g.area[q] / 9.;
        for (int ip=0; ip<3; ++ip) {
            double x=g.P3[6*(size_t)p+2*ip], y=g.P3[6*(size_t)p+2*ip+1];
            for (int iq=0; iq<3; ++iq) {
                double sx=g.P3[6*(size_t)q+2*iq], sy=g.P3[6*(size_t)q+2*iq+1];
                double dx=x-sx, dy=y-sy;
                double gg=w/sqrt(dx*dx+dy*dy+d*d);
                m[0]+=gg;
                if (!scalar) {
                    m[1]+=gg*x; m[2]+=gg*y; m[3]+=gg*sx; m[4]+=gg*sy; m[5]+=gg*(x*sx+y*sy);
                }
            }
        }
        return;
    }
    const double* points = k==0 ? g.P0 : (k==1 ? g.P1 : g.P2);
    const double* weights = k==0 ? g.W0 : (k==1 ? g.W1 : g.W2);
    int nq = k==0 ? g.n0 : (k==1 ? g.n1 : g.n2);
    for (int j=0; j<nq; ++j) {
        double x=points[2*((size_t)p*nq+j)], y=points[2*((size_t)p*nq+j)+1];
        double inner=0., irx=0., iry=0.;
        for (int e=0; e<3; ++e) {
            size_t ve=6*(size_t)q+2*e;
            double vx=g.V[ve]-x, vy=g.V[ve+1]-y;
            double sm=vx*g.tau[ve]+vy*g.tau[ve+1], sp=sm+g.L[3*(size_t)q+e];
            double t0=vx*g.un[ve]+vy*g.un[ve+1];
            double r02=t0*t0+d*d, r0=sqrt(r02);
            double rm=sqrt(r02+sm*sm), rp=sqrt(r02+sp*sp);
            double lg=r0>1e-14*g.L[3*(size_t)q+e] ? asinh(sp/r0)-asinh(sm/r0) : 0.;
            double term=t0*lg;
            if (ad>0.) term-=ad*(atan2(t0*sp,r02+ad*rp)-atan2(t0*sm,r02+ad*rm));
            inner+=term;
            if (!scalar) {
                double edge=.5*(r02*lg+sp*rp-sm*rm);
                irx+=edge*g.un[ve]; iry+=edge*g.un[ve+1];
            }
        }
        double w=g.area[p]*weights[j];
        m[0]+=w*inner;
        if (!scalar) {
            double sx=irx+x*inner, sy=iry+y*inner;
            m[1]+=w*x*inner; m[2]+=w*y*inner;
            m[3]+=w*sx; m[4]+=w*sy; m[5]+=w*(x*sx+y*sy);
        }
    }
}

#define GEOMETRY_ARGS \
    const double* V, const double* L, const double* tau, const double* un, \
    const double* area, const double* P3, \
    const double* P0, const double* W0, int n0, \
    const double* P1, const double* W1, int n1, \
    const double* P2, const double* W2, int n2
#define GEOMETRY {V, L, tau, un, area, P3, P0, W0, n0, P1, W1, n1, P2, W2, n2}

// Explicit pair lists (backend protocol). `order` (nullable) lets the host
// launch kind-sorted pairs so wavefronts do not mix far and near work.
extern "C" __global__ void static_pairs(GEOMETRY_ARGS,
    const int* pp, const int* qq, const double* dd, const int* kk, const int* order,
    int n, int scalar, double* out) {
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n) return;
    int i = order ? order[t] : t;
    Geometry g = GEOMETRY;
    int width = scalar ? 1 : 6;
    double m[6];
    pair_moments(g, pp[i], qq[i], dd[i], kk[i], scalar, m);
    for (int j=0; j<width; ++j) out[(size_t)i*width+j]=m[j];
}

// Host FillEngine.kinds() classification, without FMA contraction, so the
// device chooses the same quadrature as every host backend.
__device__ __forceinline__ int classify(const double* cent, const double* radius, int a, int b,
                                        double d, double near, int all_near, int outer) {
    double dx=__dsub_rn(cent[2*a], cent[2*b]), dy=__dsub_rn(cent[2*a+1], cent[2*b+1]);
    double distance=sqrt(__dadd_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)), __dmul_rn(d, d)));
    double size=__dadd_rn(radius[a], radius[b]);
    if (!all_near && !(distance <= __dmul_rn(near, size))) return -1;
    return distance <= size ? outer : 0;
}

// Entry (r, c) of the dense symmetric matrices, from the canonical pair
// (min, max): reciprocal entries are bitwise identical by construction.
__device__ __forceinline__ void store_entry(double* out, size_t plane, size_t i, int r, int c,
                                            const double* m, int scalar) {
    out[i] = m[0];
    if (scalar) return;
    double mx, my;
    if (r == c) { mx=(m[1]+m[3])/2; my=(m[2]+m[4])/2; }
    else if (r < c) { mx=m[1]; my=m[2]; }
    else { mx=m[3]; my=m[4]; }
    out[plane+i]=mx; out[2*plane+i]=my; out[3*plane+i]=m[5];
}

// Rows [r0, r0+rows) x all columns. Far pairs are integrated immediately;
// near pairs are compacted into per-level lists for static_rows_near.
// `out` holds planes M0, Mx, My, Mrr spaced `plane` apart.
// count[0..1]: list lengths, count[2]: analytic pairs with r <= c.
extern "C" __global__ void static_rows_far(GEOMETRY_ARGS,
    const double* z, const double* cent, const double* radius,
    int r0, int rows, int n, double near, int all_near, int outer, int scalar,
    double* out, size_t plane, int* list0, int* list1, unsigned int* count) {
    size_t total = (size_t)rows*n;
    size_t i = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;
    int r = r0 + (int)(i / n), c = (int)(i % n);
    int a = min(r, c), b = max(r, c);
    double d = z[a]-z[b];
    int k = classify(cent, radius, a, b, d, near, all_near, outer);
    if (k >= 0) {
        if (r <= c) atomicAdd(count+2, 1u);
        if (k == 0) list0[atomicAdd(count, 1u)] = (int)i;
        else list1[atomicAdd(count+1, 1u)] = (int)i;
        return;
    }
    Geometry g = GEOMETRY;
    double m[6];
    pair_moments(g, a, b, d, -1, scalar, m);
    store_entry(out, plane, i, r, c, m, scalar);
}

extern "C" __global__ void static_rows_near(GEOMETRY_ARGS,
    const double* z, int r0, int n, int level, int scalar,
    double* out, size_t plane, const int* list, const unsigned int* count) {
    unsigned int length = *count;
    Geometry g = GEOMETRY;
    for (unsigned int j = blockDim.x * blockIdx.x + threadIdx.x; j < length; j += blockDim.x * gridDim.x) {
        size_t i = (size_t)list[j];
        int r = r0 + (int)(i / n), c = (int)(i % n);
        int a = min(r, c), b = max(r, c);
        double m[6];
        pair_moments(g, a, b, z[a]-z[b], level, scalar, m);
        store_entry(out, plane, i, r, c, m, scalar);
    }
}
'''
