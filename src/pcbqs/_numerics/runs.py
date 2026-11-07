"""Private numerical snapshot; independently maintained. See PROVENANCE.json."""
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

class Run:
    """A straight run of trace on layer k: centre line from c0 along unit t, half-width a, from x0 to x1 (m)."""
    def __init__(self, k, c0, t, a, x0, x1):
        self.k, self.c0, self.t, self.a, self.x0, self.x1 = k, np.asarray(c0, float), np.asarray(t, float), a, x0, x1
        self.n = np.array([-self.t[1], self.t[0]])

    def polygon(self, pad=0.0):
        return Polygon([self.c0 + self.t * x + self.n * y for x, y in ((self.x0, -self.a - pad), (self.x1, -self.a - pad),
                                                                     (self.x1, self.a + pad), (self.x0, self.a + pad))])

    def local(self, P):
        d = np.atleast_2d(P) - self.c0
        return d @ self.t, d @ self.n


def find_runs(geoms, h_of, signal=None, w_max=2.0e-3, min_len=3.0, margin=1.0, tol=1e-9):
    """Straight runs of trace: pairs of antiparallel outline segments facing each other across copper, at most
    w_max apart and overlapping by at least min_len x max(width, h); their overlap, less `margin` x max(width, h)
    at each end. geoms: {layer: (Multi)Polygon} (m); h_of: {layer: column length}; signal: only runs on it."""
    runs = []
    for k, g in geoms.items():
        h = h_of[k]
        for p in getattr(g, 'geoms', [g]):
            if p.is_empty:
                continue
            p = orient(p, 1.0)                                     # exterior ccw, holes cw: copper on the left
            segs = []
            for ring in [p.exterior] + list(p.interiors):
                c = np.asarray(ring.coords)[:, :2]
                for a_, b_ in zip(c[:-1], c[1:]):
                    L = np.linalg.norm(b_ - a_)
                    if L > tol:
                        d = (b_ - a_) / L
                        segs.append((a_, b_, d, np.array([-d[1], d[0]]), L))
            cand = []
            for i, (a1, b1, d1, n1, L1) in enumerate(segs):
                for a2, b2, d2, _, _ in segs[i + 1:]:
                    w = (a2 - a1) @ n1                            # antiparallel, facing across the copper
                    if d1 @ d2 > -0.9999 or not (tol < w <= w_max):
                        continue
                    s0, s1 = sorted(((b2 - a1) @ d1, (a2 - a1) @ d1))
                    lo, hi = max(0.0, s0), min(L1, s1)
                    e = max(w, h)
                    if hi - lo < min_len * e + 2 * margin * e:
                        continue
                    cand.append((hi - lo, k, a1 + n1 * w / 2, d1, w / 2, lo + margin * e, hi - margin * e))
            cand.sort(key=lambda c: -c[0])
            taken = []
            for _, k_, c0, t, a, x0, x1 in cand:                  # the longest first, on the copper, not overlapping
                r = Run(k_, c0, t, a, x0, x1)
                poly = r.polygon()
                if p.buffer(tol).contains(poly.buffer(-tol)) and \
                        (signal is None or signal.buffer(tol).contains(poly.buffer(-tol))) and \
                        not any(poly.intersection(q).area > 1e-3 * poly.area for q in taken):
                    taken.append(poly)
                    runs.append(r)
    return runs
