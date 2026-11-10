"""Where a part touches its pads (C13): the contact area that becomes a terminal.

A terminal is one node, an ideal equipotential, so the copper it covers carries current for free. For a chip part
that is mostly right: the metal end cap (termination band, a few tens of um of Cu/Ni/Sn) sits on the pad from
(body length / 2 - band) out to the body's end, and the solder fillet joins the rest of the pad, out to its outer
end, to the cap's end face, where the internal electrodes start. Only the bare copper under the ceramic, between
the pad's inner edge and the cap, conducts as copper.

    contacts = part_contacts(part, mode='termination')   # {pad index: polygon}

mode
    'termination'  (default) two-pad SMD parts: each pad from (L / 2 - band) outward, measured from the part's
                   centre (the two pads' midpoint) along its axis, with the body length L and band from the package
                   code in the footprint name (C_1206_3216Metric: 3.2 mm, 0.5 mm). Without a code, or for ICs
                   (a lead or exposed pad soldered over its whole pad) and through-hole pads: the whole pad.
    'pad'          the whole pad (the behaviour before 2026-11-10)
    'edge'         a 0.1 mm sliver along each pad's inner edge, toward the other pad: the shortest current path a
                   part could have, a bound and not a model
"""
import re

import numpy as np
from shapely.geometry import Polygon

# EIA chip sizes: body length (mm) and termination band (mm), typical datasheet values (Murata GRM, TDK CGA, Yageo
# RC). The band is the cap's length along the body's bottom.
BODY_MM = {'01005': (0.4, 0.1), '0201': (0.6, 0.15), '0402': (1.0, 0.25), '0603': (1.6, 0.35), '0805': (2.0, 0.5),
           '1206': (3.2, 0.5), '1210': (3.2, 0.5), '1806': (4.5, 0.5), '1812': (4.5, 0.6), '2010': (5.0, 0.6),
           '2220': (5.7, 0.6), '2512': (6.4, 0.6)}
METRIC = {'0402': '01005', '0603': '0201', '1005': '0402', '1608': '0603', '2012': '0805', '3216': '1206',
          '3225': '1210', '4516': '1806', '4532': '1812', '5025': '2010', '5750': '2220', '6332': '2512'}
EDGE_MM = 0.1
MODES = ('termination', 'pad', 'edge')


def chip_size(lib):
    """(body length, termination band) in mm that a footprint name implies, or None: KiCad's
    '<x>_0805_2012Metric', a 'NNNNMetric' alone, or an imperial code as its own token."""
    name = lib.rsplit(':', 1)[-1]
    m = re.search(r'(\d{4})Metric', name)
    if m and m.group(1) in METRIC:
        return BODY_MM[METRIC[m.group(1)]]
    for tok in re.split(r'[_\-\s]', name):
        if tok in BODY_MM:
            return BODY_MM[tok]
    return None


def _coords(geom):
    return np.array(geom.exterior.coords) if geom.geom_type == 'Polygon' else \
        np.vstack([np.array(g.exterior.coords) for g in geom.geoms])


def beyond(geom, origin, u, s0, s1=None):
    """The part of a pad where s = (x - origin) . u lies in [s0, s1] (s1 None: no upper bound); the whole pad if
    that is all of it, and the whole pad (a footprint the sizes do not fit) if it is empty."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    s = (_coords(geom) - origin) @ u
    lo, hi = s.min(), s.max()
    a, b = max(s0, lo - 1), hi + 1 if s1 is None else s1
    if a <= lo and b >= hi:
        return geom
    v = np.array([-u[1], u[0]])
    w = np.abs((_coords(geom) - origin) @ v).max() + 1.0
    o = np.asarray(origin, float)
    cut = Polygon([o + a * u + w * v, o + b * u + w * v, o + b * u - w * v, o + a * u - w * v])
    band = geom.intersection(cut)
    return band if not band.is_empty and band.area > 0 else geom


def _through_hole(pad):
    return any(l in ('*.Cu', '*.Cu*') for l in pad.layers) or len([l for l in pad.layers if l.endswith('.Cu')]) > 1


def part_contacts(part, mode='termination'):
    """{index into part.pads: contact polygon}: each pad's contact with the part under `mode` (module docstring).
    Only two-pad SMD parts are trimmed; every other pad is returned whole."""
    if mode not in MODES:
        raise ValueError(f"pad contact mode {mode!r}: one of {', '.join(MODES)}")
    out = {k: p.geom for k, p in enumerate(part.pads)}
    if mode == 'pad' or len(part.pads) != 2 or any(_through_hole(p) for p in part.pads):
        return out
    c = [np.array(p.geom.centroid.coords[0]) for p in part.pads]
    centre = 0.5 * (c[0] + c[1])
    size = chip_size(getattr(part, 'lib', '') or '')
    if mode == 'termination' and size is None:
        return out
    for k, p in enumerate(part.pads):
        u = c[k] - centre                                            # outward, along the part's axis
        if np.linalg.norm(u) == 0:
            continue
        if mode == 'edge':
            inner = ((_coords(p.geom) - centre) @ (u / np.linalg.norm(u))).min()
            out[k] = beyond(p.geom, centre, u, inner, inner + EDGE_MM)
        else:
            length, band = size
            out[k] = beyond(p.geom, centre, u, length / 2 - band)
    return out


def contact_terminal(part, pads, mode='termination', layers=None):
    """[(layer, polygon)] for `pads` (a subset of part.pads), each trimmed to its contact. `layers`: the stack's
    copper layer names, for through-hole pads on '*.Cu' (rail.terminal's rule); None keeps the pad's own layers."""
    geo = part_contacts(part, mode)
    out = []
    for p in pads:
        k = next((i for i, q in enumerate(part.pads) if q is p), None)   # another part's pad (a rail's source): whole
        if layers is not None and any(l in ('*.Cu', '*.Cu*') for l in p.layers):
            ls = layers
        elif layers is not None:
            ls = [l for l in p.layers if l in layers]
        else:
            ls = p.layers
        out += [(l, p.geom if k is None else geo[k]) for l in ls]
    return out
