"""Where a part touches its pads (C13): the contact area that becomes a terminal.

A terminal is one node, an ideal equipotential, so a whole pad as a terminal lets current cross the pad for free:
a 20 x 10 mm loop read 37.4 nH with 1 mm pads against 38.8 nH with 0.25 mm ones. A chip part does not short its
pad. Its metal termination (a thick, solder-filled end cap) covers the pad's inner part, toward the part's body,
and the copper beyond it carries current like any other copper.

    contacts = part_contacts(part, mode='termination')   # {pad index: polygon}, for a two-pad chip part

mode
    'pad'          the whole pad (the earlier behaviour)
    'termination'  (default) for a two-pad SMD part, the band of each pad within the termination length of its edge
                   facing the other pad: the end cap's footprint from the package code in the footprint name
                   (C_0805_2012Metric: 0.5 mm), else half the pad's length along the part. Parts with more pads
                   (ICs: a lead or exposed pad soldered over its whole pad) and through-hole pads keep the whole pad.
    'edge'         a 0.1 mm sliver along the facing edge (the shortest current path a part could have)
"""
import re

import numpy as np
from shapely.geometry import Polygon

# MLCC / chip-resistor termination band (end cap length on the pad side), mm, by EIA imperial size; typical datasheet
# values (Murata GRM, Yageo RC): 0201 0.15, 0402 0.25, 0603 0.35, 0805 0.5, 1206 and larger 0.5-0.6
TERMINATION_MM = {'01005': 0.1, '0201': 0.15, '0402': 0.25, '0603': 0.35, '0805': 0.5, '1206': 0.5, '1210': 0.5,
                  '1806': 0.5, '1812': 0.6, '2010': 0.6, '2220': 0.6, '2512': 0.6}
METRIC = {'0402': '01005', '0603': '0201', '1005': '0402', '1608': '0603', '2012': '0805', '3216': '1206',
          '3225': '1210', '4516': '1806', '4532': '1812', '5025': '2010', '5750': '2220', '6332': '2512'}
EDGE_MM = 0.1
MODES = ('termination', 'pad', 'edge')


def termination_mm(lib):
    """The termination length (mm) a footprint name implies, or None: KiCad's '<x>_0805_2012Metric' names, a
    trailing 'NNNNMetric' alone, or an imperial code as its own token."""
    name = lib.rsplit(':', 1)[-1]
    m = re.search(r'(\d{4})Metric', name)
    if m and m.group(1) in METRIC:
        return TERMINATION_MM[METRIC[m.group(1)]]
    for tok in re.split(r'[_\-\s]', name):
        if tok in TERMINATION_MM:
            return TERMINATION_MM[tok]
    return None


def facing_band(geom, toward, depth):
    """The part of a pad within `depth` (mm) of its extreme toward the point `toward` (along the line from the
    pad's centroid to it). The whole pad when the band would cover it."""
    c = np.array(geom.centroid.coords[0])
    u = np.asarray(toward, float) - c
    n = np.linalg.norm(u)
    if n == 0:
        return geom
    u /= n
    xy = np.array(geom.exterior.coords) if geom.geom_type == 'Polygon' else \
        np.vstack([np.array(g.exterior.coords) for g in geom.geoms])
    s = xy @ u
    lo, hi = s.min(), s.max()
    if hi - depth <= lo:
        return geom
    v = np.array([-u[1], u[0]])
    w = np.abs(xy @ v).max() + (hi - lo) + 1.0                       # a half-plane box past the pad's extent
    cut = Polygon([(hi - depth) * u + w * v, (hi + 1) * u + w * v, (hi + 1) * u - w * v, (hi - depth) * u - w * v])
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
    depth_lib = termination_mm(getattr(part, 'lib', '') or '')
    for k, p in enumerate(part.pads):
        other = part.pads[1 - k].geom.centroid.coords[0]
        if mode == 'edge':
            depth = EDGE_MM
        elif depth_lib is not None:
            depth = depth_lib
        else:                                                        # half the pad's length along the part
            c = np.array(p.geom.centroid.coords[0])
            u = np.asarray(other) - c
            u /= np.linalg.norm(u) or 1.0
            s = np.array(p.geom.exterior.coords) @ u
            depth = 0.5 * (s.max() - s.min())
        out[k] = facing_band(p.geom, other, depth)
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
