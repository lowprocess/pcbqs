"""
The copper of a net as triangle meshes, one per layer (pcbqs's own mesher, geo.mesh_polygons), and the linear
(P1) finite-element matrices on them: what the DC drop (dc.py) and the plane-pair cavity (cavity.py) solve on.

    sheets = net_sheets(board, '+1V0', points=[(x, y, r), ...], dmax=1.0)   # {layer: Sheet}, board.json units (mm)
    K, M = stiffness(sh.nodes, sh.tris), mass(sh.nodes, sh.tris)

A point (a via) is a vertex of the mesh with an octagon of its radius around it (the via fan of mesh_polygons);
Sheet.at(i) is the vertex ids of point i on that sheet: its centre and its ring. A region (a pad) has its outline
made mesh edges; Sheet.region(k) is the vertex ids on and in it.
"""
from dataclasses import dataclass
import math

import numpy as np
import shapely
from shapely.affinity import scale
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree

from .._numerics.geo import mesh_polygons

SIGMA_CU = 5.8e7                # copper, S/m (shared by mqs, dc, cavity, stack)


@dataclass
class Sheet:
    layer: str
    z: float                    # mm
    t: float                    # copper thickness, mm
    geom: object                # shapely, mm
    nodes: np.ndarray           # (N, 2) mm
    tris: np.ndarray            # (M, 3)
    points: list                # [(x, y, r)] mm: the points asked for that lie on this copper
    rings: list                 # vertex ids of each of them (centre first), [] where it is not on this copper
    key: list                   # the index in the caller's list of each of `points`
    regions: list = None        # vertex ids of each region asked for on this layer ([]: not on this copper)

    def at(self, i):
        """The vertex ids of the caller's point i on this sheet ([]: not on it)."""
        return self.rings[self.key.index(i)] if i in self.key else []

    def region(self, k):
        return self.regions[k] if self.regions and k < len(self.regions) else []


def copper_of(board, net, layer):
    """The copper of `net` (a name, or a set of names: a rail tree's nets) on `layer` (board.json's dict), one
    shapely geometry in mm, or None."""
    nets = {net} if isinstance(net, str) else set(net)
    gs = [shape(c['polygon']) for c in board['copper'].get(layer, []) if c.get('net') in nets]
    g = unary_union(gs).buffer(0) if gs else None
    return None if g is None or g.is_empty else g


def stack_z(board):
    """{layer: (z, t)} of the copper layers, mm."""
    return {c['name']: (c['z'], c['t']) for c in board['stack']['copper']}


def copper_centres(board):
    """{layer: (centre z, t)}, mm, of each layer's copper with its thickness put into the stack. board.json (from
    kicad2ems) stacks the dielectrics and puts each copper layer at their interface with no thickness: KiCad's
    dielectric thickness is the gap between facing copper surfaces. Thick copper keeps those gaps when each layer's
    slab starts at its z plus the copper below it: [z + C, z + C + t], C the thicknesses of the layers beneath."""
    zt = stack_z(board)
    out, below = {}, 0.0
    for lay in sorted(zt, key=lambda l: zt[l][0]):
        z, t = zt[lay]
        out[lay] = (z + below + t / 2, t)
        below += t
    return out


def _outline_lines(polys, geom):
    """The outlines of `polys` (mm) inside `geom` as mesh_polygons' constraint segments (m)."""
    segs = []
    for q in polys:
        q = q.intersection(geom)
        for part in getattr(q, 'geoms', [q]):
            if part.geom_type != 'Polygon' or part.is_empty:
                continue
            for ring in [part.exterior] + list(part.interiors):
                c = np.asarray(ring.coords)[:, :2] * 1e-3
                segs += [(tuple(a), tuple(b)) for a, b in zip(c[:-1], c[1:])]
    return segs


def _regular(poly, sides):
    """An axis-aligned regular polygon with the centroid and area of `poly`."""
    c = poly.centroid
    R = math.sqrt(2*poly.area/(sides*math.sin(2*math.pi/sides)))
    a0 = math.pi/sides
    return [(c.x+R*math.cos(a0+2*math.pi*k/sides), c.y+R*math.sin(a0+2*math.pi*k/sides)) for k in range(sides)]


def _round(poly):
    return 4*math.pi*poly.area > .85*poly.length**2


def simplify_copper(geom, keep=(), points=(), drop_lone=True, area=1.0, sides=8):
    """Copper (mm) with lone pads dropped and small round pads/holes made regular polygons.

    A piece under `area` mm^2 that touches no `keep` polygon (terminal pads) and
    holds at most one of `points` (via centres) carries no lateral current: its via
    passes through to the next layer with copper, so it is dropped. Remaining small
    round pieces and holes become `sides`-gons of equal area (None/0 keeps them).
    Returns (geometry or None, counts).
    """
    counts = dict(dropped_pads=0, regular_pads=0, regular_holes=0)
    if geom is None or geom.is_empty or (not drop_lone and not sides):
        return geom, counts
    xy = np.asarray([(p[0], p[1]) for p in points], float).reshape(-1, 2)
    out = []
    for piece in getattr(geom, 'geoms', [geom]):
        if piece.geom_type != 'Polygon' or piece.is_empty:
            continue
        small = piece.area < area
        terminal = small and any(piece.intersects(k) for k in keep)
        if small and not terminal:
            vias = int(shapely.contains_xy(piece.buffer(1e-6), xy[:, 0], xy[:, 1]).sum()) if len(xy) else 0
            if drop_lone and vias <= 1:
                counts['dropped_pads'] += 1
                continue
        shell = list(piece.exterior.coords)
        if sides and small and not terminal and _round(Polygon(shell)):
            shell = _regular(Polygon(shell), sides); counts['regular_pads'] += 1
        holes = []
        for ring in piece.interiors:
            hole = Polygon(ring)
            if sides and hole.area < area and _round(hole):
                holes.append(_regular(hole, sides)); counts['regular_holes'] += 1
            else:
                holes.append(list(ring.coords))
        fixed = Polygon(shell, holes)
        if not fixed.is_valid:
            fixed = shapely.make_valid(fixed)
        out += [p for p in getattr(fixed, 'geoms', [fixed]) if p.geom_type == 'Polygon' and not p.is_empty]
    if not out:
        return None, counts
    return (out[0] if len(out) == 1 else MultiPolygon(out)), counts


def overlapping_vias(points, margin=0.1):
    """Vias whose rings would cross or nearly touch, as {absorbed index: host index} over `points` [(x, y, r)].

    Two rings closer than `margin` x the smaller radius (crossing, or one drilled into the other) leave the mesher
    acute corners and slivers of copper between them; with a minimum-angle switch Triangle then refines without
    end (wpthb's ground: two stitching vias 0.25 mm apart, 2.3 million vertices at any dmax). Such vias are one
    plated hole electrically, so the larger one's ring is meshed and the others share its vertices. Largest
    first: each via still free hosts the free vias its ring overlaps, so a chain is not collapsed onto one end.
    """
    if len(points) < 2:
        return {}
    xy = np.asarray([(p[0], p[1]) for p in points], float)
    r = np.asarray([p[2] for p in points], float)
    pairs = cKDTree(xy).query_pairs(2 * (1 + margin) * float(r.max()), output_type='ndarray')
    d = np.hypot(*(xy[pairs[:, 0]] - xy[pairs[:, 1]]).T)
    pairs = pairs[d < r[pairs].sum(1) + margin * r[pairs].min(1)]
    near = {}
    for a, b in pairs:
        near.setdefault(int(a), []).append(int(b)); near.setdefault(int(b), []).append(int(a))
    host = {}
    for i in sorted(near, key=lambda k: (-r[k], k)):
        if i in host:
            continue
        for k in near[i]:
            if k not in host and not any(h == k for h in host.values()):
                host[k] = i
    return host


def mesh_sheet(geom, points=(), dmax=1.0, w_ref=None, coarsen=None, regions=(), fine=None, dmax_fine=None,
               grow=0.5, edge_rings=1, edge_zone=None, via_sides=None, quality=None, info=None):
    """Mesh one piece of copper (mm): (nodes mm, tris, [vertex ids of each point: centre + ring], [vertex ids of
    each region]) for the points (x, y, r) inside it (None where one is not) and the regions (shapely polygons, mm:
    pads), whose outlines are made mesh edges (a region on none of this copper: []). fine (shapely, mm) is meshed at
    dmax_fine (mm). coarsen: mesh_polygons' (cells grow away from the features to coarsen x dmax). via_sides: each
    via's ring a regular polygon of that many sides with the via circle's perimeter (default: an octagon in it).
    Vias whose rings overlap are meshed as the largest one's ring, shared (overlapping_vias); info (a dict), if
    given, has their count added to info['merged_vias']."""
    for name,value in [('dmax',dmax),('w_ref',w_ref),('coarsen',coarsen),('grow',grow),('dmax_fine',dmax_fine)]:
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(name+' must be finite and positive')
    if coarsen is not None and coarsen < 1:
        raise ValueError('coarsen must be at least 1')
    if not isinstance(edge_rings,(int,np.integer)) or isinstance(edge_rings,bool) or edge_rings < 1:
        raise ValueError('edge_rings must be a positive integer')
    if via_sides is not None and (isinstance(via_sides,bool) or not isinstance(via_sides,(int,np.integer)) or via_sides < 3):
        raise ValueError('via_sides must be an integer of at least 3')
    grow_r = math.pi/(via_sides*math.sin(math.pi/via_sides)) if via_sides else 1.0
    xy = np.asarray([(x, y) for x, y, _ in points], float).reshape(-1, 2)
    hit = shapely.contains_xy(geom.buffer(1e-6), xy[:, 0], xy[:, 1]) if len(xy) else np.zeros(0, bool)
    inside = list(np.flatnonzero(hit))                     # the copper buffered once: not per point
    merged = overlapping_vias([points[i] for i in inside])
    merged = {inside[k]: inside[h] for k, h in merged.items()}     # absorbed point -> the point whose ring it shares
    if info is not None:
        info['merged_vias'] = info.get('merged_vias', 0) + len(merged)
    inside = [i for i in inside if i not in merged]
    vias = [(points[i][0] * 1e-3, points[i][1] * 1e-3, points[i][2] * 1e-3 * grow_r) + ((int(via_sides),) if via_sides else ())
            for i in inside]
    g = scale(geom, 1e-3, 1e-3, origin=(0, 0))
    live = [k for k, q in enumerate(regions) if q.intersects(geom)]
    lines = _outline_lines([regions[k] for k in live], geom)
    kw = {}
    if fine is not None and dmax_fine:
        kw = dict(fine=scale(fine, 1e-3, 1e-3, origin=(0, 0)), dmax_fine=dmax_fine * 1e-3)
    zone = scale(edge_zone,1e-3,1e-3,origin=(0,0)) if edge_zone is not None else None
    nodes, tris, vid = mesh_polygons(g, dmax * 1e-3, (w_ref or dmax) * 1e-3, ncross=edge_rings,
                                     vias=vias, coarsen=coarsen, grow=grow, edge_zone=zone, lines=lines, quality=quality,
                                     **kw)
    nodes = nodes * 1e3
    rings = [None] * len(points)
    for i, v in zip(inside, vid):
        x, y, r = points[i]
        d = np.hypot(nodes[:, 0] - x, nodes[:, 1] - y)
        ids = np.flatnonzero(d <= r * grow_r * 1.02 + 1e-9)
        c = int(np.argmin(d))
        rings[i] = [c] + [int(k) for k in ids if k != c]
    for i, h in merged.items():
        rings[i] = rings[h]
    reg = [[] for _ in regions]
    for k in live:
        q = regions[k].buffer(1e-4)
        ids = np.flatnonzero(shapely.contains_xy(q, nodes[:, 0], nodes[:, 1]))
        if not len(ids):                                    # a pad smaller than a cell: its nearest vertex
            c = regions[k].centroid
            ids = [int(np.argmin(np.hypot(nodes[:, 0] - c.x, nodes[:, 1] - c.y)))]
        reg[k] = [int(i) for i in ids]
    return nodes, tris, rings, reg


def net_sheets(board, net, points=(), dmax=1.0, layers=None, coarsen=None, regions=None, fine=None, dmax_fine=None,
               quality=None, info=None):
    """{layer: Sheet} of the copper of `net` (a name or a set: a rail tree) on each copper layer (or `layers`), every
    point (x, y, r) of `points` that lies on a layer's copper made a vertex with its ring. regions: {layer:
    [polygon]} (pads) whose outlines become mesh edges; Sheet.region(k) their vertex ids. fine, dmax_fine: a region
    (shapely, mm) meshed finer."""
    zt = stack_z(board)
    out = {}
    for lay in (layers or zt):
        g = copper_of(board, net, lay)
        if g is None:
            continue
        rg = (regions or {}).get(lay, [])
        nodes, tris, rings, reg = mesh_sheet(g, points, dmax, coarsen=coarsen, regions=rg, fine=fine,
                                             dmax_fine=dmax_fine, quality=quality, info=info)
        key = [i for i, r in enumerate(rings) if r is not None]
        out[lay] = Sheet(lay, zt[lay][0], zt[lay][1], g, nodes, tris, [points[i] for i in key],
                         [rings[i] for i in key], key, reg)
    return out


def _grads(nodes, tris):
    """Areas (M,) and the gradients of the three hat functions of each triangle (M, 3, 2)."""
    P = nodes[tris]                                         # (M, 3, 2)
    d1, d2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    area = 0.5 * (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    # grad of hat i: perpendicular of the opposite edge / (2 area)
    e = np.stack([P[:, 2] - P[:, 1], P[:, 0] - P[:, 2], P[:, 1] - P[:, 0]], 1)
    g = np.stack([-e[..., 1], e[..., 0]], -1) / (2 * area[:, None, None])
    return np.abs(area), g * np.sign(area)[:, None, None]


def stiffness(nodes, tris, coef=1.0):
    """The P1 stiffness matrix: sum over triangles of coef * area * grad(phi_i) . grad(phi_j). coef: a scalar or
    one per triangle (a sheet conductance, S per square, makes it the sheet's conductance matrix)."""
    area, g = _grads(nodes, tris)
    k = np.einsum('mia,mja->mij', g, g) * (area * np.broadcast_to(coef, area.shape))[:, None, None]
    r = np.repeat(tris, 3, axis=1).ravel()
    c = np.tile(tris, (1, 3)).ravel()
    n = len(nodes)
    return coo_matrix((k.ravel(), (r, c)), shape=(n, n)).tocsr()


def mass(nodes, tris, lumped=True):
    """The P1 mass matrix (lumped: a third of each triangle's area on each of its vertices, diagonal)."""
    area, _ = _grads(nodes, tris)
    n = len(nodes)
    if lumped:
        return coo_matrix((np.repeat(area / 3, 3), (tris.ravel(), tris.ravel())), shape=(n, n)).tocsr()
    loc = (np.ones((3, 3)) + np.eye(3)) / 12
    k = loc[None] * area[:, None, None]
    r = np.repeat(tris, 3, axis=1).ravel()
    c = np.tile(tris, (1, 3)).ravel()
    return coo_matrix((k.ravel(), (r, c)), shape=(n, n)).tocsr()


def gradient(nodes, tris, v):
    """The gradient of a P1 field v (one value per node) on each triangle, (M, 2) per unit of the node units."""
    _, g = _grads(nodes, tris)
    return np.einsum('mia,mi->ma', g, v[tris])
