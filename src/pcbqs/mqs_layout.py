"""Lightweight layout/mesh artifacts for MQS preview and field overlays (model mm)."""
import re
from pathlib import Path
import json
import numpy as np
from shapely.geometry import shape, mapping, box, Point
from shapely.affinity import rotate, translate
from shapely.ops import unary_union


def components(board, crop, active_parts):
    from ._packages import package, SIZES
    result = []
    layers = sorted(board['stack']['copper'], key=lambda c: c['z'])
    for fp in board.get('prims', {}).get('footprints', []):
        pads = [shape(p['geom']) for p in fp.get('pads', []) if p.get('geom')]
        if fp.get('dnp') or not pads:
            continue
        extent = unary_union(pads)
        if not extent.intersects(crop):
            continue
        key = package(fp.get('lib'))
        dims = re.search(r'(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)(?:x(\d+(?:\.\d+)?))?mm', fp.get('lib', ''))
        if key or dims:
            L, W, H = SIZES[key] if key else (float(dims[1]), float(dims[2]), float(dims[3] or .5))
            cx, cy = fp.get('x', extent.centroid.x), fp.get('y', extent.centroid.y)
            poly = translate(rotate(box(-L/2, -W/2, L/2, W/2), -fp.get('rot', 0), origin=(0, 0)), cx, cy)
            origin = 'nominal package dimensions; typical height' if key else 'footprint name; assumed height if absent'
        else:
            poly, H, origin = extent.envelope, .5, 'pad envelope and assumed 0.5 mm height'
        lay = layers[0] if fp.get('side') == 'B' else layers[-1]
        result.append(dict(ref=fp['ref'], value=fp.get('value', ''), layer=lay['name'], z=lay['z'],
                           side=fp.get('side', 'F'), polygon=mapping(poly), height=H,
                           active=fp['ref'] in active_parts, shape_source=origin))
    return result


def crop_layout(board, crop, nets, refs):
    layout = {k: board[k] for k in ['format', 'version', 'units', 'source', 'stack'] if k in board}
    layout.update(outline=mapping(crop), region=list(crop.bounds), ports=[], parts=[],
                  active_nets=sorted(nets), active_parts=sorted(refs), copper={}, vias=[])
    for layer, records in board['copper'].items():
        layout['copper'][layer] = []
        for record in records:
            g = shape(record['polygon']).intersection(crop)
            for poly in getattr(g, 'geoms', [g]):
                if poly.geom_type == 'Polygon' and not poly.is_empty:
                    layout['copper'][layer].append(dict(net=record['net'], polygon=mapping(poly), active=record['net'] in nets))
    layout['vias'] = [dict(v, active=v['net'] in nets) for v in board['vias'] if crop.covers(Point(v['x'], v['y']))]
    layout['components'] = components(board, crop, refs)
    layout['notes'] = ['Component silhouettes are visual package approximations, not electromagnetic solids.']
    return layout


def geometry(layout):
    """The existing dump-info geometry schema plus optional display metadata."""
    def rings(record):
        g = record['polygon']
        return [g['coordinates']] if g['type'] == 'Polygon' else g['coordinates']
    return dict(outline=layout['outline']['coordinates'][0],
                cu_z={c['name']: c['z'] for c in layout['stack']['copper']},
                thickness=layout['stack'].get('thickness', max(c['z'] for c in layout['stack']['copper'])),
                copper={l: [r for rec in records for r in rings(rec)] for l, records in layout['copper'].items()},
                active_copper={l: [r for rec in records if rec['active'] for r in rings(rec)] for l, records in layout['copper'].items()},
                inactive_copper={l: [r for rec in records if not rec['active'] for r in rings(rec)] for l, records in layout['copper'].items()},
                components=layout['components'], active_nets=layout['active_nets'],
                vias=[[v[k] for k in ['x','y','r','z0','z1']] for v in layout['vias']])


def mesh_stats(q):
    stats = {}
    for layer, ids in q.layer_tris.items():
        if getattr(q, 'sheet_layer', {}).get(layer, layer) != layer:
            continue                                     # a copy through the thickness: the same triangles
        v = q.nodes[q.tris[ids]] * 1000
        edges = np.linalg.norm(v - np.roll(v, 1, axis=1), axis=2)
        angles = []
        for i in range(3):
            a, b = v[:, (i+1)%3]-v[:, i], v[:, (i+2)%3]-v[:, i]
            angles.append(np.degrees(np.arccos(np.clip((a*b).sum(1)/(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)), -1, 1))))
        stats[layer] = dict(triangles=len(ids), max_edge_mm=float(edges.max()), min_angle_deg=float(np.min(angles)))
    return stats


def save_layout(out, board, q, crop, selected):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    layout = crop_layout(board, crop, q.nets, {selected['converter'], *selected['capacitors']})
    (out/'mqs_board.json').write_text(json.dumps(layout))
    nodes = np.column_stack([q.nodes * 1000, np.zeros(len(q.nodes))])
    # Mesh_sheet allocates distinct vertices per physical layer.
    nodes[q.tris.ravel(), 2] = np.repeat(q.z_of * 1000, 3)
    np.savez_compressed(out/'mqs_mesh_mqs.npz', kind='tris', nodes=nodes, tris=q.tris)
    return layout
