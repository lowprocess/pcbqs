"""Pictures of thermal results: the temperature of each layer, and two results side by side with their difference."""
import numpy as np


def _axes(ax, p, title):
    x0, y0, x1, y1 = p.outline.bounds
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)                 # the board's frame has y down the page
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def _draw(ax, res, layer, vmin, vmax, cmap='inferno'):
    p = res.problem
    if res.method == 'grid':
        ny, nx = res.T[layer].shape
        ext = (res.x0, res.x0 + nx * res.size, res.y0 + ny * res.size, res.y0)
        return ax.imshow(res.T[layer], extent=ext, origin='upper', vmin=vmin, vmax=vmax, cmap=cmap,
                         interpolation='nearest')
    return ax.tripcolor(res.nodes[:, 0], res.nodes[:, 1], res.tris, res.T[layer], vmin=vmin, vmax=vmax, cmap=cmap,
                        shading='gouraud')


def _copper(ax, p, layer):
    g = p.layers[layer].copper
    for q in getattr(g, 'geoms', [g] if g is not None else []):
        if q.geom_type == 'Polygon':
            for ring in [q.exterior, *q.interiors]:
                c = np.asarray(ring.coords)
                ax.plot(c[:, 0], c[:, 1], color='white', lw=0.3, alpha=0.5)


def _labels(ax, res, layer):
    seen = set()
    for s in res.problem.sources:
        if s.layer == layer and s.name not in seen:
            seen.add(s.name)
            c = s.geom.centroid
            ax.annotate(f"{s.name} {res.parts[s.name]['board_C']:.0f}", (c.x, c.y), color='white', fontsize=6,
                        ha='center', va='center')


def picture(res, path, dpi=150):
    """Each layer's temperature (top layer first)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = res.problem
    L = len(p.layers)
    x0, y0, x1, y1 = p.outline.bounds
    w = 4.2
    hgt = w * (y1 - y0) / (x1 - x0) + 0.6
    fig, axs = plt.subplots(1, L, figsize=(w * L + 0.8, hgt), squeeze=False, constrained_layout=True)
    vmin, vmax = float(np.nanmin(res.T)), float(np.nanmax(res.T))
    for ax, l in zip(axs[0], reversed(range(L))):
        im = _draw(ax, res, l, vmin, vmax)
        _copper(ax, p, l)
        _labels(ax, res, l)
        _axes(ax, p, f"{p.layers[l].name}: up to {np.nanmax(res.T[l]):.1f} C")
    fig.colorbar(im, ax=axs[0].tolist(), shrink=0.85, label='C')
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def difference(a, b, layer):
    """b minus a on `layer` at a's cell centres (a: a grid result): (ny, nx), NaN off the board."""
    ny, nx = a.T[layer].shape
    X, Y = np.meshgrid(a.x0 + (np.arange(nx) + 0.5) * a.size, a.y0 + (np.arange(ny) + 0.5) * a.size)
    return b.sample(layer, X, Y) - a.T[layer]


def compare(grid, mesh, path, dpi=150):
    """The top layer from the grid and from the mesh, and mesh minus grid."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p = grid.problem
    l = len(p.layers) - 1
    x0, y0, x1, y1 = p.outline.bounds
    w = 4.2
    hgt = w * (y1 - y0) / (x1 - x0) + 0.6
    fig, axs = plt.subplots(1, 3, figsize=(3 * w + 1.4, hgt), constrained_layout=True)
    vmin = min(np.nanmin(grid.T[l]), np.nanmin(mesh.T[l]))
    vmax = max(np.nanmax(grid.T[l]), np.nanmax(mesh.T[l]))
    for ax, r in zip(axs, (grid, mesh)):
        im = _draw(ax, r, l, vmin, vmax)
        _copper(ax, p, l)
        _labels(ax, r, l)
        _axes(ax, p, f"{r.method}, {r.size:g} mm, {r.unknowns:,} unknowns: up to {np.nanmax(r.T[l]):.1f} C")
    fig.colorbar(im, ax=axs[:2].tolist(), shrink=0.85, label='C')
    d = difference(grid, mesh, l)
    lim = max(float(np.nanmax(np.abs(d))), 1e-6)
    ny, nx = d.shape
    im = axs[2].imshow(d, extent=(grid.x0, grid.x0 + nx * grid.size, grid.y0 + ny * grid.size, grid.y0), origin='upper',
                       cmap='RdBu_r', vmin=-lim, vmax=lim, interpolation='nearest')
    _copper(axs[2], p, l)
    _axes(axs[2], p, f"mesh minus grid: within {lim:.2f} K")
    fig.colorbar(im, ax=axs[2], shrink=0.85, label='K')
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
