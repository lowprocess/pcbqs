"""One look for the validation figures: pcbqs blue, the reference tool orange, closed forms grey."""
PCBQS, REFERENCE, ANALYTIC = '#2a6fdb', '#e8710a', '#6b7280'


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.grid': True, 'grid.alpha': 0.25, 'figure.facecolor': 'white'})
    return plt


def mesh_axis(ax, sizes):
    """A log x axis of mesh sizes, finest on the right, ticks at the sizes used (no colliding minor labels)."""
    from matplotlib.ticker import NullFormatter
    s = sorted(set(sizes))
    ticks = [v for k, v in enumerate(s) if k == 0 or v / s[k - 1] > 1.12]     # neighbours within 12%: one label
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{v:g}' for v in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(max(s) * 1.25, min(s) / 1.25)
