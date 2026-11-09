"""Shared paths, CLI runner and plot style for the power-workflow examples."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGES = HERE / 'images'
OUT = Path(os.environ.get('POWER_WORKFLOW_OUT', HERE / 'out'))
DEMOS = Path(os.environ.get('KICAD_DEMOS', '/usr/share/kicad/demos'))
CM5 = DEMOS / 'cm5_minima' / 'CM5_MINIMA_3.kicad_pcb'
JETSON = DEMOS / 'jetson-agx-thor-baseboard' / 'jetson-agx-thor-baseboard.kicad_pcb'
WPTHB = Path(os.environ.get('WPTHB_BOARD', Path.home() / 'Desktop' / 'wpthb' / 'wpthb.kicad_pcb'))

INK, MUTED, GRID, SURF = '#17252a', '#56676c', '#d5dfe1', '#fcfcfb'
SERIES = ['#2a78d6', '#e07b39', '#2e9e6a', '#b0437e', '#7a6fd1']     # categorical, fixed order


def need(path, what):
    if not Path(path).exists():
        raise SystemExit(f'{what} not found at {path}; set the environment variable named in README.md')
    return Path(path)


def run(*args):
    """python -m <args> with this interpreter; echoes the command, raises on failure."""
    cmd = [sys.executable, '-m', *map(str, args)]
    print('$', ' '.join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True)


def keep(src, name):
    IMAGES.mkdir(exist_ok=True)
    shutil.copy2(src, IMAGES / name)
    print('image', IMAGES / name)


def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                         'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def grid(ax):
    ax.grid(True, which='major', color=GRID, lw=.6)
    ax.grid(True, which='minor', color=GRID, lw=.3, alpha=.5)


def load(path):
    return json.loads(Path(path).read_text())
