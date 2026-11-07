"""Private numerical snapshot; independently maintained. See PROVENANCE.json."""
import json,os,re
import numpy as np
AXES="xyz"

class Dump:
    """A field dump, as parse_dump makes it from a --dump spec: the field (E, H, J), its box (start, stop: mm), its
    frequencies (Hz; None: time domain), the normal axis of a plane (None: a volume) and its file name (name)."""
    def __init__(self, spec, field, start, stop, freqs, normal=None, label=''):
        self.spec, self.field, self.start, self.stop = spec, field, np.asarray(start, float), np.asarray(stop, float)
        self.freqs = freqs                                          # Hz list, or None for time domain
        self.normal = normal                                        # 0/1/2 for planes, None for volumes
        self.label = label
        tag = label.replace('+', 'p').replace('-', 'm').replace('=', '')
        safe = re.sub(r'[^A-Za-z0-9]+', '_', f"{field}_{tag}").strip('_')
        self.name = f"dump_{safe}_{'fd' if freqs else 'td'}"

    @property
    def domain(self):
        return 'FD' if self.freqs else 'TD'

    def cells(self, mesh):
        """The mesh points it records on a grid (mesh: the x, y, z lines, mm)."""
        n = 1
        for k, ax in enumerate('xyz'):
            lines = mesh[ax]
            n *= max(1, int(np.sum((lines >= self.start[k] - 1e-9) & (lines <= self.stop[k] + 1e-9))))
        return n

    def describe(self, mesh):
        """One line for the log: its name, the field, plane or volume, its points on this grid, the domain."""
        dom = (f"at {', '.join(f'{f / 1e9:g}' for f in self.freqs)} GHz (HDF5)" if self.freqs
               else 'time domain (VTK)')
        kind = f"plane {self.label}" if self.normal is not None else (self.label if self.label.startswith('volume') else f"volume {self.label}")
        return f"{self.name}: {self.field} {kind}, {self.cells(mesh):,} points, {dom}"


def write_info(path, dumps, m, dt_ps=None):
    """<run>/dump_info.json, written before the run so a viewer can follow it live: the dumps, the
    timestep and the geometry to outline (board outline, copper per layer, ports; model mm, y up)."""
    def rings(g):
        g = g.simplify(0.01, preserve_topology=True)
        return [[[round(x, 4), round(y, 4)] for x, y in r.coords] for r in [g.exterior] + list(g.interiors)]
    info = dict(dt_ps=dt_ps, dumps=[dict(name=d.name, spec=d.spec, field=d.field, label=d.label, normal=d.normal,
                                         freqs=d.freqs, start=d.start.tolist(), stop=d.stop.tolist()) for d in dumps],
                outline=rings(m.outline)[0], cu_z={k: float(v) for k, v in m.cu_z.items()},
                copper={lay: [rings(p) for p in ps] for lay, ps in m.copper.items()},
                ports=[dict(name=p.name, at=[float(p.c[0]), float(p.c[1])]) for p in m.ports],
                vias=[[round(float(v), 4) for v in via] for via in m.vias],        # x, y, r, z0, z1 (the 3D view)
                thickness=float(m.thickness))
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'dump_info.json'), 'w', encoding='utf-8') as fh:
        json.dump(info, fh)


def write_npz(path, mesh, freqs, data):
    """The counterpart of read_npz: data (nf, nx, ny, nz, 3) complex."""
    np.savez_compressed(path, freqs=np.asarray(freqs, float), data=np.asarray(data, np.complex64),
                        **{a: np.asarray(mesh[a], float) for a in AXES})
