"""Palace (AWS's frequency-domain FEM) runs for the validation suite: a model folder with mesh.msh and config.json,
run in the unpacked Palace image with rootless podman (K2E_PALACE_ROOTFS, default ~/opt/palace/rootfs, as
kicad2ems-palace does) or a native `palace` (K2E_PALACE).

    run(folder, config, np=8)                   # Palace's log tail; results in folder/postpro
    port_z(folder, R)                           # (f Hz, Z[f, i, j]) from Palace's port voltages and currents
"""
import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

ROOTFS = Path(os.environ.get('K2E_PALACE_ROOTFS', Path.home() / 'opt' / 'palace' / 'rootfs'))


def command(folder, np_=8):
    native = os.environ.get('K2E_PALACE') or shutil.which('palace')
    if native:
        return [native, '-np', str(np_), 'config.json'], str(folder)
    if not (ROOTFS / 'opt/views/view/bin').is_dir():
        raise RuntimeError('Palace not found: set K2E_PALACE or K2E_PALACE_ROOTFS')
    inner = f'export PATH=/opt/views/view/bin:$PATH; cd /work && palace -np {np_} config.json'
    return ['podman', 'run', '--rm', '--network', 'none', '--userns=keep-id', '--shm-size', '2g',
            '-v', f'{Path(folder).resolve()}:/work:Z', '--rootfs', f'{ROOTFS}:O', 'bash', '-c', inner], None


def run(folder, config, np_=8, timeout=7200):
    folder = Path(folder)
    (folder / 'config.json').write_text(json.dumps(config, indent=1))
    cmd, cwd = command(folder, np_)
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    (folder / 'palace.log').write_text(p.stdout + p.stderr)
    if p.returncode:
        raise RuntimeError('palace failed; see ' + str(folder / 'palace.log'))
    return p.stdout[-2000:]


def _table(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    head = [h.strip() for h in rows[0]]
    return head, np.array([[float(x) for x in r] for r in rows[1:] if r])


def port_z(folder, n, R=50.0):
    """(f Hz, Z[f, i, j]) from Palace's port-S.csv (|S| dB and arg deg., S[i][j]: port i with port j excited), every
    port of reference resistance R: Z = R (1 + S)(1 - S)^-1. (port-V/I hold the terminated port's own V and I.)"""
    head, data = _table(Path(folder) / 'postpro' / 'port-S.csv')
    S = np.zeros((len(data), n, n), complex)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            k = head.index(f'|S[{i}][{j}]| (dB)')
            S[:, i - 1, j - 1] = 10 ** (data[:, k] / 20) * np.exp(1j * np.radians(data[:, k + 1]))
    eye = np.eye(n)
    return data[:, 0] * 1e9, np.array([R * (eye + s) @ np.linalg.inv(eye - s) for s in S])
