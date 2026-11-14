"""The suite runs in parallel (-n auto, pyproject.toml): each worker's BLAS and OpenMP get two threads, not every
core each, unless they are set already."""
import os

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '2')
