"""
The steady temperature of a board from what its parts dissipate: a lightweight layered model (problem.py) with two
solvers, a regular grid (grid.py) and pcbqs's triangle mesh (mesh.py).

    from pcbqs.thermal import Problem, solve
    r = solve(Problem.of(board, {'Q1': 0.35}), method='grid', size=0.5)
"""
from .problem import Problem, Result, losses_of, K_CU, K_XY, K_Z, FILM, AMBIENT


def solve(problem, method='grid', size=None, log=print, **kw):
    """method: 'grid' (size: the pitch, mm, default 0.5) or 'mesh' (size: the longest edge, mm, default 1)."""
    if method == 'grid':
        from .grid import solve as go
        return go(problem, pitch=size or 0.5, log=log, **kw)
    if method == 'mesh':
        from .mesh import solve as go
        return go(problem, dmax=size or 1.0, log=log, **kw)
    raise ValueError(f"method {method!r}: 'grid' or 'mesh'")
