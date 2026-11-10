"""Surface-charge PEEC extension of MQS in a homogeneous dielectric.

Branch incidence A is positive at the branch's starting node. With exp(+jwt),
    Z i = A.T P q,   A i + jw q = 0.
P is a Galerkin coefficient-of-potential matrix (volts per coulomb), referenced
to infinity. A sparse loop/tree basis keeps P out of the loop equations. The
electrostatic equilibrium is separated and the tree correction scaled by jw:
this avoids both 1/w ill-conditioning and subtracting almost equal currents
to recover a small charge at low frequency.
"""
import numpy as np
from scipy.sparse import coo_matrix, eye, hstack
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
from scipy.linalg import solve as dense_solve, lu_factor, lu_solve
from collections import OrderedDict

from ._numerics.base import qform


def electric_field(model, result, fi, pts, chunk=100_000):
    """Coulomb E (V/m) at off-surface points (m), exact for each constant
    triangle charge density. No inductive -jw A contribution is included.
    The edge integrals give tangential E and the solid angle gives normal E.
    """
    from ._numerics.base import EPS0
    pts = np.asarray(pts, float)
    if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
        raise ValueError('pts must be a finite (n, 3) array in metres')
    if result.charge is None:
        raise ValueError('electric fields require a capacitance=True result')
    multi = np.ndim(fi) > 0
    charge = result.charge[np.atleast_1d(fi)].T
    _, D = node_potential(model) if getattr(model, 'charge_of', None) is not None else (None, None)
    if D is not None:                                    # face sheets: each copy's share (side panels left out)
        of = model.charge_of
        charge = (D @ charge)[:of.max() + 1]
    else:
        of = model.node_of
    area = np.bincount(of, weights=model.rwg.area, minlength=of.max() + 1)
    density = charge[of] / area[of, None]
    V = model.nodes[model.tris]
    if model.dielectric == 'homogeneous':
        E = _electric_triangles(V, model.z_of, density, pts, chunk) / model.er
    else:
        E = np.zeros((len(pts), 3, charge.shape[1]), complex)
        for z in np.unique(pts[:, 2]):
            chosen = pts[:, 2] == z
            for idx in model.layer_tris.values():
                source = model._dielectric_stack.interface(model.z_of[idx[0]])
                for image_z, coefficient in model._dielectric_stack.images(source, float(z), field=True):
                    E[chosen] += _electric_triangles(V[idx], np.full(len(idx), image_z),
                                                     density[idx]*coefficient, pts[chosen], chunk)
    E /= 4*np.pi*EPS0
    return E if multi else E[:, :, 0]


def _electric_triangles(V, z_of, density, pts, chunk):
    from ._numerics.base import edge_frames
    _, length, tau, un = edge_frames(V)
    E = np.zeros((len(pts), 3, density.shape[1]), complex)
    step = max(1, chunk // len(V))
    for start in range(0, len(pts), step):
        obs = pts[start:start + step]
        dv = V[None] - obs[:, None, None, :2]
        sm = (dv * tau[None]).sum(-1)
        sp = sm + length[None]
        t = (dv * un[None]).sum(-1)
        dz = z_of[None] - obs[:, 2, None]
        r0 = np.sqrt(t*t + dz[..., None]**2)
        # No unique field on a zero-thickness charged sheet. Coplanar points
        # outside the copper are well-defined, including edge extensions.
        on_sheet = (dz == 0) & (t >= 0).all(-1)
        if on_sheet.any():
            raise ValueError('evaluate E away from charged copper surfaces')
        nonzero = r0 > 0
        safe = np.where(nonzero, r0, 1.0)
        lg = np.arcsinh(sp / safe) - np.arcsinh(sm / safe)
        external = ~nonzero
        if external.any():
            lg[external] = np.sign(sm[external]) * np.log(np.abs(sp[external] / sm[external]))
        xy = np.einsum('pne,nec->pnc', lg, un)
        vectors = np.concatenate([dv, np.broadcast_to(dz[..., None, None], dv.shape[:-1] + (1,))], axis=-1)
        a, b, c = vectors[:, :, 0], vectors[:, :, 1], vectors[:, :, 2]
        la, lb, lc = [np.linalg.norm(v, axis=-1) for v in (a, b, c)]
        num = np.einsum('pnc,pnc->pn', a, np.cross(b, c))
        den = la*lb*lc + (a*b).sum(-1)*lc + (b*c).sum(-1)*la + (c*a).sum(-1)*lb
        normal = -2 * np.arctan2(num, den)
        E[start:start + step, :2] = np.einsum('pnc,nk->pck', xy, density)
        E[start:start + step, 2] = normal @ density
    return E


def capacitance_matrix(model):
    """Maxwell C (F), conductor node labels, terminal -> conductor map.

    A conductor is a connected component of copper and vias; lumped elements
    do not merge conductors. Potentials are relative to infinity, so row sums
    need not vanish. For floating differential capacitance enforce zero total
    charge, rather than grounding one conductor to infinity.
    """
    from scipy.sparse.csgraph import connected_components
    model._ensure_potential()
    a, b = np.asarray(model.br, int).reshape(-1, 2).T
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(model.n_nodes, model.n_nodes))
    n, labels = connected_components(graph, directed=False)
    charge_of = getattr(model, 'charge_of', None)
    if charge_of is not None:                            # face sheets: a charge per triangle copy, on its node's conductor
        rep = np.zeros(charge_of.max() + 1, int)
        rep[charge_of] = model.node_of
        side = getattr(model, 'side_owner', None)
        if side is not None:                             # side panels: their triangles' nodes
            rep = np.r_[rep, model.node_of[side]]
        B = np.eye(n)[labels[rep]]
    else:
        B = np.eye(n)[labels]
    if model.matrix_backend == 'hierarchical':
        from .peec_iterative import potential_solve
        C = B.T @ potential_solve(model,B)
    else:
        C = B.T @ dense_solve(model._potential, B, assume_a='sym')
    return (C + C.T) / 2, labels, {name: int(labels[node]) for name, node in model.term_node.items()}


def charge_nodes(model):
    """Face sheets (and side panels): the charge unknowns -> circuit nodes, as a sparse (n_charge, n_nodes)
    incidence. None without face sheets (a charge per circuit node)."""
    charge_of = getattr(model, 'charge_of', None)
    if charge_of is None:
        return None
    rep = np.zeros(charge_of.max() + 1, int)
    rep[charge_of] = model.node_of
    side = getattr(model, 'side_owner', None)
    if side is not None:
        rep = np.r_[rep, model.node_of[side]]
    return coo_matrix((np.ones(len(rep)), (np.arange(len(rep)), rep)), shape=(len(rep), model.n_nodes)).tocsc()


def node_potential(model):
    """The circuit nodes' coefficients of potential, and the charge's spread over each node's copies.

    With face sheets a node's charge is free to move between its copies (both faces, its side panels): they
    are one conductor at one potential. So for node charges q, the copies hold Pc^-1 Bn v with
    Bn^T Pc^-1 Bn v = q, i.e. node-level P_eff = (Bn^T Pc^-1 Bn)^-1 and copy charges D q, D = Pc^-1 Bn P_eff.
    Without face sheets: (model._potential, None)."""
    Bn = charge_nodes(model)
    Pc = model._ensure_potential()
    if Bn is None:
        return Pc, None
    cached = getattr(model, '_node_potential', None)
    if cached is not None and cached[0] is Pc:
        return cached[1], cached[2]
    X = dense_solve(Pc, Bn.toarray(), assume_a='sym')    # Pc^-1 Bn (complex symmetric in a lossy stack)
    C = Bn.T @ X                                         # the node-level capacitance (Maxwell, to infinity)
    C = (C + C.T) / 2
    Peff = np.linalg.inv(C)
    Peff = (Peff + Peff.T) / 2
    D = X @ Peff
    model._node_potential = (Pc, Peff, D)
    return Peff, D


def prepare(model):
    """Frequency/excitation-independent projections, reused until parts change."""
    if model._peec is not None:
        return model._peec
    m = model._assemble()
    ends = model.br + [(model.term_node[p['a']], model.term_node[p['b']]) for p in model.parts]
    nb = len(ends)
    branch = np.arange(nb)
    a, b = np.asarray(ends, int).reshape(-1, 2).T
    A = coo_matrix((np.r_[np.ones(nb), -np.ones(nb)], (np.r_[a, b], np.r_[branch, branch])),
                   shape=(model.n_nodes, nb)).tocsc()
    C, src_cols, is_src, tree = m['C'], m['src'], m['is_src'], m['tree']
    passive, sources = np.flatnonzero(~is_src), np.flatnonzero(is_src)
    cycles = [k for k in range(C.shape[1]) if k not in src_cols.values()]
    K = hstack([C[passive][:, cycles], eye(nb, format='csc')[passive][:, np.flatnonzero(tree)]], format='csc')
    nl = len(cycles)
    T = A[:, passive] @ K[:, nl:]
    P, _ = node_potential(model)
    pscale = np.max(np.abs(P))
    electro = qform(T, P, T) / pscale
    kc = K.tocoo()
    full_basis = coo_matrix((kc.data, (passive[kc.row], kc.col)),
                            shape=(nb, K.shape[1])).tocsc()
    R = (full_basis.T @ m['R'] @ full_basis).toarray()
    L = model._project_inductance(full_basis, *m['L'])
    names = [model.parts[k - len(model.br)]['name'] for k in sources]
    # Split out the electrostatic equilibrium charge of each passive graph
    # component. The remaining tree currents are jw times a correction.
    # Computing q = -(A i + drive)/jw directly would lose the tiny conduction
    # charges to cancellation at low frequency, even with a well-scaled solve.
    graph = coo_matrix((np.ones(len(passive)), (a[passive], b[passive])),
                       shape=(model.n_nodes, model.n_nodes))
    nc, labels = connected_components(graph, directed=False)
    B = np.eye(nc)[labels]
    roots = np.unique(labels, return_index=True)[1]
    keep = np.ones(model.n_nodes, bool); keep[roots] = False
    tree_lu = splu(T[keep].astype(complex).tocsc()) if T.shape[1] else None
    part_rows = np.flatnonzero(passive >= len(model.br))
    model._peec = dict(A=A, K=K, T=T, P=P, pscale=pscale, electro=electro,
                       R=R, L=L, names=names, passive=passive, sources=sources,
                       B=B, keep=keep, tree_lu=tree_lu, PT=P @ T, nl=nl,
                       source_components=A[:, sources].T @ B,
                       part_rows=part_rows, E=K[part_rows].toarray(),
                       factors=OrderedDict(), factor_bytes=0, factorizations=0, full_basis=full_basis)
    return model._peec


def solve(model, freqs, amps=None):
    from .mqs import Result
    f = np.atleast_1d(np.asarray(freqs, float))
    if f.ndim != 1 or not np.isfinite(f).all() or (f <= 0).any():
        raise ValueError('frequencies must be finite and positive')
    m = prepare(model)
    A, K, T, P, B = [m[k] for k in ('A', 'K', 'T', 'P', 'B')]
    R, L, PT, nl, pscale = [m[k] for k in ('R', 'L', 'PT', 'nl', 'pscale')]
    passive, sources, names = m['passive'], m['sources'], m['names']
    nb = A.shape[1]
    excitation = np.array([(amps or {}).get(n, 1.0) for n in names], complex)
    drive = np.asarray(A[:, sources] @ excitation).ravel()
    # Cancel source endpoints topologically before multiplying amplitudes.
    # Summing rounded node currents could create a spurious net component
    # charge, amplified by 1/w in an otherwise closed low-frequency circuit.
    totals = m['source_components'].T @ excitation
    qpart = np.zeros(model.n_nodes, complex)
    vpart = np.zeros(B.shape[1], complex)
    if np.any(totals):
        if 'PinvB' not in m:
            m['PinvB'] = dense_solve(P, B, assume_a='sym')
            m['component_lu'] = lu_factor(B.T @ m['PinvB'])
        vpart = lu_solve(m['component_lu'], -totals)
        qpart = m['PinvB'] @ vpart
    y0 = (m['tree_lu'].solve((-drive - qpart)[m['keep']])
          if m['tree_lu'] is not None else np.zeros(0, complex))
    baseline = np.r_[np.zeros(nl, complex), y0]
    source_voltage = -m['source_components'] @ vpart
    rb, lb, eb = R @ baseline, L @ baseline, m['E'] @ baseline
    currents = np.zeros((len(f), nb), complex)
    charge = np.zeros((len(f), model.n_nodes), complex)
    potential = np.zeros_like(charge)
    impedance = {n: np.zeros(len(f), complex) for n in names}
    # Parts enter the sparse basis through their actual branches only.
    for fi, freq in enumerate(f):
        jw = 2j * np.pi * freq
        dr = model._resistance_delta(freq)
        delta = (m['full_basis'].T @ dr @ m['full_basis']).toarray() if dr.nnz else 0
        zp = np.array([model.parts[passive[j] - len(model.br)]['R'] +
                       jw * model.parts[passive[j] - len(model.br)]['L'] +
                       (1 / (jw * model.parts[passive[j] - len(model.br)]['C'])
                        if model.parts[passive[j] - len(model.br)].get('C') else 0)
                       for j in m['part_rows']], complex)
        if len(passive) and K.shape[1]:                  # (a terminal covering a whole conductor leaves no unknowns)
            cached = m['factors'].pop(float(freq), None)
            if cached is None:
                Z = R + delta + jw * L + m['E'].T @ (zp[:, None] * m['E'])
                zscale = max(np.max(np.abs(Z)), np.finfo(float).tiny)
                system = Z / zscale
                system[:, nl:] = jw * Z[:, nl:] / pscale
                system[nl:, nl:] += m['electro']
                cached = lu_factor(system, overwrite_a=True), zscale
                m['factorizations'] += 1
                size = cached[0][0].nbytes + cached[0][1].nbytes
                if size <= model.factor_cache_bytes:
                    while m['factor_bytes'] + size > model.factor_cache_bytes:
                        _, old = m['factors'].popitem(last=False)
                        m['factor_bytes'] -= old[0][0].nbytes + old[0][1].nbytes
                    m['factor_bytes'] += size
                    m['factors'][float(freq)] = cached
            else:
                m['factors'][float(freq)] = cached
            factor, zscale = cached
            rhs = -(rb + (delta @ baseline if dr.nnz else 0) + jw * lb + m['E'].T @ (zp * eb)) / zscale
            x = lu_solve(factor, rhs)
            alpha = zscale / pscale
            currents[fi, passive] = K[:, :nl] @ x[:nl] + K[:, nl:] @ (y0 + jw * alpha * x[nl:])
            charge[fi] = qpart / jw - alpha * (T @ x[nl:])
            potential[fi] = B @ vpart / jw - alpha * (PT @ x[nl:])
            voltage = source_voltage / jw + alpha * (A[:, sources].T @ (PT @ x[nl:]))
        else:
            charge[fi] = qpart / jw
            potential[fi] = B @ vpart / jw
            voltage = source_voltage / jw
        currents[fi, sources] = excitation
        for j, name in enumerate(names):
            impedance[name][fi] = voltage[j] / excitation[j] if excitation[j] else np.nan
    return Result(f, currents, impedance, [p['name'] for p in model.parts],
                  charge=charge, potential=potential)
