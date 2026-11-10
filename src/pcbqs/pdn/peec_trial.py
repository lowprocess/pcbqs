"""Experimental PDN impedance using PEEC and finite-thickness skin loss.

This is a trial adapter, not the default PDN backend. RailPDN supplies the rail
and explicit/defaulted component RLC assumptions. Copper paths and mounting
geometry are solved by MQS/PEEC instead of adding cavity mounting estimates.
"""
import numpy as np

from ..mqs import MQS
from ..contacts import contact_terminal


class PEECPDN:
    def __init__(self, legacy, dmax=1., conductor_loss='slab', conductor_faces=1,
                 fill_backend='auto', fill_workers=None, contact_spacing=None, region=None,
                 max_triangles=None, log=print, coarsen=None, grow=0.5,
                 edge_rings=1, edge_width=None, edge_zone=None, matrix_backend='dense',
                 compression_tol=1e-6, compression_leaf=64, compression_eta=1.,
                 solver_tol=1e-8, solver_maxiter=500, pad_contact='termination'):
        self.legacy = legacy
        self.rail = rail = legacy.rail
        if (any(s.kind not in ('R','jumper','through') for s in rail.series)
                or len(rail.ground)!=1 or len(rail.source_parts)!=1):
            raise NotImplementedError('PEEC PDN trial supports one ground/source and resistive series rail parts; beads are unsupported')
        layers = [c['name'] for c in rail.board['stack']['copper']]
        terms = {}
        self.terminals = {}
        for ref, info in legacy.parts.items():
            power, ground = rail.pads_of(info['part'], info['role'])
            a, b = ref+':power', ref+':ground'
            part = info['part']
            terms[a] = contact_terminal(part, power, pad_contact, layers)
            terms[b] = contact_terminal(part, ground, pad_contact, layers)
            if not terms[a] or not terms[b]:
                raise ValueError(ref+': PEEC PDN trial requires explicit power and ground pads')
            self.terminals[ref] = a, b
        for link in rail.series:
            a,b=link.part.ref+':a',link.part.ref+':b'
            terms[a]=contact_terminal(link.part,link.pads('a'),pad_contact,layers)
            terms[b]=contact_terminal(link.part,link.pads('b'),pad_contact,layers)
            if not terms[a] or not terms[b]:raise ValueError(link.part.ref+': missing series-part pads')
        fine = None
        if contact_spacing is not None:
            if not np.isfinite(contact_spacing) or contact_spacing<=0:
                raise ValueError('contact_spacing must be positive and finite')
            from shapely.ops import unary_union
            from shapely.geometry import box
            fine = unary_union([box(*poly.buffer(.2).bounds) for pads in terms.values() for _,poly in pads])
        self.q = MQS(rail.board, set(rail.nets)|set(rail.ground), terminals=terms,
            dmax=dmax, region=region, capacitance=True, dielectric='stack', conductor_loss=conductor_loss,
            conductor_faces=conductor_faces, fill_backend=fill_backend, fill_workers=fill_workers,
            fine=fine, dmax_fine=contact_spacing, factor_cache_bytes=0, log=log,
            coarsen=coarsen,grow=grow,edge_rings=edge_rings,edge_width=edge_width,edge_zone=edge_zone,
            matrix_backend=matrix_backend,compression_tol=compression_tol,compression_leaf=compression_leaf,
            compression_eta=compression_eta,solver_tol=solver_tol,solver_maxiter=solver_maxiter)
        if matrix_backend == 'dense' and max_triangles is not None and len(self.q.tris)>max_triangles:
            raise RuntimeError(f'PEEC PDN trial mesh has {len(self.q.tris):,} triangles, exceeding the requested {max_triangles:,} dense-solver limit')
        self._models = {}
        self.notes = ['Experimental PEEC PDN; one source/ground, resistive series links; beads unsupported.',
            'Component RLC assumptions come from RailPDN; heuristic mounting inductance is replaced by resolved copper.',
            'Part-to-copper mutual inductance, via skin effect and arbitrary thickness proximity are not resolved.']

    def _model(self, bare):
        if bare in self._models:
            return self._models[bare]
        # A shallow geometry copy deliberately shares immutable copper kernels;
        # topology, frequency factors and component lists belong to each model.
        import copy
        q = copy.copy(self.q)
        q.parts = []; q._mats = None; q._peec = None
        for ref in self.rail.loads:
            q.source('load:'+ref, *self.terminals[ref], part_h=0)
        for info in self.legacy.series:
            link=info['s'];ref=link.part.ref
            q.element('series:'+ref,ref+':a',ref+':b',R=link.R,L=info['esl'],part_h=0)
        if not bare:
            for ref, info in self.legacy.parts.items():
                el = info['el']
                if el is None:
                    continue
                if 'C' in el:
                    q.element('part:'+ref, *self.terminals[ref], C=el['C'], ESR=el['ESR'], ESL=el['ESL'], part_h=0)
                else:
                    q.element('part:'+ref, *self.terminals[ref], R=el.get('R',0.), L=el.get('L',0.), part_h=0)
        self._models[bare] = q
        return q

    def z(self, frequencies, observe=None, bare=False):
        """{load: complex driving-point Z(f)} with decaps and small-signal VRM.

        A load's DC current is not an AC shunt resistance. One 1 A small-signal
        source is driven at a time; the other load ports have zero source current.
        """
        frequencies = np.atleast_1d(np.asarray(frequencies,float))
        if frequencies.ndim!=1 or not len(frequencies) or not np.isfinite(frequencies).all() or (frequencies<=0).any():
            raise ValueError('PEEC PDN frequencies must be finite and positive')
        names = list(self.rail.loads) if observe is None else list(observe)
        if any(n not in self.rail.loads for n in names):
            raise ValueError('observe must name configured rail loads')
        q = self._model(bare)
        out = {}
        for ref in names:
            amps = {'load:'+r:float(r==ref) for r in self.rail.loads}
            out[ref] = q.solve(frequencies, amps=amps).z['load:'+ref]
        # Subsequent bare/loaded models reuse only geometry-dependent kernels.
        self.q._copper, self.q._potential = q._copper, q._potential
        self.q._sheet_terms, self.q.fill_info = q._sheet_terms, q.fill_info
        for other in self._models.values():
            if other._copper is None:
                other._copper, other._potential = q._copper, q._potential
                other._sheet_terms = q._sheet_terms
        return out
