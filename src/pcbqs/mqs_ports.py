"""Passive terminal reduction of MQS copper for circuit simulation and field inference.

Each current-source port joins two terminals on the *same connected net*. The
reported current enters copper at its first terminal. Artificial port filaments
are excluded from field responses. No lumped parts are allowed in this reduction.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve
from .mqs import Result


@dataclass
class PortResponse:
    frequencies: np.ndarray
    names: list
    impedance: np.ndarray
    currents: np.ndarray

    def drive(self, amplitudes):
        """Branch currents for complex Fourier coefficients entering each copper port."""
        a=np.asarray(amplitudes,complex)
        if a.shape!=(len(self.frequencies),len(self.names)):
            raise ValueError('amplitudes must have shape (frequencies, ports)')
        return Result(self.frequencies,np.einsum('fbp,fp->fb',self.currents,a),{},self.names)


class CopperPorts:
    def __init__(self, q):
        if q.capacitance or not q.parts or any(p['kind']!='I' for p in q.parts):
            raise ValueError('copper ports require magnetic MQS with current ports only')
        self.q=q
        self.names=[p['name'] for p in q.parts]
        if len(set(self.names))!=len(self.names):raise ValueError('port names must be unique')

    def extract(self, frequencies, log=None):
        f=np.atleast_1d(np.asarray(frequencies,float))
        if f.ndim!=1 or not np.isfinite(f).all() or (f<0).any():
            raise ValueError('port frequencies must be finite and nonnegative')
        q=self.q
        if q.matrix_backend=='hierarchical':             # compressed: direct or GMRES (mqs_iterative.port_method)
            from .mqs_iterative import port_matrix
            q._assemble();Z,branch=port_matrix(q,f,log=log)
            branch[:,len(q.br):]=0                      # fictitious port connectors are never radiators
            return PortResponse(f,self.names,Z,branch)
        m=q._assemble();offset=len(q.br)
        sc=np.array([m['src'][offset+i] for i in range(len(q.parts))])
        Z,branch=direct_ports(q,m['C'],m['Rl'],m['Ll'],sc,f,log)
        branch[:,offset:]=0                             # fictitious port connectors are never radiators
        return PortResponse(f,self.names,Z,branch)


def direct_ports(q, C, Rl, Ll, sc, frequencies, log=None, note=None):
    """Port impedances (F, P, P) and branch currents (F, nb, P) from the dense loop matrices Rl, Ll of the loop
    basis C, the ports' loops the columns sc: the other loops eliminated by a direct solve at each frequency."""
    free=np.setdiff1d(np.arange(C.shape[1]),sc);C=C.tocsc()
    Zs=[];responses=[]
    for fi in frequencies:
        Z=Rl.astype(complex)+2j*np.pi*fi*Ll
        delta=q._resistance_delta(fi) if fi else None
        if delta is not None and delta.nnz:Z+=(C.T@delta@C).toarray()
        X=solve(Z[np.ix_(free,free)],-Z[np.ix_(free,sc)],assume_a='sym',overwrite_a=True,check_finite=False) \
            if len(free) else np.zeros((0,len(sc)),complex)
        zp=Z[np.ix_(sc,sc)]+Z[np.ix_(sc,free)]@X
        Zs.append(zp);responses.append(-(C[:,sc].toarray()+C[:,free]@X))
        if log:log(f'Copper ports: {fi:g} Hz'+(f' ({note})' if note else ''))
    return np.asarray(Zs),np.asarray(responses)


def rl_equivalent(response, index=-1, tolerance=1e-7):
    """Single-frequency passive R/L approximation; error must be checked over use band.

    An RL copper network's terminal R and Im(Z)/omega are positive semidefinite.
    Tiny numerical negative eigenvalues are projected out; material violations fail.
    """
    f=response.frequencies[index]
    if f<=0:raise ValueError('RL extraction needs a positive reference frequency')
    Z=response.impedance[index]
    mats=[]
    for A in [Z.real,Z.imag/(2*np.pi*f)]:
        A=(A+A.T)/2;e,V=np.linalg.eigh(A);scale=max(abs(e).max(),1e-30)
        if e.min() < -tolerance*scale:raise ValueError('non-passive copper reduction; refine/check topology')
        mats.append((V*np.maximum(e,scale*1e-12))@V.T)
    return dict(names=response.names,reference_Hz=float(f),R_ohm=mats[0].tolist(),L_H=mats[1].tolist())


def rl_error(model, response):
    R=np.array(model['R_ohm']);L=np.array(model['L_H'])
    if model['names']!=response.names:raise ValueError('copper port order differs')
    modeled=model_impedance(model,response.frequencies)
    return [float(np.linalg.norm(M-Z)/max(np.linalg.norm(Z),1e-30)) for M,Z in zip(modeled,response.impedance)]


def foster_equivalent(response, poles_Hz=None, iterations=1800):
    """Fit a positive-real matrix Foster RL impedance by convex PSD-constrained LS.

    Z(s)=Rdc+s*Linf+sum(s/(s+p)*Rmode); every matrix is PSD. This is a
    passive circuit, unlike an unconstrained interpolated impedance table.
    Residues are fit jointly across all ports using scaled ADMM projections.
    Accuracy is reported at each supplied frequency and must be independently checked.
    """
    f=response.frequencies;Z=response.impedance
    zeros=np.flatnonzero(f==0)
    if len(zeros)!=1:raise ValueError('Foster fitting requires exactly one DC sample')
    R0=(Z[zeros[0]].real+Z[zeros[0]].real.T)/2
    if np.linalg.eigvalsh(R0).min()<-1e-8*max(abs(R0).max(),1e-30):raise ValueError('non-passive DC port data')
    poles=np.asarray(poles_Hz if poles_Hz is not None else np.geomspace(1e2,1e8,10))*2*np.pi
    if not np.isfinite(poles).all() or (poles<=0).any():raise ValueError('Foster poles must be positive')
    take=f>0;s=2j*np.pi*f[take];reference=2*np.pi*1e5
    A=np.column_stack([s/reference,*[s/(s+p) for p in poles]])
    B=(Z[take]-R0).reshape(sum(take),-1)
    weights=1/np.maximum(np.linalg.norm(Z[take],axis=(1,2)),1e-30)
    A=A*weights[:,None];B=B*weights[:,None]
    A=np.r_[A.real,A.imag];B=np.r_[B.real,B.imag]
    scale=np.linalg.norm(A,axis=0);A/=scale
    H=A.T@A;rhs=A.T@B;rho=.03
    K=np.linalg.inv(H+rho*np.eye(len(H)));N=Z.shape[1]
    Q=np.zeros((len(H),N,N));U=Q.copy();last=Q.copy()
    for step in range(iterations):
        X=(K@(rhs+rho*(Q-U).reshape(len(H),-1))).reshape(Q.shape)
        proposed=X+U;proposed=(proposed+proposed.transpose(0,2,1))/2
        e,V=np.linalg.eigh(proposed);Q=(V*np.maximum(e,0)[:,None,:])@V.transpose(0,2,1)
        U+=X-Q
        if step%50==0:
            if np.linalg.norm(X-Q)<1e-7*max(np.linalg.norm(Q),1e-30) and np.linalg.norm(Q-last)<1e-7*max(np.linalg.norm(Q),1e-30):break
            last=Q.copy()
    matrices=Q/scale[:,None,None]
    return dict(kind='foster_rl',names=response.names,R_ohm=R0.tolist(),L_H=(matrices[0]/reference).tolist(),
                foster=[dict(pole_rad_s=float(p),R_ohm=M.tolist()) for p,M in zip(poles,matrices[1:])],
                fit_iterations=step+1,fit_frequencies_Hz=f.tolist())


def model_impedance(model, frequencies):
    f=np.atleast_1d(frequencies);s=2j*np.pi*f[:,None,None]
    Z=np.array(model['R_ohm'])[None]+s*np.array(model['L_H'])[None]
    for mode in model.get('foster',[]):Z+=s/(s+mode['pole_rad_s'])*np.asarray(mode['R_ohm'])[None]
    return Z
