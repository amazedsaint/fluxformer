
import numpy as np
def pairwise_rotate_np(x, theta):
    T,C = x.shape; M = C//2
    x2 = x.reshape(T,M,2); c=np.cos(theta); s=np.sin(theta)
    y0=c*x2[...,0]-s*x2[...,1]; y1=s*x2[...,0]+c*x2[...,1]
    return np.stack([y0,y1], axis=-1).reshape(T,C)
def split_shift_np(x):
    T,C=x.shape; M=C//2; x2=x.reshape(T,M,2); a=x2[...,0]; b=x2[...,1]
    return np.stack([np.roll(a,+1,axis=0), np.roll(b,-1,axis=0)], axis=-1).reshape(T,C)
class QWalk1DNP:
    def __init__(self,C,T):
        assert C%2==0; self.C=C; self.M=C//2; self.T=T
        rng=np.random.default_rng(123); self.theta_e=0.1*rng.standard_normal(self.M)
        self.theta_o=0.1*rng.standard_normal(self.M); self.edge=np.zeros(T); self.A0=np.zeros(T); self.use_nl=False
    def gn(self,x):
        if not self.use_nl: return x
        T,C=x.shape; M=self.M; x2=x.reshape(T,M,2); r=np.linalg.norm(x2,axis=-1)+1e-6
        g=np.maximum(r,0.0); return ((g/r)[...,None]*x2).reshape(T,C)
    def step(self,x):
        T,C=x.shape; M=self.M
        x=pairwise_rotate_np(x, self.A0[None,:].repeat(M,axis=0).T)
        x=pairwise_rotate_np(x, self.theta_e[None,:].repeat(T,axis=0))
        x=split_shift_np(x)
        x2=x.reshape(T,M,2); x2r=np.roll(x2,1,axis=1); xr=x2r.reshape(T,C)
        xr=pairwise_rotate_np(xr,self.theta_o[None,:].repeat(T,axis=0))
        x=np.roll(xr.reshape(T,M,2), -1, axis=1).reshape(T,C)
        x=pairwise_rotate_np(x, self.edge[None,:].repeat(M,axis=0).T)
        x=split_shift_np(x); x=self.gn(x); return x
def simulate(T=64,C=8,L=20,nonlinearity=False):
    blk=QWalk1DNP(C,T); blk.use_nl=nonlinearity; x=np.zeros((T,C)); x[T//2,0]=1.0
    norms=[]; energies=[]; 
    for _ in range(L): norms.append((x**2).sum()); energies.append((x**2).sum(axis=1)); x=blk.step(x)
    norms.append((x**2).sum()); energies.append((x**2).sum(axis=1)); return np.array(norms), np.stack(energies,axis=0)
