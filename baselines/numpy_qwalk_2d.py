
import numpy as np
def pairwise_rotate_np(x, theta):
    H,W,C=x.shape; M=C//2; x2=x.reshape(H,W,M,2); c=np.cos(theta); s=np.sin(theta)
    y0=c*x2[...,0]-s*x2[...,1]; y1=s*x2[...,0]+c*x2[...,1]
    return np.stack([y0,y1],axis=-1).reshape(H,W,C)
def split_shift_axis(x, axis):
    H,W,C=x.shape; M=C//2; x2=x.reshape(H,W,M,2); a=x2[...,0]; b=x2[...,1]
    return np.stack([np.roll(a,+1,axis=axis), np.roll(b,-1,axis=axis)], axis=-1).reshape(H,W,C)
class QWalk2DNP:
    def __init__(self,C,H,W): assert C%2==0; self.C=C; self.M=C//2; self.H=H; self.W=W
        # minimal params
    def step(self,x):
        xr=split_shift_axis(x, axis=0); xc=split_shift_axis(xr, axis=1); return xc
def simulate(H=16,W=16,C=8,L=10, nonlinearity=False):
    x=np.zeros((H,W,C)); x[H//2,W//2,0]=1.0; norms=[]; energies=[]
    for _ in range(L): norms.append((x**2).sum()); energies.append((x**2).sum(axis=2)); x=split_shift_axis(split_shift_axis(x,0),1)
    norms.append((x**2).sum()); energies.append((x**2).sum(axis=2)); return np.array(norms), np.stack(energies,axis=0)
