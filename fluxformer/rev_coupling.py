
import torch
import torch.nn as nn
import torch.nn.functional as F

def _pairwise_rotate(x, theta):
    B,T,C = x.shape; M = C//2
    x2 = x.view(B,T,M,2)
    c = torch.cos(theta); s = torch.sin(theta)
    y0 = c * x2[...,0] - s * x2[...,1]
    y1 = s * x2[...,0] + c * x2[...,1]
    return torch.stack([y0,y1], dim=-1).view(B,T,C)

class Ortho1x1(nn.Module):
    """
    Orthogonal 1x1 mixer via learnable Givens rotations over channel pairs.
    """
    def __init__(self, C, init=0.05):
        super().__init__()
        assert C % 2 == 0
        self.M = C//2
        self.theta = nn.Parameter(init*torch.randn(self.M))
    def forward(self, x):
        return _pairwise_rotate(x, self.theta.view(1,1,self.M))
    def inverse(self, y):
        return _pairwise_rotate(y, (-self.theta).view(1,1,self.M))

class CouplingMLP(nn.Module):
    """
    Small MLP used inside additive coupling. Not required to be invertible.
    """
    def __init__(self, C_half, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C_half, hidden),
            nn.GELU(),
            nn.Linear(hidden, C_half),
            nn.Tanh()
        )
    def forward(self, x):
        return self.net(x)

class RevCouplingSegment(nn.Module):
    """
    Exact reversible block:
      z = U x
      y2 = z2 + f(z1)
      y1 = z1 + g(y2)
      y = U^{-1} [y1,y2]
    Inverse:
      [y1,y2] = U y
      z1 = y1 - g(y2)
      z2 = y2 - f(z1)
      x = U^{-1} [z1,z2]
    """
    def __init__(self, C, hidden_mult=4):
        super().__init__()
        assert C % 2 == 0
        self.C = C; self.C2 = C//2
        self.U = Ortho1x1(C)
        self.f = CouplingMLP(self.C2, hidden_mult*self.C2)
        self.g = CouplingMLP(self.C2, hidden_mult*self.C2)

    def forward(self, x):
        B,T,C = x.shape
        z = self.U(x)
        z1, z2 = z[...,:self.C2], z[...,self.C2:]
        y2 = z2 + self.f(z1)
        y1 = z1 + self.g(y2)
        ycat = torch.cat([y1,y2], dim=-1)
        y = self.U.inverse(ycat)
        return y

    def inverse(self, y):
        B,T,C = y.shape
        ycat = self.U(y)  # apply U
        y1, y2 = ycat[...,:self.C2], ycat[...,self.C2:]
        z1 = y1 - self.g(y2)
        z2 = y2 - self.f(z1)
        z = torch.cat([z1,z2], dim=-1)
        x = self.U.inverse(z)
        return x
