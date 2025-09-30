
import numpy as np

def gelu(x): return 0.5*x*(1.0+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))
def tanh(x): return np.tanh(x)

def mlp(W1, b1, W2, b2, x):
    return tanh((x@W2)+b2)

def rand_params(rng, C2, H):
    W1 = rng.normal(0, 0.2, size=(C2, H))
    b1 = rng.normal(0, 0.1, size=(H,))
    W2 = rng.normal(0, 0.2, size=(H, C2))
    b2 = rng.normal(0, 0.1, size=(C2,))
    return W1,b1,W2,b2

def f_fun(params, x):
    W1,b1,W2,b2=params
    h=gelu(x@W1+b1)
    y=tanh(h@W2+b2)
    return y

def pair_rotate(theta, x):
    # x: [N,C], C even
    C = x.shape[1]; M=C//2
    x2 = x.reshape(-1,M,2)
    c = np.cos(theta); s=np.sin(theta)
    y0 = c * x2[...,0] - s * x2[...,1]
    y1 = s * x2[...,0] + c * x2[...,1]
    return np.stack([y0,y1], axis=-1).reshape(-1,C)

def test_roundtrip():
    rng = np.random.default_rng(1234)
    C=8; C2=4; H=16; N=128
    # Ortho mixer U
    theta = 0.05*rng.standard_normal(C2)
    # f,g params
    pf = rand_params(rng, C2, H)
    pg = rand_params(rng, C2, H)
    # data
    x = rng.normal(0,1,size=(N,C))

    # forward
    z = pair_rotate(theta, x)
    z1, z2 = z[:,:C2], z[:,C2:]
    y2 = z2 + f_fun(pf, z1)
    y1 = z1 + f_fun(pg, y2)
    ycat = np.concatenate([y1,y2], axis=-1)
    y = pair_rotate(-theta, ycat)

    # inverse
    ycat2 = pair_rotate(theta, y)
    y1b, y2b = ycat2[:,:C2], ycat2[:,C2:]
    z1b = y1b - f_fun(pg, y2b)   # g
    z2b = y2b - f_fun(pf, z1b)   # f
    zb = np.concatenate([z1b, z2b], axis=-1)
    xb = pair_rotate(-theta, zb)

    err = np.max(np.abs(x - xb))
    assert err < 1e-6, f"Reversible coupling roundtrip error too large: {err}"
    print("OK: reversible coupling exact inverse (NumPy)")

if __name__ == "__main__":
    test_roundtrip()
