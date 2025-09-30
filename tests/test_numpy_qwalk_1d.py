
from baselines.numpy_qwalk_1d import simulate

def approx_eq(a,b,tol=1e-12): return abs(a-b) <= tol

def main():
    norms, E = simulate(T=64, C=8, L=20, nonlinearity=False)
    assert all(approx_eq(float(n), 1.0, 1e-12) for n in norms[:10])
    assert approx_eq(float(norms[-1]), 1.0, 1e-12)
    print("OK: 1D QWalk invariants")
if __name__ == "__main__":
    main()
