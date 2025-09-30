
from baselines.numpy_qwalk_2d import simulate
def approx_eq(a,b,tol=1e-12): return abs(a-b) <= tol
def main():
    norms, E = simulate(H=16, W=16, C=8, L=10, nonlinearity=False)
    assert all(approx_eq(float(n), 1.0, 1e-12) for n in norms[:5])
    assert approx_eq(float(norms[-1]), 1.0, 1e-12)
    print("OK: 2D QWalk invariants")
if __name__ == "__main__":
    main()
