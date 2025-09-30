
from baselines.numpy_perhead_bias_demo import compare_mean_vs_perhead
def main():
    mean_kl, kl, p_ph, p_mean = compare_mean_vs_perhead(T=32, H=8)
    assert mean_kl > 1e-4, f"mean KL too small: {mean_kl}"
    print("OK: per-head bias specialization")
if __name__ == "__main__":
    main()
