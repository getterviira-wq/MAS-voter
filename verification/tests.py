import numpy as np
from reference_model import run, party, count_clusters

def test_hand_calculation():
    # x_i = 0.4, message 0.6, s = 0.5, tau = 0.5, e = 0.3  ->  0.4 + 0.5*0.5*0.2 = 0.45
    x, m, s, tau = 0.4, 0.6, 0.5, 0.5
    assert abs(x + s * tau * (m - x) - 0.45) < 1e-12

def test_bounds_and_trust():
    r = run(p=1, b=0.15, tau_c0=0.75, e=0.3, T=30, seed=1)
    assert r["x"].min() >= 0 and r["x"].max() <= 1
    assert r["trust"].min() >= 0 and r["trust"].max() <= 1

def test_trust_unchanged_without_computer():
    r = run(p=0, tau_c0=0.5, T=20, seed=2)
    assert np.allclose(r["trust"], 0.5)

def test_same_seed_same_result():
    a, b_ = run(seed=5, T=10), run(seed=5, T=10)
    assert np.array_equal(a["x"], b_["x"])

def test_party_assignment():
    assert list(party(np.array([0.0, 0.24, 0.26, 0.49, 0.51, 0.74, 0.76, 1.0]))) == [0, 0, 1, 1, 2, 2, 3, 3]

def test_cluster_counter():
    x = np.r_[np.full(100, 0.1), np.full(100, 0.5), np.full(10, 0.9)]  # last group too small
    assert count_clusters(x) == 2

if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)