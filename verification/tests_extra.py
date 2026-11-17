"""Extra tests that check the REAL model code (reference_model.run), not just arithmetic.
Run:  python tests_extra.py"""
import numpy as np
from reference_model import run, party

def test_e0_nobody_moves():
    # e = 0: a message is only accepted if it is exactly equal -> opinions must not change
    r = run(p=0, sigma=0, e=0.0, T=20, seed=1)
    assert np.array_equal(r["x"], r["x_start"])

def test_e1_opinions_converge():
    # e = 1: every message is accepted -> opinions come together (variance falls a lot)
    r = run(p=0, sigma=0, e=1.0, T=100, seed=1)
    assert r["var"] < 0.1 * r["x_start"].var()

def test_trust_goes_down_exactly_when_rejected():
    # p=1, e=0: computer message always rejected -> trust 0.5 falls 0.05 per tick -> 0 after 10 ticks
    r = run(p=1, e=0.0, tau_c0=0.5, T=10, seed=1)
    assert np.allclose(r["trust"], 0.0)
    # after 4 ticks it must be 0.5 - 4*0.05 = 0.30
    r = run(p=1, e=0.0, tau_c0=0.5, T=4, seed=1)
    assert np.allclose(r["trust"], 0.30)

def test_trust_goes_up_exactly_when_accepted():
    # p=1, e=1: always accepted -> trust 0.5 + 10*0.02 = 0.70
    r = run(p=1, e=1.0, tau_c0=0.5, T=10, seed=1)
    assert np.allclose(r["trust"], 0.70)

def test_trust_capped_at_1():
    r = run(p=1, e=1.0, tau_c0=0.95, T=50, seed=1)
    assert r["trust"].max() <= 1.0 and np.allclose(r["trust"], 1.0)

def test_bias_moves_median_up():
    a = np.mean([np.median(run(p=1, e=0.2, b=0.0, T=100, seed=s)["x"]) for s in range(5)])
    b = np.mean([np.median(run(p=1, e=0.2, b=0.15, T=100, seed=s)["x"]) for s in range(5)])
    assert b > a + 0.05

def test_party_boundaries():
    # exactly on a boundary (0.25, 0.5, 0.75): documented rule = lower party
    assert list(party(np.array([0.25, 0.5, 0.75]))) == [0, 1, 2]

def test_vote_shares_sum_to_1():
    r = run(p=0.5, T=20, seed=3)
    assert abs(r["shares"].sum() - 1) < 1e-12

if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)