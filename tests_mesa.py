
"""Extra tests for the FINAL Mesa model (the file that gets submitted).
Put this next to the model file and run:  python tests_mesa.py
If Role 1 renames the model file, change MODEL_FILE below."""
import importlib.util
from dataclasses import replace
import numpy as np

MODEL_FILE = "project-assignment-group-20.py"
spec = importlib.util.spec_from_file_location("model", MODEL_FILE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
P, VoterModel, run, count_clusters = m.PARAMS, m.VoterModel, m.run, m.count_clusters

def test_e0_nobody_moves():
    # e = 0: a message is only accepted if it is exactly equal -> opinions must not change
    model = VoterModel(replace(P, e=0.0, p=0.5, sigma=0.0))
    start = model.opinions()
    run(model, 20)
    assert np.array_equal(model.opinions(), start)

def test_e1_opinions_converge():
    # e = 1: every message is accepted -> opinions come together
    model = VoterModel(replace(P, e=1.0, p=0.0, sigma=0.0, network_type="complete"))
    start_var = model.opinions().var()
    run(model, 100)
    assert model.opinions().var() < 0.1 * start_var

def test_trust_falls_exactly_when_rejected():
    # p = 1, e = 0: computer always rejected -> trust 0.50 - 4 * 0.05 = 0.30 after 4 ticks
    model = VoterModel(replace(P, p=1.0, e=0.0, tau_c0=0.5))
    run(model, 4)
    assert np.allclose(model.computer_trusts(), 0.30)
    run(model, 10)
    assert np.allclose(model.computer_trusts(), 0.0)       # floor at 0

def test_trust_rises_exactly_when_accepted():
    # p = 1, e = 1: always accepted -> trust 0.50 + 10 * 0.02 = 0.70
    model = VoterModel(replace(P, p=1.0, e=1.0, tau_c0=0.5))
    run(model, 10)
    assert np.allclose(model.computer_trusts(), 0.70)

def test_trust_capped_at_1():
    model = VoterModel(replace(P, p=1.0, e=1.0, tau_c0=0.95))
    run(model, 50)
    assert np.allclose(model.computer_trusts(), 1.0)

def test_trust_unchanged_without_computer():
    model = VoterModel(replace(P, p=0.0, tau_c0=0.5))
    run(model, 20)
    assert np.allclose(model.computer_trusts(), 0.5)

def test_cluster_counter():
    x = np.r_[np.full(100, 0.1), np.full(100, 0.5), np.full(10, 0.9)]  # last group too small
    assert count_clusters(x, 0.05, 25) == 2

if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f(); print("ok", name)
