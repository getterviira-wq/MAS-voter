"""Cross-check: independent reference model vs final Mesa model (averages over seeds).
Run from the main project folder:  python verification/crosscheck_mesa_vs_reference.py
Same seed gives DIFFERENT runs in the two models (different random generators), so we compare means only."""
import sys, importlib.util
sys.path.insert(0, "verification")
import numpy as np
from dataclasses import replace
from reference_model import run

spec = importlib.util.spec_from_file_location("m1", "project-assignment-group-20.py")
m1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m1)

SEEDS, T = 10, 200
settings = {"p=0.5 e=0.2 b=0": dict(p=0.5, e=0.2, b=0),
            "p=1 e=0.3 b=0.15": dict(p=1, e=0.3, b=0.15),
            "p=1 e=0.1 b=0": dict(p=1, e=0.1, b=0)}
for sigma in (0.0, 0.05):
    print(f"\nsigma = {sigma}  ({SEEDS} seeds, T = {T})")
    print("setting | variance (ref / mesa) | mean trust (ref / mesa) | median opinion (ref / mesa) | share switched (ref / mesa)")
    for lab, kw in settings.items():
        R = [run(sigma=sigma, T=T, seed=s, **kw) for s in range(SEEDS)]
        M = []
        for s in range(SEEDS):
            mod = m1.VoterModel(replace(m1.PARAMS, seed=s, sigma=sigma, **kw)); m1.run(mod, T); M.append(mod)
        f = lambda a, b: f"{np.mean(a):.3f} / {np.mean(b):.3f}"
        print(lab, "|", f([r["var"] for r in R], [m.opinions().var() for m in M]),
              "|", f([r["trust"].mean() for r in R], [m.computer_trusts().mean() for m in M]),
              "|", f([np.median(r["x"]) for r in R], [np.median(m.opinions()) for m in M]),
              "|", f([r["switched"] for r in R], [m.share_switched() for m in M]))
