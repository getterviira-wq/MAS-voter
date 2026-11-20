"""Sensitivity analysis (Role 2): do the bias results depend on the assumed values?
Run from the main project folder:  python verification/sensitivity.py
Varies bias b and the trust rates around the defaults, at p = 0.5, tau_c0 = 0.5, T = 1000.
Same seeds in every setting, so differences come from the setting, not from a different start."""
import importlib.util, sys
from dataclasses import replace
from multiprocessing import Pool
import numpy as np, pandas as pd

spec = importlib.util.spec_from_file_location("model", "project-assignment-group-20.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

SEEDS, T = 20, 1000
E_VALUES = (0.2, 0.3)
# (label, b, trust_down, trust_up)
SETTINGS = [("no bias (b=0)", 0.0, 0.05, 0.02),
            ("b=0.10", 0.10, 0.05, 0.02),
            ("b=0.15 (default)", 0.15, 0.05, 0.02),
            ("b=0.30", 0.30, 0.05, 0.02),
            ("b=0.15, trust -0.02/+0.02", 0.15, 0.02, 0.02),
            ("b=0.15, trust -0.10/+0.02", 0.15, 0.10, 0.02),
            ("b=0.15, trust -0.05/+0.05", 0.15, 0.05, 0.05)]

def job(args):
    e, label, b, down, up, seed = args
    P = replace(m.PARAMS, p=0.5, tau_c0=0.5, e=e, b=b, trust_down=down, trust_up=up, seed=seed)
    model = m.VoterModel(P)
    x0 = model.opinions().copy()
    m.run(model, T - 10)
    last = []
    for _ in range(10):
        model.step(); last.append(model.party_shares())
    x = model.opinions()
    sh = np.mean(last, axis=0)
    return dict(e=e, setting=label, seed=seed, share_switched=model.share_switched(),
                median_shift=float(np.median(x) - np.median(x0)), var=x.var(),
                mean_trust=model.computer_trusts().mean(), **{f"party{k}": sh[k] for k in range(4)})

if __name__ == "__main__":
    jobs = [(e, l, b, d, u, s) for e in E_VALUES for (l, b, d, u) in SETTINGS for s in range(SEEDS)]
    with Pool() as pool:
        rows = pool.map(job, jobs)
    df = pd.DataFrame(rows)
    df.to_csv("results/sensitivity.csv", index=False)
    base = df[df.setting == "no bias (b=0)"].set_index(["e", "seed"])
    df["centre_right_gain"] = df.apply(lambda r: r.party2 - base.loc[(r.e, r.seed), "party2"], axis=1)
    out = df.groupby(["e", "setting"], sort=False)[["share_switched", "median_shift", "var", "mean_trust",
                                                    "party2", "centre_right_gain"]].mean().round(3)
    print(out.to_string())
    out.to_csv("results/sensitivity_summary.csv")
