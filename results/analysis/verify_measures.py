"""Independent re-computation of the outcome measures from voters.csv.gz,
compared with runs.csv (written by Role 3 as a cross-check of Role 1's code)."""
import numpy as np, pandas as pd

from pathlib import Path
import os
HERE = Path(__file__).resolve().parent
# find runs.csv whether this script sits in results/analysis/, analysis/, or results/
R = next(str(p) + os.sep for p in [HERE.parent, HERE.parent / "results", HERE / "results", HERE] if (p / "runs.csv").exists())
OUT = str(HERE / "output") + os.sep
os.makedirs(OUT, exist_ok=True)

runs = pd.read_csv(R + "runs.csv")
vot = pd.read_csv(R + "voters.csv.gz")

def clusters(x, gap=0.05, min_size=25):
    x = np.sort(np.asarray(x))
    cuts = np.flatnonzero(np.diff(x) > gap) + 1
    sizes = np.diff(np.r_[0, cuts, len(x)])
    return int((sizes >= min_size).sum())

def measures(g):
    xs, xe = g.opinion_start.values, g.opinion_end.values
    sh_s = np.bincount(g.party_start, minlength=4) / len(g)
    sh_e = np.bincount(g.party_end, minlength=4) / len(g)
    return pd.Series(dict(
        share_switched=(g.party_start != g.party_end).mean(),
        net_vote_shift=0.5 * np.abs(sh_e - sh_s).sum(),
        mean_abs_opinion_change=np.abs(xe - xs).mean(),
        mean_opinion=xe.mean(), median_opinion=np.median(xe),
        median_shift=np.median(xe) - np.median(xs), var_opinion=xe.var(),
        clusters=clusters(xe), mean_trust=g.trust_end.mean()))

mine = vot.groupby(["condition_id", "seed"]).apply(measures, include_groups=False).reset_index()
m = runs.merge(mine, on=["condition_id", "seed"], suffixes=("", "_mine"))
rows = []
for c in ["share_switched","net_vote_shift","mean_abs_opinion_change","mean_opinion",
          "median_opinion","median_shift","var_opinion","clusters","mean_trust"]:
    d = (m[c] - m[c + "_mine"]).abs()
    rows.append((c, d.max(), (d > 1e-6).sum()))
out = pd.DataFrame(rows, columns=["measure", "max_abs_diff", "n_runs_differing"])
print(len(m), "runs compared"); print(out.to_string(index=False))
out.to_csv(OUT + "verification_table.csv", index=False)
