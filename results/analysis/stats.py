import numpy as np, pandas as pd
from scipy import stats
pd.set_option('display.width',220); pd.set_option('display.max_columns',40)
from pathlib import Path
import os
HERE = Path(__file__).resolve().parent
# find runs.csv whether this script sits in results/analysis/, analysis/, or results/
R = next(str(p) + os.sep for p in [HERE.parent, HERE.parent / "results", HERE / "results", HERE] if (p / "runs.csv").exists())
OUT = str(HERE / "output") + os.sep
os.makedirs(OUT, exist_ok=True)

r = pd.read_csv(R+"runs.csv")
r["last10_p"] = r[[f"share_party{k}_last10" for k in range(4)]].values.tolist()

def ci(x):
    x = np.asarray(x, float); n=len(x); m=x.mean(); se=x.std(ddof=1)/np.sqrt(n)
    h = stats.t.ppf(.975, n-1)*se; return m, h

# ---- paired bias effect: b=0.15 minus b=0 (same seed, p, tau_c0, e) ----
a = r[r.b==0].set_index(["p","tau_c0","e","seed"])
c = r[r.b==0.15].set_index(["p","tau_c0","e","seed"])
meas = ["share_switched","net_vote_shift","mean_opinion","median_opinion","median_shift","var_opinion","mean_trust","clusters"]
d = (c[meas]-a.loc[c.index, meas]).reset_index()
d.to_csv(OUT + "paired_diffs.csv", index=False)
print("N pairs", len(d))
print("\n== Paired bias effect overall (mean diff, 95% CI, paired t p) ==")
for m in meas:
    mu,h = ci(d[m]); t,p = stats.ttest_1samp(d[m],0)
    print(f"{m:16s} {mu:+.3f} [{mu-h:+.3f},{mu+h:+.3f}] p={p:.2g}")
print("\n== by p,e : mean diff of mean_opinion vs median_shift vs net_vote_shift ==")
g = d.groupby(["p","e"])[["mean_opinion","median_shift","net_vote_shift","share_switched","var_opinion"]].mean().round(3)
print(g.to_string())
# direction: how many pairs have median shift positive
print("\nshare of pairs with median_shift diff >0:", (d.median_shift>0).mean().round(3), " mean_opinion diff >0:", (d.mean_opinion>0).mean().round(3))
