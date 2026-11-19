"""Full-factorial ANOVA (p > 0 only) and paired party-share effects of bias.
Writes anova_full_factorial.csv and prints the numbers quoted in the Results notes.
Needs: pandas, numpy, scipy, statsmodels."""
from pathlib import Path
import os
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf, statsmodels.api as sm

HERE = Path(__file__).resolve().parent
R = next(str(p) + os.sep for p in [HERE.parent, HERE.parent / "results", HERE / "results", HERE] if (p / "runs.csv").exists())
OUT = str(HERE / "output") + os.sep
os.makedirs(OUT, exist_ok=True)

r = pd.read_csv(R + "runs.csv")
for k in range(4):
    r[f"sh{k}"] = r[f"share_party{k}_last10"]
r["right_change"] = (r.sh2 + r.sh3) - (r.share_party2_start + r.share_party3_start)

# ---- paired effect of bias on party shares (b = 0.15 minus b = 0, same seed and settings) ----
key = ["p", "tau_c0", "e", "seed"]
cols = ["sh0", "sh1", "sh2", "sh3", "right_change"]
a = r[r.b == 0].set_index(key); c = r[r.b == 0.15].set_index(key)
d = (c[cols] - a.loc[c.index, cols]).reset_index()
print("Paired bias effect on final party shares (last-10-tick average)")
for m in cols:
    x = d[m]; h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    print(f"  {m:13s} {x.mean():+.3f} [{x.mean()-h:+.3f}, {x.mean()+h:+.3f}]  share of pairs > 0: {(x > 0).mean():.2f}")

# ---- full-factorial ANOVA on runs with the computer in use (p = 0.5, 1) ----
s = r[r.p > 0].copy()
for col in ["p", "tau_c0", "e", "b"]:
    s[col] = s[col].astype(str)
rows = []
for y in ["share_switched", "median_shift", "var_opinion", "mean_trust"]:
    m = smf.ols(f"{y} ~ C(p)*C(tau_c0)*C(e)*C(b)", data=s).fit()
    an = sm.stats.anova_lm(m, typ=2)
    an["partial_eta2"] = an.sum_sq / (an.sum_sq + an.loc["Residual", "sum_sq"])
    an = an.drop("Residual").reset_index().rename(columns={"index": "term", "PR(>F)": "p_value"})
    an.insert(0, "outcome", y); an["R2_model"] = m.rsquared
    rows.append(an)
res = pd.concat(rows).round(5)
res.to_csv(OUT + "anova_full_factorial.csv", index=False)
print("\nLargest effects (partial eta squared) per outcome")
for y, g in res.groupby("outcome", sort=False):
    print(y, "| R2 =", g.R2_model.iloc[0].round(3))
    print(g.sort_values("partial_eta2", ascending=False).head(4)[["term", "F", "partial_eta2"]].to_string(index=False))
