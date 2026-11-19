import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from pathlib import Path
import os
HERE = Path(__file__).resolve().parent
# find runs.csv whether this script sits in results/analysis/, analysis/, or results/
R = next(str(p) + os.sep for p in [HERE.parent, HERE.parent / "results", HERE / "results", HERE] if (p / "runs.csv").exists())
OUT = str(HERE / "output") + os.sep
os.makedirs(OUT, exist_ok=True)


r = pd.read_csv(R + "runs.csv")
for k in range(4):
    r[f"sh{k}"] = r[f"share_party{k}_last10"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200})
TAU = {0.25: ("#0b3d91", "o", "-"), 0.5: ("#8a3b00", "s", "--"), 0.75: ("#1a1a1a", "^", ":")}
BCOL = {0.0: ("#0b3d91", "o", "-"), 0.15: ("#8a3b00", "s", "--")}

def mci(x):
    x = np.asarray(x, float); n = len(x)
    return x.mean(), stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n)

# ---------------- Figure 1: share switched vs p ----------------
fig, ax = plt.subplots(2, 3, figsize=(7, 4.1), sharey=True, sharex=True)
for i, b in enumerate([0.0, 0.15]):
    for j, e in enumerate([0.1, 0.2, 0.3]):
        a = ax[i, j]
        base = r[(r.p == 0) & (r.e == e)].share_switched
        m0, h0 = mci(base)
        for tau, (c, mk, ls) in TAU.items():
            ms, hs = [m0], [h0]
            for p in (0.5, 1.0):
                m, h = mci(r[(r.p == p) & (r.e == e) & (r.b == b) & (r.tau_c0 == tau)].share_switched)
                ms.append(m); hs.append(h)
            off = {0.25: -0.012, 0.5: 0, 0.75: 0.012}[tau]
            a.errorbar(np.array([0, .5, 1]) + off, ms, yerr=hs, color=c, marker=mk, ls=ls, ms=3.5, lw=1, capsize=1.5,
                       label=f"$\\tau_{{c0}}$ = {tau}")
        a.set_title(f"e = {e}, b = {b}")
        a.set_xticks([0, .5, 1]); a.set_ylim(0, 0.9)
        if i == 1: a.set_xlabel("Computer use p")
        if j == 0: a.set_ylabel("Share of voters who switched party")
ax[0, 0].legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(OUT + "fig1_share_switched.png"); plt.close(fig)

# ---------------- Figure 2: paired bias effect, mean vs median ----------------
key = ["p", "tau_c0", "e", "seed"]
a_ = r[r.b == 0].set_index(key); c_ = r[r.b == 0.15].set_index(key)
d = (c_[["mean_opinion", "median_shift", "sh0", "sh1", "sh2", "sh3"]] - a_.loc[c_.index, ["mean_opinion", "median_shift", "sh0", "sh1", "sh2", "sh3"]]).reset_index()
ds = d.groupby(["p", "e", "seed"]).mean(numeric_only=True).reset_index()   # average over tau_c0 inside each seed
fig, ax = plt.subplots(1, 2, figsize=(7, 2.5), sharey=True)
for j, p in enumerate([0.5, 1.0]):
    a = ax[j]
    for off, col, lab, mk, c in [(-0.012, "mean_opinion", "Mean opinion", "o", "#0b3d91"), (0.012, "median_shift", "Median opinion", "s", "#8a3b00")]:
        res = [mci(ds[(ds.p == p) & (ds.e == e)][col]) for e in (0.1, 0.2, 0.3)]
        a.errorbar(np.array([0.1, 0.2, 0.3]) + off, [x[0] for x in res], yerr=[x[1] for x in res], marker=mk, color=c, ls="-" if mk == "o" else "--",
                   ms=4, lw=1, capsize=2, label=lab)
    a.axhline(0, color="k", lw=.6); a.set_xticks([.1, .2, .3]); a.set_xlabel("Confidence threshold e")
    a.set_title(f"p = {p}")
ax[0].set_ylabel("Shift caused by bias\n(b = 0.15 minus b = 0)"); ax[0].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig2_bias_mean_vs_median.png"); plt.close(fig)

# ---------------- Figure 3: party shares ----------------
fig, ax = plt.subplots(2, 3, figsize=(7, 3.9), sharey=True)
names = ["Left", "Centre-\nleft", "Centre-\nright", "Right"]
for i, p in enumerate([0.5, 1.0]):
    for j, e in enumerate([0.1, 0.2, 0.3]):
        a = ax[i, j]
        s = r[(r.p == p) & (r.e == e)].groupby(["b", "seed"])[["sh0", "sh1", "sh2", "sh3"]].mean().reset_index()
        for o, b in enumerate([0.0, 0.15]):
            c = BCOL[b][0]
            res = [mci(s[s.b == b][f"sh{k}"]) for k in range(4)]
            a.bar(np.arange(4) + (o - .5) * .38, [x[0] for x in res], .36, yerr=[x[1] for x in res], color=c, alpha=.9 if b == 0 else .75,
                  hatch="" if b == 0 else "///", edgecolor="white" if b == 0 else "#ffffff", error_kw=dict(lw=.8, capsize=1.5), label=f"b = {b}")
        a.axhline(.25, color="k", lw=.7, ls=":")
        a.set_xticks(range(4)); a.set_xticklabels(names, fontsize=6.5); a.set_title(f"p = {p}, e = {e}")
        if j == 0: a.set_ylabel("Final vote share")
ax[0, 0].legend(frameon=False, loc="upper left", fontsize=6.5)
fig.tight_layout(); fig.savefig(OUT + "fig3_party_shares.png"); plt.close(fig)

# ---------------- Figure 4: variance, clusters, trust ----------------
fig, ax = plt.subplots(3, 3, figsize=(7, 5.2), sharex=True, sharey="row")
for row, (col, lab) in enumerate([("var_opinion", "Final opinion variance"), ("clusters", "Number of clusters"), ("mean_trust", "Mean trust in computer")]):
    for j, e in enumerate([0.1, 0.2, 0.3]):
        a = ax[row, j]
        base = mci(r[(r.p == 0) & (r.e == e)][col])
        if col != "mean_trust":
            for b, (c, mk, ls) in BCOL.items():
                per = r[(r.e == e) & (r.p > 0) & (r.b == b)].groupby(["p", "seed"])[col].mean().reset_index()
                res = [base] + [mci(per[per.p == p][col]) for p in (.5, 1.)]
                a.errorbar(np.array([0, .5, 1]) + (0.01 if b else -0.01), [x[0] for x in res], yerr=[x[1] for x in res], color=c, marker=mk, ls=ls, ms=3.5, lw=1, capsize=1.5, label=f"b = {b}")
        else:
            for b, (c, mk, ls) in BCOL.items():
                per = r[(r.e == e) & (r.p > 0) & (r.b == b)].groupby(["p", "seed"])[col].mean().reset_index()
                res = [mci(per[per.p == p][col]) for p in (.5, 1.)]
                a.errorbar(np.array([.5, 1]) + (0.01 if b else -0.01), [x[0] for x in res], yerr=[x[1] for x in res], color=c, marker=mk, ls=ls, ms=3.5, lw=1, capsize=1.5, label=f"b = {b}")
            a.set_ylim(0, 1.05)
        a.set_xticks([0, .5, 1])
        if row == 0: a.set_title(f"e = {e}")
        if j == 0: a.set_ylabel(lab)
        if row == 2: a.set_xlabel("Computer use p")
ax[0, 0].legend(frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig4_variance_clusters_trust.png"); plt.close(fig)
print("done")
