# Analysis scripts (Role 3)

These scripts analyse the final experiment data in `results/` (`runs.csv` and
`voters.csv.gz`, produced by `project-assignment-group-20.py`, T = 1000,
39 conditions × 30 seeds). They do not run the model.

## How to run

From the main project folder (after `pip install -r requirements.txt`):

    python results/analysis/verify_measures.py
    python results/analysis/stats.py
    python results/analysis/anova_and_party_shares.py
    python results/analysis/make_figures.py

The scripts can be run in any order. Output is written to `results/analysis/output/`.

## What each script does

| Script | What it does | Output |
|---|---|---|
| `verify_measures.py` | Recalculates every outcome measure independently from the per-voter data and compares with `runs.csv` (cross-check of the model's measure code) | `verification_table.csv` |
| `stats.py` | Paired bias effect (b = 0.15 minus b = 0, same seed and settings): mean difference, 95% CI, paired t-test | `paired_diffs.csv` |
| `anova_and_party_shares.py` | Bias effect on each party's vote share; full-factorial ANOVA (p > 0) with partial η² | `anova_full_factorial.csv` |
| `make_figures.py` | Figures with means and 95% CIs: party switching, bias effect (mean vs median), party shares, variance / clusters / trust | `fig1_…png` to `fig4_…png` |

