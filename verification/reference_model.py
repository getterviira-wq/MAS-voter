"""Independent reference implementation (numpy + networkx) of the proposal model.
Purpose: cross-check the Mesa model and run the pilot. Not the final model."""
import numpy as np
import networkx as nx

PARTIES = np.array([0.125, 0.375, 0.625, 0.875])

def party(x):
    """Index of the nearest party for each opinion (ties go to the lower party)."""
    return np.abs(x[:, None] - PARTIES[None, :]).argmin(axis=1)

def count_clusters(x, gap=0.05, min_size=25):
    """New cluster at a gap above `gap`; clusters smaller than `min_size` ignored."""
    xs = np.sort(x)
    cuts = np.where(np.diff(xs) > gap)[0] + 1
    sizes = [len(c) for c in np.split(xs, cuts)]
    return sum(s >= min_size for s in sizes)

def run(p=0.5, tau_c0=0.5, e=0.2, b=0.0, N=500, T=200, sigma=0.05, k=10,
        tau_h=0.5, down=0.05, up=0.02, seed=0, complete=False, two_way=False,
        s_range=(0.2, 1.0)):
    rng = np.random.default_rng(seed)
    G = nx.complete_graph(N) if complete else nx.watts_strogatz_graph(N, 6, 0.1, seed=seed)
    nbrs = [np.array(list(G[i])) for i in range(N)]
    x = rng.uniform(0, 1, N)
    s = rng.uniform(*s_range, N)
    trust = np.full(N, float(tau_c0))
    x_start, party_start = x.copy(), party(x)
    mean_trust, mean_change, checkpoints = [], [], {}
    for t in range(T):
        before = x.copy()
        for i in rng.permutation(N):
            use_computer = rng.random() < p
            if use_computer:
                m = np.clip(x[rng.choice(N, k, replace=False)].mean() + b, 0, 1)
                tau = trust[i]
            else:
                j = rng.choice(nbrs[i])
                m = np.clip(x[j] + rng.normal(0, sigma), 0, 1)
                tau = tau_h
            if abs(m - x[i]) <= e:
                if two_way and not use_computer:      # original Deffuant: both move
                    x[j] += s[j] * tau * (x[i] - x[j])
                x[i] += s[i] * tau * (m - x[i])
                if use_computer:
                    trust[i] = min(1.0, trust[i] + up)
            elif use_computer:
                trust[i] = max(0.0, trust[i] - down)
        mean_trust.append(trust.mean())
        mean_change.append(np.abs(x - before).mean())
        if (t + 1) % 50 == 0:
            checkpoints[t + 1] = dict(var=x.var(), clusters=count_clusters(x),
                                      switched=(party(x) != party_start).mean(), trust=trust.mean())
    pf = party(x)
    return dict(x=x, x_start=x_start, trust=trust, mean_trust=np.array(mean_trust),
                mean_change=np.array(mean_change), checkpoints=checkpoints,
                switched=(pf != party_start).mean(), abs_change=np.abs(x - x_start).mean(),
                var=x.var(), clusters=count_clusters(x), shares=np.bincount(pf, minlength=4) / N)