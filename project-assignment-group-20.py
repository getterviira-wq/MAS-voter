"""
Group 20 – MAS project
Voter Behaviour Change Based on the Information Source:
Human–Human versus Human–Computer, and the Role of Misinformation

Annotation convention (for Appendix 1):
    # [Deffuant et al., 2000]  -> rule taken from the original bounded-confidence model
    # [Mesa]                   -> standard Mesa 3.x usage pattern (Mesa documentation)
    # [OWN]                    -> our own contribution / design choice

Build status:
    Step 1 – parameter block                      DONE
    Step 2 – plain Deffuant baseline (two-way)    DONE
    Step 3 – small-world network, receiver-only   DONE
    Steps 4–11 – computer, trust, bias, noise, parties, output: TODO
"""

from dataclasses import dataclass, replace

import networkx as nx
import numpy as np
from mesa import Agent, Model


# =============================================================================
# STEP 1 – ALL PARAMETERS IN ONE PLACE                                   [OWN]
# =============================================================================
# Every number the model uses lives here. Nothing is hard-coded further down.
# To run a different condition, make a copy with `replace(params, e=0.3)`.
# Parameters marked "(later)" are not used yet; they are listed now so the
# whole team can see the full design in one place.

@dataclass(frozen=True)
class Params:
    # --- general ---
    seed: int = 0              # random seed -> same seed gives exactly the same run
    N: int = 500               # number of voters
    T: int = 200               # number of ticks (one tick = every voter acts once)

    # --- Deffuant core ---
    e: float = 0.2             # confidence threshold: only listen if |m - x_i| <= e
    mu: float = 0.5            # convergence parameter (plain Deffuant baseline only)

    # --- network (Step 3) ---
    network_type: str = "small_world"  # "small_world" (main) or "complete" (verification)
    mean_degree: int = 6       # Watts–Strogatz: each voter linked to k nearest
    rewire_prob: float = 0.1   # Watts–Strogatz rewiring probability

    # --- voters (Step 3; computer trust from Step 5) ---
    s_min: float = 0.2         # susceptibility s_i ~ U(s_min, s_max)
    s_max: float = 1.0
    tau_h: float = 0.5         # fixed trust in humans

    # --- computer source (later: Steps 4–6) ---
    p: float = 0.0             # probability of consulting the computer
    k: int = 10                # computer averages k random voters
    tau_c0: float = 0.5        # initial computer trust
    trust_down: float = 0.05   # trust loss after rejecting computer message
    trust_up: float = 0.02     # trust gain after accepting computer message
    b: float = 0.0             # computer bias (misinformation)

    # --- noise (later: Step 7) ---
    sigma: float = 0.05        # SD of Gaussian noise on human messages

    # --- parties (later: Step 8) ---
    party_positions: tuple = (0.125, 0.375, 0.625, 0.875)

    # --- measures ---
    cluster_gap: float = 0.05  # new cluster starts at a gap larger than this
    min_cluster_size: int = 25 # clusters smaller than this are ignored


PARAMS = Params()


# =============================================================================
# MEASURES
# =============================================================================

def count_clusters(opinions, gap, min_size):
    """Number of opinion clusters.                                       [OWN]

    Sort opinions; a new cluster starts wherever two neighbouring opinions
    differ by more than `gap`. Clusters with fewer than `min_size` voters
    are ignored (they are stragglers, not real opinion groups).
    """
    x = np.sort(np.asarray(opinions))
    # positions where the gap between consecutive sorted opinions is too big
    breaks = np.where(np.diff(x) > gap)[0]
    # cluster sizes = lengths of the pieces between the breaks
    edges = np.concatenate(([0], breaks + 1, [len(x)]))
    sizes = np.diff(edges)
    return int(np.sum(sizes >= min_size))


# =============================================================================
# STEP 2 – PLAIN DEFFUANT BASELINE (random mixing, two-way update)
# =============================================================================

class DeffuantVoter(Agent):
    """A voter in the original Deffuant model: just an opinion in [0, 1]."""

    def __init__(self, model):
        super().__init__(model)                     # [Mesa] gives unique_id
        # initial opinion x_i ~ U(0, 1)              # [Deffuant et al., 2000]
        self.opinion = self.model.random.random()

    def step(self):
        """Pick a random other voter and maybe move towards each other."""
        p = self.model.params

        # random mixing: any other voter can be the partner  [Deffuant et al., 2000]
        partner = self.model.random.choice(self.model.voter_list)
        while partner is self:
            partner = self.model.random.choice(self.model.voter_list)

        xi, xj = self.opinion, partner.opinion
        # bounded confidence: only interact if opinions are close enough
        if abs(xi - xj) <= p.e:                     # [Deffuant et al., 2000]
            # two-way update, both use the OLD values  [Deffuant et al., 2000]
            self.opinion = xi + p.mu * (xj - xi)
            partner.opinion = xj + p.mu * (xi - xj)


class DeffuantModel(Model):
    """Original bounded-confidence model (Deffuant et al., 2000).

    Kept as a verification baseline: our extended model should reproduce
    its behaviour when the extensions are switched off.
    """

    def __init__(self, params: Params):
        super().__init__(rng=params.seed)           # [Mesa] seeds self.random and self.rng
        self.params = params
        for _ in range(params.N):
            DeffuantVoter(self)                      # [Mesa] auto-registers in self.agents
        self.voter_list = list(self.agents)          # fixed list for fast random choice

    def step(self):
        # every voter acts once, in a new random order each tick   [Mesa]
        self.agents.shuffle_do("step")

    def opinions(self):
        return np.array([a.opinion for a in self.voter_list])


# =============================================================================
# STEP 3 – SMALL-WORLD NETWORK + RECEIVER-ONLY UPDATE                    [OWN]
# =============================================================================
# Differences from the Deffuant baseline above:
#   (a) voters only talk to their network neighbours, not to anyone;
#   (b) only the RECEIVER changes opinion (the sender stays the same);
#   (c) the step size is s_i * tau_h instead of a fixed mu.
# The bounded-confidence rule |m - x_i| <= e is still Deffuant et al. (2000).

def build_network(params: Params):
    """Create the social network. Nodes are numbered 0 .. N-1."""
    if params.network_type == "small_world":
        # [NetworkX] Watts & Strogatz (1998) small-world graph, seeded
        return nx.watts_strogatz_graph(params.N, params.mean_degree,
                                       params.rewire_prob, seed=params.seed)
    if params.network_type == "complete":
        # everyone linked to everyone = random mixing (for verification)
        return nx.complete_graph(params.N)
    raise ValueError(f"unknown network_type: {params.network_type}")


class Voter(Agent):
    """A voter in our extended model."""

    def __init__(self, model, node):
        super().__init__(model)                     # [Mesa]
        p = model.params
        self.node = node                            # position in the network
        # initial opinion x_i ~ U(0, 1)              # [Deffuant et al., 2000]
        self.opinion = model.random.random()
        # fixed susceptibility s_i ~ U(s_min, s_max) # [OWN]
        self.susceptibility = model.random.uniform(p.s_min, p.s_max)
        self.neighbours = []                        # filled in by the model

    def step(self):
        p = self.model.params

        # ask one random network neighbour j                         [OWN]
        partner = self.model.random.choice(self.neighbours)
        message = partner.opinion        # noise sigma is added in Step 7

        # bounded confidence: only listen if close enough  [Deffuant et al., 2000]
        if abs(message - self.opinion) <= p.e:
            # receiver-only update, weighted by susceptibility and trust  [OWN]
            self.opinion += self.susceptibility * p.tau_h * (message - self.opinion)


class VoterModel(Model):
    """Our extended model. Step 3: network + receiver-only update."""

    def __init__(self, params: Params):
        super().__init__(rng=params.seed)           # [Mesa]
        self.params = params
        self.network = build_network(params)

        # one voter per network node
        self.voter_list = [Voter(self, node) for node in self.network.nodes]
        # give every voter a list of its neighbouring Voter objects
        for voter in self.voter_list:
            voter.neighbours = [self.voter_list[n]
                                for n in self.network.neighbors(voter.node)]

    def step(self):
        self.agents.shuffle_do("step")              # [Mesa]

    def opinions(self):
        return np.array([a.opinion for a in self.voter_list])


# =============================================================================
# TESTS / QUICK CHECKS
# =============================================================================

def run(model, T):
    """Run T ticks; check after every tick that opinions stay in [0, 1]."""
    for _ in range(T):
        model.step()
        x = model.opinions()
        assert x.min() >= 0 and x.max() <= 1, "opinion left [0, 1]!"
    return model

def test_deffuant_clusters():
    """Step 2 test: larger e -> fewer clusters.                          [OWN]

    Expected (Deffuant et al., 2000): roughly 1 / (2e) clusters,
    so e = 0.1 -> several clusters, e = 0.3 -> about one cluster.
    """
    print("Step 2 test – plain Deffuant, number of clusters after T ticks")
    print(f"{'e':>5} | clusters per seed (0-4)")
    for e in (0.1, 0.2, 0.3):
        counts = []
        for seed in range(5):
            params = replace(PARAMS, e=e, seed=seed)
            model = DeffuantModel(params)
            for _ in range(params.T):
                model.step()
            x = model.opinions()
            # sanity check: opinions must stay inside [0, 1]
            assert x.min() >= 0 and x.max() <= 1
            counts.append(count_clusters(x, params.cluster_gap,
                                         params.min_cluster_size))
        print(f"{e:>5} | {counts}")

    # reproducibility check: same seed -> identical result
    a = DeffuantModel(PARAMS); b = DeffuantModel(PARAMS)
    for _ in range(10):
        a.step(); b.step()
    assert np.array_equal(a.opinions(), b.opinions()), "seed not reproducible!"
    print("Reproducibility check passed (same seed -> same opinions).")


def test_step3():
    """Step 3 tests: network, one interaction by hand, bounds, clusters. [OWN]"""
    print("\nStep 3 test – small-world network, receiver-only update")

    # 1) network has the right shape
    model = VoterModel(PARAMS)
    degrees = [d for _, d in model.network.degree()]
    print(f"Network: {model.network.number_of_nodes()} voters, "
          f"mean degree {np.mean(degrees):.1f}, min degree {min(degrees)}, "
          f"connected: {nx.is_connected(model.network)}")
    assert np.mean(degrees) == PARAMS.mean_degree

    # 2) one interaction checked by hand
    #    x_i = 0.40, x_j = 0.45, s_i = 0.8, tau_h = 0.5, e = 0.2
    #    -> x_i_new = 0.40 + 0.8 * 0.5 * (0.45 - 0.40) = 0.42, x_j unchanged
    i = model.voter_list[0]
    j = i.neighbours[0]
    i.opinion, j.opinion, i.susceptibility = 0.40, 0.45, 0.8
    i.neighbours = [j]                       # force i to talk to j
    i.step()
    assert np.isclose(i.opinion, 0.42) and j.opinion == 0.45
    #    too far apart (0.40 vs 0.90 > e) -> nothing happens
    i.opinion, j.opinion = 0.40, 0.90
    i.step()
    assert i.opinion == 0.40
    print("Hand-checked interaction passed (0.40 -> 0.42, sender unchanged; "
          "no update when too far apart).")

    # 3) opinions stay in [0, 1] (checked every tick inside run())
    #    + clusters on small-world vs complete network
    print(f"{'e':>5} | {'small-world clusters':<22} | complete-network clusters")
    for e in (0.1, 0.2, 0.3):
        sw, cp = [], []
        for seed in range(5):
            for net, out in (("small_world", sw), ("complete", cp)):
                params = replace(PARAMS, e=e, seed=seed, network_type=net)
                x = run(VoterModel(params), params.T).opinions()
                out.append(count_clusters(x, params.cluster_gap,
                                          params.min_cluster_size))
        print(f"{e:>5} | {str(sw):<22} | {cp}")
    print("Opinions stayed in [0, 1] in every tick of every run.")

    # 4) is T = 200 long enough? how much do opinions still move at the end?
    params = replace(PARAMS, e=0.2)
    model = VoterModel(params)
    run(model, params.T - 10)
    before = model.opinions()
    run(model, 10)
    change = np.mean(np.abs(model.opinions() - before))
    print(f"Mean opinion change in the last 10 ticks (e=0.2): {change:.6f}")


if __name__ == "__main__":
    test_deffuant_clusters()
    test_step3()
