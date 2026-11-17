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
    Step 4 – computer source (p, k)               DONE
    Step 5 – dynamic computer trust               DONE
    Step 6 – computer bias b                      DONE
    Steps 7–11 – noise, parties, output, experiment, switches: TODO
"""

import os
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

    # --- computer source (Steps 4–6) ---
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


def clip01(value):
    """Keep a number inside [0, 1]."""
    return min(max(value, 0.0), 1.0)


class Voter(Agent):
    """A voter in our extended model."""

    def __init__(self, model, node):
        super().__init__(model)                     # [Mesa]
        params = model.params
        self.node = node                            # position in the network
        # initial opinion x_i ~ U(0, 1)              # [Deffuant et al., 2000]
        self.opinion = model.random.random()
        # fixed susceptibility s_i ~ U(s_min, s_max) # [OWN]
        self.susceptibility = model.random.uniform(params.s_min, params.s_max)
        # personal computer trust, starts at tau_c0  # [OWN] Step 5
        self.computer_trust = params.tau_c0
        self.neighbours = []                        # filled in by the model

    def step(self):
        """One action of this voter in a tick (steps 1–4 of the tick in our proposal)."""
        params = self.model.params

        # 1) choose the information source                             [OWN] Step 4
        #    (the `params.p > 0` check means no random number is drawn when
        #    p = 0, so p = 0 gives exactly the same run as Step 3)
        use_computer = params.p > 0 and self.model.random.random() < params.p

        # 2) get the message and the trust in its source               [OWN]
        if use_computer:
            message = self.model.computer.consult()
            trust = self.computer_trust              # personal, changes over time
        else:
            partner = self.model.random.choice(self.neighbours)
            message = partner.opinion                # noise sigma is added in Step 7
            trust = params.tau_h                     # fixed

        # 3) bounded confidence: only listen if close enough  [Deffuant et al., 2000]
        accepted = abs(message - self.opinion) <= params.e
        if accepted:
            # receiver-only update, weighted by susceptibility and trust  [OWN]
            self.opinion += self.susceptibility * trust * (message - self.opinion)

        # 4) after using the computer, update trust in it              [OWN] Step 5
        if use_computer:
            self.update_computer_trust(accepted)

    def update_computer_trust(self, accepted):
        """Trust goes up a little after accepting the computer's message and
        down more after rejecting it: people lose trust in algorithms faster
        than they gain it.               [OWN, based on Dietvorst et al., 2015]
        """
        params = self.model.params
        change = params.trust_up if accepted else -params.trust_down
        self.computer_trust = clip01(self.computer_trust + change)


class Computer(Agent):
    """The computer source (e.g. a search engine or chatbot).       [OWN] Step 4

    It has no opinion of its own. When a voter consults it, it returns the mean
    opinion of k randomly chosen voters (from the whole population, not just
    the network), plus a bias b (misinformation, Step 6), kept inside [0, 1].
    """

    def consult(self):
        params = self.model.params
        sample = self.model.random.sample(self.model.voter_list, params.k)
        summary = np.mean([v.opinion for v in sample])
        return clip01(summary + params.b)            # bias b: Step 6


class VoterModel(Model):
    """Our extended model: network, computer source, trust, bias."""

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

        # one computer, shared by everyone                             [OWN]
        self.computer = Computer(self)

    def step(self):
        # only voters act; the computer just answers when consulted   [Mesa]
        self.agents_by_type[Voter].shuffle_do("step")

    def opinions(self):
        return np.array([a.opinion for a in self.voter_list])

    def computer_trusts(self):
        return np.array([a.computer_trust for a in self.voter_list])


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


def test_step4():
    """Step 4 tests: p = 0 identical to Step 3; computer summary correct. [OWN]"""
    print("\nStep 4 test – computer source")

    # 1) with p = 0 the result must be EXACTLY the same as in Step 3.
    #    Reference values: sum of all final opinions, produced by the Step 3
    #    code (commit c260756) with default parameters and seeds 0, 1, 2.
    step3_reference = {0: 249.6097507906078, 1: 258.30506738389084,
                       2: 254.33135948572186}
    for seed, expected in step3_reference.items():
        params = replace(PARAMS, seed=seed, p=0.0)
        got = float(run(VoterModel(params), params.T).opinions().sum())
        assert got == expected, f"seed {seed}: p=0 differs from Step 3!"
    print("p = 0 gives exactly the Step 3 results (seeds 0, 1, 2).")

    # 2) computer summary checked by hand
    model = VoterModel(replace(PARAMS, b=0.0))
    for v in model.voter_list:
        v.opinion = 0.3
    assert np.isclose(model.computer.consult(), 0.3)  # mean of 0.3s = 0.3
    model = VoterModel(replace(PARAMS, b=0.15))
    for v in model.voter_list:
        v.opinion = 0.3
    assert np.isclose(model.computer.consult(), 0.45)  # 0.3 + bias 0.15
    for v in model.voter_list:
        v.opinion = 0.95
    assert model.computer.consult() == 1.0             # 1.10 clipped to 1
    print("Computer summary passed (0.3 -> 0.3; +bias -> 0.45; 1.10 clipped to 1.0).")

    # 3) opinions stay in [0, 1] when the computer is used
    for p in (0.5, 1.0):
        run(VoterModel(replace(PARAMS, p=p)), PARAMS.T)
    print("Opinions stayed in [0, 1] with p = 0.5 and p = 1.")


def test_step5():
    """Step 5 tests: trust update by hand; plot one voter's trust.    [OWN]"""
    print("\nStep 5 test – dynamic computer trust")

    # 1) hand check. Voter: x = 0.40, s = 0.8, computer trust = 0.5, e = 0.2.
    model = VoterModel(replace(PARAMS, p=1.0))       # p = 1: always the computer
    i = model.voter_list[0]
    i.opinion, i.susceptibility, i.computer_trust = 0.40, 0.8, 0.5
    #    computer says 0.45 -> accepted:
    #    x = 0.40 + 0.8 * 0.5 * 0.05 = 0.42, trust = 0.5 + 0.02 = 0.52
    model.computer.consult = lambda: 0.45
    i.step()
    assert np.isclose(i.opinion, 0.42) and np.isclose(i.computer_trust, 0.52)
    #    computer says 0.90 -> rejected: x stays 0.42, trust = 0.52 - 0.05 = 0.47
    model.computer.consult = lambda: 0.90
    i.step()
    assert np.isclose(i.opinion, 0.42) and np.isclose(i.computer_trust, 0.47)
    #    trust never goes below 0
    i.computer_trust = 0.03
    i.step()
    assert i.computer_trust == 0.0
    print("Hand-checked trust update passed (accept: 0.50 -> 0.52, "
          "reject: 0.52 -> 0.47, floor at 0).")

    # 2) follow trust over time and save a plot
    import matplotlib
    matplotlib.use("Agg")                  # save to file, no window
    import matplotlib.pyplot as plt

    params = replace(PARAMS, p=0.5, e=0.2)
    model = VoterModel(params)
    follow = model.voter_list[:3]          # follow three voters
    history = {v.unique_id: [v.computer_trust] for v in follow}
    mean_trust = [model.computer_trusts().mean()]
    for _ in range(params.T):
        model.step()
        for v in follow:
            history[v.unique_id].append(v.computer_trust)
        mean_trust.append(model.computer_trusts().mean())

    fig, ax = plt.subplots(figsize=(7, 4))
    for v in follow:
        ax.plot(history[v.unique_id],
                label=f"voter {v.unique_id} (final opinion {v.opinion:.2f})")
    ax.plot(mean_trust, "k--", label="mean of all voters")
    ax.set(xlabel="tick", ylabel="computer trust", ylim=(-0.02, 1.02),
           title=f"Computer trust over time (p={params.p}, e={params.e}, b={params.b})")
    ax.legend()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/test_step5_trust.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Mean computer trust: start {mean_trust[0]:.2f}, "
          f"end {mean_trust[-1]:.2f}. Plot saved to results/test_step5_trust.png")


def test_step6():
    """Step 6 test: b = 0.15, p = 1 -> the bulk of opinions moves up. [OWN]

    Note: we first planned to test the MEAN opinion, but the mean hardly moves.
    Voters on both sides of the biased message are pulled towards it, while
    voters far below it reject it and stay behind. So the majority moves up
    and a minority is left behind, and the average stays about the same.
    The MEDIAN (the middle voter) shows the shift, so we test that.
    """
    print("\nStep 6 test – computer bias (p = 1, e = 0.2, seeds 0-4)")
    print(f"{'b':>5} | {'start median':>12} | {'end median':>10} | "
          f"{'end mean':>8} | {'end mean trust':>14}")
    end_medians = {}
    for b in (0.0, 0.15):
        starts, medians, means, trusts = [], [], [], []
        for seed in range(5):
            params = replace(PARAMS, p=1.0, e=0.2, b=b, seed=seed)
            model = VoterModel(params)
            starts.append(np.median(model.opinions()))
            x = run(model, params.T).opinions()
            medians.append(np.median(x))
            means.append(x.mean())
            trusts.append(model.computer_trusts().mean())
        end_medians[b] = np.mean(medians)
        print(f"{b:>5} | {np.mean(starts):>12.3f} | {np.mean(medians):>10.3f} | "
              f"{np.mean(means):>8.3f} | {np.mean(trusts):>14.3f}")
    assert end_medians[0.15] > end_medians[0.0] + 0.05, "bias did not move opinions up!"
    print("With bias the median voter ends clearly higher (the mean barely moves).")


if __name__ == "__main__":
    test_deffuant_clusters()
    test_step3()
    test_step4()
    test_step5()
    test_step6()
