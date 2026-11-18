"""Benchmarks for ``pgmpy.models.DynamicBayesianNetwork``."""

from functools import cache

import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork

ROWS = 1_000
DBN_SLICES = 3
DBN_VARS = 5
DBN_SIMULATE_SAMPLES = 500
NEW_NODES = 10


@cache
def dbn_spec(n_vars, intra_prob=0.4, inter_prob=0.3, interface_every=2, n_states=2):
    """2-TBN with numpy CPDs (``DynamicBayesianNetwork.get_random`` breaks ``DBNInference``). The chain ``V(j-1) -> Vj``
    keeps a slice connected for the junction tree. Only every ``interface_every``-th variable has inter-slice edges, so
    sources match targets (``backward_inference``); the others have the same parents in both slices."""
    rng = np.random.default_rng(0)
    names = [f"V{i}" for i in range(n_vars)]
    intra = {j: [i for i in range(j) if rng.random() < intra_prob or i == j - 1] for j in range(n_vars)}
    interface = range(0, n_vars, interface_every)
    inter = {j: [i for i in interface if i == j or rng.random() < inter_prob] if j in interface else [] for j in range(n_vars)}
    nodes = []
    for t in (0, 1):
        for j in range(n_vars):
            parents = [(names[i], t) for i in intra[j]] + ([(names[i], 0) for i in inter[j]] if t else [])
            values = rng.dirichlet(np.ones(n_states), size=n_states ** len(parents)).T
            nodes.append({"name": (names[j], t), "parents": parents, "values": values})
    return {"n_states": n_states, "variables": names, "nodes": nodes}


def dbn(n_vars):
    spec = dbn_spec(n_vars)
    k = spec["n_states"]
    model = DynamicBayesianNetwork()
    model.add_nodes_from(spec["variables"])
    model.add_edges_from([(p, n["name"]) for n in spec["nodes"] for p in n["parents"]])
    model.add_cpds(
        *[
            TabularCPD(n["name"], k, n["values"], evidence=n["parents"] or None, evidence_card=[k] * len(n["parents"]) or None)
            for n in spec["nodes"]
        ]
    )
    return model


@fresh
def dbn_data(n_vars, n_rows, seed=0):
    return dbn(n_vars).simulate(n_rows, n_time_slices=DBN_SLICES, rng=seed, show_progress=False)


def bare_dbn(model):
    bare = DynamicBayesianNetwork()
    bare.add_nodes_from(sorted({n.node for n in model.nodes}))
    bare.add_edges_from(dbn_edges(model))
    return bare


def dbn_edges(model):
    return [(u.to_tuple(), v.to_tuple()) for u, v in model.edges()]


def missing_inter_edge(model):
    variables = sorted({n.node for n in model.nodes})
    present = set(dbn_edges(model))
    for a in variables:
        for b in variables:
            if ((a, 0), (b, 1)) not in present:
                return (a, 0), (b, 1)


class TimeDynamicBayesianNetwork(Bench):
    def setup(self):
        self.model = dbn(DBN_VARS)
        self.first = ("V0", 0)
        self.last = ("V0", 1)
        self.observed = [("V1", 0), ("V2", 0)]
        self.some_cpds = self.model.cpds[:3]

    def time_active_trail_nodes(self):
        self.model.active_trail_nodes(self.first, observed=self.observed)

    def time_check_model(self):
        self.model.check_model()

    def time_copy(self):
        self.model.copy()

    def time_get_constant_bn(self):
        self.model.get_constant_bn()

    def time_get_cpds(self):
        self.model.get_cpds(self.first)

    def time_get_cpds_all(self):
        self.model.get_cpds()

    def time_get_inter_edges(self):
        self.model.get_inter_edges()

    def time_get_interface_nodes(self):
        self.model.get_interface_nodes()

    def time_get_intra_edges(self):
        self.model.get_intra_edges()

    def time_get_markov_blanket(self):
        self.model.get_markov_blanket(self.last)

    def time_get_slice_nodes(self):
        self.model.get_slice_nodes()

    def time_moralize(self):
        self.model.moralize()

    def time_remove_cpds(self):
        self.model.remove_cpds(*self.some_cpds)

    def time_simulate(self):
        self.model.simulate(n_samples=DBN_SIMULATE_SAMPLES, n_time_slices=3, rng=0, show_progress=False)


class TimeDynamicBayesianNetworkBuild(Bench):
    def setup(self):
        model = dbn(DBN_VARS)
        self.cpds = list(model.cpds)
        self.edges = dbn_edges(model)
        self.new_edge = missing_inter_edge(model)
        self.new_nodes = [f"Z{i}" for i in range(NEW_NODES)]
        self.bare = bare_dbn(model)
        self.empty = DynamicBayesianNetwork()

    def time_add_cpds(self):
        self.bare.add_cpds(*self.cpds)

    def time_add_edge(self):
        self.bare.add_edge(*self.new_edge)

    def time_add_edges_from(self):
        self.empty.add_edges_from(self.edges)

    def time_add_node(self):
        self.bare.add_node(self.new_nodes[0])

    def time_add_nodes_from(self):
        self.bare.add_nodes_from(self.new_nodes)


class TimeDynamicBayesianNetworkInitialState(Bench):
    def setup(self):
        model = dbn(DBN_VARS)
        given = [c for c in model.cpds if c.variable[1] == 0 or any(v[1] == 0 for v in c.variables[1:])]
        self.model = bare_dbn(model)
        self.model.add_cpds(*given)

    def time_initialize_initial_state(self):
        self.model.initialize_initial_state()


class TimeDynamicBayesianNetworkFit(Bench):
    def setup(self):
        self.model = bare_dbn(dbn(DBN_VARS))
        self.data = dbn_data(DBN_VARS, ROWS)

    def time_fit(self):
        self.model.fit(self.data)
