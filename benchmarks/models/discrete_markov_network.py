"""Benchmarks for ``pgmpy.models.DiscreteMarkovNetwork``.

Skipped:
    DiscreteMarkovNetwork.states: trivial
"""

import networkx as nx
import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.models import DiscreteMarkovNetwork

MODEL = "alarm"
PRODUCT_CELLS = 20_000
INDEPENDENCE_NODES = 5


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


def bare(model):
    b = type(model)(model.edges())
    b.add_nodes_from(model.nodes)
    return b


def _cpds(name):
    model = discrete_bn(name)
    return [model.get_cpds(v) for v in topological(model)]


def factors(name):
    return [cpd.to_factor() for cpd in _cpds(name)]


def markov_network(name):
    return discrete_bn(name).to_markov_model()


def product_prefix(fs, max_cells):
    """Longest prefix of the factors (sorted by scope) whose joint state space is <= max_cells."""
    chosen, cards = [], {}
    for f in sorted(fs, key=lambda f: f.variables):
        merged = {**cards, **dict(zip(f.variables, f.cardinality))}
        if chosen and np.prod(list(merged.values()), dtype=float) > max_cells:
            break
        chosen.append(f)
        cards = merged
    return chosen


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(MODEL)


def prefix_factors(name, max_cells):
    return product_prefix(factors(name), max_cells)


def prefix_markov_network(name, max_cells):
    fs = prefix_factors(name, max_cells)
    variables = sorted({v for f in fs for v in f.variables})
    edges = {(a, b) for f in fs for a in f.variables for b in f.variables if a < b}
    model = DiscreteMarkovNetwork(sorted(edges))
    model.add_nodes_from(variables)
    model.add_factors(*fs)
    return model


def non_edge(g):
    nodes = sorted(g.nodes)
    for a in nodes:
        for b in nodes:
            if a != b and not g.has_edge(a, b) and not g.has_edge(b, a):
                return a, b


class TimeDiscreteMarkovNetwork(Bench):
    def setup(self):
        name = MODEL
        self.model = markov_network(name)
        self.nodes = sorted(self.model.nodes)
        self.first = self.nodes[0]
        self.edge = non_edge(self.model)
        self.some_factors = self.model.get_factors()[:3]
        self.factors = factors(name)
        self.bare = bare(self.model)

    def time_add_edge(self):
        self.model.add_edge(*self.edge)

    def time_add_factors(self):
        self.bare.add_factors(*self.factors)

    def time_check_model(self):
        self.model.check_model()

    def time_copy(self):
        self.model.copy()

    def time_get_cardinality(self):
        self.model.get_cardinality()

    def time_get_factors(self):
        self.model.get_factors(self.first)

    def time_markov_blanket(self):
        self.model.markov_blanket(self.first)

    def time_remove_factors(self):
        self.model.remove_factors(*self.some_factors)

    def time_to_factor_graph(self):
        self.model.to_factor_graph()

    def time_triangulate(self):
        self.model.triangulate()


class TimeDiscreteMarkovNetworkConversions(Bench):
    def setup(self):
        self.model = markov_network(MODEL)

    def time_to_bayesian_model(self):
        self.model.to_bayesian_model()

    def time_to_junction_tree(self):
        self.model.to_junction_tree()


class TimeDiscreteMarkovNetworkIndependencies(Bench):
    def setup(self):
        g = markov_network(MODEL)
        start = sorted(g.nodes)[0]
        order = [start] + [v for _, v in nx.bfs_edges(g, start, sort_neighbors=sorted)]
        nodes = order[:INDEPENDENCE_NODES]
        self.model = DiscreteMarkovNetwork(g.subgraph(nodes).edges())
        self.model.add_nodes_from(nodes)

    def time_get_local_independencies(self):
        self.model.get_local_independencies()


class TimeDiscreteMarkovNetworkPartition(Bench):
    def setup(self):
        self.model = prefix_markov_network(MODEL, PRODUCT_CELLS)

    def time_get_partition_function(self):
        self.model.get_partition_function()
