"""Benchmarks for ``pgmpy.models.FactorGraph``."""

import networkx as nx
import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.models import FactorGraph

MODEL = "alarm"
PRODUCT_CELLS = 20_000


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


def _cpds(name):
    model = discrete_bn(name)
    return [model.get_cpds(v) for v in topological(model)]


def factors(name):
    return [cpd.to_factor() for cpd in _cpds(name)]


def factor_graph(name):
    return factor_graph_from(topological(discrete_bn(name)), factors(name))


def factor_graph_from(variables, fs):
    fg = FactorGraph()
    fg.add_nodes_from(variables)
    fg.add_nodes_from(fs)
    fg.add_factors(*fs)
    fg.add_edges_from([(v, f) for f in fs for v in f.variables])
    return fg


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


class TimeFactorGraph(Bench):
    def setup(self):
        name = MODEL
        self.model = factor_graph(name)
        self.variable = sorted(self.model.get_variable_nodes())[0]
        self.some_factors = self.model.get_factors()[:3]
        self.factor_node = self.some_factors[0]
        self.factors = factors(name)
        self.bare = FactorGraph()
        self.bare.add_nodes_from(self.model.get_variable_nodes())
        self.edge = (
            self.variable,
            next(f for f in self.model.get_factors() if self.variable not in f.variables),
        )

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

    def time_get_factor_nodes(self):
        self.model.get_factor_nodes()

    def time_get_factors(self):
        self.model.get_factors(self.factor_node)

    def time_get_point_mass_message(self):
        self.model.get_point_mass_message(self.variable, 0)

    def time_get_uniform_message(self):
        self.model.get_uniform_message(self.variable)

    def time_get_variable_nodes(self):
        self.model.get_variable_nodes()

    def time_remove_factors(self):
        self.model.remove_factors(*self.some_factors)

    def time_to_markov_model(self):
        self.model.to_markov_model()


class TimeFactorGraphJunctionTree(Bench):
    def setup(self):
        self.model = factor_graph(MODEL)

    def time_to_junction_tree(self):
        self.model.to_junction_tree()


class TimeFactorGraphPartition(Bench):
    def setup(self):
        fs = prefix_factors(MODEL, PRODUCT_CELLS)
        self.model = factor_graph_from(sorted({v for f in fs for v in f.variables}), fs)

    def time_get_partition_function(self):
        self.model.get_partition_function()
