"""Benchmarks for ``pgmpy.models.ClusterGraph``.

Skipped:
    ClusterGraph.clique_beliefs: trivial
"""

import networkx as nx
import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.models import ClusterGraph

MODEL = "alarm"
PRODUCT_CELLS = 20_000


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


def junction_tree(name):
    return discrete_bn(name).to_junction_tree()


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(MODEL)


def cluster_graph(name, max_cells=None):
    jt = junction_tree(name)
    factor_of = {frozenset(f.scope()): f for f in jt.get_factors()}
    cliques = list(jt.nodes)
    if max_cells is not None:
        order = [cliques[0]] + [v for _, v in nx.bfs_edges(jt, cliques[0])]
        cards, cliques = {}, []
        for c in order:
            f = factor_of[frozenset(c)]
            merged = {**cards, **dict(zip(f.variables, f.cardinality))}
            if cliques and np.prod(list(merged.values()), dtype=float) > max_cells:
                break
            cliques.append(c)
            cards = merged
    kept = set(cliques)
    cg = ClusterGraph()
    cg.add_nodes_from(cliques)
    cg.add_edges_from([(u, v) for u, v in jt.edges if u in kept and v in kept])
    cg.add_factors(*[factor_of[frozenset(c)] for c in cliques])
    return cg


class TimeClusterGraph(Bench):
    def setup(self):
        self.model = cluster_graph(MODEL)
        self.cliques = list(self.model.nodes)
        self.first = self.cliques[0]
        self.some_factors = self.model.get_factors()[:3]
        self.factors = self.model.get_factors()
        self.bare = ClusterGraph()
        self.bare.add_nodes_from(self.cliques)
        self.new_nodes = [("N0", "N1"), ("N1", "N2")]
        self.edge = next(
            (a, b)
            for i, a in enumerate(self.cliques)
            for b in self.cliques[i + 1 :]
            if not self.model.has_edge(a, b) and set(a) & set(b)
        )

    def time_add_edge(self):
        self.model.add_edge(*self.edge)

    def time_add_factors(self):
        self.bare.add_factors(*self.factors)

    def time_add_node(self):
        self.model.add_node(self.new_nodes[0])

    def time_add_nodes_from(self):
        self.model.add_nodes_from(self.new_nodes)

    def time_check_model(self):
        self.model.check_model()

    def time_copy(self):
        self.model.copy()

    def time_get_cardinality(self):
        self.model.get_cardinality()

    def time_get_factors(self):
        self.model.get_factors(self.first)

    def time_remove_factors(self):
        self.model.remove_factors(*self.some_factors)


class TimeClusterGraphPartition(Bench):
    def setup(self):
        self.model = cluster_graph(MODEL, PRODUCT_CELLS)

    def time_get_partition_function(self):
        self.model.get_partition_function()
