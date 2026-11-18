"""Benchmarks for ``pgmpy.base.DAG``.

Skipped:
    DAG.to_daft, DAG.to_graphviz: plotting
    DAG.in_degree_iter, DAG.out_degree_iter: trivial
    _CoreGraph / _GraphRolesMixin properties (latents, observed, exposures, outcomes, directed_edges, undirected_edges): trivial
"""

from functools import cache

import networkx as nx
from benchmarks._bench import Bench, fresh

from pgmpy.base import DAG
from pgmpy.models import LinearGaussianBayesianNetwork

ROWS = 1_000
GAUSSIAN = 8
EDGE_PROB = 0.15
GRAPH_NODES = 20
INDEPENDENCIES_NODES = 10


@fresh
def dag():
    return DAG.get_random(n_nodes=GRAPH_NODES, edge_prob=EDGE_PROB, rng=0)


@cache
def syntax():
    g = dag()
    return g.to_lavaan(), g.to_dagitty()


def leading_subgraph(g, k):
    nodes = sorted(g.nodes)[:k]
    sub = DAG(g.subgraph(nodes).edges())
    sub.add_nodes_from(nodes)
    return sub


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


@fresh
def gaussian_data(name, n_rows, seed=0):
    model = gaussian_bn(name)
    return model.simulate(n_rows, rng=seed)[list(nx.topological_sort(model))]


class TimeDAG(Bench):
    def setup(self):
        self.g = dag()
        nodes = sorted(self.g.nodes)
        self.first, self.last = nodes[0], nodes[-1]
        self.given = nodes[1:4]
        self.other = dag()
        self.rg = self.g.with_role("exposures", self.first).with_role("outcomes", self.last)
        self.small = leading_subgraph(self.g, INDEPENDENCIES_NODES)
        self.lavaan, self.dagitty = syntax()
        self.strength_dag = DAG(gaussian_bn(GAUSSIAN).edges())
        self.strength_data = gaussian_data(GAUSSIAN, ROWS)

    def time_from_lavaan(self):
        DAG.from_lavaan(self.lavaan)

    def time_from_dagitty(self):
        DAG.from_dagitty(self.dagitty)

    def time_edge_strength(self):
        self.strength_dag.edge_strength(self.strength_data)

    def time_add_edge(self):
        self.g.add_edge(self.last, "new_node")

    def time_add_edges_from(self):
        self.g.add_edges_from([(n, "new_node") for n in self.given])

    def time_get_parents(self):
        self.g.get_parents(self.last)

    def time_get_children(self):
        self.g.get_children(self.first)

    def time_get_spouses(self):
        self.g.get_spouses(self.last)

    def time_get_ancestors(self):
        self.g.get_ancestors(self.last)

    def time_get_roots(self):
        self.g.get_roots()

    def time_get_leaves(self):
        self.g.get_leaves()

    def time_moralize(self):
        self.g.moralize()

    def time_get_independencies(self):
        self.small.get_independencies()

    def time_local_independencies(self):
        self.g.local_independencies(self.last)

    def time_is_iequivalent(self):
        self.g.is_iequivalent(self.other)

    def time_get_immoralities(self):
        self.g.get_immoralities()

    def time_is_dconnected(self):
        self.g.is_dconnected(self.first, self.last, observed=self.given)

    def time_minimal_dseparator(self):
        self.g.minimal_dseparator(self.first, self.last)

    def time_get_markov_blanket(self):
        self.g.get_markov_blanket(self.last)

    def time_active_trail_nodes(self):
        self.g.active_trail_nodes(self.first, observed=self.given)

    def time_to_pdag(self):
        self.g.to_pdag()

    def time_to_admg(self):
        self.g.to_admg()

    def time_to_adjacency(self):
        self.g.to_adjacency()

    def time_do(self):
        self.g.do(self.last)

    def time_get_ancestral_graph(self):
        self.g.get_ancestral_graph(self.given + [self.last])

    def time_get_random(self):
        DAG.get_random(n_nodes=20, edge_prob=0.2, rng=0)

    def time_to_lavaan(self):
        self.g.to_lavaan()

    def time_to_dagitty(self):
        self.g.to_dagitty()

    def time_copy(self):
        self.g.copy()

    def time_get_stats(self):
        self.rg.get_stats()

    def time_get_role(self):
        self.rg.get_role("exposures")

    def time_get_roles(self):
        self.rg.get_roles()

    def time_get_role_dict(self):
        self.rg.get_role_dict()

    def time_has_role(self):
        self.rg.has_role("outcomes")

    def time_with_role(self):
        self.g.with_role("latents", self.given)

    def time_without_role(self):
        self.rg.without_role("exposures")

    def time_is_valid_causal_structure(self):
        self.rg.is_valid_causal_structure()
