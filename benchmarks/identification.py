"""Benchmarks for ``pgmpy.identification``. ``Adjustment`` and ``Frontdoor`` run on the DAG only, ``Frontdoor`` on a
``FRONTDOOR_NODES`` BFS prefix (its search is exponential when no set exists).

Skipped:
    ID.__call__, Adjustment.__call__, Frontdoor.__call__ (alias of ``identify``), constructors: trivial
    ProbabilityNode.marginalize and the other tree-node classes: internal
"""

from functools import cache

import networkx as nx

from benchmarks._fixtures import bfs_prefix, graph
from benchmarks._bench import Bench

from pgmpy.identification import ID, Adjustment, Frontdoor

DAG_GRAPH = "dag"
FRONTDOOR_NODES = 12


def descendants(g, node):
    return (
        set(g.get_descendants(node))
        if hasattr(g, "get_descendants")
        else nx.descendants(g, node)
    )


def treatment_outcome(g):
    for x in sorted(g.nodes()):
        below = sorted(descendants(g, x) - {x})
        if below:
            return x, below[-1]


@cache
def causal_graph(name, n_nodes=None):
    g = graph(name)
    x, y = treatment_outcome(g)
    if n_nodes is not None:
        g = bfs_prefix(g.edges(), [x, y], n_nodes)
    return g.with_role("exposures", x).with_role("outcomes", y)


@cache
def adjusted(name):
    return Adjustment().identify(causal_graph(name))[0]


@cache
def with_frontdoor(name, n_nodes):
    return Frontdoor().identify(causal_graph(name, n_nodes))[0]


@cache
def expression(name):
    return ID().identify(causal_graph(name))


class TimeID(Bench):
    def setup(self):
        self.admg = causal_graph("admg")
        self.dag = causal_graph(DAG_GRAPH)
        self.id = ID()

    def time_identify(self):
        self.id.identify(self.admg)

    def time_identify_dag(self):
        self.id.identify(self.dag)


class TimeAdjustment(Bench):
    def setup(self):
        name = DAG_GRAPH
        self.dag = causal_graph(name)
        self.adjusted = adjusted(name)
        self.adjustment = Adjustment()

    def time_identify(self):
        self.adjustment.identify(self.dag)

    def time_validate(self):
        self.adjustment.validate(self.adjusted)


class TimeFrontdoor(Bench):
    def setup(self):
        name, n_nodes = DAG_GRAPH, FRONTDOOR_NODES
        self.dag = causal_graph(name, n_nodes)
        self.with_frontdoor = with_frontdoor(name, n_nodes)
        self.frontdoor = Frontdoor()

    def time_identify(self):
        self.frontdoor.identify(self.dag)

    def time_validate(self):
        self.frontdoor.validate(self.with_frontdoor)


class TimeProbabilityExpressionTree(Bench):
    def setup(self):
        self.tree = expression(DAG_GRAPH)

    def time_to_latex(self):
        self.tree.to_latex()

    def time_collect_node_types(self):
        self.tree.collect_node_types()

    def time_find_leaves(self):
        self.tree.find_leaves()
