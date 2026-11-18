"""Benchmarks for ``pgmpy.base.UndirectedGraph``."""

from benchmarks._bench import Bench, fresh

from pgmpy.base import DAG

EDGE_PROB = 0.15
GRAPH_NODES = 20


@fresh
def undirected_graph():
    return DAG.get_random(n_nodes=GRAPH_NODES, edge_prob=EDGE_PROB, rng=0).moralize()


class TimeUndirectedGraph(Bench):
    def setup(self):
        self.g = undirected_graph()
        nodes = sorted(self.g.nodes)
        self.first, self.last = nodes[0], nodes[-1]
        self.given = nodes[1:4]

    def time_add_edge(self):
        self.g.add_edge(self.first, self.last)

    def time_add_edges_from(self):
        self.g.add_edges_from([(self.first, n) for n in self.given])

    def time_is_clique(self):
        self.g.is_clique(sorted(self.g.nodes)[:10])

    def time_is_triangulated(self):
        self.g.is_triangulated()
