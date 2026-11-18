"""Benchmarks for ``pgmpy.base.PDAG``."""

from benchmarks._bench import Bench, fresh

from pgmpy.base import DAG

EDGE_PROB = 0.15
GRAPH_NODES = 20


@fresh
def pdag():
    return DAG.get_random(n_nodes=GRAPH_NODES, edge_prob=EDGE_PROB, rng=0).to_pdag()


class TimePDAG(Bench):
    def setup(self):
        self.g = pdag()
        nodes = sorted(self.g.nodes)
        self.first, self.last = nodes[0], nodes[-1]
        self.u, self.v = min(tuple(sorted(e)) for e in self.g.undirected_edges)

    def time_chain_component(self):
        self.g.chain_component(self.u)

    def time_is_clique(self):
        self.g.is_clique([self.u, self.v])

    def time_has_semidirected_path(self):
        self.g.has_semidirected_path(self.first, self.last)

    def time_orient_undirected_edge(self):
        self.g.orient_undirected_edge(self.u, self.v)

    def time_has_acyclic_extension(self):
        self.g.has_acyclic_extension()

    def time_apply_meeks_rules(self):
        self.g.apply_meeks_rules()

    def time_to_cpdag(self):
        self.g.to_cpdag()

    def time_to_dag(self):
        self.g.to_dag()
