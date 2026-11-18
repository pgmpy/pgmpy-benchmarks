"""Benchmarks for ``pgmpy.base.ADMG``."""

import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.base import ADMG, DAG

EDGE_PROB = 0.15
GRAPH_NODES = 20
LATENT_FRACTION = 0.1
PATHS_NODES = 12  # size of the subgraph for has_inducing_path and get_all_paths


@fresh
def admg():
    n_latent = max(1, round(GRAPH_NODES * LATENT_FRACTION))
    dag = DAG.get_random(n_nodes=GRAPH_NODES + n_latent, edge_prob=EDGE_PROB, rng=0)
    dag.latents = set(np.random.default_rng(0).choice(list(dag.nodes()), size=n_latent, replace=False).tolist())
    return dag.to_admg()


class TimeADMG(Bench):
    def setup(self):
        self.g = admg()
        nodes = sorted(self.g.nodes)
        self.first, self.last = nodes[0], nodes[-1]
        self.given = nodes[1:4]
        self.u, self.v, _ = min(self.g.get_edges(edge_types=["->"]))
        self.adj = self.g.to_adjacency()
        self.small = self.g.get_subgraph(nodes[:PATHS_NODES])
        small_nodes = sorted(self.small.nodes)
        self.small_first, self.small_last = small_nodes[0], small_nodes[-1]

    def time_to_dag(self):
        self.g.to_dag()

    def time_add_node(self):
        self.g.add_node("new_node")

    def time_add_nodes_from(self):
        self.g.add_nodes_from(["new_a", "new_b", "new_c"])

    def time_add_edge(self):
        self.g.add_edge(self.first, "new_node", "->")

    def time_add_edges_from(self):
        self.g.add_edges_from([(n, "new_node", "->") for n in self.given])

    def time_remove_edge(self):
        self.g.remove_edge(self.u, self.v, "->")

    def time_remove_edges_from(self):
        self.g.remove_edges_from([(self.u, self.v, "->")])

    def time_get_edges(self):
        self.g.get_edges()

    def time_get_unique_edge_types(self):
        self.g.get_unique_edge_types()

    def time_get_edge_type(self):
        self.g.get_edge_type(self.u, self.v)

    def time_has_edge(self):
        self.g.has_edge(self.u, self.v, "->")

    def time_replace_edge(self):
        self.g.replace_edge(self.u, self.v, "->", "<>")

    def time_get_marker(self):
        self.g.get_marker(self.u, self.v)

    def time_set_marker(self):
        self.g.set_marker(self.v, self.u, ">")

    def time_copy(self):
        self.g.copy()

    def time_get_subgraph(self):
        self.g.get_subgraph(sorted(self.g.nodes)[: len(self.g) // 2])

    def time_get_skeleton(self):
        self.g.get_skeleton()

    def time_get_directed_graph(self):
        self.g.get_directed_graph()

    def time_do(self):
        self.g.do(self.last)

    def time_get_roots(self):
        self.g.get_roots()

    def time_get_leaves(self):
        self.g.get_leaves()

    def time_get_neighbors(self):
        self.g.get_neighbors(self.last)

    def time_get_parents(self):
        self.g.get_parents(self.last)

    def time_get_children(self):
        self.g.get_children(self.first)

    def time_get_spouses(self):
        self.g.get_spouses(self.last)

    def time_get_ancestors(self):
        self.g.get_ancestors(self.last)

    def time_get_descendants(self):
        self.g.get_descendants(self.first)

    def time_get_reachable_nodes(self):
        self.g.get_reachable_nodes(self.first)

    def time_get_district(self):
        self.g.get_district()

    def time_to_adjacency(self):
        self.g.to_adjacency()

    def time_from_adjacency(self):
        ADMG.from_adjacency(self.adj)

    def time_is_collider(self):
        self.g.is_collider(self.u, self.v, self.last, shielded=False)

    def time_get_ancestral_graph(self):
        self.g.get_ancestral_graph(self.given + [self.last])

    def time_get_markov_blanket(self):
        self.g.get_markov_blanket(self.last)

    def time_has_inducing_path(self):
        self.small.has_inducing_path(self.small_first, self.small_last)

    def time_get_mconnected_nodes(self):
        self.g.get_mconnected_nodes(self.first, conditioning_set=self.given)

    def time_is_mseparated(self):
        self.g.is_mseparated(self.first, self.last, conditioning_set=self.given)

    def time_has_path(self):
        self.g.has_path(self.first, self.last)

    def time_get_all_paths(self):
        self.small.get_all_paths(self.small_first, self.small_last)

    def time_get_topological_order(self):
        self.g.get_topological_order()

    def time_has_directed_cycle(self):
        self.g.has_directed_cycle()
