"""Benchmarks for the graph classes in ``pgmpy.base``.

Skipped:
    DAG.to_daft, DAG.to_graphviz: plotting
    DAG.in_degree_iter, DAG.out_degree_iter: trivial
    _CoreGraph / _GraphRolesMixin properties (latents, observed, exposures, outcomes, directed_edges, undirected_edges): trivial
"""

from benchmarks._fixtures import gaussian_bn, gaussian_data, graph, syntax
from benchmarks._bench import GAUSSIAN, ROWS, Bench

from pgmpy.base import ADMG, DAG, SimpleCausalModel

SUBGRAPH_NODES = {
    "get_independencies": 10,
    "paths": 12,
    "mag": 15,
}


def ends(g):
    nodes = sorted(g.nodes)
    return nodes[0], nodes[-1]


def leading_subgraph(g, k):
    nodes = sorted(g.nodes)[:k]
    if isinstance(g, DAG):
        sub = DAG(g.subgraph(nodes).edges())
        sub.add_nodes_from(nodes)
        return sub
    return g.get_subgraph(nodes)


def conditioning(g):
    return sorted(g.nodes)[1:4]


class TimeDAG(Bench):
    def setup(self):
        self.g = graph("dag")
        self.first, self.last = ends(self.g)
        self.given = conditioning(self.g)
        self.other = graph("dag")
        self.rg = self.g.with_role("exposures", self.first).with_role("outcomes", self.last)
        self.small = leading_subgraph(self.g, SUBGRAPH_NODES["get_independencies"])
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


class TimePDAG(Bench):
    def setup(self):
        self.g = graph("pdag")
        self.first, self.last = ends(self.g)
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


class TimeADMG(Bench):
    def setup(self):
        self.g = graph("admg")
        self.first, self.last = ends(self.g)
        self.given = conditioning(self.g)
        self.u, self.v, _ = min(self.g.get_edges(edge_types=["->"]))
        self.adj = self.g.to_adjacency()
        self.small = leading_subgraph(self.g, SUBGRAPH_NODES["paths"])
        self.small_first, self.small_last = ends(self.small)

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


class TimeMAG(Bench):
    def setup(self):
        self.g = graph("mag")
        self.given = conditioning(self.g)
        self.small = leading_subgraph(self.g, SUBGRAPH_NODES["mag"])
        self.u, self.v, _ = min(self.small.get_edges(edge_types=["->"]))
        self.x = sorted(self.small.nodes)[:1]

    def time_is_maximal(self):
        self.small.is_maximal()

    def time_is_visible_edge(self):
        self.small.is_visible_edge(self.u, self.v)

    def time_lower_manipulation(self):
        self.small.lower_manipulation(self.x)

    def time_upper_manipulation(self):
        self.g.upper_manipulation(self.given)


class TimeUndirectedGraph(Bench):
    def setup(self):
        self.g = graph("ug")
        self.first, self.last = ends(self.g)
        self.given = conditioning(self.g)

    def time_add_edge(self):
        self.g.add_edge(self.first, self.last)

    def time_add_edges_from(self):
        self.g.add_edges_from([(self.first, n) for n in self.given])

    def time_is_clique(self):
        self.g.is_clique(sorted(self.g.nodes)[:10])

    def time_is_triangulated(self):
        self.g.is_triangulated()


class TimeSimpleCausalModel(Bench):
    def time_init(self):
        SimpleCausalModel(exposures=2, outcomes=2, confounders=4, mediators=4, instruments=4)
