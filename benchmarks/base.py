"""Benchmarks for the graph classes in ``pgmpy.base``.

Inputs are the ``dag_*``, ``pdag_*``, ``admg_*``, ``mag_*`` and ``ug_*`` graph fixtures
(20 nodes in the fast tier, 500 nodes in the slow tier). Methods that take nodes use the
first and last nodes in sorted order, plus the three nodes after the first as a conditioning
set, unless a comment says otherwise.

Several methods enumerate simple paths or all node pairs and are exponential in graph size, so
they are infeasible on the 500-node fixtures. These run on the first ``k`` nodes (sorted) of the
20-node fixtures, with ``k`` set per tier in ``SUBGRAPH_NODES`` below: ``DAG.get_independencies``,
``MAG.is_maximal``, ``MAG.is_visible_edge``, ``MAG.lower_manipulation``, and (through
``_GraphAlgorithms``) ``get_all_paths`` and ``has_inducing_path`` on ADMG.

Methods defined on the private bases ``_CoreGraph``, ``_GraphAlgorithms`` (shared by PDAG,
ADMG and MAG) and ``_GraphRolesMixin`` (shared with DAG) are benchmarked once, through ADMG
and DAG respectively. ``SimpleCausalModel`` defines no methods of its own, so only its
constructor is timed.

Skipped:
    DAG.to_daft, DAG.to_graphviz: plotting
    DAG.in_degree_iter, DAG.out_degree_iter: trivial
    _CoreGraph / _GraphRolesMixin properties (latents, observed, exposures, outcomes,
    directed_edges, undirected_edges): trivial
"""

from benchmarks._fixtures import gaussian_bn, gaussian_data, graph, syntax
from benchmarks._tiers import ROWS, Bench

from pgmpy.base import ADMG, DAG, SimpleCausalModel

DAG_FIXTURE = {"fast": "dag_20", "slow": "dag_500"}
PDAG_FIXTURE = {"fast": "pdag_20", "slow": "pdag_500"}
ADMG_FIXTURE = {"fast": "admg_20", "slow": "admg_500"}
MAG_FIXTURE = {"fast": "mag_20", "slow": "mag_500"}
# DAG.edge_strength: a linear Gaussian network's DAG and data sampled from it.
STRENGTH_MODEL = {"fast": "random_gaussian_8", "slow": "arth150"}
SYNTAX_FIXTURE = {"fast": "syntax_20", "slow": "syntax_500"}
UG_FIXTURE = {"fast": "ug_20", "slow": "ug_500"}

# Edge probabilities for get_random, chosen to match the density of the dag_* fixtures.
RANDOM_DAG = {"fast": (20, 0.2), "slow": (500, 0.005)}

# Number of leading (sorted) nodes of the 20-node fixture used by the exponential methods.
SUBGRAPH_NODES = {
    "get_independencies": {"fast": 10, "slow": 16},
    "paths": {"fast": 12, "slow": 20},
    "mag": {"fast": 15, "slow": 18},
}

# Number of variables of each role for SimpleCausalModel.
SCM_SIZES = {
    "fast": dict(exposures=2, outcomes=2, confounders=4, mediators=4, instruments=4),
    "slow": dict(exposures=10, outcomes=10, confounders=50, mediators=50, instruments=50),
}


def ends(g):
    """First and last node in sorted order."""
    nodes = sorted(g.nodes)
    return nodes[0], nodes[-1]


def leading_subgraph(g, k):
    """Induced subgraph on the first k nodes in sorted order, as a new graph of the same class."""
    nodes = sorted(g.nodes)[:k]
    if isinstance(g, DAG):
        sub = DAG(g.subgraph(nodes).edges())
        sub.add_nodes_from(nodes)
        return sub
    return g.get_subgraph(nodes)


def conditioning(g):
    """The three nodes after the first one in sorted order."""
    return sorted(g.nodes)[1:4]


class TimeDAG(Bench):
    def prepare(self, tier):
        self.name = DAG_FIXTURE[tier]
        self.g = graph(self.name)
        self.first, self.last = ends(self.g)
        self.given = conditioning(self.g)
        self.n_nodes, self.edge_prob = RANDOM_DAG[tier]
        self.other = graph(self.name)
        # Graph with exposure and outcome roles, for the role methods and get_stats.
        self.rg = self.g.with_role("exposures", self.first).with_role("outcomes", self.last)
        self.small = leading_subgraph(graph("dag_20"), SUBGRAPH_NODES["get_independencies"][tier])
        self.lavaan, self.dagitty = syntax(SYNTAX_FIXTURE[tier])
        model = STRENGTH_MODEL[tier]
        self.strength_dag = DAG(gaussian_bn(model).edges())
        self.strength_data = gaussian_data(model, ROWS[tier])

    def time_from_lavaan(self, tier):
        DAG.from_lavaan(self.lavaan)

    def time_from_dagitty(self, tier):
        DAG.from_dagitty(self.dagitty)

    def time_edge_strength(self, tier):
        self.strength_dag.edge_strength(self.strength_data)

    def time_add_edge(self, tier):
        self.g.add_edge(self.last, "new_node")

    def time_add_edges_from(self, tier):
        self.g.add_edges_from([(n, "new_node") for n in self.given])

    def time_get_parents(self, tier):
        self.g.get_parents(self.last)

    def time_get_children(self, tier):
        self.g.get_children(self.first)

    def time_get_spouses(self, tier):
        self.g.get_spouses(self.last)

    def time_get_ancestors(self, tier):
        self.g.get_ancestors(self.last)

    def time_get_roots(self, tier):
        self.g.get_roots()

    def time_get_leaves(self, tier):
        self.g.get_leaves()

    def time_moralize(self, tier):
        self.g.moralize()

    def time_get_independencies(self, tier):
        self.small.get_independencies()

    def time_local_independencies(self, tier):
        self.g.local_independencies(self.last)

    def time_is_iequivalent(self, tier):
        self.g.is_iequivalent(self.other)

    def time_get_immoralities(self, tier):
        self.g.get_immoralities()

    def time_is_dconnected(self, tier):
        self.g.is_dconnected(self.first, self.last, observed=self.given)

    def time_minimal_dseparator(self, tier):
        self.g.minimal_dseparator(self.first, self.last)

    def time_get_markov_blanket(self, tier):
        self.g.get_markov_blanket(self.last)

    def time_active_trail_nodes(self, tier):
        self.g.active_trail_nodes(self.first, observed=self.given)

    def time_to_pdag(self, tier):
        self.g.to_pdag()

    def time_to_admg(self, tier):
        self.g.to_admg()

    def time_to_adjacency(self, tier):
        self.g.to_adjacency()

    def time_do(self, tier):
        self.g.do(self.last)

    def time_get_ancestral_graph(self, tier):
        self.g.get_ancestral_graph(self.given + [self.last])

    def time_get_random(self, tier):
        DAG.get_random(n_nodes=self.n_nodes, edge_prob=self.edge_prob, rng=0)

    def time_to_lavaan(self, tier):
        self.g.to_lavaan()

    def time_to_dagitty(self, tier):
        self.g.to_dagitty()

    def time_copy(self, tier):
        self.g.copy()

    def time_get_stats(self, tier):
        self.rg.get_stats()

    # Role methods, defined on _GraphRolesMixin.
    def time_get_role(self, tier):
        self.rg.get_role("exposures")

    def time_get_roles(self, tier):
        self.rg.get_roles()

    def time_get_role_dict(self, tier):
        self.rg.get_role_dict()

    def time_has_role(self, tier):
        self.rg.has_role("outcomes")

    def time_with_role(self, tier):
        self.g.with_role("latents", self.given)

    def time_without_role(self, tier):
        self.rg.without_role("exposures")

    def time_is_valid_causal_structure(self, tier):
        self.rg.is_valid_causal_structure()


class TimePDAG(Bench):
    def prepare(self, tier):
        self.g = graph(PDAG_FIXTURE[tier])
        self.first, self.last = ends(self.g)
        # An undirected edge (u, v) and the node set {u, v}: the first undirected edge in sorted
        # order, since the first/last node pair need not be joined by one.
        self.u, self.v = min(tuple(sorted(e)) for e in self.g.undirected_edges)

    def time_chain_component(self, tier):
        self.g.chain_component(self.u)

    def time_is_clique(self, tier):
        self.g.is_clique([self.u, self.v])

    def time_has_semidirected_path(self, tier):
        self.g.has_semidirected_path(self.first, self.last)

    def time_orient_undirected_edge(self, tier):
        self.g.orient_undirected_edge(self.u, self.v)

    def time_has_acyclic_extension(self, tier):
        self.g.has_acyclic_extension()

    def time_apply_meeks_rules(self, tier):
        self.g.apply_meeks_rules()

    def time_to_cpdag(self, tier):
        self.g.to_cpdag()

    def time_to_dag(self, tier):
        self.g.to_dag()


class TimeADMG(Bench):
    """ADMG's own method plus everything from ``_CoreGraph`` and ``_GraphAlgorithms``."""

    def prepare(self, tier):
        self.g = graph(ADMG_FIXTURE[tier])
        self.first, self.last = ends(self.g)
        self.given = conditioning(self.g)
        # A directed edge, taken as the first one in sorted order, for edge-level methods.
        self.u, self.v, _ = min(self.g.get_edges(edge_types=["->"]))
        self.adj = self.g.to_adjacency()
        # get_all_paths and has_inducing_path are exponential, so they use a smaller graph.
        self.small = leading_subgraph(graph("admg_20"), SUBGRAPH_NODES["paths"][tier])
        self.small_first, self.small_last = ends(self.small)

    def time_to_dag(self, tier):
        self.g.to_dag()

    def time_add_node(self, tier):
        self.g.add_node("new_node")

    def time_add_nodes_from(self, tier):
        self.g.add_nodes_from(["new_a", "new_b", "new_c"])

    def time_add_edge(self, tier):
        self.g.add_edge(self.first, "new_node", "->")

    def time_add_edges_from(self, tier):
        self.g.add_edges_from([(n, "new_node", "->") for n in self.given])

    def time_remove_edge(self, tier):
        self.g.remove_edge(self.u, self.v, "->")

    def time_remove_edges_from(self, tier):
        self.g.remove_edges_from([(self.u, self.v, "->")])

    def time_get_edges(self, tier):
        self.g.get_edges()

    def time_get_unique_edge_types(self, tier):
        self.g.get_unique_edge_types()

    def time_get_edge_type(self, tier):
        self.g.get_edge_type(self.u, self.v)

    def time_has_edge(self, tier):
        self.g.has_edge(self.u, self.v, "->")

    def time_replace_edge(self, tier):
        self.g.replace_edge(self.u, self.v, "->", "<>")

    def time_get_marker(self, tier):
        self.g.get_marker(self.u, self.v)

    def time_set_marker(self, tier):
        self.g.set_marker(self.v, self.u, ">")

    def time_copy(self, tier):
        self.g.copy()

    def time_get_subgraph(self, tier):
        self.g.get_subgraph(sorted(self.g.nodes)[: len(self.g) // 2])

    def time_get_skeleton(self, tier):
        self.g.get_skeleton()

    def time_get_directed_graph(self, tier):
        self.g.get_directed_graph()

    def time_do(self, tier):
        self.g.do(self.last)

    def time_get_roots(self, tier):
        self.g.get_roots()

    def time_get_leaves(self, tier):
        self.g.get_leaves()

    def time_get_neighbors(self, tier):
        self.g.get_neighbors(self.last)

    def time_get_parents(self, tier):
        self.g.get_parents(self.last)

    def time_get_children(self, tier):
        self.g.get_children(self.first)

    def time_get_spouses(self, tier):
        self.g.get_spouses(self.last)

    def time_get_ancestors(self, tier):
        self.g.get_ancestors(self.last)

    def time_get_descendants(self, tier):
        self.g.get_descendants(self.first)

    def time_get_reachable_nodes(self, tier):
        self.g.get_reachable_nodes(self.first)

    def time_get_district(self, tier):
        self.g.get_district()

    def time_to_adjacency(self, tier):
        self.g.to_adjacency()

    def time_from_adjacency(self, tier):
        ADMG.from_adjacency(self.adj)

    def time_is_collider(self, tier):
        self.g.is_collider(self.u, self.v, self.last, shielded=False)

    def time_get_ancestral_graph(self, tier):
        self.g.get_ancestral_graph(self.given + [self.last])

    def time_get_markov_blanket(self, tier):
        self.g.get_markov_blanket(self.last)

    def time_has_inducing_path(self, tier):
        self.small.has_inducing_path(self.small_first, self.small_last)

    def time_get_mconnected_nodes(self, tier):
        self.g.get_mconnected_nodes(self.first, conditioning_set=self.given)

    def time_is_mseparated(self, tier):
        self.g.is_mseparated(self.first, self.last, conditioning_set=self.given)

    def time_has_path(self, tier):
        self.g.has_path(self.first, self.last)

    def time_get_all_paths(self, tier):
        self.small.get_all_paths(self.small_first, self.small_last)

    def time_get_topological_order(self, tier):
        self.g.get_topological_order()

    def time_has_directed_cycle(self, tier):
        self.g.has_directed_cycle()


class TimeMAG(Bench):
    def prepare(self, tier):
        self.g = graph(MAG_FIXTURE[tier])
        self.given = conditioning(self.g)
        # is_maximal, is_visible_edge and lower_manipulation enumerate simple paths, so they use
        # a smaller graph.
        self.small = leading_subgraph(graph("mag_20"), SUBGRAPH_NODES["mag"][tier])
        small_nodes = sorted(self.small.nodes)
        # is_visible_edge needs a directed edge: the first one in sorted order.
        self.u, self.v, _ = min(self.small.get_edges(edge_types=["->"]))
        self.x = small_nodes[:1]

    def time_is_maximal(self, tier):
        self.small.is_maximal()

    def time_is_visible_edge(self, tier):
        self.small.is_visible_edge(self.u, self.v)

    def time_lower_manipulation(self, tier):
        self.small.lower_manipulation(self.x)

    def time_upper_manipulation(self, tier):
        self.g.upper_manipulation(self.given)


class TimeUndirectedGraph(Bench):
    def prepare(self, tier):
        self.g = graph(UG_FIXTURE[tier])
        self.first, self.last = ends(self.g)
        self.given = conditioning(self.g)

    def time_add_edge(self, tier):
        self.g.add_edge(self.first, self.last)

    def time_add_edges_from(self, tier):
        self.g.add_edges_from([(self.first, n) for n in self.given])

    def time_is_clique(self, tier):
        self.g.is_clique(sorted(self.g.nodes)[:10])

    def time_is_triangulated(self, tier):
        self.g.is_triangulated()


class TimeSimpleCausalModel(Bench):
    def prepare(self, tier):
        self.sizes = SCM_SIZES[tier]

    def time_init(self, tier):
        SimpleCausalModel(**self.sizes)
