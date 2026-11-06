"""Benchmarks for ``pgmpy.inference``.

Classes: ``VariableElimination``, ``BeliefPropagation``, ``BeliefPropagationWithMessagePassing``,
``ApproxInference``, ``CausalInference``, ``DBNInference`` and ``Mplp``. The inference classes are
constructed from a frozen model and queried; the constructors are timed as ``time_init``.

Fixtures (fast / slow), fixed per class in the module-level dicts below and chosen by probing the
slow tier against the per-call budget:

* ``VariableElimination``, ``BeliefPropagationWithMessagePassing`` and ``ApproxInference``:
  ``alarm`` / ``munin`` (``MODEL``). Exact queries on ``munin`` take a few ms through variable
  elimination.
* ``BeliefPropagation``: the junction trees ``jt_alarm`` / ``jt_andes`` (``BP_TREE``; ``andes`` is not
  connected, so its junction tree fixture is its largest component, 220 nodes), with queries chosen from
  ``alarm`` / ``andes``. ``munin`` is connected and calibrates in about 0.5 s, but a query on its
  junction tree takes anywhere from 0.02 s to 4 s depending on the process's string hash seed
  (pgmpy's choice of subtree root follows set order), which would make comparisons between commits
  noisy. The ``jt_andes`` queries take 10 to 25 ms for every seed tried. A ``BeliefPropagation`` built
  from a ``JunctionTree`` is timed; building the tree from a model is timed by ``models.py``.
* ``CausalInference``: ``alarm`` / ``andes`` (``CAUSAL_MODEL``), plus ``random_gaussian_8`` /
  ``arth150`` (``ATE_MODEL``) with ``gaussian_data`` for ``estimate_ate``, which fits linear
  regressions and so needs numeric data (the discrete fixtures hold state labels).
* ``CausalInference.identification_method`` enumerates backdoor and frontdoor sets by powerset over the
  observed variables, so its cost grows about 4x per 2 nodes. Measured on connected breadth-first prefixes
  of ``alarm`` grown from ``ARTCO2``, ``CATECHOL`` (PYTHONHASHSEED=0): 10 nodes 5 ms, 12 nodes 12 ms,
  14 nodes 33 ms, 16 nodes 0.12 s, 18 nodes 0.49 s; the full ``alarm`` was not measured to finish. It is
  size-limited: ``asia`` in the fast tier and the 16-node ``alarm`` subgraph in the slow tier
  (``IDENTIFICATION_MODEL``, ``IDENTIFICATION_NODES``; ``_fixtures.bfs_subgraph``). The pair is
  ``ARTCO2`` -> ``CATECHOL``, the ``treatment_outcome`` pair of ``alarm``, which the subgraph contains by
  construction (``asia`` uses its own ``treatment_outcome`` pair, ``either`` -> ``dysp``).
* ``DBNInference``: ``dbn_5`` / ``dbn_20`` (``DBN_MODEL``), not ``dbn_50``. ``DBNInference`` joins the
  interface nodes into one clique and multiplies a whole slice's junction-tree factors into one
  ``2 ** variables_per_slice`` table, so queries on 50 variables per slice are infeasible whatever the
  structure. ``dbn_20`` takes 0.1 to 0.4 s per call.
* ``Mplp``: ``pairwise_markov_network`` of ``alarm`` / ``andes`` (``MPLP_MODEL``). ``Mplp`` accepts
  only single-node and edge factors, so the ``markov_network`` fixtures (one factor per CPD family,
  up to seven variables) fail with it; the pairwise fixture averages each CPD down to node and edge
  factors.
* ``BeliefPropagationWithMessagePassing`` is exact only on loop-free factor graphs (it recurses
  forever on loops), so it uses ``tree_factor_graph``, which cuts a network down to a forest.

Queries and evidence follow one fixed rule: the query variables are the 4 lexicographically first
non-root nodes, and the evidence is the 2 lexicographically last nodes, each at its first state
(``query_and_evidence``). In the tree factor graph, roots are the nodes with a single-variable factor, and
the evidence is a state index, as ``BeliefPropagationWithMessagePassing`` expects. ``DBNInference``
queries the 4 lexicographically first variables at time slice 2 (slice 1 for backward inference) with
the last 2 variables at their first state as evidence (slice 0 for forward inference and query, slice 2
for backward inference).

``CausalInference`` needs a treatment ``X`` and an outcome ``Y`` joined by a directed edge (the
instrumental-variable methods raise otherwise): the lexicographically first edge ``(X, Y)`` for which
``X`` has a parent and ``Y`` has at least two parents (``treatment_outcome``). Its adjustment set is
the parents of ``X``, which is always valid. ``estimate_ate`` enumerates adjustment sets by powerset,
which only terminates quickly when the empty set is valid, so it uses the first edge ``(X, Y)`` (sorted)
whose tail ``X`` is a root and is the only path from ``X`` to ``Y`` (``ate_pair``).

``VariableElimination`` uses pgmpy's default elimination orders for ``query``, ``max_marginal`` and
``map_query`` (a few ms on ``munin``). ``induced_graph`` and ``induced_width`` need an explicit order
over every variable: the ``MinNeighbors`` order, computed once outside the timed code. The
elimination-order classes themselves are not part of this module.

``Mplp`` runs ``map_query`` with its default arguments; ``andes`` takes about 1.4 s. The
``ApproxInference`` methods use ``rng=0``.

Skipped:
    CausalInference.is_valid_backdoor_adjustment_set, .get_all_backdoor_adjustment_sets,
    .is_valid_frontdoor_adjustment_set, .get_all_frontdoor_adjustment_sets: deprecated
    BeliefPropagation.get_cliques, .get_clique_beliefs, .get_sepset_beliefs: trivial (accessors)
    Mplp.get_integrality_gap: trivial (returns the gap stored by map_query)
    Mplp.query: unsupported (raises NotImplementedError)
"""

import copy
from functools import lru_cache

import networkx as nx
from benchmarks._fixtures import (
    bfs_subgraph,
    dbn,
    discrete_bn,
    discrete_data,
    gaussian_bn,
    gaussian_data,
    junction_tree,
    load_spec,
    pairwise_markov_network,
    tree_factor_graph,
)
from benchmarks._tiers import ROWS, Bench

from pgmpy.inference import (
    ApproxInference,
    BeliefPropagation,
    BeliefPropagationWithMessagePassing,
    CausalInference,
    DBNInference,
    Mplp,
    VariableElimination,
)
from pgmpy.inference.EliminationOrder import MinNeighbors

MODEL = {"fast": "alarm", "slow": "munin"}
CAUSAL_MODEL = {"fast": "alarm", "slow": "andes"}
# Junction tree fixture and the network its queries are chosen from.
BP_TREE = {"fast": "jt_alarm", "slow": "jt_andes"}
BP_MODEL = {"jt_alarm": "alarm", "jt_andes": "andes"}
IDENTIFICATION_MODEL = {"fast": "asia", "slow": "alarm"}
IDENTIFICATION_NODES = {"fast": None, "slow": 16}  # None: the whole model
ATE_MODEL = {"fast": "random_gaussian_8", "slow": "arth150"}
DBN_MODEL = {"fast": "dbn_5", "slow": "dbn_20"}
MPLP_MODEL = {"fast": "alarm", "slow": "andes"}

# Forward samples drawn by ApproxInference (alarm takes 0.35 s per 1000 samples, munin 1.1 s).
APPROX_SAMPLES = {"fast": 500, "slow": 1_000}

N_QUERY = 4
N_EVIDENCE = 2


@lru_cache(maxsize=None)
def model(name):
    """A shared Bayesian network; no inference class here modifies it."""
    return discrete_bn(name)


@lru_cache(maxsize=None)
def query_and_evidence(name):
    """The 4 first non-root nodes (sorted) and the last 2 nodes (sorted) at their first state."""
    m = model(name)
    roots = set(m.get_roots())
    nodes = sorted(m.nodes)
    query = [n for n in nodes if n not in roots][:N_QUERY]
    return query, {n: m.states[n][0] for n in nodes[-N_EVIDENCE:]}


@lru_cache(maxsize=None)
def tree_graph(name):
    """A shared tree factor graph (building the munin one takes about 3 s); inference does not modify it."""
    return tree_factor_graph(name)


@lru_cache(maxsize=None)
def tree_query_and_evidence(name):
    """The same rule on the tree factor graph: roots have a single-variable factor, evidence is state 0."""
    fg = tree_graph(name)
    roots = {f.variables[0] for f in fg.get_factors() if len(f.variables) == 1}
    nodes = sorted(n for n in fg.nodes if isinstance(n, str))
    return [n for n in nodes if n not in roots][:N_QUERY], {n: 0 for n in nodes[-N_EVIDENCE:]}


@lru_cache(maxsize=None)
def elimination_order(name):
    """A MinNeighbors elimination order over every variable of the model."""
    m = model(name)
    return MinNeighbors(m).get_elimination_order(nodes=set(m.nodes), show_progress=False)


@lru_cache(maxsize=None)
def bp_query_and_evidence(tree):
    """The query rule applied to the nodes of a junction tree fixture, with roots from the full network."""
    m = model(BP_MODEL[tree])
    roots = set(m.get_roots())
    nodes = sorted({v for clique in junction_tree(tree).nodes() for v in clique})
    return [n for n in nodes if n not in roots][:N_QUERY], {n: m.states[n][0] for n in nodes[-N_EVIDENCE:]}


@lru_cache(maxsize=None)
def calibrated_bp(tree):
    """A calibrated BeliefPropagation, to be copied: calibrate and max_calibrate change its state."""
    return BeliefPropagation(junction_tree(tree))


@lru_cache(maxsize=None)
def treatment_outcome(name):
    """The first edge (sorted) whose tail has a parent and whose head has at least two parents."""
    m = model(name)
    return next((x, y) for x, y in sorted(m.edges()) if m.in_degree(x) >= 1 and m.in_degree(y) >= 2)


@lru_cache(maxsize=None)
def ate_pair(name):
    """The first edge (sorted) whose tail is a root and is the only path between its ends."""
    m = gaussian_bn(name)
    g = nx.DiGraph(m.edges())
    return next(
        (x, y) for x, y in sorted(m.edges()) if m.in_degree(x) == 0 and len(list(nx.all_simple_paths(g, x, y))) == 1
    )


class TimeVariableElimination(Bench):
    def prepare(self, tier):
        self.name = MODEL[tier]
        self.model = model(self.name)
        self.inference = VariableElimination(self.model)
        self.query, self.evidence = query_and_evidence(self.name)
        self.order = elimination_order(self.name)

    def time_init(self, tier):
        VariableElimination(self.model)

    def time_query(self, tier):
        self.inference.query(self.query, evidence=self.evidence, show_progress=False)

    def time_max_marginal(self, tier):
        self.inference.max_marginal(self.query, evidence=self.evidence, show_progress=False)

    def time_map_query(self, tier):
        self.inference.map_query(self.query, evidence=self.evidence, show_progress=False)

    def time_induced_graph(self, tier):
        self.inference.induced_graph(self.order)

    def time_induced_width(self, tier):
        self.inference.induced_width(self.order)


class TimeBeliefPropagation(Bench):
    def prepare(self, tier):
        self.tree = BP_TREE[tier]
        self.inference = copy.deepcopy(calibrated_bp(self.tree))
        self.query, self.evidence = bp_query_and_evidence(self.tree)

    def time_init(self, tier):
        BeliefPropagation(junction_tree(self.tree))

    def time_calibrate(self, tier):
        self.inference.calibrate()

    def time_max_calibrate(self, tier):
        self.inference.max_calibrate()

    def time_query(self, tier):
        self.inference.query(self.query, evidence=self.evidence, show_progress=False)

    def time_map_query(self, tier):
        self.inference.map_query(self.query, evidence=self.evidence, show_progress=False)


class TimeBeliefPropagationWithMessagePassing(Bench):
    def prepare(self, tier):
        self.name = MODEL[tier]
        self.factor_graph = tree_graph(self.name)
        # check_model is timed by time_init only.
        self.inference = BeliefPropagationWithMessagePassing(self.factor_graph, check_model=False)
        self.query, self.evidence = tree_query_and_evidence(self.name)
        # The first factor (in graph order) over the first query variable and its uniform messages.
        self.variable = self.query[0]
        self.factor = next(f for f in self.factor_graph.get_factors() if f.variables[0] == self.variable)
        self.factor_messages = [self.factor_graph.get_uniform_message(v) for v in self.factor.variables[1:]]
        self.variable_messages = [self.factor_graph.get_uniform_message(self.variable)] * 3

    def time_init(self, tier):
        BeliefPropagationWithMessagePassing(self.factor_graph)

    def time_query(self, tier):
        self.inference.query(self.query, evidence=self.evidence)

    def time_calc_variable_node_message(self, tier):
        self.inference.calc_variable_node_message(self.variable, self.variable_messages)

    def time_calc_factor_node_message(self, tier):
        self.inference.calc_factor_node_message(self.factor, self.factor_messages, self.variable)


class TimeApproxInference(Bench):
    def prepare(self, tier):
        self.name = MODEL[tier]
        self.model = model(self.name)
        self.inference = ApproxInference(self.model)
        self.query, self.evidence = query_and_evidence(self.name)
        self.n_samples = APPROX_SAMPLES[tier]
        self.samples = discrete_data(self.name, self.n_samples)
        self.state_names = {v: self.model.states[v] for v in self.query}

    def time_init(self, tier):
        ApproxInference(self.model)

    def time_get_distribution(self, tier):
        self.inference.get_distribution(self.samples, self.query, state_names=self.state_names)

    def time_query(self, tier):
        self.inference.query(
            self.query, n_samples=self.n_samples, evidence=self.evidence, show_progress=False, rng=0
        )

    def time_map_query(self, tier):
        self.inference.map_query(
            self.query, n_samples=self.n_samples, evidence=self.evidence, show_progress=False, rng=0
        )


class TimeCausalInference(Bench):
    def prepare(self, tier):
        self.name = CAUSAL_MODEL[tier]
        self.model = model(self.name)
        self.inference = CausalInference(self.model)
        self.x, self.y = treatment_outcome(self.name)
        self.adjustment = sorted(self.model.get_parents(self.x))
        last = sorted(self.model.nodes)[-1]
        self.do = {self.x: self.model.states[self.x][0]}
        self.evidence = {last: self.model.states[last][0]} if last not in (self.x, self.y) else None

    def time_init(self, tier):
        CausalInference(self.model)

    def time_query(self, tier):
        self.inference.query([self.y], do=self.do, evidence=self.evidence, show_progress=False)

    def time_get_minimal_adjustment_set(self, tier):
        self.inference.get_minimal_adjustment_set(self.x, self.y)

    def time_get_proper_backdoor_graph(self, tier):
        self.inference.get_proper_backdoor_graph([self.x], [self.y])

    def time_is_valid_adjustment_set(self, tier):
        self.inference.is_valid_adjustment_set([self.x], [self.y], self.adjustment)

    def time_get_scaling_indicators(self, tier):
        self.inference.get_scaling_indicators()

    def time_get_ivs(self, tier):
        self.inference.get_ivs(self.x, self.y)

    def time_get_conditional_ivs(self, tier):
        self.inference.get_conditional_ivs(self.x, self.y)

    def time_get_total_conditional_ivs(self, tier):
        self.inference.get_total_conditional_ivs(self.x, self.y)


class TimeCausalInferenceIdentification(Bench):
    """``identification_method`` on a small model: it is exponential in the number of variables."""

    def prepare(self, tier):
        name = IDENTIFICATION_MODEL[tier]
        self.x, self.y = treatment_outcome(name)
        n_nodes = IDENTIFICATION_NODES[tier]
        graph = model(name) if n_nodes is None else bfs_subgraph(name, [self.x, self.y], n_nodes)
        self.inference = CausalInference(graph)

    def time_identification_method(self, tier):
        self.inference.identification_method(self.x, self.y)


class TimeCausalInferenceEstimateATE(Bench):
    """``estimate_ate`` on a linear Gaussian network, since it regresses on numeric data."""

    def prepare(self, tier):
        self.name = ATE_MODEL[tier]
        self.inference = CausalInference(gaussian_bn(self.name))
        self.x, self.y = ate_pair(self.name)
        self.data = gaussian_data(self.name, ROWS[tier])

    def time_estimate_ate(self, tier):
        self.inference.estimate_ate(self.x, self.y, self.data)


class TimeDBNInference(Bench):
    def prepare(self, tier):
        self.name = DBN_MODEL[tier]
        self.model = dbn(self.name)
        self.inference = DBNInference(self.model)
        variables = sorted(load_spec(self.name)["variables"])
        query, evidence = variables[:N_QUERY], variables[-N_EVIDENCE:]
        self.query = [(v, 2) for v in query]
        self.backward_query = [(v, 1) for v in query]
        self.evidence = {(v, 0): 0 for v in evidence}
        self.backward_evidence = {(v, 2): 0 for v in evidence}

    def time_init(self, tier):
        DBNInference(self.model)

    def time_forward_inference(self, tier):
        self.inference.forward_inference(self.query, self.evidence)

    def time_backward_inference(self, tier):
        self.inference.backward_inference(self.backward_query, self.backward_evidence)

    def time_query(self, tier):
        self.inference.query(self.query, self.evidence)


class TimeMplp(Bench):
    def prepare(self, tier):
        # map_query changes the Mplp object, and Mplp may change the network, so both are built fresh.
        self.model = pairwise_markov_network(MPLP_MODEL[tier])
        self.inference = Mplp(self.model)

    def time_init(self, tier):
        Mplp(self.model)

    def time_map_query(self, tier):
        self.inference.map_query()

    def time_find_triangles(self, tier):
        self.inference.find_triangles()
