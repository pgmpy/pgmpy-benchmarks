"""Benchmarks for ``pgmpy.models``: Bayesian, Markov, factor graph, cluster and structural equation models.

Classes: ``DiscreteBayesianNetwork``, ``DiscreteMarkovNetwork``, ``FactorGraph``, ``JunctionTree``,
``ClusterGraph``, ``LinearGaussianBayesianNetwork``, ``DynamicBayesianNetwork``, ``MarkovChain``,
``NaiveBayes``, ``SEMGraph``, ``SEMAlg`` and ``SEM``. Methods a Bayesian network inherits from ``DAG``
are benchmarked in ``benchmarks/base.py``. ``SEMEstimator``, ``FunctionalBayesianNetwork`` and
``FunctionalCPD`` are out of scope.

Fixtures (fast / slow). Each is fixed per tier and per method group in the module-level dicts
below, chosen by probing the slow tier against the per-call budget:

* ``DiscreteBayesianNetwork``: ``alarm`` / ``munin`` for the structural methods, CPD handling and
  ``to_junction_tree`` (``BN_MODEL``); ``alarm`` / ``andes`` for ``fit``, ``fit_update``,
  ``predict`` and ``predict_probability`` (``DATA_MODEL``); ``alarm`` / ``random_discrete_200``
  for ``simulate``.
* ``DiscreteMarkovNetwork`` and ``FactorGraph``: ``markov_network`` / ``factor_graph`` of
  ``alarm`` / ``andes``. ``andes`` has several connected components and a junction tree needs a
  connected graph, so the conversions that build one use a connected network instead:
  ``DiscreteMarkovNetwork.to_junction_tree`` and ``to_bayesian_model`` use ``munin``
  (``CONNECTED_MODEL``), and ``FactorGraph.to_junction_tree`` uses ``pathfinder``
  (``FACTOR_GRAPH_TREE_MODEL``; on ``munin`` it takes about 6 s).
* ``JunctionTree`` and ``ClusterGraph``: ``jt_alarm`` / ``jt_andes`` (``ClusterGraph`` is built from
  the junction tree's cliques, edges and factors).
* ``LinearGaussianBayesianNetwork``: ``random_gaussian_8`` / ``arth150`` with ``gaussian_data``
  (``GAUSSIAN_MODEL``). Every method fits the budget at ``arth150`` (the slowest, ``fit`` and
  ``predict``, take about 0.3 s).
* ``DynamicBayesianNetwork``: the frozen 2-TBNs ``dbn_5`` / ``dbn_50`` (``DBN_MODEL``; ``dbn_20`` is the largest that ``DBNInference`` can query) and
  ``dbn_data``, which unrolls them over three time slices.
* ``MarkovChain``: 3 variables x 3 states / 20 variables x 5 states (``CHAIN_SIZE``), with
  transition matrices from a seeded numpy generator.
* ``NaiveBayes``: ``asia`` / ``alarm`` (``NAIVE_BAYES_MODEL``) with ``discrete_data``. The first column
  is the class variable and every other column is a feature.
* ``SEMGraph``, ``SEMAlg`` and ``SEM``: the edges of ``random_gaussian_8`` / ``arth150``
  (``GAUSSIAN_MODEL``) through ``_fixtures.sem_inputs``: nodes with children are latent, the leaves
  are observed, and ``SEM_ERR_CORR`` pairs of observed error terms are correlated. The SEM methods
  take under 10 ms at ``arth150``, so their slow-tier calls are short.

Reduced inputs (exponential or whole-network operations):

* ``get_partition_function`` (Markov network, factor graph, cluster graph) multiplies every factor
  into one joint table, which is astronomically large for ``alarm`` or ``andes``. These run on
  a network built from the longest prefix of the factors (sorted by scope) whose joint state space
  stays under ``PARTITION_CELLS``; for ``ClusterGraph``, the leading cliques in breadth-first
  order from the first clique.
* ``DiscreteMarkovNetwork.get_local_independencies`` closes a set of independence assertions,
  which grows exponentially with the number of variables (7 nodes take about 3 s on both
  fixtures). It runs on the first ``INDEPENDENCE_NODES`` nodes in breadth-first order from the
  first node (sorted), as an induced subgraph.
* ``DiscreteBayesianNetwork.get_state_probability`` sums over every state of the variables left
  out of the query; all but the last ``FREE_VARIABLES`` nodes (sorted) are fixed to their first
  state.
* ``DiscreteBayesianNetwork.predict`` and ``predict_probability`` run one exact inference query
  per distinct row, and the joint of the missing variables is built in memory (predicting all
  of ``alarm`` needs 12.8 GiB). They drop the last ``MISSING_COLUMNS`` columns (sorted) of
  ``discrete_data(model, ROWS[tier])`` and use its first ``PREDICT_ROWS`` / ``PROBABILITY_ROWS``
  rows, so the cost per call stays near the budget.
* ``DiscreteBayesianNetwork.is_imap`` builds the joint of a binary chain, so it uses a chain of
  ``IMAP_VARS`` variables from ``_fixtures.joint_chain``.
* ``DiscreteBayesianNetwork.simulate`` draws ``SIMULATE_SAMPLES`` rows (100000 rows of
  ``random_discrete_200`` take about 10 s).

``DiscreteBayesianNetwork.get_random`` builds a network of ``RANDOM_BN`` size. It and ``simulate`` are timed, never used to produce inputs. Mutating methods work on
a model built fresh by ``prepare``.

Notes on the remaining models:

* ``DynamicBayesianNetwork.simulate`` draws ``DBN_SIMULATE_SAMPLES`` rows over three time slices
  (100000 rows of ``dbn_50`` take about 7 s). ``fit`` uses ``ROWS[tier]`` rows of ``dbn_data``.
  ``initialize_initial_state`` starts from a network that has every slice-0 CPD and the slice-1
  CPDs of the variables with an inter-slice parent, so it derives the rest.
* ``DynamicBayesianNetwork.add_edges_from`` adds every edge to an empty network, and
  ``add_nodes_from`` adds ``NEW_NODES`` new variables.
* ``LinearGaussianBayesianNetwork.get_random`` builds a network of ``LINEAR_GAUSSIAN_RANDOM`` size.
* ``MarkovChain.add_transition_model`` adds the matrix of every variable, ``add_variables_from`` adds
  every variable. ``generate_sample`` consumes ``CHAIN_STEPS`` steps and ``sample`` draws
  ``CHAIN_SAMPLES`` rows. ``prob_from_sample`` reads a frozen numpy sample of ``CHAIN_SAMPLES`` rows,
  in ten windows.
* ``SEMAlg.generate_samples`` draws ``ROWS[tier]`` rows; ``np.random`` is seeded in ``prepare``.
* ``NaiveBayes.fit`` uses ``ROWS[tier]`` rows of ``discrete_data``.

Skipped:
    DiscreteBayesianNetwork.get_factorized_product: trivial (empty stub)
    DiscreteBayesianNetwork.states, DiscreteMarkovNetwork.states, JunctionTree.states,
    ClusterGraph.clique_beliefs: trivial (properties)
    LinearGaussianBayesianNetwork.get_cardinality, .is_imap, .to_markov_model: unsupported (raise)
    MarkovChain.is_stationarity: infeasible (it draws and scans a 10000-row sample per state, ignoring
    its ``sample`` argument: 0.4 s for one variable with two states, 1.8 s for 3 x 3, 32 s for 20 x 5)
    SEM.fit: trivial (empty stub)
"""

import networkx as nx
import numpy as np
from benchmarks._fixtures import (
    dbn,
    dbn_data,
    discrete_bn,
    discrete_data,
    factor_graph,
    factors,
    gaussian_bn,
    gaussian_data,
    joint_chain,
    junction_tree,
    load_spec,
    markov_chain,
    markov_chain_matrices,
    markov_chain_sample,
    markov_network,
    sem_inputs,
)
from benchmarks._tiers import ROWS, Bench
from benchmarks.factors import product_prefix

from pgmpy.factors.discrete import State
from pgmpy.models import (
    SEM,
    ClusterGraph,
    DiscreteBayesianNetwork,
    DiscreteMarkovNetwork,
    DynamicBayesianNetwork,
    FactorGraph,
    LinearGaussianBayesianNetwork,
    MarkovChain,
    NaiveBayes,
    SEMAlg,
    SEMGraph,
)

BN_MODEL = {"fast": "alarm", "slow": "munin"}
# andes (and random_discrete_200) are not connected, and junction trees need a connected graph.
CONNECTED_MODEL = {"fast": "alarm", "slow": "munin"}
FACTOR_GRAPH_TREE_MODEL = {"fast": "alarm", "slow": "pathfinder"}
DATA_MODEL = {"fast": "alarm", "slow": "andes"}
SIMULATE_MODEL = {"fast": "alarm", "slow": "random_discrete_200"}
SIMULATE_SAMPLES = {"fast": 1_000, "slow": 10_000}
MARKOV_MODEL = {"fast": "alarm", "slow": "andes"}
TREE_FIXTURE = {"fast": "jt_alarm", "slow": "jt_andes"}

INDEPENDENCE_NODES = {"fast": 5, "slow": 7}
FREE_VARIABLES = 3
MISSING_COLUMNS = 2
PREDICT_ROWS = {"fast": 40, "slow": 30}
PROBABILITY_ROWS = {"fast": 50, "slow": 50}
IMAP_VARS = {"fast": 4, "slow": 16}
PARTITION_CELLS = {"fast": 20_000, "slow": 1_000_000}
# get_random: (n_nodes, edge_prob, n_states)
RANDOM_BN = {"fast": (20, 0.2, 3), "slow": (500, 0.01, 3)}

GAUSSIAN_MODEL = {"fast": "random_gaussian_8", "slow": "arth150"}
# LinearGaussianBayesianNetwork.get_random: (n_nodes, edge_prob)
LINEAR_GAUSSIAN_RANDOM = {"fast": (20, 0.2), "slow": (1_000, 0.005)}
DBN_MODEL = {"fast": "dbn_5", "slow": "dbn_50"}
DBN_SIMULATE_SAMPLES = {"fast": 500, "slow": 10_000}
NEW_NODES = 10
# (n_vars, n_states) of the Markov chain
CHAIN_SIZE = {"fast": (3, 3), "slow": (20, 5)}
CHAIN_STEPS = {"fast": 1_000, "slow": 1_000}
CHAIN_SAMPLES = {"fast": 1_000, "slow": 10_000}
NAIVE_BAYES_MODEL = {"fast": "asia", "slow": "alarm"}
SEM_ERR_CORR = {"fast": 1, "slow": 10}


def prefix_factors(name, max_cells):
    """The leading factors (sorted by scope) of a discrete spec whose joint state space is small."""
    return product_prefix(factors(name), max_cells)


def prefix_markov_network(name, max_cells):
    fs = prefix_factors(name, max_cells)
    variables = sorted({v for f in fs for v in f.variables})
    edges = {(a, b) for f in fs for a in f.variables for b in f.variables if a < b}
    model = DiscreteMarkovNetwork(sorted(edges))
    model.add_nodes_from(variables)
    model.add_factors(*fs)
    return model


def non_edge(g):
    """First pair of nodes (sorted) that are not adjacent."""
    nodes = sorted(g.nodes)
    for a in nodes:
        for b in nodes:
            if a != b and not g.has_edge(a, b) and not g.has_edge(b, a):
                return a, b


def dag_non_edge(model):
    """First and last node in topological order: an edge between them can never form a cycle."""
    order = list(nx.topological_sort(model))
    assert not model.has_edge(order[0], order[-1])
    return order[0], order[-1]


def first_states(model, nodes):
    """Map each node to its first state."""
    return {n: model.states[n][0] for n in nodes}


class TimeDiscreteBayesianNetwork(Bench):
    def prepare(self, tier):
        self.model = discrete_bn(BN_MODEL[tier])
        self.nodes = sorted(self.model.nodes)
        self.first = self.nodes[0]
        self.edge = dag_non_edge(self.model)
        self.some_nodes = self.nodes[:3]
        self.assignment = first_states(self.model, self.nodes[:-FREE_VARIABLES])

    def time_add_edge(self, tier):
        self.model.add_edge(*self.edge)

    def time_check_model(self, tier):
        self.model.check_model()

    def time_copy(self, tier):
        self.model.copy()

    def time_do(self, tier):
        self.model.do(self.some_nodes)

    def time_get_cardinality(self, tier):
        self.model.get_cardinality()

    def time_get_cpds(self, tier):
        self.model.get_cpds(self.first)

    def time_get_markov_blanket(self, tier):
        self.model.get_markov_blanket(self.first)

    def time_get_state_probability(self, tier):
        self.model.get_state_probability(self.assignment)

    def time_remove_node(self, tier):
        self.model.remove_node(self.first)

    def time_remove_nodes_from(self, tier):
        self.model.remove_nodes_from(self.some_nodes)

    def time_to_junction_tree(self, tier):
        self.model.to_junction_tree()

    def time_to_markov_model(self, tier):
        self.model.to_markov_model()


class TimeDiscreteBayesianNetworkCPDs(Bench):
    def prepare(self, tier):
        self.model = discrete_bn(BN_MODEL[tier])
        self.cpds = self.model.get_cpds()
        self.some_cpds = self.cpds[:3]
        self.bare = DiscreteBayesianNetwork(self.model.edges())
        self.bare.add_nodes_from(self.model.nodes)

    def time_add_cpds(self, tier):
        self.bare.add_cpds(*self.cpds)

    def time_get_random_cpds(self, tier):
        self.bare.get_random_cpds(n_states=3, rng=0)

    def time_remove_cpds(self, tier):
        self.model.remove_cpds(*self.some_cpds)


class TimeDiscreteBayesianNetworkFit(Bench):
    """``fit`` and ``fit_update`` on ``ROWS[tier]`` rows of data sampled from the network."""

    def prepare(self, tier):
        name = DATA_MODEL[tier]
        self.data = discrete_data(name, ROWS[tier])
        self.model = DiscreteBayesianNetwork(discrete_bn(name).edges())
        self.model.add_nodes_from(self.data.columns)
        self.fitted = discrete_bn(name)

    def time_fit(self, tier):
        self.model.fit(self.data)

    def time_fit_update(self, tier):
        self.fitted.fit_update(self.data)


class TimeDiscreteBayesianNetworkPredict(Bench):
    def prepare(self, tier):
        name = DATA_MODEL[tier]
        self.model = discrete_bn(name)
        data = discrete_data(name, ROWS[tier])
        columns = sorted(data.columns)
        known = data[columns[:-MISSING_COLUMNS]]
        self.predict_data = known.head(PREDICT_ROWS[tier])
        self.probability_data = known.head(PROBABILITY_ROWS[tier])

    def time_predict(self, tier):
        self.model.predict(self.predict_data, n_jobs=1, rng=0)

    def time_predict_probability(self, tier):
        self.model.predict_probability(self.probability_data)


class TimeDiscreteBayesianNetworkSimulate(Bench):
    def prepare(self, tier):
        self.model = discrete_bn(SIMULATE_MODEL[tier])
        self.n_samples = SIMULATE_SAMPLES[tier]

    def time_simulate(self, tier):
        self.model.simulate(self.n_samples, rng=0, show_progress=False)


class TimeDiscreteBayesianNetworkRandom(Bench):
    def prepare(self, tier):
        self.n_nodes, self.edge_prob, self.n_states = RANDOM_BN[tier]

    def time_get_random(self, tier):
        DiscreteBayesianNetwork.get_random(
            self.n_nodes, edge_prob=self.edge_prob, n_states=self.n_states, rng=0
        )


class TimeDiscreteBayesianNetworkIsImap(Bench):
    def prepare(self, tier):
        self.joint, self.model = joint_chain(IMAP_VARS[tier])

    def time_is_imap(self, tier):
        self.model.is_imap(self.joint)


class TimeDiscreteMarkovNetwork(Bench):
    def prepare(self, tier):
        name = MARKOV_MODEL[tier]
        self.model = markov_network(name)
        self.nodes = sorted(self.model.nodes)
        self.first = self.nodes[0]
        self.edge = non_edge(self.model)
        self.some_factors = self.model.get_factors()[:3]
        self.factors = factors(name)
        self.bare = DiscreteMarkovNetwork(self.model.edges())
        self.bare.add_nodes_from(self.nodes)

    def time_add_edge(self, tier):
        self.model.add_edge(*self.edge)

    def time_add_factors(self, tier):
        self.bare.add_factors(*self.factors)

    def time_check_model(self, tier):
        self.model.check_model()

    def time_copy(self, tier):
        self.model.copy()

    def time_get_cardinality(self, tier):
        self.model.get_cardinality()

    def time_get_factors(self, tier):
        self.model.get_factors(self.first)

    def time_markov_blanket(self, tier):
        self.model.markov_blanket(self.first)

    def time_remove_factors(self, tier):
        self.model.remove_factors(*self.some_factors)

    def time_to_factor_graph(self, tier):
        self.model.to_factor_graph()

    def time_triangulate(self, tier):
        self.model.triangulate()


class TimeDiscreteMarkovNetworkConversions(Bench):
    def prepare(self, tier):
        self.model = markov_network(CONNECTED_MODEL[tier])

    def time_to_bayesian_model(self, tier):
        self.model.to_bayesian_model()

    def time_to_junction_tree(self, tier):
        self.model.to_junction_tree()


class TimeDiscreteMarkovNetworkIndependencies(Bench):
    def prepare(self, tier):
        g = markov_network(MARKOV_MODEL[tier])
        start = sorted(g.nodes)[0]
        order = [start] + [v for _, v in nx.bfs_edges(g, start, sort_neighbors=sorted)]
        nodes = order[: INDEPENDENCE_NODES[tier]]
        self.model = DiscreteMarkovNetwork(g.subgraph(nodes).edges())
        self.model.add_nodes_from(nodes)

    def time_get_local_independencies(self, tier):
        self.model.get_local_independencies()


class TimeDiscreteMarkovNetworkPartition(Bench):
    def prepare(self, tier):
        self.model = prefix_markov_network(MARKOV_MODEL[tier], PARTITION_CELLS[tier])

    def time_get_partition_function(self, tier):
        self.model.get_partition_function()


class TimeFactorGraph(Bench):
    def prepare(self, tier):
        name = MARKOV_MODEL[tier]
        self.model = factor_graph(name)
        self.variable = sorted(self.model.get_variable_nodes())[0]
        self.some_factors = self.model.get_factors()[:3]
        self.factor_node = self.some_factors[0]
        self.factors = factors(name)
        self.bare = FactorGraph()
        self.bare.add_nodes_from(self.model.get_variable_nodes())
        # Edge between the first variable and the first factor that does not contain it.
        self.edge = (
            self.variable,
            next(f for f in self.model.get_factors() if self.variable not in f.variables),
        )

    def time_add_edge(self, tier):
        self.model.add_edge(*self.edge)

    def time_add_factors(self, tier):
        self.bare.add_factors(*self.factors)

    def time_check_model(self, tier):
        self.model.check_model()

    def time_copy(self, tier):
        self.model.copy()

    def time_get_cardinality(self, tier):
        self.model.get_cardinality()

    def time_get_factor_nodes(self, tier):
        self.model.get_factor_nodes()

    def time_get_factors(self, tier):
        self.model.get_factors(self.factor_node)

    def time_get_point_mass_message(self, tier):
        self.model.get_point_mass_message(self.variable, 0)

    def time_get_uniform_message(self, tier):
        self.model.get_uniform_message(self.variable)

    def time_get_variable_nodes(self, tier):
        self.model.get_variable_nodes()

    def time_remove_factors(self, tier):
        self.model.remove_factors(*self.some_factors)

    def time_to_markov_model(self, tier):
        self.model.to_markov_model()


class TimeFactorGraphJunctionTree(Bench):
    def prepare(self, tier):
        self.model = factor_graph(FACTOR_GRAPH_TREE_MODEL[tier])

    def time_to_junction_tree(self, tier):
        self.model.to_junction_tree()


class TimeFactorGraphPartition(Bench):
    def prepare(self, tier):
        fs = prefix_factors(MARKOV_MODEL[tier], PARTITION_CELLS[tier])
        self.model = FactorGraph()
        self.model.add_nodes_from(sorted({v for f in fs for v in f.variables}))
        self.model.add_nodes_from(fs)
        self.model.add_factors(*fs)
        self.model.add_edges_from([(v, f) for f in fs for v in f.variables])

    def time_get_partition_function(self, tier):
        self.model.get_partition_function()


def cluster_graph(name, max_cells=None):
    """ClusterGraph with the cliques, edges and factors of a junction tree fixture.

    With ``max_cells``, only the leading cliques in breadth-first order from the first clique,
    stopping before the joint state space of their variables would exceed ``max_cells``.
    """
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


class TimeJunctionTree(Bench):
    def prepare(self, tier):
        self.model = junction_tree(TREE_FIXTURE[tier])
        # A second tree missing its last edge, which time_add_edge puts back.
        self.edge = list(self.model.edges)[-1]
        self.cut = junction_tree(TREE_FIXTURE[tier])
        self.cut.remove_edge(*self.edge)

    def time_add_edge(self, tier):
        self.cut.add_edge(*self.edge)

    def time_check_model(self, tier):
        self.model.check_model()

    def time_copy(self, tier):
        self.model.copy()


class TimeClusterGraph(Bench):
    def prepare(self, tier):
        name = TREE_FIXTURE[tier]
        self.model = cluster_graph(name)
        self.cliques = list(self.model.nodes)
        self.first = self.cliques[0]
        self.some_factors = self.model.get_factors()[:3]
        self.factors = self.model.get_factors()
        self.bare = ClusterGraph()
        self.bare.add_nodes_from(self.cliques)
        self.new_nodes = [("N0", "N1"), ("N1", "N2")]
        # First pair of cliques that share a variable but are not yet joined.
        self.edge = next(
            (a, b)
            for i, a in enumerate(self.cliques)
            for b in self.cliques[i + 1 :]
            if not self.model.has_edge(a, b) and set(a) & set(b)
        )

    def time_add_edge(self, tier):
        self.model.add_edge(*self.edge)

    def time_add_factors(self, tier):
        self.bare.add_factors(*self.factors)

    def time_add_node(self, tier):
        self.model.add_node(self.new_nodes[0])

    def time_add_nodes_from(self, tier):
        self.model.add_nodes_from(self.new_nodes)

    def time_check_model(self, tier):
        self.model.check_model()

    def time_copy(self, tier):
        self.model.copy()

    def time_get_cardinality(self, tier):
        self.model.get_cardinality()

    def time_get_factors(self, tier):
        self.model.get_factors(self.first)

    def time_remove_factors(self, tier):
        self.model.remove_factors(*self.some_factors)


class TimeClusterGraphPartition(Bench):
    def prepare(self, tier):
        self.model = cluster_graph(TREE_FIXTURE[tier], PARTITION_CELLS[tier])

    def time_get_partition_function(self, tier):
        self.model.get_partition_function()


def bare_gaussian_bn(model):
    """A LinearGaussianBayesianNetwork with the structure of ``model`` and no CPDs."""
    bare = LinearGaussianBayesianNetwork(model.edges())
    bare.add_nodes_from(model.nodes)
    return bare


class TimeLinearGaussianBayesianNetwork(Bench):
    def prepare(self, tier):
        self.model = gaussian_bn(GAUSSIAN_MODEL[tier])
        self.first = sorted(self.model.nodes)[0]
        self.cpds = self.model.get_cpds()
        self.some_cpds = self.cpds[:3]
        self.bare = bare_gaussian_bn(self.model)

    def time_add_cpds(self, tier):
        self.bare.add_cpds(*self.cpds)

    def time_check_model(self, tier):
        self.model.check_model()

    def time_copy(self, tier):
        self.model.copy()

    def time_get_cpds(self, tier):
        self.model.get_cpds(self.first)

    def time_get_random(self, tier):
        n_nodes, edge_prob = LINEAR_GAUSSIAN_RANDOM[tier]
        LinearGaussianBayesianNetwork.get_random(n_nodes=n_nodes, edge_prob=edge_prob, rng=0)

    def time_get_random_cpds(self, tier):
        self.bare.get_random_cpds(rng=0)

    def time_remove_cpds(self, tier):
        self.model.remove_cpds(*self.some_cpds)

    def time_simulate(self, tier):
        self.model.simulate(n_samples=ROWS[tier], rng=0)

    def time_to_joint_gaussian(self, tier):
        self.model.to_joint_gaussian()


class TimeLinearGaussianBayesianNetworkFit(Bench):
    """``fit`` and ``log_likelihood`` on ``ROWS[tier]`` rows of ``gaussian_data``."""

    def prepare(self, tier):
        name = GAUSSIAN_MODEL[tier]
        self.model = gaussian_bn(name)
        self.bare = bare_gaussian_bn(self.model)
        self.data = gaussian_data(name, ROWS[tier])

    def time_fit(self, tier):
        self.bare.fit(self.data)

    def time_log_likelihood(self, tier):
        self.model.log_likelihood(self.data)


class TimeLinearGaussianBayesianNetworkPredict(Bench):
    """``predict`` and ``predict_probability`` with the last two columns (sorted) missing."""

    def prepare(self, tier):
        name = GAUSSIAN_MODEL[tier]
        self.model = gaussian_bn(name)
        data = gaussian_data(name, ROWS[tier])
        self.data = data.drop(columns=sorted(data.columns)[-2:])

    def time_predict(self, tier):
        self.model.predict(self.data)

    def time_predict_probability(self, tier):
        self.model.predict_probability(self.data)


def bare_dbn(model):
    """A DynamicBayesianNetwork with the nodes and edges of ``model`` and no CPDs."""
    bare = DynamicBayesianNetwork()
    bare.add_nodes_from(sorted({n.node for n in model.nodes}))
    bare.add_edges_from(dbn_edges(model))
    return bare


def dbn_edges(model):
    return [(u.to_tuple(), v.to_tuple()) for u, v in model.edges()]


def missing_inter_edge(model):
    """First inter-slice edge ``(Vi, 0) -> (Vj, 1)`` (variables sorted) that ``model`` lacks."""
    variables = sorted({n.node for n in model.nodes})
    present = set(dbn_edges(model))
    for a in variables:
        for b in variables:
            if ((a, 0), (b, 1)) not in present:
                return (a, 0), (b, 1)


class TimeDynamicBayesianNetwork(Bench):
    def prepare(self, tier):
        self.model = dbn(DBN_MODEL[tier])
        self.first = ("V0", 0)
        self.last = ("V0", 1)
        self.observed = [("V1", 0), ("V2", 0)]
        self.some_cpds = self.model.cpds[:3]

    def time_active_trail_nodes(self, tier):
        self.model.active_trail_nodes(self.first, observed=self.observed)

    def time_check_model(self, tier):
        self.model.check_model()

    def time_copy(self, tier):
        self.model.copy()

    def time_get_constant_bn(self, tier):
        self.model.get_constant_bn()

    def time_get_cpds(self, tier):
        self.model.get_cpds(self.first)

    def time_get_cpds_all(self, tier):
        self.model.get_cpds()

    def time_get_inter_edges(self, tier):
        self.model.get_inter_edges()

    def time_get_interface_nodes(self, tier):
        self.model.get_interface_nodes()

    def time_get_intra_edges(self, tier):
        self.model.get_intra_edges()

    def time_get_markov_blanket(self, tier):
        self.model.get_markov_blanket(self.last)

    def time_get_slice_nodes(self, tier):
        self.model.get_slice_nodes()

    def time_moralize(self, tier):
        self.model.moralize()

    def time_remove_cpds(self, tier):
        self.model.remove_cpds(*self.some_cpds)

    def time_simulate(self, tier):
        self.model.simulate(n_samples=DBN_SIMULATE_SAMPLES[tier], n_time_slices=3, rng=0, show_progress=False)


class TimeDynamicBayesianNetworkBuild(Bench):
    """Adding nodes, edges and CPDs to a network built by ``prepare``."""

    def prepare(self, tier):
        model = dbn(DBN_MODEL[tier])
        self.cpds = list(model.cpds)
        self.edges = dbn_edges(model)
        self.new_edge = missing_inter_edge(model)
        self.new_nodes = [f"Z{i}" for i in range(NEW_NODES)]
        self.bare = bare_dbn(model)
        self.empty = DynamicBayesianNetwork()

    def time_add_cpds(self, tier):
        self.bare.add_cpds(*self.cpds)

    def time_add_edge(self, tier):
        self.bare.add_edge(*self.new_edge)

    def time_add_edges_from(self, tier):
        self.empty.add_edges_from(self.edges)

    def time_add_node(self, tier):
        self.bare.add_node(self.new_nodes[0])

    def time_add_nodes_from(self, tier):
        self.bare.add_nodes_from(self.new_nodes)


class TimeDynamicBayesianNetworkInitialState(Bench):
    def prepare(self, tier):
        model = dbn(DBN_MODEL[tier])
        # Every slice-0 CPD, and the slice-1 CPDs of variables with an inter-slice parent.
        given = [c for c in model.cpds if c.variable[1] == 0 or any(v[1] == 0 for v in c.variables[1:])]
        self.model = bare_dbn(model)
        self.model.add_cpds(*given)

    def time_initialize_initial_state(self, tier):
        self.model.initialize_initial_state()


class TimeDynamicBayesianNetworkFit(Bench):
    """``fit`` on ``ROWS[tier]`` rows of ``dbn_data`` (three time slices)."""

    def prepare(self, tier):
        name = DBN_MODEL[tier]
        self.model = bare_dbn(dbn(name))
        self.data = dbn_data(name, ROWS[tier])

    def time_fit(self, tier):
        self.model.fit(self.data)


class TimeMarkovChain(Bench):
    def prepare(self, tier):
        n_vars, n_states = CHAIN_SIZE[tier]
        self.chain = markov_chain(n_vars, n_states)
        self.matrices = markov_chain_matrices(n_vars, n_states)
        self.variables = list(self.matrices)
        self.bare = MarkovChain(self.variables, [n_states] * n_vars)
        self.empty = MarkovChain()
        self.cards = [n_states] * n_vars
        self.start = [State(v, 1) for v in self.variables]

    def time_add_transition_model(self, tier):
        for variable, matrix in self.matrices.items():
            self.bare.add_transition_model(variable, matrix)

    def time_add_variable(self, tier):
        self.empty.add_variable("Z", 3)

    def time_add_variables_from(self, tier):
        self.empty.add_variables_from(self.variables, self.cards)

    def time_copy(self, tier):
        self.chain.copy()

    def time_generate_sample(self, tier):
        list(self.chain.generate_sample(size=CHAIN_STEPS[tier], rng=0))

    def time_random_state(self, tier):
        self.chain.random_state()

    def time_sample(self, tier):
        self.chain.sample(size=CHAIN_SAMPLES[tier], rng=0)

    def time_set_start_state(self, tier):
        self.chain.set_start_state(self.start)


class TimeMarkovChainProbFromSample(Bench):
    """``prob_from_sample`` of the first two variables being in state 0, over a frozen sample."""

    def prepare(self, tier):
        n_vars, n_states = CHAIN_SIZE[tier]
        self.chain = markov_chain(n_vars, n_states)
        self.sample = markov_chain_sample(n_vars, n_states, CHAIN_SAMPLES[tier])
        self.state = [State("X0", 0), State("X1", 0)]
        self.window_size = CHAIN_SAMPLES[tier] // 10

    def time_prob_from_sample(self, tier):
        self.chain.prob_from_sample(self.state, sample=self.sample, window_size=self.window_size)


def class_and_features(name):
    """The first column of the data (spec order) is the class; all the others are features."""
    columns = [n["name"] for n in load_spec(name)["nodes"]]
    return columns[0], columns[1:]


class TimeNaiveBayes(Bench):
    def prepare(self, tier):
        self.cls, self.features = class_and_features(NAIVE_BAYES_MODEL[tier])
        self.model = NaiveBayes(self.features, self.cls)
        self.almost = NaiveBayes(self.features[:-1], self.cls)
        self.empty = NaiveBayes()
        self.edges = [(self.cls, f) for f in self.features]

    def time_active_trail_nodes(self, tier):
        self.model.active_trail_nodes(self.features[0], observed=self.features[1:3])

    def time_add_edge(self, tier):
        self.almost.add_edge(self.cls, self.features[-1])

    def time_add_edges_from(self, tier):
        self.empty.add_edges_from(self.edges)

    def time_local_independencies(self, tier):
        self.model.local_independencies(self.features)


class TimeNaiveBayesFit(Bench):
    """``fit`` on ``ROWS[tier]`` rows of ``discrete_data``."""

    def prepare(self, tier):
        name = NAIVE_BAYES_MODEL[tier]
        cls, features = class_and_features(name)
        self.model = NaiveBayes(features, cls)
        self.data = discrete_data(name, ROWS[tier])

    def time_fit(self, tier):
        self.model.fit(self.data)


class TimeSEMGraph(Bench):
    def prepare(self, tier):
        sem = sem_inputs(GAUSSIAN_MODEL[tier], SEM_ERR_CORR[tier])
        self.model = SEMGraph(ebunch=sem["ebunch"], latents=sem["latents"], err_corr=sem["err_corr"])
        observed = sorted(self.model.observed)
        self.start = observed[0]
        self.observed = observed[1:3]

    def time_active_trail_nodes(self, tier):
        self.model.active_trail_nodes(self.start, observed=self.observed)

    def time_get_scaling_indicators(self, tier):
        self.model.get_scaling_indicators()

    def time_moralize(self, tier):
        self.model.moralize()

    def time_to_lisrel(self, tier):
        self.model.to_lisrel()

    def time_to_standard_lisrel(self, tier):
        self.model.to_standard_lisrel()


class TimeSEMAlg(Bench):
    def prepare(self, tier):
        sem = sem_inputs(GAUSSIAN_MODEL[tier], SEM_ERR_CORR[tier])
        self.B_values = sem["B_values"]
        self.zeta_values = sem["zeta_values"]
        self.model = SEMAlg(eta=sem["variables"], B=sem["B"], zeta=sem["zeta"], wedge_y=sem["wedge_y"])
        self.model.set_params(self.B_values, self.zeta_values)
        np.random.seed(0)  # generate_samples draws from the global generator

    def time_generate_samples(self, tier):
        self.model.generate_samples(n_samples=ROWS[tier])

    def time_set_params(self, tier):
        self.model.set_params(self.B_values, self.zeta_values)

    def time_to_SEMGraph(self, tier):
        self.model.to_SEMGraph()


class TimeSEM(Bench):
    def prepare(self, tier):
        self.sem = sem_inputs(GAUSSIAN_MODEL[tier], SEM_ERR_CORR[tier])

    def time_from_graph(self, tier):
        SEM.from_graph(ebunch=self.sem["ebunch"], latents=self.sem["latents"], err_corr=self.sem["err_corr"])

    def time_from_lavaan(self, tier):
        SEM.from_lavaan(string=self.sem["lavaan"])

    def time_from_lisrel(self, tier):
        SEM.from_lisrel(self.sem["var_names"], self.sem["params"], self.sem["fixed_masks"])

    def time_from_RAM(self, tier):
        SEM.from_RAM(self.sem["variables"], self.sem["B"], self.sem["zeta"], wedge_y=self.sem["wedge_y"])
