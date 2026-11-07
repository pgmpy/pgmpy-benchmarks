"""Benchmarks for ``pgmpy.models``. Methods a Bayesian network inherits from ``DAG`` are timed in ``base.py``.

Whole-network operations run on reduced inputs: ``get_partition_function`` on a ``PRODUCT_CELLS`` prefix of the factors,
``get_local_independencies`` on ``INDEPENDENCE_NODES`` nodes, ``get_state_probability`` with all but ``FREE_VARIABLES`` nodes
fixed, ``predict``/``predict_probability`` on ``PREDICT_ROWS``/``PROBABILITY_ROWS`` rows with ``MISSING_COLUMNS`` columns dropped,
``is_imap`` on an ``IMAP_VARS`` chain. Junction-tree conversions need a connected graph. ``NaiveBayes``: first column is the class.
Mutating methods work on a model built fresh by ``setup``. ``SEMEstimator``, ``FunctionalBayesianNetwork`` and ``FunctionalCPD``
are out of scope.

Skipped:
    DiscreteBayesianNetwork.get_factorized_product: trivial (empty stub)
    DiscreteBayesianNetwork.states, DiscreteMarkovNetwork.states, JunctionTree.states, ClusterGraph.clique_beliefs: trivial
    LinearGaussianBayesianNetwork.get_cardinality, .is_imap, .to_markov_model: unsupported (raise)
    MarkovChain.is_stationarity: infeasible (draws a 10000-row sample per state: 32 s for 20 x 5)
    SEM.fit: trivial (empty stub)
"""

import networkx as nx
import numpy as np
from benchmarks._fixtures import (
    dbn,
    bare,
    dbn_data,
    discrete_bn,
    discrete_data,
    factor_graph,
    factors,
    first_states,
    gaussian_bn,
    gaussian_data,
    joint_chain,
    junction_tree,
    markov_chain,
    markov_chain_matrices,
    markov_chain_sample,
    markov_network,
    sem_inputs,
    topological,
)
from benchmarks._bench import DISCRETE, GAUSSIAN, MODEL, ROWS, Bench
from benchmarks.factors import PRODUCT_CELLS, product_prefix

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

TREE_FIXTURE = "jt_alarm"

INDEPENDENCE_NODES = 5
FREE_VARIABLES = 3
MISSING_COLUMNS = 2
PREDICT_ROWS = 40
PROBABILITY_ROWS = 50
IMAP_VARS = 4
RANDOM_BN = (20, 0.2, 3)

LINEAR_GAUSSIAN_RANDOM = (20, 0.2)
DBN_MODEL = "dbn_5"
DBN_SIMULATE_SAMPLES = 500
NEW_NODES = 10
CHAIN_SIZE = (3, 3)
SEM_ERR_CORR = 1


def prefix_factors(name, max_cells):
    """The leading factors (sorted by scope) of a discrete model whose joint state space is small."""
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


class TimeDiscreteBayesianNetwork(Bench):
    def setup(self):
        self.model = discrete_bn(MODEL)
        self.nodes = sorted(self.model.nodes)
        self.first = self.nodes[0]
        self.edge = dag_non_edge(self.model)
        self.some_nodes = self.nodes[:3]
        self.assignment = first_states(self.model, self.nodes[:-FREE_VARIABLES])

    def time_add_edge(self):
        self.model.add_edge(*self.edge)

    def time_check_model(self):
        self.model.check_model()

    def time_copy(self):
        self.model.copy()

    def time_do(self):
        self.model.do(self.some_nodes)

    def time_get_cardinality(self):
        self.model.get_cardinality()

    def time_get_cpds(self):
        self.model.get_cpds(self.first)

    def time_get_markov_blanket(self):
        self.model.get_markov_blanket(self.first)

    def time_get_state_probability(self):
        self.model.get_state_probability(self.assignment)

    def time_remove_node(self):
        self.model.remove_node(self.first)

    def time_remove_nodes_from(self):
        self.model.remove_nodes_from(self.some_nodes)

    def time_to_junction_tree(self):
        self.model.to_junction_tree()

    def time_to_markov_model(self):
        self.model.to_markov_model()


class TimeDiscreteBayesianNetworkCPDs(Bench):
    def setup(self):
        self.model = discrete_bn(MODEL)
        self.cpds = self.model.get_cpds()
        self.some_cpds = self.cpds[:3]
        self.bare = bare(self.model)

    def time_add_cpds(self):
        self.bare.add_cpds(*self.cpds)

    def time_get_random_cpds(self):
        self.bare.get_random_cpds(n_states=3, rng=0)

    def time_remove_cpds(self):
        self.model.remove_cpds(*self.some_cpds)


class TimeDiscreteBayesianNetworkFit(Bench):
    def setup(self):
        name = MODEL
        self.data = discrete_data(name, ROWS)
        self.fitted = discrete_bn(name)
        self.model = bare(self.fitted)

    def time_fit(self):
        self.model.fit(self.data)

    def time_fit_update(self):
        self.fitted.fit_update(self.data)


class TimeDiscreteBayesianNetworkPredict(Bench):
    def setup(self):
        name = MODEL
        self.model = discrete_bn(name)
        data = discrete_data(name, ROWS)
        columns = sorted(data.columns)
        known = data[columns[:-MISSING_COLUMNS]]
        self.predict_data = known.head(PREDICT_ROWS)
        self.probability_data = known.head(PROBABILITY_ROWS)

    def time_predict(self):
        self.model.predict(self.predict_data, n_jobs=1, rng=0)

    def time_predict_probability(self):
        self.model.predict_probability(self.probability_data)


class TimeDiscreteBayesianNetworkSimulate(Bench):
    def setup(self):
        self.model = discrete_bn(MODEL)
        self.n_samples = ROWS

    def time_simulate(self):
        self.model.simulate(self.n_samples, rng=0, show_progress=False)


class TimeDiscreteBayesianNetworkRandom(Bench):
    def setup(self):
        self.n_nodes, self.edge_prob, self.n_states = RANDOM_BN

    def time_get_random(self):
        DiscreteBayesianNetwork.get_random(
            self.n_nodes, edge_prob=self.edge_prob, n_states=self.n_states, rng=0
        )


class TimeDiscreteBayesianNetworkIsImap(Bench):
    def setup(self):
        self.joint, self.model = joint_chain(IMAP_VARS)

    def time_is_imap(self):
        self.model.is_imap(self.joint)


class TimeDiscreteMarkovNetwork(Bench):
    def setup(self):
        name = MODEL
        self.model = markov_network(name)
        self.nodes = sorted(self.model.nodes)
        self.first = self.nodes[0]
        self.edge = non_edge(self.model)
        self.some_factors = self.model.get_factors()[:3]
        self.factors = factors(name)
        self.bare = bare(self.model)

    def time_add_edge(self):
        self.model.add_edge(*self.edge)

    def time_add_factors(self):
        self.bare.add_factors(*self.factors)

    def time_check_model(self):
        self.model.check_model()

    def time_copy(self):
        self.model.copy()

    def time_get_cardinality(self):
        self.model.get_cardinality()

    def time_get_factors(self):
        self.model.get_factors(self.first)

    def time_markov_blanket(self):
        self.model.markov_blanket(self.first)

    def time_remove_factors(self):
        self.model.remove_factors(*self.some_factors)

    def time_to_factor_graph(self):
        self.model.to_factor_graph()

    def time_triangulate(self):
        self.model.triangulate()


class TimeDiscreteMarkovNetworkConversions(Bench):
    def setup(self):
        self.model = markov_network(MODEL)

    def time_to_bayesian_model(self):
        self.model.to_bayesian_model()

    def time_to_junction_tree(self):
        self.model.to_junction_tree()


class TimeDiscreteMarkovNetworkIndependencies(Bench):
    def setup(self):
        g = markov_network(MODEL)
        start = sorted(g.nodes)[0]
        order = [start] + [v for _, v in nx.bfs_edges(g, start, sort_neighbors=sorted)]
        nodes = order[:INDEPENDENCE_NODES]
        self.model = DiscreteMarkovNetwork(g.subgraph(nodes).edges())
        self.model.add_nodes_from(nodes)

    def time_get_local_independencies(self):
        self.model.get_local_independencies()


class TimeDiscreteMarkovNetworkPartition(Bench):
    def setup(self):
        self.model = prefix_markov_network(MODEL, PRODUCT_CELLS)

    def time_get_partition_function(self):
        self.model.get_partition_function()


class TimeFactorGraph(Bench):
    def setup(self):
        name = MODEL
        self.model = factor_graph(name)
        self.variable = sorted(self.model.get_variable_nodes())[0]
        self.some_factors = self.model.get_factors()[:3]
        self.factor_node = self.some_factors[0]
        self.factors = factors(name)
        self.bare = FactorGraph()
        self.bare.add_nodes_from(self.model.get_variable_nodes())
        self.edge = (
            self.variable,
            next(f for f in self.model.get_factors() if self.variable not in f.variables),
        )

    def time_add_edge(self):
        self.model.add_edge(*self.edge)

    def time_add_factors(self):
        self.bare.add_factors(*self.factors)

    def time_check_model(self):
        self.model.check_model()

    def time_copy(self):
        self.model.copy()

    def time_get_cardinality(self):
        self.model.get_cardinality()

    def time_get_factor_nodes(self):
        self.model.get_factor_nodes()

    def time_get_factors(self):
        self.model.get_factors(self.factor_node)

    def time_get_point_mass_message(self):
        self.model.get_point_mass_message(self.variable, 0)

    def time_get_uniform_message(self):
        self.model.get_uniform_message(self.variable)

    def time_get_variable_nodes(self):
        self.model.get_variable_nodes()

    def time_remove_factors(self):
        self.model.remove_factors(*self.some_factors)

    def time_to_markov_model(self):
        self.model.to_markov_model()


class TimeFactorGraphJunctionTree(Bench):
    def setup(self):
        self.model = factor_graph(MODEL)

    def time_to_junction_tree(self):
        self.model.to_junction_tree()


class TimeFactorGraphPartition(Bench):
    def setup(self):
        fs = prefix_factors(MODEL, PRODUCT_CELLS)
        self.model = FactorGraph()
        self.model.add_nodes_from(sorted({v for f in fs for v in f.variables}))
        self.model.add_nodes_from(fs)
        self.model.add_factors(*fs)
        self.model.add_edges_from([(v, f) for f in fs for v in f.variables])

    def time_get_partition_function(self):
        self.model.get_partition_function()


def cluster_graph(name, max_cells=None):
    """ClusterGraph with the cliques, edges and factors of a junction tree fixture."""
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
    def setup(self):
        self.model = junction_tree(TREE_FIXTURE)
        self.edge = list(self.model.edges)[-1]
        self.cut = junction_tree(TREE_FIXTURE)
        self.cut.remove_edge(*self.edge)

    def time_add_edge(self):
        self.cut.add_edge(*self.edge)

    def time_check_model(self):
        self.model.check_model()

    def time_copy(self):
        self.model.copy()


class TimeClusterGraph(Bench):
    def setup(self):
        name = TREE_FIXTURE
        self.model = cluster_graph(name)
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
        self.model = cluster_graph(TREE_FIXTURE, PRODUCT_CELLS)

    def time_get_partition_function(self):
        self.model.get_partition_function()


class TimeLinearGaussianBayesianNetwork(Bench):
    def setup(self):
        self.model = gaussian_bn(GAUSSIAN)
        self.first = sorted(self.model.nodes)[0]
        self.cpds = self.model.get_cpds()
        self.some_cpds = self.cpds[:3]
        self.bare = bare(self.model)

    def time_add_cpds(self):
        self.bare.add_cpds(*self.cpds)

    def time_check_model(self):
        self.model.check_model()

    def time_copy(self):
        self.model.copy()

    def time_get_cpds(self):
        self.model.get_cpds(self.first)

    def time_get_random(self):
        n_nodes, edge_prob = LINEAR_GAUSSIAN_RANDOM
        LinearGaussianBayesianNetwork.get_random(n_nodes=n_nodes, edge_prob=edge_prob, rng=0)

    def time_get_random_cpds(self):
        self.bare.get_random_cpds(rng=0)

    def time_remove_cpds(self):
        self.model.remove_cpds(*self.some_cpds)

    def time_simulate(self):
        self.model.simulate(n_samples=ROWS, rng=0)

    def time_to_joint_gaussian(self):
        self.model.to_joint_gaussian()


class TimeLinearGaussianBayesianNetworkFit(Bench):
    def setup(self):
        name = GAUSSIAN
        self.model = gaussian_bn(name)
        self.bare = bare(self.model)
        self.data = gaussian_data(name, ROWS)

    def time_fit(self):
        self.bare.fit(self.data)

    def time_log_likelihood(self):
        self.model.log_likelihood(self.data)


class TimeLinearGaussianBayesianNetworkPredict(Bench):
    def setup(self):
        name = GAUSSIAN
        self.model = gaussian_bn(name)
        data = gaussian_data(name, ROWS)
        self.data = data.drop(columns=sorted(data.columns)[-2:])

    def time_predict(self):
        self.model.predict(self.data)

    def time_predict_probability(self):
        self.model.predict_probability(self.data)


def bare_dbn(model):
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
    def setup(self):
        self.model = dbn(DBN_MODEL)
        self.first = ("V0", 0)
        self.last = ("V0", 1)
        self.observed = [("V1", 0), ("V2", 0)]
        self.some_cpds = self.model.cpds[:3]

    def time_active_trail_nodes(self):
        self.model.active_trail_nodes(self.first, observed=self.observed)

    def time_check_model(self):
        self.model.check_model()

    def time_copy(self):
        self.model.copy()

    def time_get_constant_bn(self):
        self.model.get_constant_bn()

    def time_get_cpds(self):
        self.model.get_cpds(self.first)

    def time_get_cpds_all(self):
        self.model.get_cpds()

    def time_get_inter_edges(self):
        self.model.get_inter_edges()

    def time_get_interface_nodes(self):
        self.model.get_interface_nodes()

    def time_get_intra_edges(self):
        self.model.get_intra_edges()

    def time_get_markov_blanket(self):
        self.model.get_markov_blanket(self.last)

    def time_get_slice_nodes(self):
        self.model.get_slice_nodes()

    def time_moralize(self):
        self.model.moralize()

    def time_remove_cpds(self):
        self.model.remove_cpds(*self.some_cpds)

    def time_simulate(self):
        self.model.simulate(n_samples=DBN_SIMULATE_SAMPLES, n_time_slices=3, rng=0, show_progress=False)


class TimeDynamicBayesianNetworkBuild(Bench):
    def setup(self):
        model = dbn(DBN_MODEL)
        self.cpds = list(model.cpds)
        self.edges = dbn_edges(model)
        self.new_edge = missing_inter_edge(model)
        self.new_nodes = [f"Z{i}" for i in range(NEW_NODES)]
        self.bare = bare_dbn(model)
        self.empty = DynamicBayesianNetwork()

    def time_add_cpds(self):
        self.bare.add_cpds(*self.cpds)

    def time_add_edge(self):
        self.bare.add_edge(*self.new_edge)

    def time_add_edges_from(self):
        self.empty.add_edges_from(self.edges)

    def time_add_node(self):
        self.bare.add_node(self.new_nodes[0])

    def time_add_nodes_from(self):
        self.bare.add_nodes_from(self.new_nodes)


class TimeDynamicBayesianNetworkInitialState(Bench):
    def setup(self):
        model = dbn(DBN_MODEL)
        given = [c for c in model.cpds if c.variable[1] == 0 or any(v[1] == 0 for v in c.variables[1:])]
        self.model = bare_dbn(model)
        self.model.add_cpds(*given)

    def time_initialize_initial_state(self):
        self.model.initialize_initial_state()


class TimeDynamicBayesianNetworkFit(Bench):
    def setup(self):
        name = DBN_MODEL
        self.model = bare_dbn(dbn(name))
        self.data = dbn_data(name, ROWS)

    def time_fit(self):
        self.model.fit(self.data)


class TimeMarkovChain(Bench):
    def setup(self):
        n_vars, n_states = CHAIN_SIZE
        self.chain = markov_chain(n_vars, n_states)
        self.matrices = markov_chain_matrices(n_vars, n_states)
        self.variables = list(self.matrices)
        self.bare = MarkovChain(self.variables, [n_states] * n_vars)
        self.empty = MarkovChain()
        self.cards = [n_states] * n_vars
        self.start = [State(v, 1) for v in self.variables]

    def time_add_transition_model(self):
        for variable, matrix in self.matrices.items():
            self.bare.add_transition_model(variable, matrix)

    def time_add_variable(self):
        self.empty.add_variable("Z", 3)

    def time_add_variables_from(self):
        self.empty.add_variables_from(self.variables, self.cards)

    def time_copy(self):
        self.chain.copy()

    def time_generate_sample(self):
        list(self.chain.generate_sample(size=ROWS, rng=0))

    def time_random_state(self):
        self.chain.random_state()

    def time_sample(self):
        self.chain.sample(size=ROWS, rng=0)

    def time_set_start_state(self):
        self.chain.set_start_state(self.start)


class TimeMarkovChainProbFromSample(Bench):
    def setup(self):
        n_vars, n_states = CHAIN_SIZE
        self.chain = markov_chain(n_vars, n_states)
        self.sample = markov_chain_sample(n_vars, n_states, ROWS)
        self.state = [State("X0", 0), State("X1", 0)]
        self.window_size = ROWS // 10

    def time_prob_from_sample(self):
        self.chain.prob_from_sample(self.state, sample=self.sample, window_size=self.window_size)


def class_and_features(name):
    columns = topological(discrete_bn(name))
    return columns[0], columns[1:]


class TimeNaiveBayes(Bench):
    def setup(self):
        self.cls, self.features = class_and_features(DISCRETE)
        self.model = NaiveBayes(self.features, self.cls)
        self.almost = NaiveBayes(self.features[:-1], self.cls)
        self.empty = NaiveBayes()
        self.edges = [(self.cls, f) for f in self.features]

    def time_active_trail_nodes(self):
        self.model.active_trail_nodes(self.features[0], observed=self.features[1:3])

    def time_add_edge(self):
        self.almost.add_edge(self.cls, self.features[-1])

    def time_add_edges_from(self):
        self.empty.add_edges_from(self.edges)

    def time_local_independencies(self):
        self.model.local_independencies(self.features)


class TimeNaiveBayesFit(Bench):
    def setup(self):
        name = DISCRETE
        cls, features = class_and_features(name)
        self.model = NaiveBayes(features, cls)
        self.data = discrete_data(name, ROWS)

    def time_fit(self):
        self.model.fit(self.data)


class TimeSEMGraph(Bench):
    def setup(self):
        sem = sem_inputs(GAUSSIAN, SEM_ERR_CORR)
        self.model = SEMGraph(ebunch=sem["ebunch"], latents=sem["latents"], err_corr=sem["err_corr"])
        observed = sorted(self.model.observed)
        self.start = observed[0]
        self.observed = observed[1:3]

    def time_active_trail_nodes(self):
        self.model.active_trail_nodes(self.start, observed=self.observed)

    def time_get_scaling_indicators(self):
        self.model.get_scaling_indicators()

    def time_moralize(self):
        self.model.moralize()

    def time_to_lisrel(self):
        self.model.to_lisrel()

    def time_to_standard_lisrel(self):
        self.model.to_standard_lisrel()


class TimeSEMAlg(Bench):
    def setup(self):
        sem = sem_inputs(GAUSSIAN, SEM_ERR_CORR)
        self.B_values = sem["B_values"]
        self.zeta_values = sem["zeta_values"]
        self.model = SEMAlg(eta=sem["variables"], B=sem["B"], zeta=sem["zeta"], wedge_y=sem["wedge_y"])
        self.model.set_params(self.B_values, self.zeta_values)
        np.random.seed(0)  # generate_samples draws from the global generator

    def time_generate_samples(self):
        self.model.generate_samples(n_samples=ROWS)

    def time_set_params(self):
        self.model.set_params(self.B_values, self.zeta_values)

    def time_to_SEMGraph(self):
        self.model.to_SEMGraph()


class TimeSEM(Bench):
    def setup(self):
        self.sem = sem_inputs(GAUSSIAN, SEM_ERR_CORR)

    def time_from_graph(self):
        SEM.from_graph(ebunch=self.sem["ebunch"], latents=self.sem["latents"], err_corr=self.sem["err_corr"])

    def time_from_lavaan(self):
        SEM.from_lavaan(string=self.sem["lavaan"])

    def time_from_lisrel(self):
        SEM.from_lisrel(self.sem["var_names"], self.sem["params"], self.sem["fixed_masks"])

    def time_from_RAM(self):
        SEM.from_RAM(self.sem["variables"], self.sem["B"], self.sem["zeta"], wedge_y=self.sem["wedge_y"])
