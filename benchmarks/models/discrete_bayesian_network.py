"""Benchmarks for ``pgmpy.models.DiscreteBayesianNetwork``. The methods that it inherits from ``DAG`` are timed in ``benchmarks/base/dag.py``.

Skipped:
    DiscreteBayesianNetwork.get_factorized_product: trivial (empty stub)
    DiscreteBayesianNetwork.states: trivial
"""

import networkx as nx
import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.factors.discrete import JointProbabilityDistribution, TabularCPD
from pgmpy.models import DiscreteBayesianNetwork

ROWS = 1_000
MODEL = "alarm"
FREE_VARIABLES = 3
MISSING_COLUMNS = 2
PREDICT_ROWS = 40
PROBABILITY_ROWS = 50
IMAP_VARS = 4
RANDOM_BN = (20, 0.2, 3)


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


def bare(model):
    b = type(model)(model.edges())
    b.add_nodes_from(model.nodes)
    return b


def first_states(model, nodes):
    return {n: model.states[n][0] for n in nodes}


def joint_chain(n_vars, seed=0):
    """Binary chain X0 -> X1 -> ... as (JointProbabilityDistribution, DiscreteBayesianNetwork), the network an I-map of the joint."""
    rng = np.random.default_rng(seed)
    names = [f"X{i}" for i in range(n_vars)]
    root = rng.dirichlet([1, 1])  # P(X0)
    cpts = [rng.dirichlet([1, 1], size=2).T for _ in range(n_vars - 1)]
    joint = root
    for cpt in cpts:
        joint = joint[..., None] * cpt.T  # axes: X0, ..., X(i-1), Xi
    model = DiscreteBayesianNetwork([(names[i], names[i + 1]) for i in range(n_vars - 1)])
    model.add_cpds(
        TabularCPD(names[0], 2, root.reshape(2, 1)),
        *[
            TabularCPD(names[i + 1], 2, cpt, evidence=[names[i]], evidence_card=[2])
            for i, cpt in enumerate(cpts)
        ],
    )
    return JointProbabilityDistribution(names, [2] * n_vars, joint.ravel()), model


@fresh
def discrete_data(name, n_rows, seed=0):
    model = discrete_bn(name)
    return model.simulate(n_rows, rng=seed, show_progress=False)[topological(model)]


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(MODEL)


def dag_non_edge(model):
    order = topological(model)
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

    def time_simulate(self):
        self.model.simulate(ROWS, rng=0, show_progress=False)


class TimeDiscreteBayesianNetworkRandom(Bench):
    def time_get_random(self):
        n_nodes, edge_prob, n_states = RANDOM_BN
        DiscreteBayesianNetwork.get_random(n_nodes, edge_prob=edge_prob, n_states=n_states, rng=0)


class TimeDiscreteBayesianNetworkIsImap(Bench):
    def setup(self):
        self.joint, self.model = joint_chain(IMAP_VARS)

    def time_is_imap(self):
        self.model.is_imap(self.joint)
