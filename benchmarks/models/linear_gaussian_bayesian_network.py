"""Benchmarks for ``pgmpy.models.LinearGaussianBayesianNetwork``. The methods that it inherits from ``DAG`` are timed in ``benchmarks/base/dag.py``.

Skipped:
    LinearGaussianBayesianNetwork.get_cardinality, .is_imap, .to_markov_model: unsupported (raise)
"""

import networkx as nx
from benchmarks._bench import Bench, fresh

from pgmpy.models import LinearGaussianBayesianNetwork

ROWS = 1_000
GAUSSIAN = 8
LINEAR_GAUSSIAN_RANDOM = (20, 0.2)


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


def bare(model):
    b = type(model)(model.edges())
    b.add_nodes_from(model.nodes)
    return b


@fresh
def gaussian_data(name, n_rows, seed=0):
    model = gaussian_bn(name)
    return model.simulate(n_rows, rng=seed)[topological(model)]


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
