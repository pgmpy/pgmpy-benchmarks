"""Benchmarks for ``pgmpy.parameter_estimator``: ``fit`` per estimator, each call on a new estimator, on a shared bare structure.

Skipped:
    constructors, DiscreteEM.set_params, ``_``-prefixed helpers: trivial / private
    BaseParameterEstimator, DiscreteParameterEstimator, GaussianParameterEstimator: abstract (``fit`` covered via subclasses)
"""

import warnings
from functools import cache

import networkx as nx
from benchmarks._bench import Bench, fresh
from sklearn.exceptions import ConvergenceWarning

from pgmpy.example_models import load_model
from pgmpy.models import DiscreteBayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteEM, DiscreteMLE, LinearGaussianMLE

ROWS = 1_000
DISCRETE = "asia"
GAUSSIAN = 8
warnings.filterwarnings("ignore", category=ConvergenceWarning, message=r".*max_iter.*")

EM_ITERATIONS = 5


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


def bare(model):
    b = type(model)(model.edges())
    b.add_nodes_from(model.nodes)
    return b


@fresh
def discrete_data(name, n_rows, seed=0):
    model = discrete_bn(name)
    return model.simulate(n_rows, rng=seed, show_progress=False)[topological(model)]


@fresh
def gaussian_data(name, n_rows, seed=0):
    model = gaussian_bn(name)
    return model.simulate(n_rows, rng=seed)[topological(model)]


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


@cache
def discrete_structure(name):
    return bare(discrete_bn(name))


@cache
def gaussian_structure(name):
    return bare(gaussian_bn(name))


@cache
def latent_variable(name):
    model = discrete_bn(name)
    return min(n for n in model.nodes() if model.get_parents(n))


@cache
def em_inputs(name, n_rows):
    latent = latent_variable(name)
    edges = list(discrete_bn(name).edges())
    model = DiscreteBayesianNetwork(edges, latents={latent})
    return model, discrete_data(name, n_rows).drop(columns=[latent])


class TimeDiscreteMLE(Bench):
    def setup(self):
        name = DISCRETE
        self.model = discrete_structure(name)
        self.data = discrete_data(name, ROWS)

    def time_fit(self):
        DiscreteMLE().fit(self.model, self.data)


class TimeDiscreteBayesianEstimator(TimeDiscreteMLE):
    def time_fit(self):
        DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=5).fit(self.model, self.data)


class TimeDiscreteEM(Bench):
    def setup(self):
        self.model, self.data = em_inputs(DISCRETE, ROWS)

    def time_fit(self):
        DiscreteEM(max_iter=EM_ITERATIONS, rng=0, show_progress=False).fit(self.model, self.data)


class TimeLinearGaussianMLE(Bench):
    def setup(self):
        name = GAUSSIAN
        self.model = gaussian_structure(name)
        self.data = gaussian_data(name, ROWS)

    def time_fit(self):
        LinearGaussianMLE().fit(self.model, self.data)
