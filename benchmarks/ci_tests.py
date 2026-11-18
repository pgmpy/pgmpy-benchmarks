"""Benchmarks for ``pgmpy.ci_tests``: only ``is_independent``, for every test (param = test name)."""

from functools import cache

import networkx as nx
import pandas as pd
from benchmarks._bench import Bench, fresh

from pgmpy import ci_tests
from pgmpy.example_models import load_model
from pgmpy.models import LinearGaussianBayesianNetwork

ROWS = 1_000
DISCRETE = "asia"
GAUSSIAN = 8
SEEDED = {"GeneralizedCov", "HSIC"}

TEST_ROWS = {"KCI": 600}  # kernel matrices are cubic in n; 1,000 rows took 0.32 s


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


@fresh
def discrete_data(name, n_rows, seed=0):
    model = discrete_bn(name)
    return model.simulate(n_rows, rng=seed, show_progress=False)[topological(model)]


@fresh
def gaussian_data(name, n_rows, seed=0):
    model = gaussian_bn(name)
    return model.simulate(n_rows, rng=seed)[topological(model)]


def mixed_data(name, n_rows, seed=0):
    df = gaussian_data(name, n_rows, seed)
    for col in df.columns[1::2]:
        df[col] = pd.qcut(df[col], 3, labels=["low", "mid", "high"]).astype(object)
    return df


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


def triple(model):
    nodes = sorted(model.nodes())
    y = nodes[-1]
    return nodes[0], y, [sorted(model.predecessors(y))[0]]


@cache
def ci_test(name, data, source):
    kwargs = {"rng": 0} if name in SEEDED else {}
    return getattr(ci_tests, name)(data(source, TEST_ROWS.get(name, ROWS)), use_cache=False, **kwargs)


@cache
def independence_match():
    model = discrete_bn(DISCRETE)
    return ci_tests.IndependenceMatch(model.local_independencies(sorted(model.nodes())), use_cache=False)


class _CITest(Bench):
    param_names = ["test"]

    def setup(self, name):
        bn, data, source = self.inputs
        self.test = ci_test(name, data, source)
        self.x, self.y, self.z = triple(bn(source))

    def time_is_independent(self, name):
        self.test.is_independent(self.x, self.y, self.z)


class TimeDiscreteCITest(_CITest):
    params = [["ChiSquare", "GSq", "LogLikelihood", "ModifiedLogLikelihood", "PowerDivergence"]]
    inputs = discrete_bn, discrete_data, DISCRETE


class TimeContinuousCITest(_CITest):
    params = [["FisherZ", "Pearsonr", "PearsonrEquivalence", "GCM", "GeneralizedCov", "HSIC", "KCI"]]
    inputs = gaussian_bn, gaussian_data, GAUSSIAN


class TimeMixedCITest(_CITest):
    params = [["HotellingLawley", "PillaiTrace", "RoysLargestRoot", "WilksLambda"]]
    inputs = gaussian_bn, mixed_data, GAUSSIAN


class TimeIndependenceMatch(Bench):
    def setup(self):
        self.test = independence_match()
        self.x, self.y, self.z = triple(discrete_bn(DISCRETE))

    def time_is_independent(self):
        self.test.is_independent(self.x, self.y, self.z)

