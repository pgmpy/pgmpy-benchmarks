"""Benchmarks for ``pgmpy.structure_score``: only ``score`` and ``local_score``, for every score (one class per data
type, param = score name).
"""

from functools import cache

import networkx as nx
import pandas as pd
from benchmarks._bench import Bench, fresh

from pgmpy import structure_score
from pgmpy.example_models import load_model
from pgmpy.models import LinearGaussianBayesianNetwork

ROWS = 1_000
DISCRETE = "asia"
GAUSSIAN = 8
GAUSSIAN_ROWS = 20_000  # at 1,000 rows a call takes ~25 us, close to timer noise


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


def local_inputs(model):
    node = sorted(model.nodes())[-1]
    return node, tuple(model.predecessors(node))


@cache
def make_scorer(name, data, source, n_rows):
    return getattr(structure_score, name)(data(source, n_rows))


def clean(scorer):  # private cache, no public API: if pgmpy renames it, the scores time cache hits
    cache = getattr(scorer, "_cache", None)
    if cache is not None:
        cache.clear()
    return scorer


class _Score(Bench):
    param_names = ["score"]

    def setup(self, name):
        bn, data, source, n_rows = self.inputs
        self.model = bn.shared(source)
        self.scorer = clean(make_scorer(name, data, source, n_rows))
        self.node, self.parents = local_inputs(self.model)

    def time_score(self, name):
        self.scorer.score(self.model)

    def time_local_score(self, name):
        self.scorer.local_score(self.node, self.parents)


class TimeDiscreteScore(_Score):
    params = [["K2", "BDeu", "BDs", "LogLikelihood", "AIC", "BIC"]]
    inputs = discrete_bn, discrete_data, DISCRETE, ROWS


class TimeGaussianScore(_Score):
    params = [["LogLikelihoodGauss", "AICGauss", "BICGauss"]]
    inputs = gaussian_bn, gaussian_data, GAUSSIAN, GAUSSIAN_ROWS


class TimeCondGaussScore(_Score):
    params = [["LogLikelihoodCondGauss", "AICCondGauss", "BICCondGauss"]]
    inputs = gaussian_bn, mixed_data, GAUSSIAN, ROWS

