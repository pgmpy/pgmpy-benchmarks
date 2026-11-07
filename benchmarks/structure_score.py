"""Benchmarks for ``pgmpy.structure_score``: ``score`` and ``local_score`` per score (one class per data type, param = score name), ``BDs`` priors, ``get_scoring_method``.

The scored structure is the generating DAG; ``local_score`` takes the last sorted node with its true parents. ``TimeGaussianScore``
uses ``GAUSSIAN_ROWS`` = 20000 rows because at 1000 rows calls take ~25 us and the noise check flags them.

Skipped:
    BDs.structure_prior_ratio, inherited structure_prior / structure_prior_ratio: trivial (constants)
"""

from functools import cache

from benchmarks._fixtures import discrete_bn, discrete_data, gaussian_bn, gaussian_data, mixed_data
from benchmarks._bench import DISCRETE, GAUSSIAN, ROWS, Bench

from pgmpy import structure_score

DISCRETE_SCORES = ["K2", "BDeu", "BDs", "LogLikelihood", "AIC", "BIC"]
GAUSSIAN_SCORES = ["LogLikelihoodGauss", "AICGauss", "BICGauss"]
GAUSSIAN_ROWS = 20_000
COND_GAUSS_SCORES = ["LogLikelihoodCondGauss", "AICCondGauss", "BICCondGauss"]


@cache
def discrete_model():
    return discrete_bn(DISCRETE)


@cache
def gaussian_model():
    return gaussian_bn(GAUSSIAN)


def local_inputs(model):
    """The last node in sorted order with its parents (the ``local_score`` arguments)."""
    node = sorted(model.nodes())[-1]
    return node, tuple(model.predecessors(node))


@cache
def discrete_score(name):
    return getattr(structure_score, name)(discrete_data(DISCRETE, ROWS))


@cache
def gaussian_score(name):
    return getattr(structure_score, name)(gaussian_data(GAUSSIAN, GAUSSIAN_ROWS))


@cache
def cond_gauss_score(name):
    return getattr(structure_score, name)(mixed_data(GAUSSIAN, ROWS))


def clean(scorer):
    """Clear the score's private local-score cache (no public API); a rename makes the benchmarks time cache hits."""
    cache = getattr(scorer, "_cache", None)
    if cache is not None:
        cache.clear()
    return scorer


class TimeDiscreteScore(Bench):
    params = [DISCRETE_SCORES]
    param_names = ["score"]

    def setup(self, name):
        self.model = discrete_model()
        self.scorer = clean(discrete_score(name))
        self.node, self.parents = local_inputs(self.model)

    def time_score(self, name):
        self.scorer.score(self.model)

    def time_local_score(self, name):
        self.scorer.local_score(self.node, self.parents)


class TimeGaussianScore(Bench):
    params = [GAUSSIAN_SCORES]
    param_names = ["score"]

    def setup(self, name):
        self.model = gaussian_model()
        self.scorer = clean(gaussian_score(name))
        self.node, self.parents = local_inputs(self.model)

    def time_score(self, name):
        self.scorer.score(self.model)

    def time_local_score(self, name):
        self.scorer.local_score(self.node, self.parents)


class TimeCondGaussScore(Bench):
    params = [COND_GAUSS_SCORES]
    param_names = ["score"]

    def setup(self, name):
        self.model = gaussian_model()
        self.scorer = clean(cond_gauss_score(name))
        self.node, self.parents = local_inputs(self.model)

    def time_score(self, name):
        self.scorer.score(self.model)

    def time_local_score(self, name):
        self.scorer.local_score(self.node, self.parents)


class TimeBDs(Bench):
    def setup(self):
        self.model = discrete_model()
        self.scorer = discrete_score("BDs")

    def time_structure_prior(self):
        self.scorer.structure_prior(self.model)


class TimeScoreFunctions(Bench):
    def setup(self):
        self.data = discrete_data(DISCRETE, ROWS)

    def time_get_scoring_method(self):
        structure_score.get_scoring_method("bdeu", self.data)

    def time_get_scoring_method_default(self):
        structure_score.get_scoring_method(None, self.data)
