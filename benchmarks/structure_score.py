"""Benchmarks for ``pgmpy.structure_score``: ``score`` and ``local_score`` per score (one class per data type, param = score name), ``BDs`` priors, ``get_scoring_method``.

Skipped:
    BDs.structure_prior_ratio, inherited structure_prior / structure_prior_ratio: trivial (constants)
"""

from functools import cache

from benchmarks._fixtures import discrete_bn, discrete_data, gaussian_bn, gaussian_data, mixed_data
from benchmarks._bench import DISCRETE, GAUSSIAN, ROWS, Bench

from pgmpy import structure_score

GAUSSIAN_ROWS = 20_000  # at 1,000 rows a call takes ~25 us, close to timer noise


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


class TimeBDs(Bench):
    def setup(self):
        self.model = discrete_bn.shared(DISCRETE)
        self.scorer = make_scorer("BDs", discrete_data, DISCRETE, ROWS)

    def time_structure_prior(self):
        self.scorer.structure_prior(self.model)


class TimeScoreFunctions(Bench):
    def setup(self):
        self.data = discrete_data(DISCRETE, ROWS)

    def time_get_scoring_method(self):
        structure_score.get_scoring_method("bdeu", self.data)

    def time_get_scoring_method_default(self):
        structure_score.get_scoring_method(None, self.data)
