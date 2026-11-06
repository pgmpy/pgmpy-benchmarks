"""Benchmarks for ``pgmpy.structure_score``.

The 12 score classes define no public method of their own except ``BDs.structure_prior`` and
``BDs.structure_prior_ratio``. They share the public ``score`` / ``local_score`` /
``structure_prior`` / ``structure_prior_ratio`` that ``BaseStructureScore`` defines, and each class's
own ``_local_score`` is what those two time. So ``score`` and ``local_score`` are timed for every
score (one benchmark class per data type, with the score name as the second param), and the
structure prior of ``BDs`` (the only non-constant one) in ``TimeBDs``.

Inputs (``ROWS`` rows, the generating model's DAG as the scored structure):

* ``TimeDiscreteScore``: ``discrete_data`` and ``discrete_bn`` of ``asia`` (fast) / ``alarm`` (slow).
* ``TimeGaussianScore``: ``gaussian_data`` and ``gaussian_bn`` of ``random_gaussian_8`` (fast) /
  ``arth150`` (slow).
* ``TimeCondGaussScore``: ``mixed_data`` (every other column cut into 3 string categories) of the
  same Gaussian models, scored on the same DAG.
* ``local_score(variable, parents)`` is the last node in sorted order with its true parents.
* A score caches its local scores, so ``prepare`` clears that cache  before every call.
  The score instance itself is memoized per tier and score, since building one preprocesses all the
  data (2.5 s for ``arth150``).

The fast ``TimeGaussianScore`` uses ``GAUSSIAN_ROWS`` = 20000 rows instead of ``ROWS``: at 1000 rows
``local_score`` took 25 us (``score`` 0.25 ms) and the noise check flagged ``BICGauss`` (25.4 us vs
39.3 us on the same code). At 20000 rows ``local_score`` takes 0.25-0.28 ms and ``score`` 2.1-2.5 ms.

Measured slow-tier times: ``score`` is 8 ms for every discrete score on ``alarm``, 0.2 s for the
Gaussian scores on ``arth150`` and 3.1 s for the CondGauss scores on ``arth150``. The CondGauss
``score`` is the heaviest benchmark here (it fits a regression per node and mixed parent
configuration); it stays under the 15 s cap. Every ``local_score`` takes 0.4 ms to 4 ms.

``TimeScoreFunctions`` times ``get_scoring_method`` by name (``"bdeu"``) and with ``None`` (default
score for the data type) on ``discrete_data``; it includes building the score.

Skipped:
    ``BDs.structure_prior_ratio`` and the inherited ``structure_prior`` /
    ``structure_prior_ratio`` (trivial: constants).
"""

from functools import cache

from benchmarks._fixtures import discrete_bn, discrete_data, gaussian_bn, gaussian_data, mixed_data
from benchmarks._tiers import ROWS, TIERS, Bench

from pgmpy import structure_score

DISCRETE = {"fast": "asia", "slow": "alarm"}
GAUSSIAN = {"fast": "random_gaussian_8", "slow": "arth150"}

DISCRETE_SCORES = ["K2", "BDeu", "BDs", "LogLikelihood", "AIC", "BIC"]
GAUSSIAN_SCORES = ["LogLikelihoodGauss", "AICGauss", "BICGauss"]
GAUSSIAN_ROWS = {"fast": 20_000, "slow": ROWS["slow"]}  # see the module docstring
COND_GAUSS_SCORES = ["LogLikelihoodCondGauss", "AICCondGauss", "BICCondGauss"]


@cache
def discrete_model(tier):
    return discrete_bn(DISCRETE[tier])


@cache
def gaussian_model(tier):
    return gaussian_bn(GAUSSIAN[tier])


def local_inputs(model):
    """The last node in sorted order with its parents (the ``local_score`` arguments)."""
    node = sorted(model.nodes())[-1]
    return node, tuple(model.predecessors(node))


@cache
def discrete_score(tier, name):
    return getattr(structure_score, name)(discrete_data(DISCRETE[tier], ROWS[tier]))


@cache
def gaussian_score(tier, name):
    return getattr(structure_score, name)(gaussian_data(GAUSSIAN[tier], GAUSSIAN_ROWS[tier]))


@cache
def cond_gauss_score(tier, name):
    return getattr(structure_score, name)(mixed_data(GAUSSIAN[tier], ROWS[tier]))


def clean(scorer):
    """Empty the local score cache, so every call computes its scores.

    pgmpy has no public way to do this, and rebuilding the scorer in ``prepare`` would copy and
    preprocess the whole dataset before every call. So this clears the private ``_cache`` if it
    exists; if pgmpy renames it, the benchmarks keep running (and then time cache hits) instead of
    failing every unrelated PR.
    """
    cache = getattr(scorer, "_cache", None)
    if cache is not None:
        cache.clear()
    return scorer


class TimeDiscreteScore(Bench):
    params = [TIERS, DISCRETE_SCORES]
    param_names = ["tier", "score"]

    def prepare(self, tier, name):
        self.model = discrete_model(tier)
        self.scorer = clean(discrete_score(tier, name))
        self.node, self.parents = local_inputs(self.model)

    def time_score(self, tier, name):
        self.scorer.score(self.model)

    def time_local_score(self, tier, name):
        self.scorer.local_score(self.node, self.parents)


class TimeGaussianScore(Bench):
    params = [TIERS, GAUSSIAN_SCORES]
    param_names = ["tier", "score"]

    def prepare(self, tier, name):
        self.model = gaussian_model(tier)
        self.scorer = clean(gaussian_score(tier, name))
        self.node, self.parents = local_inputs(self.model)

    def time_score(self, tier, name):
        self.scorer.score(self.model)

    def time_local_score(self, tier, name):
        self.scorer.local_score(self.node, self.parents)


class TimeCondGaussScore(Bench):
    params = [TIERS, COND_GAUSS_SCORES]
    param_names = ["tier", "score"]

    def prepare(self, tier, name):
        self.model = gaussian_model(tier)
        self.scorer = clean(cond_gauss_score(tier, name))
        self.node, self.parents = local_inputs(self.model)

    def time_score(self, tier, name):
        self.scorer.score(self.model)

    def time_local_score(self, tier, name):
        self.scorer.local_score(self.node, self.parents)


class TimeBDs(Bench):
    def prepare(self, tier):
        self.model = discrete_model(tier)
        self.scorer = discrete_score(tier, "BDs")

    def time_structure_prior(self, tier):
        self.scorer.structure_prior(self.model)


class TimeScoreFunctions(Bench):
    def prepare(self, tier):
        self.data = discrete_data(DISCRETE[tier], ROWS[tier])

    def time_get_scoring_method(self, tier):
        structure_score.get_scoring_method("bdeu", self.data)

    def time_get_scoring_method_default(self, tier):
        structure_score.get_scoring_method(None, self.data)
