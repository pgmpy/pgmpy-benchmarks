"""Benchmarks for ``pgmpy.metrics``.

The metric classes define no public method of their own: each one is evaluated through ``evaluate``
(``__call__`` is the same method), which ``BaseSupervisedMetric`` / ``BaseUnsupervisedMetric`` define.
So ``evaluate`` is timed once per metric class, in the benchmark class named after it, and
``get_metric`` is timed by name and by default (``None`` with ``requires_true_graph=False``).

Supervised metrics (``TimeAdjacencyConfusionMatrix``, ``TimeOrientationConfusionMatrix``, ``TimeSHD``)
compare the true graph ``graph("dag_20")`` (fast) / ``graph("dag_500")`` (slow) with the frozen estimate
``graph("dag_20_perturbed")`` / ``graph("dag_500_perturbed")``: 10% of the edges reversed and 10% of
the others removed (seed 0, ``numpy.random.default_rng``). A reversal that would close a cycle is
skipped (see ``tools/build_fixtures.perturbed_dag_spec``), so the dense ``dag_20`` keeps one of its
four chosen reversals and ``dag_500`` 62 of 63. The metrics only read the graphs, so they are memoized
per tier.

Unsupervised metrics (``TimeCorrelationScore``, ``TimeImpliedCIs``, ``TimeFisherC``,
``TimeStructureScore``) evaluate the generating DAG of the model (``gaussian_bn(name)``, i.e. the
"estimated graph" is the true DAG itself) against ``gaussian_data`` of ``random_gaussian_8`` (fast) /
``arth150`` (slow). They use the ``pearsonr`` CI test (explicit, so the data-type detection of
``get_ci_test`` is not timed), ``show_progress=False`` where accepted, and ``bic-g`` for
``StructureScore`` (building the score, which preprocesses the data, is part of the timed call).

Row cap (``METRIC_ROWS``): the CI-test based metrics run one test per node pair (``FisherC`` and
``ImpliedCIs`` one per non-adjacent pair, ``CorrelationScore`` one per pair) on ``arth150``'s 107
nodes. With 100000 rows ``FisherC`` takes 31 s, ``ImpliedCIs`` 15 s and ``StructureScore`` 3 s, over
the 15 s cap, so the slow tier uses 5000 rows (``FisherC`` about 3 s, ``ImpliedCIs`` about 3 s,
``CorrelationScore`` about 1 s, ``StructureScore`` 0.1 s; the per-pair overhead dominates from there
down). The fast tier keeps ``ROWS`` (1000).

``CorrelationScore`` on ``random_gaussian_8`` has no pair of uncorrelated variables, so scikit-learn's
f1 is ill-defined and warns; that warning is filtered by category and message.

Skipped: none (every public method is timed: ``evaluate``, and ``get_metric``).
"""

import warnings
from functools import cache

from benchmarks._fixtures import gaussian_bn, gaussian_data, graph
from benchmarks._tiers import ROWS, TIERS, Bench
from sklearn.exceptions import UndefinedMetricWarning

from pgmpy import metrics

warnings.filterwarnings("ignore", category=UndefinedMetricWarning, message=r"F-score is ill-defined")

TRUE_GRAPH = {"fast": "dag_20", "slow": "dag_500"}
GAUSSIAN = {"fast": "random_gaussian_8", "slow": "arth150"}
METRIC_ROWS = {"fast": ROWS["fast"], "slow": 5_000}


@cache
def graphs(tier):
    name = TRUE_GRAPH[tier]
    return graph(name), graph(f"{name}_perturbed")


@cache
def unsupervised_inputs(tier):
    """``(data, true DAG)`` of the Gaussian model; both are only read by the metrics."""
    name = GAUSSIAN[tier]
    return gaussian_data(name, METRIC_ROWS[tier]), gaussian_bn(name)


class SupervisedMetric(Bench):
    """Shared ``prepare``; the subclasses name the metric and time ``evaluate``."""

    metric = None

    def prepare(self, tier):
        self.true_graph, self.est_graph = graphs(tier)
        self.instance = self.metric()

    def evaluate(self):
        return self.instance.evaluate(true_causal_graph=self.true_graph, est_causal_graph=self.est_graph)


class TimeAdjacencyConfusionMatrix(SupervisedMetric):
    metric = metrics.AdjacencyConfusionMatrix

    def time_evaluate(self, tier):
        self.evaluate()


class TimeOrientationConfusionMatrix(SupervisedMetric):
    metric = metrics.OrientationConfusionMatrix

    def time_evaluate(self, tier):
        self.evaluate()


class TimeSHD(SupervisedMetric):
    metric = metrics.SHD

    def time_evaluate(self, tier):
        self.evaluate()


class UnsupervisedMetric(Bench):
    """Shared ``prepare``; the subclasses build the metric and time ``evaluate``."""

    def make(self):
        raise NotImplementedError

    def prepare(self, tier):
        self.data, self.model = unsupervised_inputs(tier)
        self.instance = self.make()

    def evaluate(self):
        return self.instance.evaluate(X=self.data, causal_graph=self.model)


class TimeCorrelationScore(UnsupervisedMetric):
    def make(self):
        return metrics.CorrelationScore(ci_test="pearsonr")

    def time_evaluate(self, tier):
        self.evaluate()


class TimeImpliedCIs(UnsupervisedMetric):
    def make(self):
        return metrics.ImpliedCIs(ci_test="pearsonr", show_progress=False)

    def time_evaluate(self, tier):
        self.evaluate()


class TimeFisherC(UnsupervisedMetric):
    def make(self):
        return metrics.FisherC(ci_test="pearsonr", show_progress=False)

    def time_evaluate(self, tier):
        self.evaluate()


class TimeStructureScore(UnsupervisedMetric):
    def make(self):
        return metrics.StructureScore(scoring_method="bic-g")

    def time_evaluate(self, tier):
        self.evaluate()


class TimeGetMetric(Bench):
    def time_get_metric_by_name(self, tier):
        metrics.get_metric("SHD")

    def time_get_metric_default(self, tier):
        metrics.get_metric(requires_true_graph=False)
