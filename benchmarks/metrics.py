"""Benchmarks for ``pgmpy.metrics``: ``evaluate`` once per metric class, and ``get_metric`` by name and by default.

Supervised metrics compare ``dag_20`` with ``dag_20_perturbed``. Unsupervised metrics evaluate the generating DAG of
``random_gaussian_8`` against its data, with the ``pearsonr`` CI test and ``bic-g``.

Skipped: none
"""

import warnings
from functools import cache

from benchmarks._fixtures import gaussian_bn, gaussian_data, graph
from benchmarks._bench import GAUSSIAN, ROWS, Bench
from sklearn.exceptions import UndefinedMetricWarning

from pgmpy import metrics

warnings.filterwarnings("ignore", category=UndefinedMetricWarning, message=r"F-score is ill-defined")


@cache
def graphs():
    return graph("dag_20"), graph("dag_20_perturbed")


@cache
def unsupervised_inputs():
    return gaussian_data(GAUSSIAN, ROWS), gaussian_bn(GAUSSIAN)


class SupervisedMetric(Bench):
    metric = None

    def setup(self):
        self.true_graph, self.est_graph = graphs()
        self.instance = self.metric()

    def evaluate(self):
        return self.instance.evaluate(true_causal_graph=self.true_graph, est_causal_graph=self.est_graph)


class TimeAdjacencyConfusionMatrix(SupervisedMetric):
    metric = metrics.AdjacencyConfusionMatrix

    def time_evaluate(self):
        self.evaluate()


class TimeOrientationConfusionMatrix(SupervisedMetric):
    metric = metrics.OrientationConfusionMatrix

    def time_evaluate(self):
        self.evaluate()


class TimeSHD(SupervisedMetric):
    metric = metrics.SHD

    def time_evaluate(self):
        self.evaluate()


class UnsupervisedMetric(Bench):
    def make(self):
        raise NotImplementedError

    def setup(self):
        self.data, self.model = unsupervised_inputs()
        self.instance = self.make()

    def evaluate(self):
        return self.instance.evaluate(X=self.data, causal_graph=self.model)


class TimeCorrelationScore(UnsupervisedMetric):
    def make(self):
        return metrics.CorrelationScore(ci_test="pearsonr")

    def time_evaluate(self):
        self.evaluate()


class TimeImpliedCIs(UnsupervisedMetric):
    def make(self):
        return metrics.ImpliedCIs(ci_test="pearsonr", show_progress=False)

    def time_evaluate(self):
        self.evaluate()


class TimeFisherC(UnsupervisedMetric):
    def make(self):
        return metrics.FisherC(ci_test="pearsonr", show_progress=False)

    def time_evaluate(self):
        self.evaluate()


class TimeStructureScore(UnsupervisedMetric):
    def make(self):
        return metrics.StructureScore(scoring_method="bic-g")

    def time_evaluate(self):
        self.evaluate()


class TimeGetMetric(Bench):
    def time_get_metric_by_name(self):
        metrics.get_metric("SHD")

    def time_get_metric_default(self):
        metrics.get_metric(requires_true_graph=False)
