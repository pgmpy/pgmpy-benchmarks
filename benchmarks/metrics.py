"""Benchmarks for ``pgmpy.metrics``: ``evaluate`` once per metric class, and ``get_metric`` by name and by default."""

import warnings
from functools import cache

from benchmarks._fixtures import gaussian_bn, gaussian_data, graph
from benchmarks._bench import GAUSSIAN, ROWS, Bench
from sklearn.exceptions import UndefinedMetricWarning

from pgmpy import metrics

warnings.filterwarnings("ignore", category=UndefinedMetricWarning, message=r"F-score is ill-defined")


@cache
def unsupervised_inputs():
    return gaussian_data(GAUSSIAN, ROWS), gaussian_bn(GAUSSIAN)


class _Supervised(Bench):
    def setup(self):
        self.true_graph, self.est_graph = graph.shared("dag"), graph.shared("perturbed")
        self.instance = self.metric()

    def time_evaluate(self):
        self.instance.evaluate(true_causal_graph=self.true_graph, est_causal_graph=self.est_graph)


class TimeAdjacencyConfusionMatrix(_Supervised):
    metric = metrics.AdjacencyConfusionMatrix


class TimeOrientationConfusionMatrix(_Supervised):
    metric = metrics.OrientationConfusionMatrix


class TimeSHD(_Supervised):
    metric = metrics.SHD


class _Unsupervised(Bench):
    def setup(self):
        self.data, self.model = unsupervised_inputs()
        self.instance = self.metric(**self.kwargs)

    def time_evaluate(self):
        self.instance.evaluate(X=self.data, causal_graph=self.model)


class TimeCorrelationScore(_Unsupervised):
    metric, kwargs = metrics.CorrelationScore, {"ci_test": "pearsonr"}


class TimeImpliedCIs(_Unsupervised):
    metric, kwargs = metrics.ImpliedCIs, {"ci_test": "pearsonr", "show_progress": False}


class TimeFisherC(_Unsupervised):
    metric, kwargs = metrics.FisherC, {"ci_test": "pearsonr", "show_progress": False}


class TimeStructureScore(_Unsupervised):
    metric, kwargs = metrics.StructureScore, {"scoring_method": "bic-g"}


class TimeGetMetric(Bench):
    def time_get_metric_by_name(self):
        metrics.get_metric("SHD")

    def time_get_metric_default(self):
        metrics.get_metric(requires_true_graph=False)
