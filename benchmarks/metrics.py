"""Benchmarks for ``pgmpy.metrics``: ``evaluate`` once per metric class, and ``get_metric`` by name and by default."""

import warnings
from functools import cache

import networkx as nx
import numpy as np
from benchmarks._bench import Bench, fresh
from sklearn.exceptions import UndefinedMetricWarning

from pgmpy import metrics
from pgmpy.base import DAG, MAG
from pgmpy.models import LinearGaussianBayesianNetwork

ROWS = 1_000
GAUSSIAN = 8
EDGE_PROB = 0.15
GRAPH_NODES = 20
LATENT_FRACTION = 0.1
warnings.filterwarnings("ignore", category=UndefinedMetricWarning, message=r"F-score is ill-defined")


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


@fresh
def graph(kind):
    if kind in ("admg", "mag"):
        admg = _admg(GRAPH_NODES)
        return admg if kind == "admg" else _mag(admg)
    dag = DAG.get_random(n_nodes=GRAPH_NODES, edge_prob=EDGE_PROB, rng=0)
    return {"dag": lambda: dag, "pdag": dag.to_pdag, "ug": dag.moralize, "perturbed": lambda: _perturbed(dag)}[kind]()


def _admg(n):
    n_latent = max(1, round(n * LATENT_FRACTION))
    dag = DAG.get_random(n_nodes=n + n_latent, edge_prob=EDGE_PROB, rng=0)
    dag.latents = set(np.random.default_rng(0).choice(list(dag.nodes()), size=n_latent, replace=False).tolist())
    return dag.to_admg()


def _mag(admg):
    ancestors = {v: admg.get_ancestors(v) for v in admg.nodes()}
    edges = [
        (u, v, t)
        for u, v, t in admg.get_edges(data=True)
        if t == "->" or (u not in ancestors[v] and v not in ancestors[u])
    ]
    mag = MAG(edge_list=edges)
    mag.add_nodes_from(admg.nodes())
    return mag


def _perturbed(dag, fraction=0.1, rng=0):
    edges = sorted(dag.edges())
    n = round(fraction * len(edges))
    picked = np.random.default_rng(rng).permutation(len(edges))[: 2 * n]
    removed = {edges[i] for i in picked[n:]}
    out = DAG([e for e in edges if e not in removed])
    out.add_nodes_from(dag.nodes())
    for u, v in sorted(edges[i] for i in picked[:n]):
        out.remove_edge(u, v)
        out.add_edge(*((u, v) if nx.has_path(out, u, v) else (v, u)))
    return out


@fresh
def gaussian_data(name, n_rows, seed=0):
    model = gaussian_bn(name)
    return model.simulate(n_rows, rng=seed)[topological(model)]


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
