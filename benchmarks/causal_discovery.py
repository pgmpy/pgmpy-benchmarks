"""Benchmarks for ``pgmpy.causal_discovery``: only ``fit``, for each algorithm.

Skipped:
    LLMPairwise, ExpertInLoop: llm
"""

import warnings
from functools import partial

import networkx as nx
from benchmarks._bench import Bench, fresh

from pgmpy.causal_discovery import (
    ANM,
    GES,
    IGCI,
    PC,
    SP,
    TAN,
    TOPIC,
    BootstrapEstimator,
    ChowLiu,
    ExpertKnowledge,
    HillClimbSearch,
    R2Sort,
    VarSort,
)
from pgmpy.example_models import load_model
from pgmpy.models import LinearGaussianBayesianNetwork

ROWS = 1_000
DISCRETE = "asia"
GAUSSIAN = 8
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"\d+ pair\(s\) of variables passed the threshold but were left out of the consensus graph",
)

ANM_ROWS = 300  # the GP fit is O(n^2)
SP_MAX_ITER = 2_000
N_BOOTSTRAPS = 10


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


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


def discrete(rows=ROWS):
    return discrete_data(DISCRETE, rows)


def gaussian(rows=ROWS):
    return gaussian_data(GAUSSIAN, rows)


def gaussian_pair(rows):
    """The two endpoints of the first sorted edge of the true network, as a two-column frame."""
    first_edge = sorted(gaussian_bn(GAUSSIAN).edges())[0]
    return gaussian_data(GAUSSIAN, rows)[list(first_edge)]


class _Fit(Bench):
    def setup(self):
        self.data = self.load()

    def time_fit(self):
        self.make().fit(self.data)


class TimePC(_Fit):
    load, make = staticmethod(discrete), staticmethod(partial(PC, n_jobs=1, show_progress=False))


class TimePCGaussian(_Fit):
    load, make = staticmethod(gaussian), staticmethod(partial(PC, n_jobs=1, show_progress=False))


class TimeGES(_Fit):
    load, make = staticmethod(discrete), GES


class TimeGESGaussian(_Fit):
    load, make = staticmethod(gaussian), GES


class TimeHillClimbSearch(_Fit):
    load, make = staticmethod(discrete), staticmethod(partial(HillClimbSearch, show_progress=False))


class TimeChowLiu(_Fit):
    load, make = staticmethod(discrete), staticmethod(partial(ChowLiu, n_jobs=1, show_progress=False))


class TimeTAN(Bench):
    def setup(self):
        self.data = discrete()
        self.class_node = self.data.columns[0]

    def time_fit(self):
        TAN(class_node=self.class_node, n_jobs=1, show_progress=False).fit(self.data)


class TimeSP(_Fit):
    load, make = staticmethod(discrete), staticmethod(partial(SP, max_iter=SP_MAX_ITER, rng=0, show_progress=False))


class TimeANM(_Fit):
    load, make = staticmethod(partial(gaussian_pair, ANM_ROWS)), ANM


class TimeIGCI(_Fit):
    load, make = staticmethod(partial(gaussian_pair, ROWS)), IGCI


class TimeR2Sort(_Fit):
    load, make = staticmethod(gaussian), R2Sort


class TimeVarSort(_Fit):
    load, make = staticmethod(gaussian), VarSort


class TimeTOPIC(_Fit):
    load, make = staticmethod(gaussian), staticmethod(partial(TOPIC, show_progress=False))


def bootstrap_estimator():
    return BootstrapEstimator(
        HillClimbSearch(return_type="dag", show_progress=False),
        n_bootstraps=N_BOOTSTRAPS,
        n_jobs=1,
        show_progress=False,
        rng=0,
    )


class TimeBootstrapEstimator(_Fit):
    load = staticmethod(discrete)

    def time_fit(self):
        bootstrap_estimator().fit(self.data)


def temporal_order(name):
    return [sorted(generation) for generation in nx.topological_generations(discrete_bn(name))]


class TimeExpertKnowledge(Bench):
    def setup(self):
        name = DISCRETE
        edges = sorted(discrete_bn(name).edges())
        order = temporal_order(name)
        self.data = discrete()
        self.kwargs = dict(
            required_edges=edges[:3],
            forbidden_edges=[(v, u) for u, v in edges[3:13]],
            temporal_order=order,
            root_nodes=order[0],
        )

    def time_fit(self):
        ExpertKnowledge(**self.kwargs).fit(self.data)

    def time_fit_search_space(self):
        ExpertKnowledge(search_space="marginally_dependent").fit(self.data)

