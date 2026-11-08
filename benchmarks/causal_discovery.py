"""Benchmarks for ``pgmpy.causal_discovery``: ``fit`` per concrete class, ``score`` once on a fitted ``HillClimbSearch``.

Skipped:
    LLMPairwise, ExpertInLoop: llm
"""

import warnings
from functools import cache, partial
from itertools import combinations

import networkx as nx

from benchmarks._fixtures import discrete_bn, discrete_data, gaussian_bn, gaussian_data, graph
from benchmarks._bench import DISCRETE, GAUSSIAN, ROWS, Bench

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

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"\d+ pair\(s\) of variables passed the threshold but were left out of the consensus graph",
)

PDAG = "pdag"

ANM_ROWS = 300  # the GP fit is O(n^2)
SP_MAX_ITER = 2_000
N_BOOTSTRAPS = 10


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


class TimeGESOperations(Bench):
    def setup(self):
        self.model = graph(PDAG)
        self.ges = GES()
        nodes = sorted(self.model.nodes())
        self.non_adjacent = next(
            (u, v)
            for u, v in combinations(nodes, 2)
            if not self.model.has_edge(u, v) and not self.model.has_edge(v, u)
        )
        self.directed = sorted(self.model.directed_edges)[0]

    def time_insert(self):
        u, v = self.non_adjacent
        self.ges.insert(u, v, set(), self.model)

    def time_delete(self):
        u, v = self.directed
        self.ges.delete(u, v, set(), self.model)

    def time_turn(self):
        u, v = self.directed
        self.ges.turn(v, u, set(), self.model)


class TimeHillClimbSearch(_Fit):
    load, make = staticmethod(discrete), staticmethod(partial(HillClimbSearch, show_progress=False))


@cache
def fitted_hill_climb():
    return HillClimbSearch(return_type="dag", show_progress=False).fit(discrete())


class TimeHillClimbSearchScore(Bench):
    def setup(self):
        self.estimator = fitted_hill_climb()
        self.data = discrete()
        self.true_graph = discrete_bn(DISCRETE)

    def time_score(self):
        self.estimator.score(X=self.data)

    def time_score_true_graph(self):
        self.estimator.score(true_graph=self.true_graph)


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


@cache
def fitted_bootstrap():
    return bootstrap_estimator().fit(discrete())


class TimeBootstrapEstimatorConsensus(Bench):
    def setup(self):
        self.estimator = fitted_bootstrap()

    def time_get_consensus_graph(self):
        self.estimator.get_consensus_graph(0.5)

    def time_get_adjacency_matrix(self):
        self.estimator.get_adjacency_matrix(0.5)


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


class TimeExpertKnowledgeApplyTo(Bench):
    def setup(self):
        self.model = graph(PDAG)
        undirected = sorted(self.model.undirected_edges)
        self.expert_knowledge = ExpertKnowledge(
            required_edges=undirected[0::2], forbidden_edges=undirected[1::2]
        ).fit()

    def time_apply_to(self):
        self.expert_knowledge.apply_to(self.model)
