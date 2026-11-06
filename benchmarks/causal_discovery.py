"""Benchmarks for ``pgmpy.causal_discovery``.

Every call passes ``n_jobs=1`` (parallelism makes timings non-comparable), ``show_progress=False``
and ``rng=0`` wherever the class accepts them, and CI exports ``PYTHONHASHSEED=0`` (PC and other
constraint-based searches depend on set order). Measured numbers below use that setup.

``fit`` is defined once on ``BaseCausalDiscovery`` (and on ``_ConstraintMixin`` for PC), but every
concrete class implements its own ``_fit``, which is what ``fit`` runs. So ``fit`` is timed once per
concrete class. ``score`` lives on ``BaseCausalDiscovery`` and is timed once, on a fitted
``HillClimbSearch`` (``TimeHillClimbSearchScore``).

Inputs (module-level dicts, mirrored by the ``MmhcEstimator`` benchmark in ``estimators.py``):

* ``DISCRETE`` / ``ROWS``: ``asia`` with 1000 rows (fast) and ``alarm`` with 100000 rows (slow) for
  ``PC``, ``GES``, ``HillClimbSearch`` and ``BootstrapEstimator`` (on ``HillClimbSearch``).
  ``PC``, ``GES`` and ``HillClimbSearch`` fit the full 37-variable ``alarm`` data in 1.7 to 3 s.
* ``GAUSSIAN``: ``random_gaussian_8`` (fast) / ``ecoli70`` (slow). pgmpy picks the Gaussian CI test
  and score automatically from the float columns, so ``PC``, ``GES`` and ``TOPIC`` have Gaussian
  variants on this data.
* ``TAN``'s class variable is the first column of the data.
* ``ANM`` and ``IGCI`` need exactly two variables: the endpoints of the first sorted edge of the true
  network (``X_0 -> X_1`` and ``asnA -> icdA``).

Reductions from the brief's slow sizes (each measured with ``PYTHONHASHSEED=0``, one call):

* ``ChowLiu`` and ``TAN`` on ``alarm`` use ``TREE_ROWS`` = 1000 rows, not 100000. Their cost is in the
  666 pairwise (conditional) mutual informations: ``ChowLiu`` takes 47 s at 100000 rows, 4.4 s at
  10000 and about 0.8 s at 1000; ``TAN`` 10 s at 10000 and 2.3 s at 1000.
* ``ANM`` fits Gaussian processes (O(n^2) memory; 100000 rows would need 75 GiB). It uses
  ``ANM_ROWS`` = 300 (fast, about 0.2 s) and 700 (slow, about 1.5 s) rows; 1000 rows takes 2 to 4 s.
* ``R2Sort`` and ``VarSort`` on ``ecoli70`` use ``ORDER_ROWS`` = 10000 rows: at 100000 they take 9.4 s
  and 5.1 s, at 10000 1.6 s and 1.0 s.
* ``PC``, ``GES`` and ``TOPIC`` on ``ecoli70`` (46 variables) use the first ``GAUSSIAN_COLUMNS`` = 12
  columns at 100000 rows (0.8 s, 1.4 s and 1.3 s). The full 46 columns take 8 s (``PC``) and 4 s
  (``GES``) already at 1000 rows, so rows cannot fix this. The fast tier uses all 8 columns.
* ``SP`` tries every permutation of the variables, which is infeasible for ``alarm`` (37 variables).
  It uses ``asia`` in both tiers (8 variables, 1000 / 100000 rows) with ``max_iter`` = ``SP_MAX_ITER``
  = 2000 of the 40320 permutations (0.1 s fast, about 0.5 s slow).
* ``BootstrapEstimator`` wraps ``HillClimbSearch(return_type="dag")``. Slow tier: ``alarm`` with
  ``BOOTSTRAP_ROWS`` = 10000 rows and 4 bootstraps (about 1.5 s, versus 8.9 s for 4 bootstraps at
  100000 rows). Fast tier: ``asia`` with 1000 rows and 10 bootstraps. ``get_consensus_graph`` and
  ``get_adjacency_matrix`` use a fitted estimator that is built once per process.
* ``GES.insert``, ``delete`` and ``turn`` run on the ``pdag_20`` / ``pdag_500`` fixture with
  ``T`` / ``H`` / ``C`` empty: ``insert`` joins the first non-adjacent pair of sorted nodes,
  ``delete`` removes the first sorted directed edge, ``turn`` reverses it.
* ``ExpertKnowledge.fit`` uses the true ``asia`` / ``alarm`` network: ``required_edges`` are the first
  3 sorted edges, ``forbidden_edges`` the reversals of the next 10, ``temporal_order`` the
  topological generations and ``root_nodes`` the first generation. ``fit_search_space`` adds
  ``search_space="marginally_dependent"`` (marginal CI tests on the data). ``apply_to`` orients the
  ``pdag_20`` / ``pdag_500`` fixture, requiring every even and forbidding every odd sorted undirected
  edge; it mutates the graph, so each call gets a fresh one.

The ``UserWarning`` from ``BootstrapEstimator.get_consensus_graph`` ("N pair(s) of variables passed the
threshold but were left out of the consensus graph because their edge would create a directed
cycle") is silenced by a module-level filter on that exact message. No deprecation warning is
filtered.

Skipped:
    (none) ``LLMPairwise`` and ``ExpertInLoop`` are out of scope (llm) and are not benchmarked.
"""

import warnings
from functools import cache
from itertools import combinations

import networkx as nx

from benchmarks._fixtures import discrete_bn, discrete_data, gaussian_bn, gaussian_data, graph
from benchmarks._tiers import ROWS, Bench

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

DISCRETE = {"fast": "asia", "slow": "alarm"}
GAUSSIAN = {"fast": "random_gaussian_8", "slow": "ecoli70"}
PDAG = {"fast": "pdag_20", "slow": "pdag_500"}

TREE_ROWS = {"fast": 1_000, "slow": 1_000}
ANM_ROWS = {"fast": 300, "slow": 700}
ORDER_ROWS = {"fast": 1_000, "slow": 10_000}
GAUSSIAN_COLUMNS = {"fast": 8, "slow": 12}
SP_MAX_ITER = {"fast": 2_000, "slow": 2_000}
SP_DATA = {"fast": "asia", "slow": "asia"}
BOOTSTRAP_ROWS = {"fast": 1_000, "slow": 10_000}
N_BOOTSTRAPS = {"fast": 10, "slow": 4}


def discrete(tier, rows=None):
    return discrete_data(DISCRETE[tier], ROWS[tier] if rows is None else rows)


def gaussian(tier, rows=None, columns=None):
    data = gaussian_data(GAUSSIAN[tier], ROWS[tier] if rows is None else rows)
    return data if columns is None else data.iloc[:, :columns]


def gaussian_pair(tier, rows):
    """The two endpoints of the first sorted edge of the true network, as a two-column frame."""
    first_edge = sorted(gaussian_bn(GAUSSIAN[tier]).edges())[0]
    return gaussian_data(GAUSSIAN[tier], rows)[list(first_edge)]


class TimePC(Bench):
    def prepare(self, tier):
        self.data = discrete(tier)

    def time_fit(self, tier):
        PC(n_jobs=1, show_progress=False).fit(self.data)


class TimePCGaussian(Bench):
    def prepare(self, tier):
        self.data = gaussian(tier, columns=GAUSSIAN_COLUMNS[tier])

    def time_fit(self, tier):
        PC(n_jobs=1, show_progress=False).fit(self.data)


class TimeGES(Bench):
    def prepare(self, tier):
        self.data = discrete(tier)

    def time_fit(self, tier):
        GES().fit(self.data)


class TimeGESGaussian(Bench):
    def prepare(self, tier):
        self.data = gaussian(tier, columns=GAUSSIAN_COLUMNS[tier])

    def time_fit(self, tier):
        GES().fit(self.data)


class TimeGESOperations(Bench):
    """``insert``, ``delete`` and ``turn`` on a PDAG fixture."""

    def prepare(self, tier):
        self.model = graph(PDAG[tier])
        self.ges = GES()
        nodes = sorted(self.model.nodes())
        self.non_adjacent = next(
            (u, v)
            for u, v in combinations(nodes, 2)
            if not self.model.has_edge(u, v) and not self.model.has_edge(v, u)
        )
        self.directed = sorted(self.model.directed_edges)[0]

    def time_insert(self, tier):
        u, v = self.non_adjacent
        self.ges.insert(u, v, set(), self.model)

    def time_delete(self, tier):
        u, v = self.directed
        self.ges.delete(u, v, set(), self.model)

    def time_turn(self, tier):
        u, v = self.directed
        self.ges.turn(v, u, set(), self.model)


class TimeHillClimbSearch(Bench):
    def prepare(self, tier):
        self.data = discrete(tier)

    def time_fit(self, tier):
        HillClimbSearch(show_progress=False).fit(self.data)


@cache
def fitted_hill_climb(tier):
    return HillClimbSearch(return_type="dag", show_progress=False).fit(discrete(tier))


class TimeHillClimbSearchScore(Bench):
    """``score`` (defined on ``BaseCausalDiscovery``) of a fitted DAG, against data and a true graph."""

    def prepare(self, tier):
        self.estimator = fitted_hill_climb(tier)
        self.data = discrete(tier)
        self.true_graph = discrete_bn(DISCRETE[tier])

    def time_score(self, tier):
        self.estimator.score(X=self.data)

    def time_score_true_graph(self, tier):
        self.estimator.score(true_graph=self.true_graph)


class TimeChowLiu(Bench):
    def prepare(self, tier):
        self.data = discrete(tier, TREE_ROWS[tier])

    def time_fit(self, tier):
        ChowLiu(n_jobs=1, show_progress=False).fit(self.data)


class TimeTAN(Bench):
    def prepare(self, tier):
        self.data = discrete(tier, TREE_ROWS[tier])
        self.class_node = self.data.columns[0]

    def time_fit(self, tier):
        TAN(class_node=self.class_node, n_jobs=1, show_progress=False).fit(self.data)


class TimeSP(Bench):
    def prepare(self, tier):
        self.data = discrete_data(SP_DATA[tier], ROWS[tier])

    def time_fit(self, tier):
        SP(max_iter=SP_MAX_ITER[tier], rng=0, show_progress=False).fit(self.data)


class TimeANM(Bench):
    def prepare(self, tier):
        self.data = gaussian_pair(tier, ANM_ROWS[tier])

    def time_fit(self, tier):
        ANM().fit(self.data)


class TimeIGCI(Bench):
    def prepare(self, tier):
        self.data = gaussian_pair(tier, ROWS[tier])

    def time_fit(self, tier):
        IGCI().fit(self.data)


class TimeR2Sort(Bench):
    def prepare(self, tier):
        self.data = gaussian(tier, ORDER_ROWS[tier])

    def time_fit(self, tier):
        R2Sort().fit(self.data)


class TimeVarSort(Bench):
    def prepare(self, tier):
        self.data = gaussian(tier, ORDER_ROWS[tier])

    def time_fit(self, tier):
        VarSort().fit(self.data)


class TimeTOPIC(Bench):
    def prepare(self, tier):
        self.data = gaussian(tier, columns=GAUSSIAN_COLUMNS[tier])

    def time_fit(self, tier):
        TOPIC(show_progress=False).fit(self.data)


def bootstrap_estimator(tier):
    return BootstrapEstimator(
        HillClimbSearch(return_type="dag", show_progress=False),
        n_bootstraps=N_BOOTSTRAPS[tier],
        n_jobs=1,
        show_progress=False,
        rng=0,
    )


class TimeBootstrapEstimator(Bench):
    def prepare(self, tier):
        self.data = discrete(tier, BOOTSTRAP_ROWS[tier])

    def time_fit(self, tier):
        bootstrap_estimator(tier).fit(self.data)


@cache
def fitted_bootstrap(tier):
    return bootstrap_estimator(tier).fit(discrete(tier, BOOTSTRAP_ROWS[tier]))


class TimeBootstrapEstimatorConsensus(Bench):
    """``get_consensus_graph`` and ``get_adjacency_matrix``; neither changes the estimator."""

    def prepare(self, tier):
        self.estimator = fitted_bootstrap(tier)

    def time_get_consensus_graph(self, tier):
        self.estimator.get_consensus_graph(0.5)

    def time_get_adjacency_matrix(self, tier):
        self.estimator.get_adjacency_matrix(0.5)


def temporal_order(name):
    return [sorted(generation) for generation in nx.topological_generations(discrete_bn(name))]


class TimeExpertKnowledge(Bench):
    def prepare(self, tier):
        name = DISCRETE[tier]
        edges = sorted(discrete_bn(name).edges())
        order = temporal_order(name)
        self.data = discrete(tier)
        self.kwargs = dict(
            required_edges=edges[:3],
            forbidden_edges=[(v, u) for u, v in edges[3:13]],
            temporal_order=order,
            root_nodes=order[0],
        )

    def time_fit(self, tier):
        ExpertKnowledge(**self.kwargs).fit(self.data)

    def time_fit_search_space(self, tier):
        ExpertKnowledge(search_space="marginally_dependent").fit(self.data)


class TimeExpertKnowledgeApplyTo(Bench):
    def prepare(self, tier):
        self.model = graph(PDAG[tier])
        undirected = sorted(self.model.undirected_edges)
        self.expert_knowledge = ExpertKnowledge(
            required_edges=undirected[0::2], forbidden_edges=undirected[1::2]
        ).fit()

    def time_apply_to(self, tier):
        self.expert_knowledge.apply_to(self.model)
