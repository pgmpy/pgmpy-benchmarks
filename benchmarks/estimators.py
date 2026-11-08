"""Benchmarks for the non-deprecated ``pgmpy.estimators``.

Skipped:
    MmhcEstimator.__init__ and the other constructors: trivial
"""

import warnings
from collections import deque
from functools import cache

from benchmarks._fixtures import discrete_bn, gaussian_data, markov_network, sem_inputs
from benchmarks._bench import DISCRETE, GAUSSIAN, ROWS, Bench
from benchmarks.causal_discovery import discrete
from benchmarks.models import SEM_ERR_CORR

from pgmpy.estimators import ExhaustiveSearch, IVEstimator, MirrorDescentEstimator, MmhcEstimator
from pgmpy.inference import CausalInference
from pgmpy.models import SEM

warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"invalid value encountered in divide")

N_MARGINALS = 5
ITERATIONS = 100


class TimeMmhcEstimator(Bench):
    def setup(self):
        self.estimator = MmhcEstimator(discrete())

    def time_estimate(self):
        self.estimator.estimate()

    def time_mmpc(self):
        self.estimator.mmpc()


class TimeExhaustiveSearch(Bench):
    def setup(self):
        self.estimator = ExhaustiveSearch(discrete().iloc[:, :3])

    def time_all_dags(self):
        deque(self.estimator.all_dags(), maxlen=0)

    def time_all_scores(self):
        self.estimator.all_scores()

    def time_estimate(self):
        self.estimator.estimate()


@cache
def marginals(name):
    return sorted(discrete_bn(name).edges())[:N_MARGINALS]


@cache
def mirror_descent_model(name):
    return markov_network(name)


class TimeMirrorDescentEstimator(Bench):
    def setup(self):
        name = DISCRETE
        self.estimator = MirrorDescentEstimator(mirror_descent_model(name), discrete())
        self.marginals = marginals(name)

    def time_estimate(self):
        self.estimator.estimate(marginals=self.marginals, iterations=ITERATIONS, show_progress=False)


@cache
def iv_inputs():
    """(SEM, data, X, Y); X -> Y is the first sorted edge where X has a parent, get_ivs finds one and get_conditional_ivs none."""
    name = GAUSSIAN
    sem = sem_inputs(name, SEM_ERR_CORR)
    model = SEM.from_graph(ebunch=sem["ebunch"], latents=[], err_corr=sem["err_corr"])
    inference = CausalInference(model)
    for x, y in sorted(sem["ebunch"]):
        if (
            list(model.graph.predecessors(x))
            and inference.get_ivs(x, y)
            and not inference.get_conditional_ivs(x, y)
        ):
            return model, gaussian_data(name, ROWS), x, y
    raise ValueError(f"no edge with instruments in {name}")


class TimeIVEstimator(Bench):
    def setup(self):
        model, data, self.x, self.y = iv_inputs()
        self.estimator = IVEstimator(model)
        self.data = data.copy(deep=False)

    def time_fit(self):
        self.estimator.fit(self.x, self.y, self.data)
