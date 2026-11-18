"""Benchmarks for ``pgmpy.sampling``; ``DiscreteBayesianNetwork.simulate`` is timed in ``models.py``.

Skipped:
    GibbsSampling.set_start_state: trivial
"""

import warnings
from functools import cache

from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelInference, BayesianModelSampling, GibbsSampling

ROWS = 1_000
DISCRETE = "asia"
MODEL = "alarm"
N_QUERY = 4
N_EVIDENCE = 2
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=UserWarning, message="At least one state name was not found")


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


def first_states(model, nodes):
    return {n: model.states[n][0] for n in nodes}


def largest_cpd(model):
    """CPD with the most values, ties broken by variable name."""
    return max(discrete_bn(model).cpds, key=lambda c: (c.values.size, str(c.variable)))


def split(nodes, roots):
    return [n for n in nodes if n not in roots][:N_QUERY], nodes[-N_EVIDENCE:]


@cache
def query_and_evidence(name):
    """The 4 first non-root nodes (sorted) and the last 2 nodes (sorted) at their first state."""
    m = discrete_bn.shared(name)
    query, last = split(sorted(m.nodes), set(m.get_roots()))
    return query, first_states(m, last)


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
for _name in (DISCRETE, MODEL):
    discrete_bn(_name)


REJECTION_ROWS = ROWS // 50
GIBBS_ROWS = 500


class TimeBayesianModelSampling(Bench):
    def setup(self):
        name = MODEL
        self.model = discrete_bn(name)
        self.sampler = BayesianModelSampling(self.model)
        self.evidence = [State(var, state) for var, state in query_and_evidence(name)[1].items()]

    def time_init(self):
        BayesianModelSampling(self.model)

    def time_forward_sample(self):
        self.sampler.forward_sample(ROWS, rng=0, show_progress=False)

    def time_rejection_sample(self):
        self.sampler.rejection_sample(self.evidence, REJECTION_ROWS, rng=0, show_progress=False)

    def time_likelihood_weighted_sample(self):
        self.sampler.likelihood_weighted_sample(self.evidence, ROWS, rng=0, show_progress=False)


class TimeGibbsSampling(Bench):
    def setup(self):
        self.model = discrete_bn(DISCRETE)
        self.sampler = GibbsSampling(self.model)
        self.start = [(var, 0) for var in self.model.nodes]

    def time_init(self):
        GibbsSampling(self.model)

    def time_sample(self):
        self.sampler.sample(start_state=self.start, size=GIBBS_ROWS, rng=0)

    def time_generate_sample(self):
        for _ in self.sampler.generate_sample(start_state=self.start, size=GIBBS_ROWS, rng=0):
            pass


class TimeBayesianModelInference(Bench):
    def setup(self):
        name = MODEL
        self.inference = BayesianModelInference(discrete_bn(name))
        self.variable = largest_cpd(name).variable

    def time_pre_compute_reduce(self):
        self.inference.pre_compute_reduce(self.variable)

    def time_pre_compute_reduce_maps(self):
        self.inference.pre_compute_reduce_maps(self.variable)
