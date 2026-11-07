"""Benchmarks for ``pgmpy.sampling``; ``DiscreteBayesianNetwork.simulate`` is timed in ``models.py``.

``GibbsSampling`` builds a kernel over all assignments (exponential), so it runs on ``asia`` with ``GIBBS_ROWS`` rows (a Python loop per row)
and a fresh sampler per call (the state is mutated). Rejection sampling is slow with the standard evidence, so it draws ``ROWS // 50``.
``BayesianModelInference`` is timed on the variable with the largest CPD in ``alarm``. Evidence follows ``inference.query_and_evidence``.

Skipped:
    GibbsSampling.set_start_state: trivial
"""

import warnings

from benchmarks._fixtures import discrete_bn
from benchmarks._bench import DISCRETE, MODEL, ROWS, Bench
from benchmarks.factors import largest_cpd
from benchmarks.inference import query_and_evidence

from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelInference, BayesianModelSampling, GibbsSampling

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=UserWarning, message="At least one state name was not found")

REJECTION_ROWS = ROWS // 50
GIBBS_ROWS = 500


class TimeBayesianModelSampling(Bench):
    def setup(self):
        name = MODEL
        self.model = discrete_bn(name)
        self.sampler = BayesianModelSampling(self.model)
        self.evidence = [State(var, state) for var, state in query_and_evidence(name)[1].items()]
        self.n_rows = ROWS
        self.n_rejection_rows = REJECTION_ROWS

    def time_init(self):
        BayesianModelSampling(self.model)

    def time_forward_sample(self):
        self.sampler.forward_sample(self.n_rows, rng=0, show_progress=False)

    def time_rejection_sample(self):
        self.sampler.rejection_sample(self.evidence, self.n_rejection_rows, rng=0, show_progress=False)

    def time_likelihood_weighted_sample(self):
        self.sampler.likelihood_weighted_sample(self.evidence, self.n_rows, rng=0, show_progress=False)


class TimeGibbsSampling(Bench):
    def setup(self):
        self.model = discrete_bn(DISCRETE)
        self.sampler = GibbsSampling(self.model)
        self.start = [(var, 0) for var in self.model.nodes]
        self.n_rows = GIBBS_ROWS

    def time_init(self):
        GibbsSampling(self.model)

    def time_sample(self):
        self.sampler.sample(start_state=self.start, size=self.n_rows, rng=0)

    def time_generate_sample(self):
        for _ in self.sampler.generate_sample(start_state=self.start, size=self.n_rows, rng=0):
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
