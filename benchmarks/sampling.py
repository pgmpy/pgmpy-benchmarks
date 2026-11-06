"""Benchmarks for ``pgmpy.sampling``.

Classes: ``BayesianModelSampling``, ``GibbsSampling`` and ``BayesianModelInference`` (also exported
by ``pgmpy.sampling``). ``DiscreteBayesianNetwork.simulate`` is timed by ``models.py``; this module
times the sampler classes themselves.

``BayesianModelInference`` (``pre_compute_reduce``, ``pre_compute_reduce_maps``) is timed on the
variable whose CPD has the most values (``largest_cpd`` from ``benchmarks.factors``) in ``alarm``
(fast, 54 parent state combinations, 1 ms) and ``pathfinder`` (slow, 2016 combinations, 20 to 60 ms).
Its constructor is not timed: the base class ``Inference`` is out of scope for this module.

``BayesianModelSampling`` (``forward_sample``, ``rejection_sample``, ``likelihood_weighted_sample``)
uses ``alarm`` in both tiers (``MODEL``). The brief allowed ``andes`` in the slow tier if ``alarm``
at 100000 rows took under 1 s; it takes 1.0 s (``andes`` 6.5 s), so ``alarm`` stays
(PYTHONHASHSEED=0). Forward and likelihood-weighted sampling draw ``ROWS[tier]`` rows. Evidence for
rejection and likelihood-weighted sampling follows the inference rule (``query_and_evidence`` in
``benchmarks.inference``): the 2 lexicographically last nodes, each at its first state. On ``alarm``
that evidence is ``VENTMACH`` and ``VENTTUBE`` at ``ZERO``. Rejection sampling with it costs 40 s for
100000 rows (3.4 s for 10000), over the 15 s per-call cap. Rejection sampling therefore draws ``REJECTION_ROWS[tier]`` rows instead (1/50 of ``ROWS``,
about 0.6 s in the slow tier). Sampling does not change a ``BayesianModelSampling``. All calls pass ``rng=0`` and ``show_progress=False``.

``GibbsSampling`` builds a transition kernel for every variable over every assignment of all the other
variables, so its constructor is exponential in the number of variables: ``alarm`` (37 variables)
cannot be built. Both tiers use ``asia`` (8 binary variables, ``GIBBS_MODEL``; the constructor takes
about 7 ms). ``sample`` and ``generate_sample`` run one Python loop iteration per row, about 4000
rows per second, so they draw ``GIBBS_ROWS[tier]`` rows (500 / 4000) instead of ``ROWS``. Both start
from the all-zeros state, so the cost of drawing a random start state is not timed. Both methods update the sampler's current state, so ``prepare`` builds a fresh sampler.
``sample`` draws a tqdm progress bar that ``show_progress`` and pgmpy's config do not control; the
CI scripts set ``TQDM_DISABLE=1`` (with ``PYTHONHASHSEED``), which hides it. Building the kernel of
``asia`` emits a ``RuntimeWarning`` (0/0 for impossible assignments) and a ``UserWarning`` ("At least
one state name was not found"); module-level filters below ignore exactly those two messages. The
``Pandas4Warning`` from ``drop(..., inplace=True)`` inside ``GibbsSampling.sample`` is pgmpy's own
third-party deprecation and is left alone.

Skipped:
    GibbsSampling.set_start_state: trivial (inherited from MarkovChain; sets the stored state)
"""

import warnings

from benchmarks._fixtures import discrete_bn
from benchmarks._tiers import ROWS, Bench
from benchmarks.factors import largest_cpd
from benchmarks.inference import query_and_evidence

from pgmpy.factors.discrete import State
from pgmpy.sampling import BayesianModelInference, BayesianModelSampling, GibbsSampling

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=UserWarning, message="At least one state name was not found")

MODEL = {"fast": "alarm", "slow": "alarm"}
GIBBS_MODEL = {"fast": "asia", "slow": "asia"}
INFERENCE_MODEL = {"fast": "alarm", "slow": "pathfinder"}

# Rejection sampling of 100000 alarm rows takes 40 s.
REJECTION_ROWS = {tier: rows // 50 for tier, rows in ROWS.items()}
GIBBS_ROWS = {"fast": 500, "slow": 4_000}


class TimeBayesianModelSampling(Bench):
    def prepare(self, tier):
        name = MODEL[tier]
        self.model = discrete_bn(name)
        self.sampler = BayesianModelSampling(self.model)
        self.evidence = [State(var, state) for var, state in query_and_evidence(name)[1].items()]
        self.n_rows = ROWS[tier]
        self.n_rejection_rows = REJECTION_ROWS[tier]

    def time_init(self, tier):
        BayesianModelSampling(self.model)

    def time_forward_sample(self, tier):
        self.sampler.forward_sample(self.n_rows, rng=0, show_progress=False)

    def time_rejection_sample(self, tier):
        self.sampler.rejection_sample(self.evidence, self.n_rejection_rows, rng=0, show_progress=False)

    def time_likelihood_weighted_sample(self, tier):
        self.sampler.likelihood_weighted_sample(self.evidence, self.n_rows, rng=0, show_progress=False)


class TimeGibbsSampling(Bench):
    def prepare(self, tier):
        self.model = discrete_bn(GIBBS_MODEL[tier])
        self.sampler = GibbsSampling(self.model)
        self.start = [(var, 0) for var in self.model.nodes]
        self.n_rows = GIBBS_ROWS[tier]

    def time_init(self, tier):
        GibbsSampling(self.model)

    def time_sample(self, tier):
        self.sampler.sample(start_state=self.start, size=self.n_rows, rng=0)

    def time_generate_sample(self, tier):
        for _ in self.sampler.generate_sample(start_state=self.start, size=self.n_rows, rng=0):
            pass


class TimeBayesianModelInference(Bench):
    def prepare(self, tier):
        name = INFERENCE_MODEL[tier]
        self.inference = BayesianModelInference(discrete_bn(name))
        self.variable = largest_cpd(name).variable

    def time_pre_compute_reduce(self, tier):
        self.inference.pre_compute_reduce(self.variable)

    def time_pre_compute_reduce_maps(self, tier):
        self.inference.pre_compute_reduce_maps(self.variable)
