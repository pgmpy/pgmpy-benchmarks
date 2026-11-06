"""Benchmarks for ``pgmpy.parameter_estimator``.

Each class defines only ``fit`` itself (the constructors, ``_estimate_cpd`` and the other helpers are
private or trivial), so every benchmark class times ``fit``. ``DiscreteBayesianNetwork.fit`` and
``fit_update`` are timed in ``models.py`` and not repeated here. Every estimator gets
``show_progress=False`` / ``rng=0`` wherever it accepts them, and each call gets a new estimator
(``fit`` stores its results on the estimator). The model structure is read-only for ``fit``
(``DiscreteEM.fit`` copies it), so one bare structure per fixture is shared by all calls.

Inputs:

* ``DiscreteMLE`` / ``DiscreteBayesianEstimator``: the structure of ``asia`` (fast) / ``alarm``
  (slow) with ``ROWS[tier]`` rows of ``discrete_data`` (1000 / 100000). The Bayesian estimator uses
  the default BDeu prior with ``equivalent_sample_size=5``.
* ``DiscreteEM``: the same structure and data, but the lexicographically first non-root node
  (``asia``: ``bronc``; ``alarm``: ``ARTCO2``) is latent: it is marked latent in the model and
  dropped from the data. Each EM iteration scores every distinct row of the data under every latent
  state in pure Python, so cost grows with the number of distinct rows, not just with rows. Fixed
  ``max_iter=EM_ITERATIONS`` (default atol never triggers early exit, so the work is fixed) with
  ``rng=0``. The fast tier uses 1000 rows (0.13 s). The slow tier uses 200 rows of
  ``alarm`` data, not 100000 (1.4 s): the pure-Python E-step makes 5 iterations take 5.7 s at 1000
  rows (0.9 / 1.4 / 2.0 s at 100 / 200 / 300 rows), and 100000 rows are far out of reach.
  ``ConvergenceWarning`` for stopping at ``max_iter`` is silenced by a module-level filter.
* ``LinearGaussianMLE``: the structure of ``random_gaussian_8`` (fast) / ``arth150`` (slow) with
  ``ROWS[tier]`` rows of ``gaussian_data``.

Skipped:
    Constructors and ``_``-prefixed helpers (trivial / private); the abstract bases
    ``BaseParameterEstimator``, ``DiscreteParameterEstimator`` and ``GaussianParameterEstimator``
    (their ``fit`` is covered once through the concrete classes); ``DiscreteEM.set_params`` (trivial).
"""

import warnings
from functools import cache

from benchmarks._fixtures import discrete_bn, discrete_data, gaussian_bn, gaussian_data
from benchmarks._tiers import ROWS, Bench
from sklearn.exceptions import ConvergenceWarning

from pgmpy.models import DiscreteBayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteEM, DiscreteMLE, LinearGaussianMLE

warnings.filterwarnings("ignore", category=ConvergenceWarning, message=r".*max_iter.*")

DISCRETE_MODEL = {"fast": "asia", "slow": "alarm"}
GAUSSIAN_MODEL = {"fast": "random_gaussian_8", "slow": "arth150"}
EM_ITERATIONS = 5
EM_ROWS = {"fast": 1_000, "slow": 200}
EQUIVALENT_SAMPLE_SIZE = 5


@cache
def discrete_structure(name):
    """Bare structure (no CPDs) of a discrete fixture; read-only for ``fit``."""
    return DiscreteBayesianNetwork(discrete_bn(name).edges())


@cache
def discrete_frame(name, n_rows):
    return discrete_data(name, n_rows)


@cache
def gaussian_structure(name):
    return LinearGaussianBayesianNetwork(gaussian_bn(name).edges())


@cache
def gaussian_frame(name, n_rows):
    return gaussian_data(name, n_rows)


@cache
def latent_variable(name):
    """Lexicographically first node with at least one parent."""
    model = discrete_bn(name)
    return min(n for n in model.nodes() if model.get_parents(n))


@cache
def em_inputs(name, n_rows):
    """Structure with the latent variable marked, and the data without its column."""
    latent = latent_variable(name)
    edges = list(discrete_bn(name).edges())
    model = DiscreteBayesianNetwork(edges, latents={latent})
    return model, discrete_frame(name, n_rows).drop(columns=[latent])


class TimeDiscreteMLE(Bench):
    def prepare(self, tier):
        name = DISCRETE_MODEL[tier]
        self.model = discrete_structure(name)
        self.data = discrete_frame(name, ROWS[tier])

    def time_fit(self, tier):
        DiscreteMLE().fit(self.model, self.data)


class TimeDiscreteBayesianEstimator(Bench):
    def prepare(self, tier):
        name = DISCRETE_MODEL[tier]
        self.model = discrete_structure(name)
        self.data = discrete_frame(name, ROWS[tier])

    def time_fit(self, tier):
        DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=EQUIVALENT_SAMPLE_SIZE).fit(
            self.model, self.data
        )


class TimeDiscreteEM(Bench):
    def prepare(self, tier):
        self.model, self.data = em_inputs(DISCRETE_MODEL[tier], EM_ROWS[tier])

    def time_fit(self, tier):
        DiscreteEM(max_iter=EM_ITERATIONS, rng=0, show_progress=False).fit(self.model, self.data)


class TimeLinearGaussianMLE(Bench):
    def prepare(self, tier):
        name = GAUSSIAN_MODEL[tier]
        self.model = gaussian_structure(name)
        self.data = gaussian_frame(name, ROWS[tier])

    def time_fit(self, tier):
        LinearGaussianMLE().fit(self.model, self.data)
