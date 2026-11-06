"""Benchmarks for the non-deprecated ``pgmpy.estimators``.

Every call passes ``show_progress=False`` wherever the method accepts it, and CI exports
``PYTHONHASHSEED=0``. Measured numbers below use that setup. None of the four classes emits a
deprecation warning on construction (checked with the harness's rule: only "``name`` is deprecated"
messages count).

Inputs:

* ``MmhcEstimator``: the discrete inputs of ``causal_discovery.py`` (``DISCRETE`` model, ``ROWS``
  rows: ``asia`` with 1000 rows fast, ``alarm`` with 100000 rows slow), cut to the first
  ``MMHC_COLUMNS`` columns. ``mmpc`` takes the minimum association over every subset of the
  neighbours found so far, so its cost is exponential in the number of variables, and the full
  37-variable ``alarm`` does not finish in minutes, even at 100 rows. Rows do not fix it, columns
  do: the first 8 / 10 / 12 / 14 columns of ``alarm`` at 100000 rows take 0.2 / 1.4 / 6.5 / 27 s.
  The slow tier uses 10 columns (``mmpc`` 1.4 s, ``estimate`` 1.5 s, which includes ``mmpc``); the
  fast tier uses all 8 columns of ``asia`` (0.05 s each).
* ``ExhaustiveSearch``: ``discrete_data`` of ``asia`` cut to the first ``EXHAUSTIVE_COLUMNS`` = 3
  (fast, 25 DAGs) / 4 (slow, 543 DAGs) columns. The number of DAGs grows super-exponentially. With 5
  slow columns (29281 DAGs) ``all_dags``, ``all_scores`` and ``estimate`` each took about 10.5 s (85 s of
  asv time apiece), which put the slow tier at 26.4 min, above its 25-minute budget (ruling R22); 4
  columns take 0.045 s at 100000 rows (0.039 s at 1000 rows), 3 columns 2-3 ms. Rows barely change it,
  since almost all of the time is generating the DAGs. Five and six columns are out of reach. Each call gets a new estimator, because the structure score
  caches local scores.
* ``MirrorDescentEstimator``: the ``asia`` / ``alarm`` Markov network (``_fixtures.markov_network``)
  with ``discrete_data``. The marginals are the first ``N_MARGINALS`` sorted edges of the true
  network, each as a pair, and the optimisation runs ``ITERATIONS`` iterations with the default
  line search (0.13 s fast, 0.55 s slow). Each call gets a new estimator, which owns the belief
  propagation state that ``estimate`` updates. The ``RuntimeWarning`` "invalid value encountered in
  divide" from the factor division in belief propagation on ``alarm`` is silenced by a module-level
  filter on that exact message.
* ``IVEstimator``: the structure of ``random_gaussian_8`` / ``arth150`` from ``_fixtures.sem_inputs``
  (its pruned edges and correlated errors) as a ``SEM`` with no latent variables, and
  ``gaussian_data`` of the same model. ``sem_inputs`` itself makes every node with children latent,
  so no observed variable could have an observed parent and ``X -> Y`` between observed variables
  would not exist; with ``latents=[]`` every edge can be estimated. ``X -> Y`` is the first sorted
  edge ``(X, Y)`` of the model such that ``X`` has a parent, ``get_ivs(X, Y)`` finds at least one
  instrumental variable, and ``get_conditional_ivs(X, Y)`` finds none (a conditional IV that is
  also a parent of ``X`` makes ``fit`` fail on duplicate regressors). That is ``X_3 -> X_7`` (6
  instruments) and ``111 -> 496`` (67 instruments). ``fit`` adds a column to the data it is given,
  so each call gets a shallow copy of the data (the new column lands on the copy only).

Skipped:
    ``MmhcEstimator.__init__`` and the other constructors (trivial).
"""

import warnings
from collections import deque
from functools import cache

from benchmarks._fixtures import discrete_bn, discrete_data, gaussian_data, markov_network, sem_inputs
from benchmarks._tiers import ROWS, Bench
from benchmarks.causal_discovery import DISCRETE, discrete
from benchmarks.models import GAUSSIAN_MODEL, SEM_ERR_CORR

from pgmpy.estimators import ExhaustiveSearch, IVEstimator, MirrorDescentEstimator, MmhcEstimator
from pgmpy.inference import CausalInference
from pgmpy.models import SEM

warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"invalid value encountered in divide")

MMHC_COLUMNS = {"fast": 8, "slow": 10}
EXHAUSTIVE_COLUMNS = {"fast": 3, "slow": 4}
N_MARGINALS = 5
ITERATIONS = 100


class TimeMmhcEstimator(Bench):
    def prepare(self, tier):
        self.estimator = MmhcEstimator(discrete(tier).iloc[:, : MMHC_COLUMNS[tier]])

    def time_estimate(self, tier):
        self.estimator.estimate()

    def time_mmpc(self, tier):
        self.estimator.mmpc()


class TimeExhaustiveSearch(Bench):
    def prepare(self, tier):
        self.estimator = ExhaustiveSearch(discrete_data("asia", ROWS[tier]).iloc[:, : EXHAUSTIVE_COLUMNS[tier]])

    def time_all_dags(self, tier):
        deque(self.estimator.all_dags(), maxlen=0)

    def time_all_scores(self, tier):
        self.estimator.all_scores()

    def time_estimate(self, tier):
        self.estimator.estimate()


@cache
def marginals(name):
    return sorted(discrete_bn(name).edges())[:N_MARGINALS]


@cache
def mirror_descent_model(name):
    return markov_network(name)


class TimeMirrorDescentEstimator(Bench):
    def prepare(self, tier):
        name = DISCRETE[tier]
        self.estimator = MirrorDescentEstimator(mirror_descent_model(name), discrete(tier))
        self.marginals = marginals(name)

    def time_estimate(self, tier):
        self.estimator.estimate(marginals=self.marginals, iterations=ITERATIONS, show_progress=False)


@cache
def iv_inputs(tier):
    """The ``SEM``, its data and the (X, Y) pair described in the module docstring."""
    name = GAUSSIAN_MODEL[tier]
    sem = sem_inputs(name, SEM_ERR_CORR[tier])
    model = SEM.from_graph(ebunch=sem["ebunch"], latents=[], err_corr=sem["err_corr"])
    inference = CausalInference(model)
    for x, y in sorted(sem["ebunch"]):
        if (
            list(model.graph.predecessors(x))
            and inference.get_ivs(x, y)
            and not inference.get_conditional_ivs(x, y)
        ):
            return model, gaussian_data(name, ROWS[tier]), x, y
    raise ValueError(f"no edge with instruments in {name}")


class TimeIVEstimator(Bench):
    def prepare(self, tier):
        model, data, self.x, self.y = iv_inputs(tier)
        self.estimator = IVEstimator(model)
        self.data = data.copy(deep=False)

    def time_fit(self, tier):
        self.estimator.fit(self.x, self.y, self.data)
