"""Benchmarks for ``pgmpy.ci_tests``.

The CI tests define no public method of their own except ``PearsonrEquivalence.is_independent`` and
``GeneralizedCov.set_params``. All of them share ``is_independent`` and ``run_test`` from
``BaseCITest`` (``__call__`` is ``is_independent``). ``is_independent`` is timed for every test, one
benchmark class per data type with the test name as the second param. ``run_test`` is inherited, so
it is timed once, on ``Pearsonr`` (``TimeBaseCITest``). ``get_ci_test`` is timed by name and with
``None`` (default test for the data type).

Every test is built with ``use_cache=False`` (a cached result would turn every call after the
first into a dictionary lookup) and the instance is memoized per tier and test, since building one
preprocesses the data. Tests that take ``rng`` get ``rng=0``.

The tested triple is fixed: ``X`` the first node in sorted order, ``Y`` the last, and ``Z`` the
first parent of ``Y`` in sorted order, all in the generating model. That gives
``asia``: ``asia _||_ xray | either``; ``alarm``: ``ANAPHYLAXIS _||_ VENTTUBE | DISCONNECT``;
``random_gaussian_8``: ``X_0, X_7 | X_3``; ``arth150``: ``100, 96 | 558`` (node names sort as
strings).

Inputs (``ROWS`` rows unless said otherwise):

* Discrete tests: ``discrete_data`` of ``asia`` (fast) / ``alarm`` (slow).
* Continuous tests: ``gaussian_data`` of ``random_gaussian_8`` (fast) / ``arth150`` (slow).
* Mixed tests: ``mixed_data`` of the same Gaussian models (every other column cut into 3 string
  categories), with the generating DAG from ``gaussian_bn``. The triple rule is valid for all four
  tests: they take any mix of categorical and numeric variables (they residualize ``X`` and ``Y``
  with a random forest, one-hot encoding categorical ones). On ``random_gaussian_8`` ``X`` and ``Z``
  are categorical and ``Y`` numeric, on ``arth150`` only ``X`` is categorical.
* ``IndependenceMatch``: no data. The independencies are the local Markov assertions of every node
  (``model.local_independencies(sorted nodes)``) of the ``asia`` / ``alarm`` model, built once per
  tier outside the timed call. The triple is not one of them, so the lookup scans all of them.

Row caps (``TEST_ROWS``; the slow tier would otherwise exceed the 15 s cap or run out of memory;
sizes measured with ``PYTHONHASHSEED=0``, one call):

* ``GeneralizedCov`` (1000 permutations over random-forest residuals) takes 0.2 s at 1000 rows,
  0.5 s at 3000, 1.8 s at 10000 and 38 s at 100000. The slow tier uses ``FOREST_ROWS`` = 10000.
* The four mixed tests fit random forests on all rows: 0.5 s at 3000 rows, 2.0 s at 10000 and 43 s
  at 100000. The slow tier uses ``FOREST_ROWS`` = 10000.
* ``HSIC`` and ``KCI`` build n x n kernel matrices (100000 rows fail with a 37 GiB allocation) and
  ``KCI`` grows with n cubed: 0.7 s at 2000 rows, 1.9 s at 3000, 3.9 s at 4000 and 7.6 s at 5000
  (``HSIC`` 0.05 / 0.11 / 0.16 / 0.25 s). The slow tier uses ``KERNEL_ROWS`` = 3000. The fast ``KCI``
  at ``ROWS`` = 1000 took 0.32 s per call, above the 0.25 s margin for slower CI runners, so its fast
  tier uses 600 rows (0.085 s; 750 rows 0.15 s, 500 rows 0.05 s).
* ``GCM`` is not kernel-based (it residualizes with linear regression by default): 0.02 s at 100000
  rows, so it keeps ``ROWS``.

Skipped:
    ``GeneralizedCov.set_params`` (trivial: renames the deprecated ``random_state`` and sets an
    attribute).
"""

from functools import cache

from benchmarks._fixtures import discrete_bn, discrete_data, gaussian_bn, gaussian_data, mixed_data
from benchmarks._tiers import ROWS, TIERS, Bench

from pgmpy import ci_tests

DISCRETE = {"fast": "asia", "slow": "alarm"}
GAUSSIAN = {"fast": "random_gaussian_8", "slow": "arth150"}

DISCRETE_TESTS = ["ChiSquare", "GSq", "LogLikelihood", "ModifiedLogLikelihood", "PowerDivergence"]
CONTINUOUS_TESTS = ["FisherZ", "Pearsonr", "PearsonrEquivalence", "GCM", "GeneralizedCov", "HSIC", "KCI"]
MIXED_TESTS = ["HotellingLawley", "PillaiTrace", "RoysLargestRoot", "WilksLambda"]

# Tests that take ``rng``.
SEEDED = {"GeneralizedCov", "HSIC"}

# Rows per tier for the tests that cannot use ``ROWS`` (see the module docstring).
FOREST_ROWS = {"fast": ROWS["fast"], "slow": 10_000}
KERNEL_ROWS = {"fast": ROWS["fast"], "slow": 3_000}
TEST_ROWS = {
    "GeneralizedCov": FOREST_ROWS,
    "HSIC": KERNEL_ROWS,
    "KCI": {**KERNEL_ROWS, "fast": 600},
    **dict.fromkeys(MIXED_TESTS, FOREST_ROWS),
}


def n_rows(tier, name):
    return TEST_ROWS.get(name, ROWS)[tier]


def triple(model):
    """``X`` the first node, ``Y`` the last, ``Z`` the first parent of ``Y`` (sorted order)."""
    nodes = sorted(model.nodes())
    y = nodes[-1]
    return nodes[0], y, [sorted(model.predecessors(y))[0]]


@cache
def discrete_test(tier, name):
    data = discrete_data(DISCRETE[tier], n_rows(tier, name))
    return getattr(ci_tests, name)(data, use_cache=False)


@cache
def continuous_test(tier, name):
    data = gaussian_data(GAUSSIAN[tier], n_rows(tier, name))
    kwargs = {"rng": 0} if name in SEEDED else {}
    return getattr(ci_tests, name)(data, use_cache=False, **kwargs)


@cache
def mixed_test(tier, name):
    data = mixed_data(GAUSSIAN[tier], n_rows(tier, name))
    return getattr(ci_tests, name)(data, use_cache=False)


@cache
def independence_match(tier):
    model = discrete_bn(DISCRETE[tier])
    return ci_tests.IndependenceMatch(model.local_independencies(sorted(model.nodes())), use_cache=False)


class TimeDiscreteCITest(Bench):
    params = [TIERS, DISCRETE_TESTS]
    param_names = ["tier", "test"]

    def prepare(self, tier, name):
        self.test = discrete_test(tier, name)
        self.x, self.y, self.z = triple(discrete_bn(DISCRETE[tier]))

    def time_is_independent(self, tier, name):
        self.test.is_independent(self.x, self.y, self.z)


class TimeContinuousCITest(Bench):
    params = [TIERS, CONTINUOUS_TESTS]
    param_names = ["tier", "test"]

    def prepare(self, tier, name):
        self.test = continuous_test(tier, name)
        self.x, self.y, self.z = triple(gaussian_bn(GAUSSIAN[tier]))

    def time_is_independent(self, tier, name):
        self.test.is_independent(self.x, self.y, self.z)


class TimeMixedCITest(Bench):
    params = [TIERS, MIXED_TESTS]
    param_names = ["tier", "test"]

    def prepare(self, tier, name):
        self.test = mixed_test(tier, name)
        self.x, self.y, self.z = triple(gaussian_bn(GAUSSIAN[tier]))

    def time_is_independent(self, tier, name):
        self.test.is_independent(self.x, self.y, self.z)


class TimeIndependenceMatch(Bench):
    def prepare(self, tier):
        self.test = independence_match(tier)
        self.x, self.y, self.z = triple(discrete_bn(DISCRETE[tier]))

    def time_is_independent(self, tier):
        self.test.is_independent(self.x, self.y, self.z)


class TimeBaseCITest(Bench):
    def prepare(self, tier):
        self.test = continuous_test(tier, "Pearsonr")
        self.x, self.y, self.z = triple(gaussian_bn(GAUSSIAN[tier]))

    def time_run_test(self, tier):
        self.test.run_test(self.x, self.y, self.z)


class TimeCITestFunctions(Bench):
    def prepare(self, tier):
        self.data = gaussian_data(GAUSSIAN[tier], ROWS[tier])

    def time_get_ci_test(self, tier):
        ci_tests.get_ci_test("pearsonr", self.data)

    def time_get_ci_test_default(self, tier):
        ci_tests.get_ci_test(None, self.data)
