"""Benchmarks for ``pgmpy.ci_tests``: ``is_independent`` for every test (param = test name), ``run_test`` once on ``Pearsonr``, ``get_ci_test``.

``use_cache=False`` (a cache hit is a dict lookup); tests taking ``rng`` get ``rng=0``. Triple: X first node, Y last, Z first parent of Y.
``KCI`` uses 600 rows: 1000 rows took 0.32 s per call (kernel matrices, cubic in n).

Skipped:
    GeneralizedCov.set_params: trivial
"""

from functools import cache

from benchmarks._fixtures import discrete_bn, discrete_data, gaussian_bn, gaussian_data, mixed_data
from benchmarks._bench import DISCRETE, GAUSSIAN, ROWS, Bench

from pgmpy import ci_tests

DISCRETE_TESTS = ["ChiSquare", "GSq", "LogLikelihood", "ModifiedLogLikelihood", "PowerDivergence"]
CONTINUOUS_TESTS = ["FisherZ", "Pearsonr", "PearsonrEquivalence", "GCM", "GeneralizedCov", "HSIC", "KCI"]
MIXED_TESTS = ["HotellingLawley", "PillaiTrace", "RoysLargestRoot", "WilksLambda"]

SEEDED = {"GeneralizedCov", "HSIC"}

TEST_ROWS = {"KCI": 600}


def n_rows(name):
    return TEST_ROWS.get(name, ROWS)


def triple(model):
    """``X`` the first node, ``Y`` the last, ``Z`` the first parent of ``Y`` (sorted order)."""
    nodes = sorted(model.nodes())
    y = nodes[-1]
    return nodes[0], y, [sorted(model.predecessors(y))[0]]


@cache
def discrete_test(name):
    data = discrete_data(DISCRETE, n_rows(name))
    return getattr(ci_tests, name)(data, use_cache=False)


@cache
def continuous_test(name):
    data = gaussian_data(GAUSSIAN, n_rows(name))
    kwargs = {"rng": 0} if name in SEEDED else {}
    return getattr(ci_tests, name)(data, use_cache=False, **kwargs)


@cache
def mixed_test(name):
    data = mixed_data(GAUSSIAN, n_rows(name))
    return getattr(ci_tests, name)(data, use_cache=False)


@cache
def independence_match():
    model = discrete_bn(DISCRETE)
    return ci_tests.IndependenceMatch(model.local_independencies(sorted(model.nodes())), use_cache=False)


class TimeDiscreteCITest(Bench):
    params = [DISCRETE_TESTS]
    param_names = ["test"]

    def setup(self, name):
        self.test = discrete_test(name)
        self.x, self.y, self.z = triple(discrete_bn(DISCRETE))

    def time_is_independent(self, name):
        self.test.is_independent(self.x, self.y, self.z)


class TimeContinuousCITest(Bench):
    params = [CONTINUOUS_TESTS]
    param_names = ["test"]

    def setup(self, name):
        self.test = continuous_test(name)
        self.x, self.y, self.z = triple(gaussian_bn(GAUSSIAN))

    def time_is_independent(self, name):
        self.test.is_independent(self.x, self.y, self.z)


class TimeMixedCITest(Bench):
    params = [MIXED_TESTS]
    param_names = ["test"]

    def setup(self, name):
        self.test = mixed_test(name)
        self.x, self.y, self.z = triple(gaussian_bn(GAUSSIAN))

    def time_is_independent(self, name):
        self.test.is_independent(self.x, self.y, self.z)


class TimeIndependenceMatch(Bench):
    def setup(self):
        self.test = independence_match()
        self.x, self.y, self.z = triple(discrete_bn(DISCRETE))

    def time_is_independent(self):
        self.test.is_independent(self.x, self.y, self.z)


class TimeBaseCITest(Bench):
    def setup(self):
        self.test = continuous_test("Pearsonr")
        self.x, self.y, self.z = triple(gaussian_bn(GAUSSIAN))

    def time_run_test(self):
        self.test.run_test(self.x, self.y, self.z)


class TimeCITestFunctions(Bench):
    def setup(self):
        self.data = gaussian_data(GAUSSIAN, ROWS)

    def time_get_ci_test(self):
        ci_tests.get_ci_test("pearsonr", self.data)

    def time_get_ci_test_default(self):
        ci_tests.get_ci_test(None, self.data)
