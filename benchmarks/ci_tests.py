"""Benchmarks for ``pgmpy.ci_tests``: ``is_independent`` for every test (param = test name), ``run_test`` once on ``Pearsonr``, ``get_ci_test``.

Skipped:
    GeneralizedCov.set_params: trivial
"""

from functools import cache

from benchmarks._fixtures import discrete_bn, discrete_data, gaussian_bn, gaussian_data, mixed_data
from benchmarks._bench import DISCRETE, GAUSSIAN, ROWS, Bench

from pgmpy import ci_tests

SEEDED = {"GeneralizedCov", "HSIC"}

TEST_ROWS = {"KCI": 600}  # kernel matrices are cubic in n; 1,000 rows took 0.32 s


def triple(model):
    nodes = sorted(model.nodes())
    y = nodes[-1]
    return nodes[0], y, [sorted(model.predecessors(y))[0]]


@cache
def ci_test(name, data, source):
    kwargs = {"rng": 0} if name in SEEDED else {}
    return getattr(ci_tests, name)(data(source, TEST_ROWS.get(name, ROWS)), use_cache=False, **kwargs)


@cache
def independence_match():
    model = discrete_bn(DISCRETE)
    return ci_tests.IndependenceMatch(model.local_independencies(sorted(model.nodes())), use_cache=False)


class _CITest(Bench):
    param_names = ["test"]

    def setup(self, name):
        bn, data, source = self.inputs
        self.test = ci_test(name, data, source)
        self.x, self.y, self.z = triple(bn(source))

    def time_is_independent(self, name):
        self.test.is_independent(self.x, self.y, self.z)


class TimeDiscreteCITest(_CITest):
    params = [["ChiSquare", "GSq", "LogLikelihood", "ModifiedLogLikelihood", "PowerDivergence"]]
    inputs = discrete_bn, discrete_data, DISCRETE


class TimeContinuousCITest(_CITest):
    params = [["FisherZ", "Pearsonr", "PearsonrEquivalence", "GCM", "GeneralizedCov", "HSIC", "KCI"]]
    inputs = gaussian_bn, gaussian_data, GAUSSIAN


class TimeMixedCITest(_CITest):
    params = [["HotellingLawley", "PillaiTrace", "RoysLargestRoot", "WilksLambda"]]
    inputs = gaussian_bn, mixed_data, GAUSSIAN


class TimeIndependenceMatch(Bench):
    def setup(self):
        self.test = independence_match()
        self.x, self.y, self.z = triple(discrete_bn(DISCRETE))

    def time_is_independent(self):
        self.test.is_independent(self.x, self.y, self.z)


class TimeBaseCITest(Bench):
    def setup(self):
        self.test = ci_test("Pearsonr", gaussian_data, GAUSSIAN)
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
