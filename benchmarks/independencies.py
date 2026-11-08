"""Benchmarks for ``pgmpy.independencies``. Inputs are ``_fixtures.independencies``: ``closure`` is exponential, so the
fixture is 5 assertions over 7 variables.

Skipped:
    Independencies.get_assertions, IndependenceAssertion.get_assertion, Independencies.get_factorized_product (stub): trivial
"""

from functools import cache

from benchmarks._fixtures import independencies
from benchmarks._bench import Bench

from pgmpy.independencies import IndependenceAssertion, Independencies


@cache
def shared():
    """The ``Independencies`` and its last assertion; only read by the benchmarks."""
    ind = independencies()
    return ind, ind.get_assertions()[-1]


class TimeIndependencies(Bench):
    def setup(self):
        self.ind, self.last = shared()

    def time_init(self):
        Independencies(*[a.get_assertion() for a in self.ind.get_assertions()])

    def time_add_assertions(self):
        independencies().add_assertions(*[a.get_assertion() for a in self.ind.get_assertions()[:2]])

    def time_contains(self):
        self.last in self.ind

    def time_get_all_variables(self):
        self.ind.get_all_variables()

    def time_latex_string(self):
        self.ind.latex_string()

    def time_closure(self):
        self.ind.closure()

    def time_entails(self):
        self.ind.entails(Independencies(*self.ind.get_assertions()[:2]))

    def time_is_equivalent(self):
        self.ind.is_equivalent(independencies())

    def time_reduce(self):
        self.ind.reduce()


class TimeIndependenceAssertion(Bench):
    def setup(self):
        _, self.assertion = shared()
        self.events = self.assertion.get_assertion()

    def time_init(self):
        IndependenceAssertion(*self.events)

    def time_latex_string(self):
        self.assertion.latex_string()
