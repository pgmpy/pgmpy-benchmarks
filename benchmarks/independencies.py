"""Benchmarks for ``pgmpy.independencies``. Inputs are ``independencies()``: ``closure`` is exponential, so the
fixture is 5 assertions over 7 variables.

Skipped:
    Independencies.get_assertions, IndependenceAssertion.get_assertion, Independencies.get_factorized_product (stub): trivial
"""

from functools import cache

import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.independencies import IndependenceAssertion, Independencies


@fresh
def independencies():
    rng = np.random.default_rng(0)
    names = [f"V{i}" for i in range(7)]
    assertions = []
    while len(assertions) < 5:
        a, b, c = int(rng.integers(1, 3)), int(rng.integers(1, 4)), int(rng.integers(0, 3))
        pick = [str(v) for v in rng.choice(names, size=a + b + c, replace=False)]
        assertion = [sorted(pick[:a]), sorted(pick[a : a + b]), sorted(pick[a + b :])]
        if assertion not in assertions:
            assertions.append(assertion)
    return Independencies(*assertions)


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
