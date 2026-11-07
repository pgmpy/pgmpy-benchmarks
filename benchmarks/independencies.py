"""Benchmarks for ``pgmpy.independencies``.

Inputs are the frozen assertion list ``indep_small`` (5 assertions over 7 variables), built with ``_fixtures.independencies``
(``tools/build_fixtures.independencies_spec``: events of 1-2, 1-3 and 0-2 variables, seeded).

``closure`` is exponential in the number of variables and assertions (semi-graphoid axioms applied until
no new assertion appears), and ``entails``, ``is_equivalent`` and ``reduce`` are built on it. The
fixture's closure has about 130 assertions and takes 0.05 s (``PYTHONHASHSEED=0``;
``is_equivalent`` 0.09 s, two closures against an equal copy; ``entails`` is checked against the
first two assertions).

Read-only methods share one ``Independencies`` (memoized). ``add_assertions`` mutates, so it
gets a fresh object on every call. ``contains`` looks up the last assertion of the list.

Skipped:
    ``Independencies.get_assertions`` and ``IndependenceAssertion.get_assertion`` (trivial: return a
    stored attribute), ``Independencies.get_factorized_product`` (stub: its body is ``pass``).
"""

from functools import cache

from benchmarks._fixtures import independencies
from benchmarks._bench import Bench

from pgmpy.independencies import IndependenceAssertion, Independencies

FIXTURE = "indep_small"


@cache
def shared():
    """The ``Independencies`` and its last assertion; only read by the benchmarks."""
    ind = independencies(FIXTURE)
    return ind, ind.get_assertions()[-1]


class TimeIndependencies(Bench):
    def setup(self):
        self.ind, self.last = shared()

    def time_init(self):
        Independencies(*[a.get_assertion() for a in self.ind.get_assertions()])

    def time_add_assertions(self):
        independencies(FIXTURE).add_assertions(*[a.get_assertion() for a in self.ind.get_assertions()[:2]])

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
        self.ind.is_equivalent(independencies(FIXTURE))

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
