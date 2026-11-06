"""Benchmarks for ``pgmpy.independencies``.

Inputs are the frozen assertion lists ``indep_small`` (fast: 5 assertions over 7 variables) and
``indep_large`` (slow: 12 assertions over 12 variables), built with ``_fixtures.independencies``
(``tools/build_fixtures.independencies_spec``: events of 1-2, 1-3 and 0-2 variables, seeded).

``closure`` is exponential in the number of variables and assertions (semi-graphoid axioms applied until
no new assertion appears), and ``entails``, ``is_equivalent`` and ``reduce`` are built on it. The slow
fixture is sized to the per-call cap: ``indep_large`` has a closure of 787 assertions that takes 1.6 s
(``PYTHONHASHSEED=0``). Neighbouring seeds and sizes reach 2 s to 30 s (for example 8 variables and 12
assertions: 22 s), so 12 variables and 12 assertions with seed 1 is the largest chosen that keeps
the heaviest methods under 5 s. Measured slow-tier calls: ``closure`` 1.6 s, ``entails`` 1.6 s (the
entailed set is the first two assertions), ``is_equivalent`` 3.2 s (two closures, against an equal
copy), ``reduce`` 4.5 s. The fast fixture's closure has about 130 assertions and takes 0.05 s
(``is_equivalent`` 0.09 s).

Read-only methods share one ``Independencies`` per tier (memoized). ``add_assertions`` mutates, so it
gets a fresh object on every call. ``contains`` looks up the last assertion of the list.

Skipped:
    ``Independencies.get_assertions`` and ``IndependenceAssertion.get_assertion`` (trivial: return a
    stored attribute), ``Independencies.get_factorized_product`` (stub: its body is ``pass``).
"""

from functools import cache

from benchmarks._fixtures import independencies
from benchmarks._tiers import TIERS, Bench

from pgmpy.independencies import IndependenceAssertion, Independencies

FIXTURE = {"fast": "indep_small", "slow": "indep_large"}


@cache
def shared(tier):
    """The tier's ``Independencies`` and its last assertion; only read by the benchmarks."""
    ind = independencies(FIXTURE[tier])
    return ind, ind.get_assertions()[-1]


class TimeIndependencies(Bench):
    def prepare(self, tier):
        self.ind, self.last = shared(tier)

    def time_init(self, tier):
        Independencies(*[a.get_assertion() for a in self.ind.get_assertions()])

    def time_add_assertions(self, tier):
        independencies(FIXTURE[tier]).add_assertions(*[a.get_assertion() for a in self.ind.get_assertions()[:2]])

    def time_contains(self, tier):
        self.last in self.ind

    def time_get_all_variables(self, tier):
        self.ind.get_all_variables()

    def time_latex_string(self, tier):
        self.ind.latex_string()

    def time_closure(self, tier):
        self.ind.closure()

    def time_entails(self, tier):
        self.ind.entails(Independencies(*self.ind.get_assertions()[:2]))

    def time_is_equivalent(self, tier):
        self.ind.is_equivalent(independencies(FIXTURE[tier]))

    def time_reduce(self, tier):
        self.ind.reduce()


class TimeIndependenceAssertion(Bench):
    def prepare(self, tier):
        _, self.assertion = shared(tier)
        self.events = self.assertion.get_assertion()

    def time_init(self, tier):
        IndependenceAssertion(*self.events)

    def time_latex_string(self, tier):
        self.assertion.latex_string()
