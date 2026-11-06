"""Benchmarks for ``pgmpy.factors``.

Classes: ``DiscreteFactor``, ``TabularCPD``, ``NoisyORCPD``, ``JointProbabilityDistribution``,
``LinearGaussianCPD``, ``FactorSet`` and ``FactorDict``, plus the module-level functions
(``factor_product``, ``factor_divide``, ``factor_sum_product``, ``factorset_product``,
``factorset_divide``) in ``TimeFactorFunctions``.

Fixtures (fast / slow):

* ``DiscreteFactor``: ``random_factor(4, 3)`` / ``random_factor(12, 3)``. The binary methods use a
  second factor over the first half of the variables.
* ``TabularCPD``: the CPD with the most values (ties by variable name) in ``alarm`` / ``pathfinder``.
* ``NoisyORCPD``: 4 / 16 parents; ``JointProbabilityDistribution``: a binary chain over 4 / 16
  variables from ``_fixtures.joint_chain``; ``LinearGaussianCPD``: 3 / 200 evidence variables.
* ``FactorSet`` / ``FactorDict`` / the ``factor*`` functions: the CPD factors of ``alarm`` /
  ``munin``.

Exponential or huge operations run on reduced inputs, documented where they are defined below:

* ``JointProbabilityDistribution.get_independencies`` and ``minimal_imap`` enumerate subsets of
  variables, so they use a chain over ``SMALL_JOINT`` variables instead of 4 / 16.
* ``FactorDict.product`` and ``factor_product`` multiply a prefix of the CPD factors (sorted by
  scope) whose joint state space stays under ``PRODUCT_CELLS``, since the product of every CPD of
  a network is astronomically large.
* ``factor_sum_product`` sums out all but one variable over the first ``SUM_PRODUCT_FACTORS``
  factors (30 / 600) only: on all 1041 munin factors it does not finish and exhausts memory.
* ``FactorDict.from_dataframe`` uses ``alarm`` in both tiers (1000 / 100000 rows) because copying
  100000 rows of the 1041-column ``munin`` data costs seconds before every call.

``TabularCPD`` inherits most methods from ``DiscreteFactor``; only those it defines or overrides are
benchmarked on it, except ``is_valid_cpd``: it is defined on ``DiscreteFactor`` but calls
``to_factor``, which only exists on CPDs, so it fails on a plain factor and is timed on
``TabularCPD`` instead. ``NoisyORCPD`` defines no methods, so only its constructor is timed. Mutating
methods default to ``inplace=True``; ``prepare`` builds fresh objects before every call.

Skipped:
    DiscreteFactor.scope, TabularCPD.get_evidence, FactorDict.get_factors: trivial
    JointProbabilityDistribution.pmap: trivial (empty stub)
"""

import numpy as np
from benchmarks._fixtures import (
    discrete_bn,
    discrete_data,
    factors,
    joint_chain,
    random_factor,
)
from benchmarks._tiers import ROWS, Bench

from pgmpy.factors import (
    FactorDict,
    FactorSet,
    factor_divide,
    factor_product,
    factor_sum_product,
    factorset_divide,
    factorset_product,
)
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import (
    DiscreteFactor,
    JointProbabilityDistribution,
    NoisyORCPD,
    TabularCPD,
)

FACTOR_VARS = {"fast": 4, "slow": 12}
FACTOR_CARD = 3
CPD_MODEL = {"fast": "alarm", "slow": "pathfinder"}
NOISY_OR_PARENTS = {"fast": 4, "slow": 16}
JOINT_VARS = {"fast": 4, "slow": 16}
SMALL_JOINT = {"fast": 4, "slow": 7}
LINEAR_EVIDENCE = {"fast": 3, "slow": 200}
NETWORK = {"fast": "alarm", "slow": "munin"}
# Largest joint state space of the factors multiplied by FactorDict.product and factor_product.
PRODUCT_CELLS = {"fast": 20_000, "slow": 1_000_000}
# Number of leading factors summed out by factor_sum_product. Its cost grows steeply and erratically
# with the factor count on munin (300: 4 ms, 600: 24 ms, 640: 2.8 s, 700+: minutes and GBs of memory),
# so the slow tier takes the largest round count measured well inside the budget.
SUM_PRODUCT_FACTORS = {"fast": 30, "slow": 600}
# Indices passed to DiscreteFactor.assignment, and samples drawn by DiscreteFactor.sample.
N_INDICES = {"fast": 1_000, "slow": 100_000}


def largest_cpd(model):
    """CPD with the most values, ties broken by variable name."""
    return max(discrete_bn(model).cpds, key=lambda c: (c.values.size, str(c.variable)))


def product_prefix(fs, max_cells):
    """Longest prefix of the factors (sorted by scope) whose joint state space is <= max_cells."""
    chosen, cards = [], {}
    for f in sorted(fs, key=lambda f: f.variables):
        merged = {**cards, **dict(zip(f.variables, f.cardinality))}
        if chosen and np.prod(list(merged.values()), dtype=float) > max_cells:
            break
        chosen.append(f)
        cards = merged
    return chosen


class TimeDiscreteFactor(Bench):
    def prepare(self, tier):
        n = FACTOR_VARS[tier]
        self.n = n
        self.phi = random_factor(n, FACTOR_CARD)
        # Divisors and addends must have a scope within phi's; use the first half of the variables.
        self.half = random_factor(n // 2, FACTOR_CARD, seed=1)
        self.variables = self.phi.variables
        self.first, self.second = self.variables[:2]
        self.n_indices = N_INDICES[tier]
        self.indices = np.arange(min(self.n_indices, self.phi.values.size))
        self.point = {v: 0 for v in self.variables}

    def time_init(self, tier):
        DiscreteFactor(self.variables, self.phi.cardinality, self.phi.values.ravel())

    def time_assignment(self, tier):
        self.phi.assignment(self.indices)

    def time_copy(self, tier):
        self.phi.copy()

    def time_divide(self, tier):
        self.phi.divide(self.half)

    def time_get_cardinality(self, tier):
        self.phi.get_cardinality(self.variables)

    def time_get_value(self, tier):
        self.phi.get_value(**self.point)

    def time_identity_factor(self, tier):
        self.phi.identity_factor()

    def time_marginalize(self, tier):
        self.phi.marginalize(self.variables[: self.n // 2])

    def time_maximize(self, tier):
        self.phi.maximize(self.variables[: self.n // 2])

    def time_normalize(self, tier):
        self.phi.normalize()

    def time_product(self, tier):
        self.phi.product(self.half)

    def time_reduce(self, tier):
        self.phi.reduce([(self.first, 0), (self.second, 1)])

    def time_sample(self, tier):
        self.phi.sample(self.n_indices, rng=0)

    def time_set_value(self, tier):
        self.phi.set_value(1.0, **self.point)

    def time_sum(self, tier):
        self.phi.sum(self.half)


class TimeTabularCPD(Bench):
    def prepare(self, tier):
        self.cpd = largest_cpd(CPD_MODEL[tier])
        self.variable = self.cpd.variable
        self.parents = list(self.cpd.get_evidence())
        self.cardinality = dict(zip(self.cpd.variables, self.cpd.cardinality))
        self.state_names = dict(self.cpd.state_names)
        self.values = self.cpd.get_values()
        self.evidence_card = [self.cardinality[p] for p in self.parents]
        self.parent_state = self.state_names[self.parents[0]][0]

    def time_init(self, tier):
        TabularCPD(
            self.variable,
            self.cardinality[self.variable],
            self.values,
            evidence=self.parents,
            evidence_card=self.evidence_card,
            state_names=self.state_names,
        )

    def time_copy(self, tier):
        self.cpd.copy()

    def time_get_random(self, tier):
        TabularCPD.get_random(
            self.variable,
            evidence=self.parents,
            cardinality=self.cardinality,
            state_names=self.state_names,
            rng=0,
        )

    def time_get_uniform(self, tier):
        TabularCPD.get_uniform(
            self.variable,
            evidence=self.parents,
            cardinality=self.cardinality,
            state_names=self.state_names,
        )

    def time_get_values(self, tier):
        self.cpd.get_values()

    def time_is_valid_cpd(self, tier):
        self.cpd.is_valid_cpd()

    def time_marginalize(self, tier):
        self.cpd.marginalize(self.parents[:1])

    def time_normalize(self, tier):
        self.cpd.normalize()

    def time_reduce(self, tier):
        self.cpd.reduce([(self.parents[0], self.parent_state)], show_warnings=False)

    def time_reorder_parents(self, tier):
        self.cpd.reorder_parents(self.parents[::-1])

    def time_to_csv(self, tier):
        self.cpd.to_csv("/dev/null")

    def time_to_dataframe(self, tier):
        self.cpd.to_dataframe()

    def time_to_factor(self, tier):
        self.cpd.to_factor()


class TimeNoisyORCPD(Bench):
    def prepare(self, tier):
        n = NOISY_OR_PARENTS[tier]
        self.prob_values = np.random.default_rng(0).uniform(0.1, 0.9, size=n)
        self.evidence = [f"P{i}" for i in range(n)]

    def time_init(self, tier):
        NoisyORCPD("Y", self.prob_values, self.evidence)


class TimeJointProbabilityDistribution(Bench):
    def prepare(self, tier):
        self.jpd, self.model = joint_chain(JOINT_VARS[tier])
        self.variables = self.jpd.variables
        self.cardinality = self.jpd.cardinality
        self.values = self.jpd.values.ravel()
        # Exponential methods: a smaller chain.
        self.small, _ = joint_chain(SMALL_JOINT[tier])
        self.order = self.small.variables

    def time_init(self, tier):
        JointProbabilityDistribution(self.variables, self.cardinality, self.values)

    def time_check_independence(self, tier):
        self.jpd.check_independence([self.variables[0]], [self.variables[-1]], self.variables[1:3], True)

    def time_conditional_distribution(self, tier):
        self.jpd.conditional_distribution([(self.variables[0], 0), (self.variables[1], 1)])

    def time_copy(self, tier):
        self.jpd.copy()

    def time_get_independencies(self, tier):
        self.small.get_independencies()

    def time_is_imap(self, tier):
        self.jpd.is_imap(self.model)

    def time_marginal_distribution(self, tier):
        self.jpd.marginal_distribution(self.variables[: len(self.variables) // 2])

    def time_minimal_imap(self, tier):
        self.small.minimal_imap(self.order)

    def time_to_factor(self, tier):
        self.jpd.to_factor()


class TimeLinearGaussianCPD(Bench):
    def prepare(self, tier):
        n = LINEAR_EVIDENCE[tier]
        self.evidence = [f"E{i}" for i in range(n)]
        self.beta = np.random.default_rng(0).normal(size=n + 1)
        self.cpd = LinearGaussianCPD("Y", self.beta, 1.0, evidence=self.evidence)

    def time_init(self, tier):
        LinearGaussianCPD("Y", self.beta, 1.0, evidence=self.evidence)

    def time_copy(self, tier):
        self.cpd.copy()

    def time_get_random(self, tier):
        LinearGaussianCPD.get_random("Y", self.evidence, rng=0)


class TimeFactorSet(Bench):
    def prepare(self, tier):
        self.factor_list = factors(NETWORK[tier])
        self.fs = FactorSet(*self.factor_list)
        self.other = FactorSet(*self.factor_list)
        self.extra = self.factor_list[0].copy()
        # Deterministic member to remove: the one with the smallest scope.
        self.victim = min(self.fs.factors, key=lambda f: f.variables)
        # Marginalize the first variable of the first few factors' scopes.
        self.marginal_vars = sorted({f.variables[0] for f in self.factor_list[:5]})

    def time_init(self, tier):
        FactorSet(*self.factor_list)

    def time_add_factors(self, tier):
        self.fs.add_factors(self.extra)

    def time_copy(self, tier):
        self.fs.copy()

    def time_divide(self, tier):
        self.fs.divide(self.other)

    def time_get_factors(self, tier):
        self.fs.get_factors()

    def time_marginalize(self, tier):
        self.fs.marginalize(self.marginal_vars)

    def time_product(self, tier):
        self.fs.product(self.other)

    def time_remove_factors(self, tier):
        self.fs.remove_factors(self.victim)


class TimeFactorDict(Bench):
    def prepare(self, tier):
        fl = factors(NETWORK[tier])
        self.fd = FactorDict({tuple(f.variables): f for f in fl})
        self.other = FactorDict({tuple(f.variables): f.copy() for f in fl})
        self.sub = FactorDict({tuple(f.variables): f for f in product_prefix(fl, PRODUCT_CELLS[tier])})

    def time_dot(self, tier):
        self.fd.dot(self.other)

    def time_product(self, tier):
        self.sub.product()


class TimeFactorDictFromDataframe(Bench):
    def prepare(self, tier):
        self.data = discrete_data("alarm", ROWS[tier])
        self.marginals = [tuple(f.variables) for f in factors("alarm")]

    def time_from_dataframe(self, tier):
        FactorDict.from_dataframe(self.data, self.marginals)


class TimeFactorFunctions(Bench):
    def prepare(self, tier):
        fl = factors(NETWORK[tier])
        self.sum_product_factors = fl[: SUM_PRODUCT_FACTORS[tier]]
        self.chosen = product_prefix(fl, PRODUCT_CELLS[tier])
        self.output_vars = list(fl[0].variables[:1])
        n = FACTOR_VARS[tier]
        self.phi = random_factor(n, FACTOR_CARD)
        self.half = random_factor(n // 2, FACTOR_CARD, seed=1)
        self.fs1 = FactorSet(*fl)
        self.fs2 = FactorSet(*fl)

    def time_factor_product(self, tier):
        factor_product(*self.chosen)

    def time_factor_divide(self, tier):
        factor_divide(self.phi, self.half)

    def time_factor_sum_product(self, tier):
        factor_sum_product(self.output_vars, self.sum_product_factors)

    def time_factorset_product(self, tier):
        factorset_product(self.fs1, self.fs2)

    def time_factorset_divide(self, tier):
        factorset_divide(self.fs1, self.fs2)
