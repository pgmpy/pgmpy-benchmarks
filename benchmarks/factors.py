"""Benchmarks for ``pgmpy.factors``, including the module-level ``factor_*`` functions.

``TabularCPD.is_valid_cpd`` is timed on the CPD because it calls ``to_factor``, which plain factors lack.

Skipped:
    DiscreteFactor.scope, TabularCPD.get_evidence, FactorDict.get_factors, JointProbabilityDistribution.pmap (empty stub): trivial
"""

import numpy as np
from benchmarks._fixtures import (
    discrete_bn,
    discrete_data,
    factors,
    joint_chain,
    random_factor,
)
from benchmarks._bench import MODEL, ROWS, Bench

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

FACTOR_CARD = 3
JOINT_VARS = 4
PRODUCT_CELLS = 20_000
SUM_PRODUCT_FACTORS = 30


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
    def setup(self):
        n = self.n = 4
        self.phi = random_factor(n, FACTOR_CARD)
        self.half = random_factor(n // 2, FACTOR_CARD, seed=1)
        self.variables = self.phi.variables
        self.first, self.second = self.variables[:2]
        self.indices = np.arange(min(ROWS, self.phi.values.size))
        self.point = {v: 0 for v in self.variables}

    def time_init(self):
        DiscreteFactor(self.variables, self.phi.cardinality, self.phi.values.ravel())

    def time_assignment(self):
        self.phi.assignment(self.indices)

    def time_copy(self):
        self.phi.copy()

    def time_divide(self):
        self.phi.divide(self.half)

    def time_get_cardinality(self):
        self.phi.get_cardinality(self.variables)

    def time_get_value(self):
        self.phi.get_value(**self.point)

    def time_identity_factor(self):
        self.phi.identity_factor()

    def time_marginalize(self):
        self.phi.marginalize(self.variables[: self.n // 2])

    def time_maximize(self):
        self.phi.maximize(self.variables[: self.n // 2])

    def time_normalize(self):
        self.phi.normalize()

    def time_product(self):
        self.phi.product(self.half)

    def time_reduce(self):
        self.phi.reduce([(self.first, 0), (self.second, 1)])

    def time_sample(self):
        self.phi.sample(ROWS, rng=0)

    def time_set_value(self):
        self.phi.set_value(1.0, **self.point)

    def time_sum(self):
        self.phi.sum(self.half)


class TimeTabularCPD(Bench):
    def setup(self):
        self.cpd = largest_cpd(MODEL)
        self.variable = self.cpd.variable
        self.parents = list(self.cpd.get_evidence())
        self.cardinality = dict(zip(self.cpd.variables, self.cpd.cardinality))
        self.state_names = dict(self.cpd.state_names)
        self.values = self.cpd.get_values()
        self.evidence_card = [self.cardinality[p] for p in self.parents]
        self.parent_state = self.state_names[self.parents[0]][0]

    def time_init(self):
        TabularCPD(
            self.variable,
            self.cardinality[self.variable],
            self.values,
            evidence=self.parents,
            evidence_card=self.evidence_card,
            state_names=self.state_names,
        )

    def time_copy(self):
        self.cpd.copy()

    def time_get_random(self):
        TabularCPD.get_random(
            self.variable,
            evidence=self.parents,
            cardinality=self.cardinality,
            state_names=self.state_names,
            rng=0,
        )

    def time_get_uniform(self):
        TabularCPD.get_uniform(
            self.variable,
            evidence=self.parents,
            cardinality=self.cardinality,
            state_names=self.state_names,
        )

    def time_get_values(self):
        self.cpd.get_values()

    def time_is_valid_cpd(self):
        self.cpd.is_valid_cpd()

    def time_marginalize(self):
        self.cpd.marginalize(self.parents[:1])

    def time_normalize(self):
        self.cpd.normalize()

    def time_reduce(self):
        self.cpd.reduce([(self.parents[0], self.parent_state)], show_warnings=False)

    def time_reorder_parents(self):
        self.cpd.reorder_parents(self.parents[::-1])

    def time_to_csv(self):
        self.cpd.to_csv("/dev/null")

    def time_to_dataframe(self):
        self.cpd.to_dataframe()

    def time_to_factor(self):
        self.cpd.to_factor()


class TimeNoisyORCPD(Bench):
    def setup(self):
        n = 4
        self.prob_values = np.random.default_rng(0).uniform(0.1, 0.9, size=n)
        self.evidence = [f"P{i}" for i in range(n)]

    def time_init(self):
        NoisyORCPD("Y", self.prob_values, self.evidence)


class TimeJointProbabilityDistribution(Bench):
    def setup(self):
        self.jpd, self.model = joint_chain(JOINT_VARS)
        self.variables = self.jpd.variables
        self.cardinality = self.jpd.cardinality
        self.values = self.jpd.values.ravel()
        self.small, _ = joint_chain(JOINT_VARS)
        self.order = self.small.variables

    def time_init(self):
        JointProbabilityDistribution(self.variables, self.cardinality, self.values)

    def time_check_independence(self):
        self.jpd.check_independence([self.variables[0]], [self.variables[-1]], self.variables[1:3], True)

    def time_conditional_distribution(self):
        self.jpd.conditional_distribution([(self.variables[0], 0), (self.variables[1], 1)])

    def time_copy(self):
        self.jpd.copy()

    def time_get_independencies(self):
        self.small.get_independencies()

    def time_is_imap(self):
        self.jpd.is_imap(self.model)

    def time_marginal_distribution(self):
        self.jpd.marginal_distribution(self.variables[: len(self.variables) // 2])

    def time_minimal_imap(self):
        self.small.minimal_imap(self.order)

    def time_to_factor(self):
        self.jpd.to_factor()


class TimeLinearGaussianCPD(Bench):
    def setup(self):
        n = 3
        self.evidence = [f"E{i}" for i in range(n)]
        self.beta = np.random.default_rng(0).normal(size=n + 1)
        self.cpd = LinearGaussianCPD("Y", self.beta, 1.0, evidence=self.evidence)

    def time_init(self):
        LinearGaussianCPD("Y", self.beta, 1.0, evidence=self.evidence)

    def time_copy(self):
        self.cpd.copy()

    def time_get_random(self):
        LinearGaussianCPD.get_random("Y", self.evidence, rng=0)


class TimeFactorSet(Bench):
    def setup(self):
        self.factor_list = factors(MODEL)
        self.fs = FactorSet(*self.factor_list)
        self.other = FactorSet(*self.factor_list)
        self.extra = self.factor_list[0].copy()
        self.victim = min(self.fs.factors, key=lambda f: f.variables)
        self.marginal_vars = sorted({f.variables[0] for f in self.factor_list[:5]})

    def time_init(self):
        FactorSet(*self.factor_list)

    def time_add_factors(self):
        self.fs.add_factors(self.extra)

    def time_copy(self):
        self.fs.copy()

    def time_divide(self):
        self.fs.divide(self.other)

    def time_get_factors(self):
        self.fs.get_factors()

    def time_marginalize(self):
        self.fs.marginalize(self.marginal_vars)

    def time_product(self):
        self.fs.product(self.other)

    def time_remove_factors(self):
        self.fs.remove_factors(self.victim)


class TimeFactorDict(Bench):
    def setup(self):
        fl = factors(MODEL)
        self.fd = FactorDict({tuple(f.variables): f for f in fl})
        self.other = FactorDict({tuple(f.variables): f.copy() for f in fl})
        self.sub = FactorDict({tuple(f.variables): f for f in product_prefix(fl, PRODUCT_CELLS)})

    def time_dot(self):
        self.fd.dot(self.other)

    def time_product(self):
        self.sub.product()


class TimeFactorDictFromDataframe(Bench):
    def setup(self):
        self.data = discrete_data(MODEL, ROWS)
        self.marginals = [tuple(f.variables) for f in factors(MODEL)]

    def time_from_dataframe(self):
        FactorDict.from_dataframe(self.data, self.marginals)


class TimeFactorFunctions(Bench):
    def setup(self):
        fl = factors(MODEL)
        self.sum_product_factors = fl[:SUM_PRODUCT_FACTORS]
        self.chosen = product_prefix(fl, PRODUCT_CELLS)
        self.output_vars = list(fl[0].variables[:1])
        n = 4
        self.phi = random_factor(n, FACTOR_CARD)
        self.half = random_factor(n // 2, FACTOR_CARD, seed=1)
        self.fs1 = FactorSet(*fl)
        self.fs2 = FactorSet(*fl)

    def time_factor_product(self):
        factor_product(*self.chosen)

    def time_factor_divide(self):
        factor_divide(self.phi, self.half)

    def time_factor_sum_product(self):
        factor_sum_product(self.output_vars, self.sum_product_factors)

    def time_factorset_product(self):
        factorset_product(self.fs1, self.fs2)

    def time_factorset_divide(self):
        factorset_divide(self.fs1, self.fs2)
