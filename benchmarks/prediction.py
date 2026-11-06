"""Benchmarks for ``pgmpy.prediction``.

Inputs: ``gaussian_data`` of ``random_gaussian_8`` (1000 rows, fast) / ``arth150`` (100000 rows, slow)
(``GAUSSIAN_MODEL``, ``ROWS``) and the structure of the same model from ``_fixtures.gaussian_bn``, as a
``DAG`` with roles. The pair is the one ``estimators.iv_inputs`` uses for ``IVEstimator``: the first
sorted edge ``X -> Y`` such that ``X`` has a parent and ``get_ivs(X, Y)`` finds an instrument (and
``get_conditional_ivs`` none), which is ``X_3 -> X_7`` and ``111 -> 496``. The roles follow one fixed
rule:

* ``NaiveAdjustmentRegressor`` and ``DoubleMLRegressor``: exposure ``X``, outcome ``Y``, adjustment =
  the parents of ``X`` (always a valid backdoor set; ``X_4`` and ``570, 61, 81``). The features are
  ``[X] + adjustment``.
* ``NaiveIVRegressor``: exposure ``X``, outcome ``Y``, instruments = the sorted ``get_ivs(X, Y)`` of the
  DAG (6 and 70 variables, no pretreatment variables). The features are ``[X] + instruments``.

The feature frames and the outcome are sliced from the data once per tier. ``fit`` and ``predict`` are
timed separately: ``fit`` on a fresh regressor, ``predict`` on a fitted one built once per tier (neither
modifies its inputs). All regressors use their default ``LinearRegression`` estimators, so they take no
random state; ``DoubleMLRegressor`` gets ``rng=0`` (cross-fitting splits, 5 folds). With these defaults
no call needs capping: at 100000 rows ``fit`` takes 8 ms (adjustment), 0.13-0.28 s (IV, 70 instruments) and
0.05-0.4 s (DoubleML; the larger figures are the harness's first call), and each ``predict`` under 10 ms.

``DoubleMLRegressor.predict`` makes scikit-learn warn "X has feature names, but LinearRegression was fitted
without feature names", because ``fit`` trains the effect estimator on the unnamed residual column. It is
silenced by a module-level filter on that exact message.

Skipped:
    ``NaiveAdjustmentRegressor.get_feature_names_out`` (trivial), ``DoubleMLRegressor.set_params``
    (trivial, scikit-learn's ``set_params`` plus a ``seed`` alias), the constructors (trivial).
"""

import warnings
from functools import cache

from benchmarks._fixtures import gaussian_bn, gaussian_data
from benchmarks._tiers import ROWS, Bench
from benchmarks.estimators import iv_inputs

from pgmpy.base import DAG
from pgmpy.inference import CausalInference
from pgmpy.prediction import (
    DoubleMLRegressor,
    NaiveAdjustmentRegressor,
    NaiveIVRegressor,
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"X has feature names, but LinearRegression was fitted without feature names",
)

GAUSSIAN_MODEL = {"fast": "random_gaussian_8", "slow": "arth150"}


@cache
def inputs(tier, kind):
    """``(causal graph with roles, feature frame, outcome)`` for ``kind`` in ``adjustment`` / ``iv``."""
    name = GAUSSIAN_MODEL[tier]
    _, _, x, y = iv_inputs(tier)
    model = gaussian_bn(name)
    dag = DAG(model.edges())
    dag.add_nodes_from(model.nodes())
    dag = dag.with_role("exposures", x).with_role("outcomes", y)
    if kind == "adjustment":
        columns = sorted(dag.predecessors(x))
        dag = dag.with_role("adjustment", columns)
    else:
        columns = sorted(CausalInference(dag).get_ivs(x, y))
        dag = dag.with_role("instrument", columns)
    data = gaussian_data(name, ROWS[tier])
    return dag, data[[x] + columns], data[y]


@cache
def fitted(tier, regressor):
    kind = "iv" if regressor is NaiveIVRegressor else "adjustment"
    dag, features, outcome = inputs(tier, kind)
    return make(regressor, dag).fit(features, outcome)


def make(regressor, dag):
    return regressor(dag, rng=0) if regressor is DoubleMLRegressor else regressor(dag)


class RegressorBench(Bench):
    """Prepares a fresh regressor, a fitted one and the inputs; the subclasses define the timed methods."""

    regressor = None
    kind = "adjustment"

    def prepare(self, tier):
        dag, self.features, self.outcome = inputs(tier, self.kind)
        self.estimator = make(self.regressor, dag)
        self.fitted = fitted(tier, self.regressor)


class TimeNaiveAdjustmentRegressor(RegressorBench):
    regressor = NaiveAdjustmentRegressor

    def time_fit(self, tier):
        self.estimator.fit(self.features, self.outcome)

    def time_predict(self, tier):
        self.fitted.predict(self.features)


class TimeNaiveIVRegressor(RegressorBench):
    regressor = NaiveIVRegressor
    kind = "iv"

    def time_fit(self, tier):
        self.estimator.fit(self.features, self.outcome)

    def time_predict(self, tier):
        self.fitted.predict(self.features)


class TimeDoubleMLRegressor(RegressorBench):
    regressor = DoubleMLRegressor

    def time_fit(self, tier):
        self.estimator.fit(self.features, self.outcome)

    def time_predict(self, tier):
        self.fitted.predict(self.features)
