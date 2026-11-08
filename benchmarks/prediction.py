"""Benchmarks for ``pgmpy.prediction``: ``fit`` (fresh regressor) and ``predict`` (fitted one) per regressor.

Skipped:
    NaiveAdjustmentRegressor.get_feature_names_out, DoubleMLRegressor.set_params, constructors: trivial
"""

import warnings
from functools import cache

from benchmarks._fixtures import gaussian_bn, gaussian_data
from benchmarks._bench import GAUSSIAN, ROWS, Bench
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


@cache
def inputs(kind):
    _, _, x, y = iv_inputs()
    model = gaussian_bn(GAUSSIAN)
    dag = DAG(model.edges())
    dag.add_nodes_from(model.nodes())
    dag = dag.with_role("exposures", x).with_role("outcomes", y)
    if kind == "adjustment":
        columns = sorted(dag.predecessors(x))
        dag = dag.with_role("adjustment", columns)
    else:
        columns = sorted(CausalInference(dag).get_ivs(x, y))
        dag = dag.with_role("instrument", columns)
    data = gaussian_data(GAUSSIAN, ROWS)
    return dag, data[[x] + columns], data[y]


@cache
def fitted(regressor, kind):
    dag, features, outcome = inputs(kind)
    return make(regressor, dag).fit(features, outcome)


def make(regressor, dag):
    return regressor(dag, rng=0) if regressor is DoubleMLRegressor else regressor(dag)


class _Regressor(Bench):
    kind = "adjustment"

    def setup(self):
        dag, self.features, self.outcome = inputs(self.kind)
        self.estimator = make(self.regressor, dag)
        self.fitted = fitted(self.regressor, self.kind)

    def time_fit(self):
        self.estimator.fit(self.features, self.outcome)

    def time_predict(self):
        self.fitted.predict(self.features)


class TimeNaiveAdjustmentRegressor(_Regressor):
    regressor = NaiveAdjustmentRegressor


class TimeNaiveIVRegressor(_Regressor):
    regressor, kind = NaiveIVRegressor, "iv"


class TimeDoubleMLRegressor(_Regressor):
    regressor = DoubleMLRegressor
