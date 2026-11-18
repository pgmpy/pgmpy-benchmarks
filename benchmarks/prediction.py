"""Benchmarks for ``pgmpy.prediction``: ``fit`` (fresh regressor) and ``predict`` (fitted one) per regressor.

Skipped:
    NaiveAdjustmentRegressor.get_feature_names_out, DoubleMLRegressor.set_params, constructors: trivial
"""

import warnings
from functools import cache

import networkx as nx
import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.base import DAG
from pgmpy.inference import CausalInference
from pgmpy.models import SEM, LinearGaussianBayesianNetwork
from pgmpy.prediction import (
    DoubleMLRegressor,
    NaiveAdjustmentRegressor,
    NaiveIVRegressor,
)

ROWS = 1_000
GAUSSIAN = 8
SEM_ERR_CORR = 1
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"X has feature names, but LinearRegression was fitted without feature names",
)


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


@fresh
def gaussian_data(name, n_rows, seed=0):
    model = gaussian_bn(name)
    return model.simulate(n_rows, rng=seed)[topological(model)]


@fresh
def sem_inputs(name, n_err_corr):
    """SEM inputs on a gaussian model's edges: nodes with children are latent, edges pruned so ``to_standard_lisrel`` works."""
    model = gaussian_bn(name)
    cpds = [model.get_cpds(v) for v in topological(model)]
    roots = {cpd.variable for cpd in cpds if not cpd.evidence}
    coefficient = {(p, cpd.variable): float(c) for cpd in cpds for p, c in zip(cpd.evidence, cpd.beta[1:])}
    changed = True
    while changed:  # dropping edges can make new leaves
        changed = False
        has_child = {p for p, _ in coefficient}
        for child in {c for _, c in coefficient if c not in has_child}:
            parents = [p for p, c in coefficient if c == child]
            if any(p in roots for p in parents) and not all(p in roots for p in parents):
                for p in parents:
                    if p not in roots:
                        del coefficient[(p, child)]
                changed = True
    ebunch = list(coefficient)
    used = {v for e in ebunch for v in e}
    std = {cpd.variable: float(cpd.std) for cpd in cpds}
    variables = [cpd.variable for cpd in cpds if cpd.variable in used]
    index = {v: i for i, v in enumerate(variables)}
    latents = [v for v in variables if v in {p for p, _ in ebunch}]
    observed = [v for v in variables if v not in latents]
    err_corr = [(observed[2 * k], observed[2 * k + 1]) for k in range(n_err_corr)]
    n = len(variables)
    B = np.zeros((n, n))
    B_values = np.zeros((n, n))
    for (p, c), value in coefficient.items():
        B[index[c], index[p]] = 1
        B_values[index[c], index[p]] = value
    zeta = np.eye(n)
    zeta_values = np.diag([std[v] ** 2 for v in variables])
    for a, b in err_corr:
        zeta[index[a], index[b]] = zeta[index[b], index[a]] = 1
        cov = 0.1 * min(zeta_values[index[a], index[a]], zeta_values[index[b], index[b]])
        zeta_values[index[a], index[b]] = zeta_values[index[b], index[a]] = cov
    wedge_y = np.zeros((len(observed), n))
    for row, v in enumerate(observed):
        wedge_y[row, index[v]] = 1
    # lavaan names are alphanumeric, and a regressor "111" would parse as an intercept ("y ~ 1").
    plain = {v: "v" + v.replace("_", "") for v in variables}
    lavaan = "\n".join(
        f"{plain[c]} ~ {' + '.join(plain[p] for p, c2 in ebunch if c2 == c)}"
        for c in variables
        if any(c2 == c for _, c2 in ebunch)
    )
    xi = [v for v in latents if not B[index[v]].any()]
    eta = [v for v in latents if v not in xi]
    x = [v for v in observed if any(B[index[v], index[r]] for r in xi)]
    y = [v for v in observed if v not in x]

    def blocks(matrix_B, matrix_zeta):
        def pick(m, rows, cols):
            return m[np.ix_([index[r] for r in rows], [index[c] for c in cols])]

        return {
            "B": pick(matrix_B, eta, eta),
            "gamma": pick(matrix_B, eta, xi),
            "wedge_y": pick(matrix_B, y, eta),
            "wedge_x": pick(matrix_B, x, xi),
            "phi": pick(matrix_zeta, xi, xi),
            "theta_e": pick(matrix_zeta, y, y),
            "theta_del": pick(matrix_zeta, x, x),
            "psi": pick(matrix_zeta, eta, eta),
        }

    return {
        "ebunch": ebunch,
        "lavaan": lavaan,
        "var_names": {"eta": eta, "xi": xi, "y": y, "x": x},
        "params": blocks(B, zeta),
        "fixed_masks": blocks(B_values, zeta_values),
        "latents": latents,
        "err_corr": err_corr,
        "variables": variables,
        "B": B,
        "zeta": zeta,
        "wedge_y": wedge_y,
        "B_values": B_values,
        "zeta_values": zeta_values,
    }


@cache
def iv_inputs():
    """(SEM, data, X, Y); X -> Y is the first sorted edge where X has a parent, get_ivs finds one and get_conditional_ivs none."""
    name = GAUSSIAN
    sem = sem_inputs(name, SEM_ERR_CORR)
    model = SEM.from_graph(ebunch=sem["ebunch"], latents=[], err_corr=sem["err_corr"])
    inference = CausalInference(model)
    for x, y in sorted(sem["ebunch"]):
        if (
            list(model.graph.predecessors(x))
            and inference.get_ivs(x, y)
            and not inference.get_conditional_ivs(x, y)
        ):
            return model, gaussian_data(name, ROWS), x, y
    raise ValueError(f"no edge with instruments in {name}")


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
