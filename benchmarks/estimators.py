"""Benchmarks for the non-deprecated ``pgmpy.estimators``.

Skipped:
    MmhcEstimator.__init__ and the other constructors: trivial
"""

import warnings
from collections import deque
from functools import cache

import networkx as nx
import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.estimators import ExhaustiveSearch, IVEstimator, MirrorDescentEstimator, MmhcEstimator
from pgmpy.example_models import load_model
from pgmpy.inference import CausalInference
from pgmpy.models import SEM, LinearGaussianBayesianNetwork

ROWS = 1_000
DISCRETE = "asia"
GAUSSIAN = 8
SEM_ERR_CORR = 1
warnings.filterwarnings("ignore", category=RuntimeWarning, message=r"invalid value encountered in divide")

N_MARGINALS = 5
ITERATIONS = 100


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


def markov_network(name):
    return discrete_bn(name).to_markov_model()


@fresh
def discrete_data(name, n_rows, seed=0):
    model = discrete_bn(name)
    return model.simulate(n_rows, rng=seed, show_progress=False)[topological(model)]


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


def discrete(rows=ROWS):
    return discrete_data(DISCRETE, rows)


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


class TimeMmhcEstimator(Bench):
    def setup(self):
        self.estimator = MmhcEstimator(discrete())

    def time_estimate(self):
        self.estimator.estimate()

    def time_mmpc(self):
        self.estimator.mmpc()


class TimeExhaustiveSearch(Bench):
    def setup(self):
        self.estimator = ExhaustiveSearch(discrete().iloc[:, :3])

    def time_all_dags(self):
        deque(self.estimator.all_dags(), maxlen=0)

    def time_all_scores(self):
        self.estimator.all_scores()

    def time_estimate(self):
        self.estimator.estimate()


@cache
def marginals(name):
    return sorted(discrete_bn(name).edges())[:N_MARGINALS]


@cache
def mirror_descent_model(name):
    return markov_network(name)


class TimeMirrorDescentEstimator(Bench):
    def setup(self):
        name = DISCRETE
        self.estimator = MirrorDescentEstimator(mirror_descent_model(name), discrete())
        self.marginals = marginals(name)

    def time_estimate(self):
        self.estimator.estimate(marginals=self.marginals, iterations=ITERATIONS, show_progress=False)


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


class TimeIVEstimator(Bench):
    def setup(self):
        model, data, self.x, self.y = iv_inputs()
        self.estimator = IVEstimator(model)
        self.data = data.copy(deep=False)

    def time_fit(self):
        self.estimator.fit(self.x, self.y, self.data)
