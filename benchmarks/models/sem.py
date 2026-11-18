"""Benchmarks for ``pgmpy.models.SEM``: ``SEMGraph``, ``SEMAlg`` and ``SEM``.

Skipped:
    SEM.fit: trivial (empty stub)
"""

import networkx as nx
import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.models import SEM, LinearGaussianBayesianNetwork, SEMAlg, SEMGraph

ROWS = 1_000
GAUSSIAN = 8
SEM_ERR_CORR = 1


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


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


class TimeSEMGraph(Bench):
    def setup(self):
        sem = sem_inputs(GAUSSIAN, SEM_ERR_CORR)
        self.model = SEMGraph(ebunch=sem["ebunch"], latents=sem["latents"], err_corr=sem["err_corr"])
        observed = sorted(self.model.observed)
        self.start = observed[0]
        self.observed = observed[1:3]

    def time_active_trail_nodes(self):
        self.model.active_trail_nodes(self.start, observed=self.observed)

    def time_get_scaling_indicators(self):
        self.model.get_scaling_indicators()

    def time_moralize(self):
        self.model.moralize()

    def time_to_lisrel(self):
        self.model.to_lisrel()

    def time_to_standard_lisrel(self):
        self.model.to_standard_lisrel()


class TimeSEMAlg(Bench):
    def setup(self):
        sem = sem_inputs(GAUSSIAN, SEM_ERR_CORR)
        self.B_values = sem["B_values"]
        self.zeta_values = sem["zeta_values"]
        self.model = SEMAlg(eta=sem["variables"], B=sem["B"], zeta=sem["zeta"], wedge_y=sem["wedge_y"])
        self.model.set_params(self.B_values, self.zeta_values)
        np.random.seed(0)  # generate_samples draws from the global generator

    def time_generate_samples(self):
        self.model.generate_samples(n_samples=ROWS)

    def time_set_params(self):
        self.model.set_params(self.B_values, self.zeta_values)

    def time_to_SEMGraph(self):
        self.model.to_SEMGraph()


class TimeSEM(Bench):
    def setup(self):
        self.sem = sem_inputs(GAUSSIAN, SEM_ERR_CORR)

    def time_from_graph(self):
        SEM.from_graph(ebunch=self.sem["ebunch"], latents=self.sem["latents"], err_corr=self.sem["err_corr"])

    def time_from_lavaan(self):
        SEM.from_lavaan(string=self.sem["lavaan"])

    def time_from_lisrel(self):
        SEM.from_lisrel(self.sem["var_names"], self.sem["params"], self.sem["fixed_masks"])

    def time_from_RAM(self):
        SEM.from_RAM(self.sem["variables"], self.sem["B"], self.sem["zeta"], wedge_y=self.sem["wedge_y"])
