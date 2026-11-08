"""Benchmark inputs, built once per process with pgmpy and fixed seeds."""

import copy
import random
from collections import defaultdict
from functools import cache, wraps
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from benchmarks._bench import DISCRETE, MODEL
from pgmpy.base import DAG, MAG
from pgmpy.example_models import load_model
from pgmpy.factors.discrete import DiscreteFactor, JointProbabilityDistribution, State, TabularCPD
from pgmpy.independencies import Independencies
from pgmpy.models import (
    DiscreteBayesianNetwork,
    DiscreteMarkovNetwork,
    DynamicBayesianNetwork,
    FactorGraph,
    LinearGaussianBayesianNetwork,
    MarkovChain,
)
from pgmpy.readwrite import BIFWriter, LGBNJSONWriter, NETWriter, PomdpXReader, UAIWriter, XBNWriter, XDSLWriter, XMLBIFWriter

EDGE_PROB = 0.15
GRAPH_NODES = 20
LATENT_FRACTION = 0.1
DBN_SLICES = 3


def fresh(build):
    """Cache ``build``. Each call returns a deep copy (some benchmarks mutate their input); ``.shared`` caches one copy."""
    cached = cache(build)
    copied = wraps(build)(lambda *args, **kwargs: copy.deepcopy(cached(*args, **kwargs)))
    copied.shared = cache(copied)
    return copied


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


@fresh
def graph(kind):
    if kind in ("admg", "mag"):
        admg = _admg(GRAPH_NODES)
        return admg if kind == "admg" else _mag(admg)
    dag = DAG.get_random(n_nodes=GRAPH_NODES, edge_prob=EDGE_PROB, rng=0)
    return {"dag": lambda: dag, "pdag": dag.to_pdag, "ug": dag.moralize, "perturbed": lambda: _perturbed(dag)}[kind]()


def _admg(n):
    n_latent = max(1, round(n * LATENT_FRACTION))
    dag = DAG.get_random(n_nodes=n + n_latent, edge_prob=EDGE_PROB, rng=0)
    dag.latents = set(np.random.default_rng(0).choice(list(dag.nodes()), size=n_latent, replace=False).tolist())
    return dag.to_admg()


def _mag(admg):
    ancestors = {v: admg.get_ancestors(v) for v in admg.nodes()}
    edges = [
        (u, v, t)
        for u, v, t in admg.get_edges(data=True)
        if t == "->" or (u not in ancestors[v] and v not in ancestors[u])
    ]
    mag = MAG(edge_list=edges)
    mag.add_nodes_from(admg.nodes())
    return mag


def _perturbed(dag, fraction=0.1, rng=0):
    edges = sorted(dag.edges())
    n = round(fraction * len(edges))
    picked = np.random.default_rng(rng).permutation(len(edges))[: 2 * n]
    removed = {edges[i] for i in picked[n:]}
    out = DAG([e for e in edges if e not in removed])
    out.add_nodes_from(dag.nodes())
    for u, v in sorted(edges[i] for i in picked[:n]):
        out.remove_edge(u, v)
        out.add_edge(*((u, v) if nx.has_path(out, u, v) else (v, u)))
    return out


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
def syntax():
    dag = graph("dag")
    return dag.to_lavaan(), dag.to_dagitty()


def bare(model):
    b = type(model)(model.edges())
    b.add_nodes_from(model.nodes)
    return b


def first_states(model, nodes):
    return {n: model.states[n][0] for n in nodes}


def _cpds(name):
    model = discrete_bn(name)
    return [model.get_cpds(v) for v in topological(model)]


def factors(name):
    return [cpd.to_factor() for cpd in _cpds(name)]


def markov_network(name):
    return discrete_bn(name).to_markov_model()


def factor_graph(name):
    return factor_graph_from(topological(discrete_bn(name)), factors(name))


def factor_graph_from(variables, fs):
    fg = FactorGraph()
    fg.add_nodes_from(variables)
    fg.add_nodes_from(fs)
    fg.add_factors(*fs)
    fg.add_edges_from([(v, f) for f in fs for v in f.variables])
    return fg


def _reduced_factor(cpd, kept):
    parents = cpd.variables[1:]
    table = cpd.values
    dropped = tuple(1 + i for i, p in enumerate(parents) if p not in kept)
    if dropped:
        table = table.mean(axis=dropped)
    scope = [cpd.variable, *(p for p in parents if p in kept)]
    return DiscreteFactor(scope, list(table.shape), table.ravel(), state_names={v: cpd.state_names[v] for v in scope})


def bfs_prefix(edges, start, n_nodes):
    neighbours = defaultdict(set)
    for p, c in edges:
        neighbours[p].add(c)
        neighbours[c].add(p)
    order, seen, queue = [], set(), list(start)
    while queue and len(order) < n_nodes:
        node = queue.pop(0)
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        queue.extend(sorted(neighbours[node] - seen))
    dag = DAG([(p, c) for p, c in edges if p in seen and c in seen])
    dag.add_nodes_from(order)
    return dag


def pairwise_markov_network(name):
    """Pairwise network for ``Mplp``: a factor per node and per BN edge, each CPD averaged over the other parents."""
    cpds = _cpds(name)
    edges, fs = [], []
    for cpd in cpds:
        fs.append(_reduced_factor(cpd, ()))
        for p in cpd.variables[1:]:
            fs.append(_reduced_factor(cpd, (p,)))
            edges.append((p, cpd.variable))
    model = DiscreteMarkovNetwork(edges)
    model.add_nodes_from(cpd.variable for cpd in cpds)
    model.add_factors(*fs)
    return model


@cache
def _tree_kept_parents(name):
    model = discrete_bn(name)
    components, kept = nx.utils.UnionFind(), {}
    for node in sorted(model.nodes()):
        kept[node] = set()
        for p in sorted(model.get_parents(node)):
            if components[p] != components[node]:
                components.union(p, node)
                kept[node].add(p)
    return kept


def tree_factor_graph(name):
    """Loop-free factor graph for ``BeliefPropagationWithMessagePassing``: each node keeps the parents that close no cycle."""
    kept = _tree_kept_parents(name)
    cpds = _cpds(name)
    return factor_graph_from([cpd.variable for cpd in cpds], [_reduced_factor(cpd, kept[cpd.variable]) for cpd in cpds])


def junction_tree(name):
    return discrete_bn(name).to_junction_tree()


def random_factor(n_vars, card, seed=0):
    values = np.random.default_rng(seed).random(card**n_vars)
    return DiscreteFactor([f"V{i}" for i in range(n_vars)], [card] * n_vars, values)


def joint_chain(n_vars, seed=0):
    """Binary chain X0 -> X1 -> ... as (JointProbabilityDistribution, DiscreteBayesianNetwork), the network an I-map of the joint."""
    rng = np.random.default_rng(seed)
    names = [f"X{i}" for i in range(n_vars)]
    root = rng.dirichlet([1, 1])  # P(X0)
    cpts = [rng.dirichlet([1, 1], size=2).T for _ in range(n_vars - 1)]
    joint = root
    for cpt in cpts:
        joint = joint[..., None] * cpt.T  # axes: X0, ..., X(i-1), Xi
    model = DiscreteBayesianNetwork([(names[i], names[i + 1]) for i in range(n_vars - 1)])
    model.add_cpds(
        TabularCPD(names[0], 2, root.reshape(2, 1)),
        *[
            TabularCPD(names[i + 1], 2, cpt, evidence=[names[i]], evidence_card=[2])
            for i, cpt in enumerate(cpts)
        ],
    )
    return JointProbabilityDistribution(names, [2] * n_vars, joint.ravel()), model


@fresh
def discrete_data(name, n_rows, seed=0):
    model = discrete_bn(name)
    return model.simulate(n_rows, rng=seed, show_progress=False)[topological(model)]


@fresh
def gaussian_data(name, n_rows, seed=0):
    model = gaussian_bn(name)
    return model.simulate(n_rows, rng=seed)[topological(model)]


def mixed_data(name, n_rows, seed=0):
    df = gaussian_data(name, n_rows, seed)
    for col in df.columns[1::2]:
        df[col] = pd.qcut(df[col], 3, labels=["low", "mid", "high"]).astype(object)
    return df


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
def dbn_spec(n_vars, intra_prob=0.4, inter_prob=0.3, interface_every=2, n_states=2):
    """2-TBN with numpy CPDs (``DynamicBayesianNetwork.get_random`` breaks ``DBNInference``). The chain ``V(j-1) -> Vj``
    keeps a slice connected for the junction tree. Only every ``interface_every``-th variable has inter-slice edges, so
    sources match targets (``backward_inference``); the others have the same parents in both slices."""
    rng = np.random.default_rng(0)
    names = [f"V{i}" for i in range(n_vars)]
    intra = {j: [i for i in range(j) if rng.random() < intra_prob or i == j - 1] for j in range(n_vars)}
    interface = range(0, n_vars, interface_every)
    inter = {j: [i for i in interface if i == j or rng.random() < inter_prob] if j in interface else [] for j in range(n_vars)}
    nodes = []
    for t in (0, 1):
        for j in range(n_vars):
            parents = [(names[i], t) for i in intra[j]] + ([(names[i], 0) for i in inter[j]] if t else [])
            values = rng.dirichlet(np.ones(n_states), size=n_states ** len(parents)).T
            nodes.append({"name": (names[j], t), "parents": parents, "values": values})
    return {"n_states": n_states, "variables": names, "nodes": nodes}


def dbn(n_vars):
    spec = dbn_spec(n_vars)
    k = spec["n_states"]
    model = DynamicBayesianNetwork()
    model.add_nodes_from(spec["variables"])
    model.add_edges_from([(p, n["name"]) for n in spec["nodes"] for p in n["parents"]])
    model.add_cpds(
        *[
            TabularCPD(n["name"], k, n["values"], evidence=n["parents"] or None, evidence_card=[k] * len(n["parents"]) or None)
            for n in spec["nodes"]
        ]
    )
    return model


@fresh
def dbn_data(n_vars, n_rows, seed=0):
    return dbn(n_vars).simulate(n_rows, n_time_slices=DBN_SLICES, rng=seed, show_progress=False)


def markov_chain_matrices(n_vars, n_states, seed=0):
    rng = np.random.default_rng(seed)
    return {f"X{i}": rng.dirichlet(np.ones(n_states), size=n_states) for i in range(n_vars)}


def markov_chain(n_vars, n_states, seed=0):
    matrices = markov_chain_matrices(n_vars, n_states, seed)
    chain = MarkovChain(list(matrices), [n_states] * n_vars, start_state=[State(v, 0) for v in matrices])
    for variable, matrix in matrices.items():
        chain.add_transition_model(variable, matrix)
    return chain


@fresh
def markov_chain_sample(n_vars, n_states, n_rows, seed=0):
    return markov_chain(n_vars, n_states, seed).sample(size=n_rows, rng=seed + 1)


@cache
def readwrite_text(fmt, name):
    """A reader's input: ``name`` written by pgmpy's writer for ``fmt``."""
    if fmt == "pomdpx":
        return (Path(__file__).parent / "data" / f"{name}.pomdpx").read_text(encoding="utf-8")
    if fmt == "json":
        return str(LGBNJSONWriter(gaussian_bn(name)))
    writers = {"bif": BIFWriter, "xmlbif": XMLBIFWriter, "xbn": XBNWriter, "xdsl": XDSLWriter, "net": NETWriter,
               "uai": lambda model: UAIWriter(model, round_values=4)}  # UAIReader cannot parse "1e-05"
    random.seed(0)  # XDSLWriter draws node positions from the global random module
    return str(writers[fmt](discrete_bn(name)))


@fresh
def pomdpx_model_data(name):
    reader = PomdpXReader(string=readwrite_text("pomdpx", name))
    return {
        "description": reader.get_description(),
        "discount": str(reader.get_discount()),
        "variables": reader.get_variables(),
        "initial_state_belief": reader.get_initial_beliefs(),
        "state_transition_function": reader.get_state_transition_function(),
        "obs_function": reader.get_obs_function(),
        "reward_function": reader.get_reward_function(),
    }


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
for _name in (DISCRETE, MODEL):
    discrete_bn(_name)
