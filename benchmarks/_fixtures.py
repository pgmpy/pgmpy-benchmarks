"""Benchmark inputs rebuilt from the frozen specs in benchmarks/data, with pgmpy constructors and numpy only (no simulate/forward_sample/get_random/readers)."""

import gzip
import json
from bisect import bisect_left
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pgmpy.base import ADMG, DAG, MAG, PDAG, UndirectedGraph
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import DiscreteFactor, JointProbabilityDistribution, State, TabularCPD
from pgmpy.independencies import Independencies
from pgmpy.models import (
    DiscreteBayesianNetwork,
    DiscreteMarkovNetwork,
    DynamicBayesianNetwork,
    FactorGraph,
    JunctionTree,
    LinearGaussianBayesianNetwork,
    MarkovChain,
)

DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=None)
def load_spec(name):
    with gzip.open(DATA_DIR / f"{name}.json.gz", "rt") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def readwrite_text(fmt, name):
    with gzip.open(DATA_DIR / "readwrite" / f"{name}.{fmt}.gz", "rt", encoding="utf-8") as f:
        return f.read()


def pomdpx_model_data(name):
    return json.loads(readwrite_text("model.json", name))


def _edges(spec):
    return [(p, n["name"]) for n in spec["nodes"] for p in n["parents"]]


def discrete_bn(name):
    spec = load_spec(name)
    states = {n["name"]: n["states"] for n in spec["nodes"]}
    model = DiscreteBayesianNetwork(_edges(spec))
    model.add_nodes_from(states)
    cpds = []
    for n in spec["nodes"]:
        parents = n["parents"]
        cpds.append(
            TabularCPD(
                n["name"],
                len(n["states"]),
                n["values"],
                evidence=parents or None,
                evidence_card=[len(states[p]) for p in parents] or None,
                state_names={v: states[v] for v in [n["name"], *parents]},
            )
        )
    model.add_cpds(*cpds)
    return model


def gaussian_bn(name):
    spec = load_spec(name)
    model = LinearGaussianBayesianNetwork(_edges(spec))
    model.add_nodes_from(n["name"] for n in spec["nodes"])
    model.add_cpds(
        *[
            LinearGaussianCPD(
                n["name"], [n["intercept"], *n["coefficients"]], n["std"], evidence=n["parents"]
            )
            for n in spec["nodes"]
        ]
    )
    return model


def graph(name):
    spec = load_spec(name)
    kind = spec["kind"]
    directed = [tuple(e) for e in spec["directed"]]
    undirected = [tuple(e) for e in spec["undirected"]]
    bidirected = [tuple(e) for e in spec["bidirected"]]
    if kind == "dag":
        g = DAG(directed)
        g.add_nodes_from(spec["nodes"])
    elif kind == "ug":
        g = UndirectedGraph(undirected)
        g.add_nodes_from(spec["nodes"])
    else:
        cls = {"pdag": PDAG, "admg": ADMG, "mag": MAG}[kind]
        g = cls(
            edge_list=[(u, v, "->") for u, v in directed]
            + [(u, v, "--") for u, v in undirected]
            + [(u, v, "<>") for u, v in bidirected]
        )
        g.add_nodes_from(spec["nodes"])
    return g


def independencies(name):
    return Independencies(*load_spec(name)["assertions"])


def syntax(name):
    spec = load_spec(name)
    return spec["lavaan"], spec["dagitty"]


def bare(model):
    """A model of the same class with the nodes and edges of ``model`` and no CPDs or factors."""
    b = type(model)(model.edges())
    b.add_nodes_from(model.nodes)
    return b


def first_states(model, nodes):
    return {n: model.states[n][0] for n in nodes}


def _factors(spec):
    states = {n["name"]: n["states"] for n in spec["nodes"]}
    return [_reduced_factor(n, states, n["parents"]) for n in spec["nodes"]]


def _moral_edges(spec):
    edges = set()
    for n in spec["nodes"]:
        parents = n["parents"]
        edges.update((p, n["name"]) for p in parents)
        edges.update((a, b) for i, a in enumerate(parents) for b in parents[i + 1 :])
    return sorted(edges)


def factors(name):
    """One DiscreteFactor per CPD family of a discrete spec, in spec order."""
    return _factors(load_spec(name))


def markov_network(name):
    spec = load_spec(name)
    model = DiscreteMarkovNetwork(_moral_edges(spec))
    model.add_nodes_from(n["name"] for n in spec["nodes"])
    model.add_factors(*_factors(spec))
    return model


def factor_graph(name):
    spec = load_spec(name)
    return _factor_graph([n["name"] for n in spec["nodes"]], _factors(spec))


def _factor_graph(variables, fs):
    fg = FactorGraph()
    fg.add_nodes_from(variables)
    fg.add_nodes_from(fs)
    fg.add_factors(*fs)
    fg.add_edges_from([(v, f) for f in fs for v in f.variables])
    return fg


def _reduced_factor(node, states, kept):
    """Factor over ``[node, *kept]`` from a spec node's CPD, averaged over the parents not in ``kept``."""
    parents = node["parents"]
    cards = [len(states[v]) for v in (node["name"], *parents)]
    table = np.asarray(node["values"], dtype=float).reshape(cards)
    dropped = tuple(1 + i for i, p in enumerate(parents) if p not in kept)
    if dropped:
        table = table.mean(axis=dropped)
    scope = [node["name"], *(p for p in parents if p in kept)]
    return DiscreteFactor(scope, list(table.shape), table.ravel(), state_names={v: states[v] for v in scope})


def bfs_prefix(edges, start, n_nodes):
    """Induced DAG on the first ``n_nodes`` nodes of a sorted-neighbour BFS over the skeleton from ``start``."""
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
    """Pairwise network for ``Mplp``: a factor per node (CPD averaged over parents) and per BN edge (child CPD averaged over other parents)."""
    spec = load_spec(name)
    states = {n["name"]: n["states"] for n in spec["nodes"]}
    edges, fs = [], []
    for node in spec["nodes"]:
        fs.append(_reduced_factor(node, states, ()))
        for p in node["parents"]:
            fs.append(_reduced_factor(node, states, (p,)))
            edges.append((p, node["name"]))
    model = DiscreteMarkovNetwork(edges)
    model.add_nodes_from(states)
    model.add_factors(*fs)
    return model


@lru_cache(maxsize=None)
def _tree_kept_parents(name):
    spec = load_spec(name)
    root = {n["name"]: n["name"] for n in spec["nodes"]}

    def find(v):
        while root[v] != v:
            root[v] = root[root[v]]
            v = root[v]
        return v

    kept = {}
    for node in sorted(spec["nodes"], key=lambda n: n["name"]):
        kept[node["name"]] = set()
        for p in sorted(node["parents"]):
            a, b = find(p), find(node["name"])
            if a != b:
                root[a] = b
                kept[node["name"]].add(p)
    return kept


def tree_factor_graph(name):
    """Forest-shaped factor graph for ``BeliefPropagationWithMessagePassing``: each node keeps the parents that close no cycle (sorted order); dropped parents are averaged out."""
    spec = load_spec(name)
    states = {n["name"]: n["states"] for n in spec["nodes"]}
    kept = _tree_kept_parents(name)
    fs = [_reduced_factor(n, states, kept[n["name"]]) for n in spec["nodes"]]
    return _factor_graph(states, fs)


def junction_tree(name):
    spec = load_spec(name)
    cliques = [tuple(c) for c in spec["cliques"]]
    jt = JunctionTree()
    jt.add_nodes_from(cliques)
    jt.add_edges_from([(cliques[i], cliques[j]) for i, j in spec["edges"]])
    jt.add_factors(
        *[
            DiscreteFactor(f["scope"], f["cardinality"], f["values"], state_names=f["state_names"])
            for f in spec["factors"]
        ]
    )
    return jt


def random_factor(n_vars, card, seed=0):
    values = np.random.default_rng(seed).random(card**n_vars)
    return DiscreteFactor([f"V{i}" for i in range(n_vars)], [card] * n_vars, values)


def joint_chain(n_vars, seed=0):
    """Binary chain X0 -> X1 -> ... as (JointProbabilityDistribution, DiscreteBayesianNetwork), the network an I-map of the joint."""
    rng = np.random.default_rng(seed)
    names = [f"X{i}" for i in range(n_vars)]
    root = rng.dirichlet([1, 1])  # P(X0)
    # cpts[i][child, parent]: columns sum to one.
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


def _draw(probs, rng):
    code = (np.cumsum(probs, axis=1) < rng.random(len(probs))[:, None]).sum(axis=1)
    return np.minimum(code, probs.shape[1] - 1)  # guards cumsum round-off


@lru_cache(maxsize=None)
def _discrete_array(name, n_rows, seed):
    spec = load_spec(name)
    rng = np.random.default_rng(seed)
    n_states = {n["name"]: len(n["states"]) for n in spec["nodes"]}
    codes = {}
    out = np.empty((n_rows, len(spec["nodes"])), dtype=object)
    for col, n in enumerate(spec["nodes"]):
        values = np.asarray(n["values"])
        if n["parents"]:
            cards = tuple(n_states[p] for p in n["parents"])
            idx = np.ravel_multi_index(tuple(codes[p] for p in n["parents"]), cards)
        else:
            idx = np.zeros(n_rows, dtype=np.intp)
        probs = values[:, idx].T
        state_idx = _draw(probs, rng)
        codes[n["name"]] = state_idx
        out[:, col] = np.array(n["states"], dtype=object)[state_idx]
    return out


def discrete_data(name, n_rows, seed=0):
    columns = [n["name"] for n in load_spec(name)["nodes"]]
    return pd.DataFrame(_discrete_array(name, n_rows, seed).copy(), columns=columns)


@lru_cache(maxsize=None)
def _gaussian_array(name, n_rows, seed):
    spec = load_spec(name)
    rng = np.random.default_rng(seed)
    cols = {}
    for n in spec["nodes"]:
        x = n["intercept"] + rng.normal(0, n["std"], n_rows)
        if n["parents"]:
            x = x + np.column_stack([cols[p] for p in n["parents"]]) @ np.asarray(n["coefficients"])
        cols[n["name"]] = x
    return np.column_stack([cols[n["name"]] for n in spec["nodes"]])


def gaussian_data(name, n_rows, seed=0):
    columns = [n["name"] for n in load_spec(name)["nodes"]]
    return pd.DataFrame(_gaussian_array(name, n_rows, seed).copy(), columns=columns)


def mixed_data(name, n_rows, seed=0):
    df = gaussian_data(name, n_rows, seed)
    for col in df.columns[1::2]:
        df[col] = pd.qcut(df[col], 3, labels=["low", "mid", "high"]).astype(object)
    return df


@lru_cache(maxsize=None)
def _sem_inputs(name, n_err_corr):
    """SEM inputs on a gaussian spec's edges (graph, lavaan, LISREL and RAM forms). Nodes with children are latent, leaves observed; edges are pruned so ``to_standard_lisrel`` works; the first ``2 * n_err_corr`` observed nodes get correlated errors."""
    spec = load_spec(name)
    roots = {nd["name"] for nd in spec["nodes"] if not nd["parents"]}
    coefficient = {
        (p, nd["name"]): c for nd in spec["nodes"] for p, c in zip(nd["parents"], nd["coefficients"])
    }
    changed = True
    while changed:  # dropping edges can turn a node into a leaf, so repeat until stable
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
    std = {nd["name"]: nd["std"] for nd in spec["nodes"]}
    variables = [nd["name"] for nd in spec["nodes"] if nd["name"] in used]
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
    # lavaan names are alphanumeric, and a regressor named like "111" would parse as an intercept ("y ~ 1").
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
        "observed": observed,
        "B": B,
        "zeta": zeta,
        "wedge_y": wedge_y,
        "B_values": B_values,
        "zeta_values": zeta_values,
    }


def sem_inputs(name, n_err_corr):
    """Structural equation model inputs of a gaussian spec (see ``_sem_inputs``); fresh copies on every call."""
    return {k: v.copy() if isinstance(v, (np.ndarray, list)) else v for k, v in _sem_inputs(name, n_err_corr).items()}


def dbn(name):
    spec = load_spec(name)
    k = spec["n_states"]
    model = DynamicBayesianNetwork()
    model.add_nodes_from(spec["variables"])
    model.add_edges_from([(tuple(p), (n["name"], n["t"])) for n in spec["nodes"] for p in n["parents"]])
    model.add_cpds(
        *[
            TabularCPD(
                (n["name"], n["t"]),
                k,
                n["values"],
                evidence=[tuple(p) for p in n["parents"]] or None,
                evidence_card=[k] * len(n["parents"]) or None,
            )
            for n in spec["nodes"]
        ]
    )
    return model


DBN_SLICES = 3


@lru_cache(maxsize=None)
def _dbn_array(name, n_rows, seed):
    spec = load_spec(name)
    rng = np.random.default_rng(seed)
    k = spec["n_states"]
    states = {}  # (variable, slice) -> state codes
    for t in range(DBN_SLICES):
        for node in spec["nodes"]:
            if node["t"] != min(t, 1):
                continue
            parents = [(p, t - 1 + pt) if t > 0 else (p, pt) for p, pt in node["parents"]]
            values = np.asarray(node["values"])
            idx = np.ravel_multi_index(tuple(states[q] for q in parents), (k,) * len(parents)) if parents else 0
            probs = values[:, idx].T if parents else np.broadcast_to(values[:, 0], (n_rows, k))
            states[(node["name"], t)] = _draw(probs, rng)
    return np.column_stack([states[(v, t)] for t in range(DBN_SLICES) for v in spec["variables"]])


def dbn_data(name, n_rows, seed=0):
    spec = load_spec(name)
    columns = [(v, t) for t in range(DBN_SLICES) for v in spec["variables"]]
    return pd.DataFrame(_dbn_array(name, n_rows, seed).copy(), columns=columns)


def markov_chain_matrices(n_vars, n_states, seed=0):
    rng = np.random.default_rng(seed)
    return {f"X{i}": rng.dirichlet(np.ones(n_states), size=n_states) for i in range(n_vars)}


def markov_chain(n_vars, n_states, seed=0):
    matrices = markov_chain_matrices(n_vars, n_states, seed)
    chain = MarkovChain(list(matrices), [n_states] * n_vars, start_state=[State(v, 0) for v in matrices])
    for variable, matrix in matrices.items():
        chain.add_transition_model(variable, matrix)
    return chain


@lru_cache(maxsize=None)
def _markov_chain_array(n_vars, n_states, n_rows, seed):
    matrices = markov_chain_matrices(n_vars, n_states, seed)
    rng = np.random.default_rng(seed + 1)
    out = np.zeros((n_rows, n_vars), dtype=np.int64)  # every variable starts in state 0
    for col, matrix in enumerate(matrices.values()):
        cumulative = np.cumsum(matrix, axis=1).tolist()
        uniform = rng.random(n_rows).tolist()
        state = 0
        for row in range(1, n_rows):
            state = min(bisect_left(cumulative[state], uniform[row]), n_states - 1)
            out[row, col] = state
    return out


def markov_chain_sample(n_vars, n_states, n_rows, seed=0):
    columns = list(markov_chain_matrices(n_vars, n_states, seed))
    return pd.DataFrame(_markov_chain_array(n_vars, n_states, n_rows, seed).copy(), columns=columns)
