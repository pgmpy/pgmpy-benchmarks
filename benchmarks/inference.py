"""Benchmarks for ``pgmpy.inference``: only ``query`` and ``map_query``.

Skipped:
    Mplp.query: unsupported (raises NotImplementedError)
"""

import copy
from functools import cache

import networkx as nx
import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.factors.discrete import DiscreteFactor, TabularCPD
from pgmpy.inference import (
    ApproxInference,
    BeliefPropagation,
    BeliefPropagationWithMessagePassing,
    CausalInference,
    DBNInference,
    Mplp,
    VariableElimination,
)
from pgmpy.models import DiscreteMarkovNetwork, DynamicBayesianNetwork, FactorGraph

MODEL = "alarm"
DBN_VARS = 5
APPROX_SAMPLES = 500

N_QUERY = 4
N_EVIDENCE = 2


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


def first_states(model, nodes):
    return {n: model.states[n][0] for n in nodes}


def _cpds(name):
    model = discrete_bn(name)
    return [model.get_cpds(v) for v in topological(model)]


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


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(MODEL)


def split(nodes, roots):
    return [n for n in nodes if n not in roots][:N_QUERY], nodes[-N_EVIDENCE:]


@cache
def query_and_evidence(name):
    """The 4 first non-root nodes (sorted) and the last 2 nodes (sorted) at their first state."""
    m = discrete_bn.shared(name)
    query, last = split(sorted(m.nodes), set(m.get_roots()))
    return query, first_states(m, last)


tree_graph = cache(tree_factor_graph)


@cache
def tree_query_and_evidence(name):
    fg = tree_graph(name)
    roots = {f.variables[0] for f in fg.get_factors() if len(f.variables) == 1}
    nodes = sorted(n for n in fg.nodes if isinstance(n, str))
    query, last = split(nodes, roots)
    return query, dict.fromkeys(last, 0)


@cache
def bp_query_and_evidence(tree):
    m = discrete_bn.shared(MODEL)
    query, last = split(sorted({v for clique in junction_tree(tree).nodes() for v in clique}), set(m.get_roots()))
    return query, first_states(m, last)


@cache
def calibrated_bp(tree):
    return BeliefPropagation(junction_tree(tree))


@cache
def treatment_outcome(name):
    m = discrete_bn.shared(name)
    return next((x, y) for x, y in sorted(m.edges()) if m.in_degree(x) >= 1 and m.in_degree(y) >= 2)


class TimeVariableElimination(Bench):
    def setup(self):
        self.name = MODEL
        self.model = discrete_bn.shared(self.name)
        self.inference = VariableElimination(self.model)
        self.query, self.evidence = query_and_evidence(self.name)

    def time_query(self):
        self.inference.query(self.query, evidence=self.evidence, show_progress=False)

    def time_map_query(self):
        self.inference.map_query(self.query, evidence=self.evidence, show_progress=False)


class TimeBeliefPropagation(Bench):
    def setup(self):
        self.tree = MODEL
        self.inference = copy.deepcopy(calibrated_bp(self.tree))
        self.query, self.evidence = bp_query_and_evidence(self.tree)

    def time_query(self):
        self.inference.query(self.query, evidence=self.evidence, show_progress=False)

    def time_map_query(self):
        self.inference.map_query(self.query, evidence=self.evidence, show_progress=False)


class TimeBeliefPropagationWithMessagePassing(Bench):
    def setup(self):
        self.name = MODEL
        self.inference = BeliefPropagationWithMessagePassing(tree_graph(self.name), check_model=False)
        self.query, self.evidence = tree_query_and_evidence(self.name)

    def time_query(self):
        self.inference.query(self.query, evidence=self.evidence)


class TimeApproxInference(Bench):
    def setup(self):
        self.name = MODEL
        self.inference = ApproxInference(discrete_bn.shared(self.name))
        self.query, self.evidence = query_and_evidence(self.name)

    def time_query(self):
        self.inference.query(
            self.query, n_samples=APPROX_SAMPLES, evidence=self.evidence, show_progress=False, rng=0
        )

    def time_map_query(self):
        self.inference.map_query(
            self.query, n_samples=APPROX_SAMPLES, evidence=self.evidence, show_progress=False, rng=0
        )


class TimeCausalInference(Bench):
    def setup(self):
        self.name = MODEL
        self.model = discrete_bn.shared(self.name)
        self.inference = CausalInference(self.model)
        self.x, self.y = treatment_outcome(self.name)
        last = sorted(self.model.nodes)[-1]
        self.do = first_states(self.model, [self.x])
        self.evidence = first_states(self.model, [last]) if last not in (self.x, self.y) else None

    def time_query(self):
        self.inference.query([self.y], do=self.do, evidence=self.evidence, show_progress=False)


class TimeDBNInference(Bench):
    def setup(self):
        self.name = DBN_VARS
        self.inference = DBNInference(dbn(self.name))
        variables = sorted(dbn_spec(self.name)["variables"])
        query, evidence = variables[:N_QUERY], variables[-N_EVIDENCE:]
        self.query = [(v, 2) for v in query]
        self.evidence = {(v, 0): 0 for v in evidence}

    def time_query(self):
        self.inference.query(self.query, self.evidence)


class TimeMplp(Bench):
    def setup(self):
        self.model = pairwise_markov_network(MODEL)
        self.inference = Mplp(self.model)

    def time_map_query(self):
        self.inference.map_query()
