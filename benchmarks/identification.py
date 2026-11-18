"""Benchmarks for ``pgmpy.identification``: only ``identify`` and ``validate``. ``Adjustment`` and ``Frontdoor`` run on
the DAG only, ``Frontdoor`` on a ``FRONTDOOR_NODES`` BFS prefix (its search is exponential when no set exists).
"""

from collections import defaultdict
from functools import cache

import networkx as nx
import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.base import DAG, MAG
from pgmpy.identification import ID, Adjustment, Frontdoor

EDGE_PROB = 0.15
GRAPH_NODES = 20
LATENT_FRACTION = 0.1
DAG_GRAPH = "dag"
FRONTDOOR_NODES = 12


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


def descendants(g, node):
    return (
        set(g.get_descendants(node))
        if hasattr(g, "get_descendants")
        else nx.descendants(g, node)
    )


def treatment_outcome(g):
    for x in sorted(g.nodes()):
        below = sorted(descendants(g, x) - {x})
        if below:
            return x, below[-1]


@cache
def causal_graph(name, n_nodes=None):
    g = graph(name)
    x, y = treatment_outcome(g)
    if n_nodes is not None:
        g = bfs_prefix(g.edges(), [x, y], n_nodes)
    return g.with_role("exposures", x).with_role("outcomes", y)


@cache
def adjusted(name):
    return Adjustment().identify(causal_graph(name))[0]


@cache
def with_frontdoor(name, n_nodes):
    return Frontdoor().identify(causal_graph(name, n_nodes))[0]


class TimeID(Bench):
    def setup(self):
        self.admg = causal_graph("admg")
        self.dag = causal_graph(DAG_GRAPH)
        self.id = ID()

    def time_identify(self):
        self.id.identify(self.admg)

    def time_identify_dag(self):
        self.id.identify(self.dag)


class TimeAdjustment(Bench):
    def setup(self):
        name = DAG_GRAPH
        self.dag = causal_graph(name)
        self.adjusted = adjusted(name)
        self.adjustment = Adjustment()

    def time_identify(self):
        self.adjustment.identify(self.dag)

    def time_validate(self):
        self.adjustment.validate(self.adjusted)


class TimeFrontdoor(Bench):
    def setup(self):
        name, n_nodes = DAG_GRAPH, FRONTDOOR_NODES
        self.dag = causal_graph(name, n_nodes)
        self.with_frontdoor = with_frontdoor(name, n_nodes)
        self.frontdoor = Frontdoor()

    def time_identify(self):
        self.frontdoor.identify(self.dag)

    def time_validate(self):
        self.frontdoor.validate(self.with_frontdoor)

