"""Benchmarks for ``pgmpy.base.MAG``."""

import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.base import DAG, MAG

EDGE_PROB = 0.15
GRAPH_NODES = 20
LATENT_FRACTION = 0.1
SMALL_NODES = 15  # size of the subgraph for is_maximal, is_visible_edge and lower_manipulation


def admg(n):
    n_latent = max(1, round(n * LATENT_FRACTION))
    dag = DAG.get_random(n_nodes=n + n_latent, edge_prob=EDGE_PROB, rng=0)
    dag.latents = set(np.random.default_rng(0).choice(list(dag.nodes()), size=n_latent, replace=False).tolist())
    return dag.to_admg()


@fresh
def mag():
    g = admg(GRAPH_NODES)
    ancestors = {v: g.get_ancestors(v) for v in g.nodes()}
    edges = [
        (u, v, t)
        for u, v, t in g.get_edges(data=True)
        if t == "->" or (u not in ancestors[v] and v not in ancestors[u])
    ]
    out = MAG(edge_list=edges)
    out.add_nodes_from(g.nodes())
    return out


class TimeMAG(Bench):
    def setup(self):
        self.g = mag()
        nodes = sorted(self.g.nodes)
        self.given = nodes[1:4]
        self.small = self.g.get_subgraph(nodes[:SMALL_NODES])
        self.u, self.v, _ = min(self.small.get_edges(edge_types=["->"]))
        self.x = sorted(self.small.nodes)[:1]

    def time_is_maximal(self):
        self.small.is_maximal()

    def time_is_visible_edge(self):
        self.small.is_visible_edge(self.u, self.v)

    def time_lower_manipulation(self):
        self.small.lower_manipulation(self.x)

    def time_upper_manipulation(self):
        self.g.upper_manipulation(self.given)
