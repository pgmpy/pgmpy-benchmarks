"""Benchmarks for ``pgmpy.identification``.

Inputs are the ``dag_*`` / ``admg_*`` graph fixtures (``DAG_GRAPH`` / ``ADMG_GRAPH``): ``dag_20`` /
``admg_20`` fast, ``dag_500`` / ``admg_500`` slow. The roles follow one fixed rule on every graph: the
treatment is the first node in sorted order that has a descendant (the first node itself on the ADMGs,
``X_0``; ``X_10`` on the DAGs, because ``X_0`` and ``X_1`` are leaves), and the outcome is the last
descendant of the treatment in sorted order. That gives ``X_10 -> X_8`` (``dag_20``), ``X_10 -> X_78``
(``dag_500``), ``X_0 -> X_9`` (``admg_20``) and ``X_0 -> X_319`` (``admg_500``). Descendants of an ADMG
come from ``ADMG.get_descendants`` (directed edges only), those of a DAG from ``networkx.descendants``.

Public methods: ``ID``, ``Adjustment`` and ``Frontdoor`` define none of their own except ``__init__`` and
the private ``_identify`` / ``_validate`` that the inherited public ``identify`` and ``validate`` run (the
base classes are abstract, so these two are timed on the concrete classes, whose algorithms differ).

* ``ID.identify`` runs on the ADMG and on the DAG (which it first projects to an ADMG). The identified
  expression is built once per tier and shared; ``identify`` does not modify the graph it is given.
  PYTHONHASHSEED=0: 1 ms (``admg_20``), 2 ms (``admg_500``), 1 ms (``dag_20``) and 8 ms (``dag_500``).
* ``Adjustment`` (variant ``minimal``) runs on the DAG only. On an ADMG ``identify`` raises
  ``ValueError: edge_type must be one of ...`` in ``_get_proper_backdoor_graph`` (it removes edges
  with a bare networkx edge key), so it cannot be timed there. ``identify`` takes 1 ms on ``dag_20``
  and 7 ms on ``dag_500``, ``validate`` 1 ms and 6 ms. ``variant="all"`` is not timed: it raises
  ``NetworkXError`` on ``dag_500`` and is an exponential powerset otherwise.
* ``Frontdoor`` runs on the DAG only (the only supported type). It tries every subset of the other
  observed nodes in order of size until one passes, so when no frontdoor set exists it is exponential
  in the number of nodes (``dag_20`` as a whole: > 10 s, and no set exists): the 14 / 15 / 16 / 17 / 18-node
  breadth-first prefixes of ``dag_20`` (start nodes treatment, outcome) take 0.45 / 0.9 / 1.9 / 4.0 / 8.3 s.
  Fast therefore uses the ``FRONTDOOR_NODES["fast"]`` = 12-node prefix (0.1 s, no set exists, so the whole
  powerset is searched). Slow uses all of ``dag_500``, where a set exists and is found early (0.9 s in a
  probe, 1.7 s in the harness's first call).
  The prefix is the induced subgraph on the first nodes of a breadth-first search of the skeleton from
  (treatment, outcome), visiting neighbours in sorted order (``_fixtures.bfs_prefix``, the rule
  ``bfs_subgraph`` applies to discrete specs).
* ``ProbabilityExpressionTree`` is the expression ``ID`` returns for the DAG (171 and 1063 characters of
  LaTeX, 8 and 32 leaves on ``dag_20`` / ``dag_500``); its three public methods are timed.

Skipped:
    ``ID.__call__``, ``Adjustment.__call__``, ``Frontdoor.__call__`` (trivial, alias of ``identify``),
    the constructors (trivial), ``ProbabilityNode.marginalize`` and the other tree-node classes
    (internal).
"""

from functools import cache

import networkx as nx

from benchmarks._fixtures import bfs_prefix, graph
from benchmarks._tiers import Bench

from pgmpy.identification import ID, Adjustment, Frontdoor

DAG_GRAPH = {"fast": "dag_20", "slow": "dag_500"}
ADMG_GRAPH = {"fast": "admg_20", "slow": "admg_500"}
FRONTDOOR_NODES = {"fast": 12, "slow": None}  # None: the whole graph


def descendants(g, node):
    return (
        set(g.get_descendants(node))
        if hasattr(g, "get_descendants")
        else nx.descendants(g, node)
    )


def treatment_outcome(g):
    """First node (sorted) with a descendant, and the last of its descendants (sorted)."""
    for x in sorted(g.nodes()):
        below = sorted(descendants(g, x) - {x})
        if below:
            return x, below[-1]
    raise ValueError("graph has no directed edge")


@cache
def causal_graph(name, n_nodes=None):
    """The fixture (optionally its BFS prefix) with ``exposures`` / ``outcomes`` roles; not modified by the calls."""
    g = graph(name)
    x, y = treatment_outcome(g)
    if n_nodes is not None:
        g = bfs_prefix(g.edges(), [x, y], n_nodes)
    return g.with_role("exposures", x).with_role("outcomes", y)


@cache
def adjusted(name):
    """``name`` with the ``Adjustment`` result's ``adjustment`` role."""
    return Adjustment().identify(causal_graph(name))[0]


@cache
def with_frontdoor(name, n_nodes):
    """The frontdoor graph found by ``Frontdoor`` (with its ``frontdoor`` role), or the input if none exists."""
    return Frontdoor().identify(causal_graph(name, n_nodes))[0]


@cache
def expression(name):
    return ID().identify(causal_graph(name))


class TimeID(Bench):
    def prepare(self, tier):
        self.admg = causal_graph(ADMG_GRAPH[tier])
        self.dag = causal_graph(DAG_GRAPH[tier])
        self.id = ID()

    def time_identify(self, tier):
        self.id.identify(self.admg)

    def time_identify_dag(self, tier):
        self.id.identify(self.dag)


class TimeAdjustment(Bench):
    def prepare(self, tier):
        name = DAG_GRAPH[tier]
        self.dag = causal_graph(name)
        self.adjusted = adjusted(name)
        self.adjustment = Adjustment()

    def time_identify(self, tier):
        self.adjustment.identify(self.dag)

    def time_validate(self, tier):
        self.adjustment.validate(self.adjusted)


class TimeFrontdoor(Bench):
    def prepare(self, tier):
        name, n_nodes = DAG_GRAPH[tier], FRONTDOOR_NODES[tier]
        self.dag = causal_graph(name, n_nodes)
        self.with_frontdoor = with_frontdoor(name, n_nodes)
        self.frontdoor = Frontdoor()

    def time_identify(self, tier):
        self.frontdoor.identify(self.dag)

    def time_validate(self, tier):
        self.frontdoor.validate(self.with_frontdoor)


class TimeProbabilityExpressionTree(Bench):
    def prepare(self, tier):
        self.tree = expression(DAG_GRAPH[tier])

    def time_to_latex(self, tier):
        self.tree.to_latex()

    def time_collect_node_types(self, tier):
        self.tree.collect_node_types()

    def time_find_leaves(self, tier):
        self.tree.find_leaves()
