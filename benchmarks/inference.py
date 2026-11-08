"""Benchmarks for ``pgmpy.inference``.

Skipped:
    CausalInference.is_valid_backdoor_adjustment_set, .get_all_backdoor_adjustment_sets,
    .is_valid_frontdoor_adjustment_set, .get_all_frontdoor_adjustment_sets: deprecated
    BeliefPropagation.get_cliques, .get_clique_beliefs, .get_sepset_beliefs, Mplp.get_integrality_gap: trivial
    Mplp.query: unsupported (raises NotImplementedError)
"""

import copy
from functools import cache

import networkx as nx
from benchmarks._fixtures import (
    dbn,
    dbn_spec,
    discrete_bn,
    discrete_data,
    first_states,
    gaussian_bn,
    gaussian_data,
    junction_tree,
    pairwise_markov_network,
    tree_factor_graph,
)
from benchmarks._bench import DISCRETE, GAUSSIAN, MODEL, ROWS, Bench
from benchmarks.models import DBN_VARS

from pgmpy.inference import (
    ApproxInference,
    BeliefPropagation,
    BeliefPropagationWithMessagePassing,
    CausalInference,
    DBNInference,
    Mplp,
    VariableElimination,
)
from pgmpy.inference.EliminationOrder import MinNeighbors

APPROX_SAMPLES = 500

N_QUERY = 4
N_EVIDENCE = 2


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
def elimination_order(name):
    m = discrete_bn.shared(name)
    return MinNeighbors(m).get_elimination_order(nodes=set(m.nodes), show_progress=False)


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


@cache
def ate_pair(name):
    m = gaussian_bn(name)
    g = nx.DiGraph(m.edges())
    return next(
        (x, y) for x, y in sorted(m.edges()) if m.in_degree(x) == 0 and len(list(nx.all_simple_paths(g, x, y))) == 1
    )


class TimeVariableElimination(Bench):
    def setup(self):
        self.name = MODEL
        self.model = discrete_bn.shared(self.name)
        self.inference = VariableElimination(self.model)
        self.query, self.evidence = query_and_evidence(self.name)
        self.order = elimination_order(self.name)

    def time_init(self):
        VariableElimination(self.model)

    def time_query(self):
        self.inference.query(self.query, evidence=self.evidence, show_progress=False)

    def time_max_marginal(self):
        self.inference.max_marginal(self.query, evidence=self.evidence, show_progress=False)

    def time_map_query(self):
        self.inference.map_query(self.query, evidence=self.evidence, show_progress=False)

    def time_induced_graph(self):
        self.inference.induced_graph(self.order)

    def time_induced_width(self):
        self.inference.induced_width(self.order)


class TimeBeliefPropagation(Bench):
    def setup(self):
        self.tree = MODEL
        self.inference = copy.deepcopy(calibrated_bp(self.tree))
        self.query, self.evidence = bp_query_and_evidence(self.tree)

    def time_init(self):
        BeliefPropagation(junction_tree(self.tree))

    def time_calibrate(self):
        self.inference.calibrate()

    def time_max_calibrate(self):
        self.inference.max_calibrate()

    def time_query(self):
        self.inference.query(self.query, evidence=self.evidence, show_progress=False)

    def time_map_query(self):
        self.inference.map_query(self.query, evidence=self.evidence, show_progress=False)


class TimeBeliefPropagationWithMessagePassing(Bench):
    def setup(self):
        self.name = MODEL
        self.factor_graph = tree_graph(self.name)
        self.inference = BeliefPropagationWithMessagePassing(self.factor_graph, check_model=False)
        self.query, self.evidence = tree_query_and_evidence(self.name)
        self.variable = self.query[0]
        self.factor = next(f for f in self.factor_graph.get_factors() if f.variables[0] == self.variable)
        self.factor_messages = [self.factor_graph.get_uniform_message(v) for v in self.factor.variables[1:]]
        self.variable_messages = [self.factor_graph.get_uniform_message(self.variable)] * 3

    def time_init(self):
        BeliefPropagationWithMessagePassing(self.factor_graph)

    def time_query(self):
        self.inference.query(self.query, evidence=self.evidence)

    def time_calc_variable_node_message(self):
        self.inference.calc_variable_node_message(self.variable, self.variable_messages)

    def time_calc_factor_node_message(self):
        self.inference.calc_factor_node_message(self.factor, self.factor_messages, self.variable)


class TimeApproxInference(Bench):
    def setup(self):
        self.name = MODEL
        self.model = discrete_bn.shared(self.name)
        self.inference = ApproxInference(self.model)
        self.query, self.evidence = query_and_evidence(self.name)
        self.samples = discrete_data(self.name, APPROX_SAMPLES)
        self.state_names = {v: self.model.states[v] for v in self.query}

    def time_init(self):
        ApproxInference(self.model)

    def time_get_distribution(self):
        self.inference.get_distribution(self.samples, self.query, state_names=self.state_names)

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
        self.adjustment = sorted(self.model.get_parents(self.x))
        last = sorted(self.model.nodes)[-1]
        self.do = first_states(self.model, [self.x])
        self.evidence = first_states(self.model, [last]) if last not in (self.x, self.y) else None

    def time_init(self):
        CausalInference(self.model)

    def time_query(self):
        self.inference.query([self.y], do=self.do, evidence=self.evidence, show_progress=False)

    def time_get_minimal_adjustment_set(self):
        self.inference.get_minimal_adjustment_set(self.x, self.y)

    def time_get_proper_backdoor_graph(self):
        self.inference.get_proper_backdoor_graph([self.x], [self.y])

    def time_is_valid_adjustment_set(self):
        self.inference.is_valid_adjustment_set([self.x], [self.y], self.adjustment)

    def time_get_scaling_indicators(self):
        self.inference.get_scaling_indicators()

    def time_get_ivs(self):
        self.inference.get_ivs(self.x, self.y)

    def time_get_conditional_ivs(self):
        self.inference.get_conditional_ivs(self.x, self.y)

    def time_get_total_conditional_ivs(self):
        self.inference.get_total_conditional_ivs(self.x, self.y)


class TimeCausalInferenceIdentification(Bench):
    def setup(self):
        name = DISCRETE
        self.x, self.y = treatment_outcome(name)
        self.inference = CausalInference(discrete_bn.shared(name))

    def time_identification_method(self):
        self.inference.identification_method(self.x, self.y)


class TimeCausalInferenceEstimateATE(Bench):
    def setup(self):
        self.name = GAUSSIAN
        self.inference = CausalInference(gaussian_bn(self.name))
        self.x, self.y = ate_pair(self.name)
        self.data = gaussian_data(self.name, ROWS)

    def time_estimate_ate(self):
        self.inference.estimate_ate(self.x, self.y, self.data)


class TimeDBNInference(Bench):
    def setup(self):
        self.name = DBN_VARS
        self.model = dbn(self.name)
        self.inference = DBNInference(self.model)
        variables = sorted(dbn_spec(self.name)["variables"])
        query, evidence = variables[:N_QUERY], variables[-N_EVIDENCE:]
        self.query = [(v, 2) for v in query]
        self.backward_query = [(v, 1) for v in query]
        self.evidence = {(v, 0): 0 for v in evidence}
        self.backward_evidence = {(v, 2): 0 for v in evidence}

    def time_init(self):
        DBNInference(self.model)

    def time_forward_inference(self):
        self.inference.forward_inference(self.query, self.evidence)

    def time_backward_inference(self):
        self.inference.backward_inference(self.backward_query, self.backward_evidence)

    def time_query(self):
        self.inference.query(self.query, self.evidence)


class TimeMplp(Bench):
    def setup(self):
        self.model = pairwise_markov_network(MODEL)
        self.inference = Mplp(self.model)

    def time_init(self):
        Mplp(self.model)

    def time_map_query(self):
        self.inference.map_query()

    def time_find_triangles(self):
        self.inference.find_triangles()
