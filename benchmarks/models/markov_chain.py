"""Benchmarks for ``pgmpy.models.MarkovChain``.

Skipped:
    MarkovChain.is_stationarity: infeasible (draws a 10000-row sample per state: 32 s for 20 x 5)
"""

import numpy as np
from benchmarks._bench import Bench, fresh

from pgmpy.factors.discrete import State
from pgmpy.models import MarkovChain

ROWS = 1_000
CHAIN_SIZE = (3, 3)


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


class TimeMarkovChain(Bench):
    def setup(self):
        n_vars, n_states = CHAIN_SIZE
        self.chain = markov_chain(n_vars, n_states)
        self.matrices = markov_chain_matrices(n_vars, n_states)
        self.variables = list(self.matrices)
        self.bare = MarkovChain(self.variables, [n_states] * n_vars)
        self.empty = MarkovChain()
        self.cards = [n_states] * n_vars
        self.start = [State(v, 1) for v in self.variables]

    def time_add_transition_model(self):
        for variable, matrix in self.matrices.items():
            self.bare.add_transition_model(variable, matrix)

    def time_add_variable(self):
        self.empty.add_variable("Z", 3)

    def time_add_variables_from(self):
        self.empty.add_variables_from(self.variables, self.cards)

    def time_copy(self):
        self.chain.copy()

    def time_generate_sample(self):
        list(self.chain.generate_sample(size=ROWS, rng=0))

    def time_random_state(self):
        self.chain.random_state()

    def time_sample(self):
        self.chain.sample(size=ROWS, rng=0)

    def time_set_start_state(self):
        self.chain.set_start_state(self.start)


class TimeMarkovChainProbFromSample(Bench):
    def setup(self):
        n_vars, n_states = CHAIN_SIZE
        self.chain = markov_chain(n_vars, n_states)
        self.sample = markov_chain_sample(n_vars, n_states, ROWS)
        self.state = [State("X0", 0), State("X1", 0)]
        self.window_size = ROWS // 10

    def time_prob_from_sample(self):
        self.chain.prob_from_sample(self.state, sample=self.sample, window_size=self.window_size)
