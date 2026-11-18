"""Benchmarks for ``pgmpy.utils``: one ``time_<function>`` per public function.

Skipped:
    optimize, pinverse: torch
    manual_pairwise_orient: interactive (calls ``input()``)
    get_example_model: deprecated
    ExperimentalWarning, StateNameMixin: not functions
    private ``_check_*`` helpers
"""

from functools import cache

import networkx as nx
import numpy as np
import pandas as pd
from benchmarks._bench import Bench, fresh

from pgmpy import utils
from pgmpy.example_models import load_model
from pgmpy.models import LinearGaussianBayesianNetwork

ROWS = 1_000
DISCRETE = "asia"
GAUSSIAN = 8
VARIABLES = 2
STEPS = 4
ARRAYS = 5
CARD = 6
STATES = 10


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


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


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


@cache
def counting_inputs():
    data = discrete_data(DISCRETE, ROWS)
    state_names = utils.build_state_names(data)
    codes, cardinalities = utils.encode_columns(data, state_names)
    return data, state_names, codes, cardinalities


@cache
def continuous_inputs():
    return gaussian_data(GAUSSIAN, ROWS), mixed_data(GAUSSIAN, ROWS)


@cache
def timeseries_frame():
    data = gaussian_data(GAUSSIAN, ROWS).iloc[:, : VARIABLES * STEPS]
    data.columns = [(f"V{j}", t) for j in range(VARIABLES) for t in range(STEPS)]
    return data


@cache
def pmfs():
    rng = np.random.default_rng(0)
    table = rng.multinomial(16, np.ones(STATES) / STATES, size=4) / 16  # multiples of 1/16 sum to exactly 1
    return table[0], table[np.arange(ROWS) % 4]


class TimeUtils(Bench):
    def setup(self):
        self.data, self.state_names, self.codes, self.cardinalities = counting_inputs()
        self.gaussian, self.mixed = continuous_inputs()
        self.variable, *self.parents = self.data.columns[:3]

    def time_collect_state_names(self):
        utils.collect_state_names(self.data, self.variable)

    def time_build_state_names(self):
        utils.build_state_names(self.data)

    def time_get_state_counts(self):
        utils.get_state_counts(self.data, self.state_names, self.variable, self.parents)

    def time_encode_columns(self):
        utils.encode_columns(self.data, self.state_names)

    def time_get_state_counts_array(self):
        utils.get_state_counts_array(self.codes, self.cardinalities, self.variable, self.parents)

    def time_discretize(self):
        utils.discretize(self.gaussian, dict.fromkeys(self.gaussian.columns, 5))

    def time_discretize_quantile(self):
        utils.discretize(self.gaussian, dict.fromkeys(self.gaussian.columns, 5), method="quantile")

    def time_preprocess_data(self):
        utils.preprocess_data(self.mixed)

    def time_get_dataset_type(self):
        utils.get_dataset_type(self.mixed)

    def time_cartesian(self):
        utils.cartesian([np.arange(CARD)] * ARRAYS)

    def time_sample_discrete(self):
        utils.sample_discrete(np.arange(STATES), pmfs()[0], size=ROWS, rng=0)

    def time_sample_discrete_per_draw(self):
        utils.sample_discrete(np.arange(STATES), pmfs()[1], size=ROWS, rng=0)


class TimeToTimeseriesFormat(Bench):
    params = [["pd-multiindex", "numpy3d", "pd-list", "sorted"]]
    param_names = ["return_format"]

    def setup(self, return_format):
        self.data = timeseries_frame()

    def time_to_timeseries_format(self, return_format):
        utils.to_timeseries_format(self.data, return_format=return_format)
