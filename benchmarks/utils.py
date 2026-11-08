"""Benchmarks for ``pgmpy.utils``: one ``time_<function>`` per public function.

Skipped:
    optimize, pinverse: torch
    manual_pairwise_orient: interactive (calls ``input()``)
    get_example_model: deprecated
    ExperimentalWarning, StateNameMixin: not functions
    private ``_check_*`` helpers
"""

from functools import cache

import numpy as np
from benchmarks._fixtures import discrete_data, gaussian_data, mixed_data
from benchmarks._bench import DISCRETE, GAUSSIAN, ROWS, Bench

from pgmpy import utils

VARIABLES = 2
STEPS = 4
ARRAYS = 5
CARD = 6
STATES = 10


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
