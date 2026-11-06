"""Benchmarks for ``pgmpy.utils``.

Every public function exported by ``pgmpy.utils`` is timed, one ``time_<function>`` method each on
``TimeUtils`` (the functions take data frames or small arrays, so one class with a lean ``prepare``
serves them all), except the ones under ``Skipped``. Data inputs come from ``_fixtures``:

* ``discrete_data`` of ``asia`` (fast) / ``alarm`` (slow) at ``ROWS`` rows for the state-count
  functions (``build_state_names``, ``collect_state_names``, ``encode_columns``, ``get_state_counts``,
  ``get_state_counts_array``). The counted variable is the first column and its parents the next two,
  in column order.
* ``gaussian_data`` of ``random_gaussian_8`` (fast) / ``arth150`` (slow) at ``ROWS`` rows for
  ``discretize`` (every column to 5 bins, ``rounding`` and ``quantile``).
* ``mixed_data`` of the same Gaussian models (every other column cut into 3 string categories) for
  ``preprocess_data`` and ``get_dataset_type``.
* ``to_timeseries_format`` needs wide frames with ``(variable, timestep)`` columns. The input is the
  first ``VARIABLES * STEPS`` columns of ``gaussian_data`` relabelled ``(variable j, timestep t)`` in
  column order, with the first ``TS_ROWS`` rows (``pd-list`` builds one data frame per row, so the
  slow tier uses 5000 rows, not 100000). One method per return format.
* ``cartesian`` takes ``ARRAYS`` arrays of ``arange(CARD)``; ``sample_discrete`` draws ``ROWS`` values
  from fixed pmfs (multinomial draws of 16 over 10 states from ``numpy.random.default_rng(0)``, divided by 16 so they sum to exactly 1), either one pmf
  or a ``(ROWS, 10)`` matrix cycling through 4 distinct pmfs (one pmf per draw).

Skipped:
    ``optimize`` and ``pinverse`` (torch), ``manual_pairwise_orient`` (interactive: it calls
    ``input()``), ``get_example_model`` (deprecated), ``ExperimentalWarning`` and ``StateNameMixin``
    (not functions), the private ``_check_*`` helpers.
"""

from functools import cache

import numpy as np
from benchmarks._fixtures import discrete_data, gaussian_data, mixed_data
from benchmarks._tiers import ROWS, Bench

from pgmpy import utils

DISCRETE = {"fast": "asia", "slow": "alarm"}
GAUSSIAN = {"fast": "random_gaussian_8", "slow": "arth150"}
VARIABLES = {"fast": 2, "slow": 20}
STEPS = {"fast": 4, "slow": 5}
TS_ROWS = {"fast": 1_000, "slow": 5_000}
ARRAYS = {"fast": 5, "slow": 7}
CARD = {"fast": 6, "slow": 8}
STATES = 10


@cache
def counting_inputs(tier):
    data = discrete_data(DISCRETE[tier], ROWS[tier])
    state_names = utils.build_state_names(data)
    codes, cardinalities = utils.encode_columns(data, state_names)
    return data, state_names, codes, cardinalities


@cache
def continuous_inputs(tier):
    """``(gaussian frame, mixed frame)``, built once per tier; none of the timed functions modifies them."""
    return gaussian_data(GAUSSIAN[tier], ROWS[tier]), mixed_data(GAUSSIAN[tier], ROWS[tier])


@cache
def timeseries_frame(tier):
    data = gaussian_data(GAUSSIAN[tier], TS_ROWS[tier]).iloc[:, : VARIABLES[tier] * STEPS[tier]]
    data.columns = [(f"V{j}", t) for j in range(VARIABLES[tier]) for t in range(STEPS[tier])]
    return data


@cache
def pmfs(tier):
    rng = np.random.default_rng(0)
    # Multiples of 1/16, so each pmf sums to exactly 1 (sample_discrete adjusts inexact ones in place).
    table = rng.multinomial(16, np.ones(STATES) / STATES, size=4) / 16
    return table[0], table[np.arange(ROWS[tier]) % 4]


class TimeUtils(Bench):
    def prepare(self, tier):
        self.data, self.state_names, self.codes, self.cardinalities = counting_inputs(tier)
        self.gaussian, self.mixed = continuous_inputs(tier)
        self.variable, *self.parents = self.data.columns[:3]

    def time_collect_state_names(self, tier):
        utils.collect_state_names(self.data, self.variable)

    def time_build_state_names(self, tier):
        utils.build_state_names(self.data)

    def time_get_state_counts(self, tier):
        utils.get_state_counts(self.data, self.state_names, self.variable, self.parents)

    def time_encode_columns(self, tier):
        utils.encode_columns(self.data, self.state_names)

    def time_get_state_counts_array(self, tier):
        utils.get_state_counts_array(self.codes, self.cardinalities, self.variable, self.parents)

    def time_discretize(self, tier):
        utils.discretize(self.gaussian, dict.fromkeys(self.gaussian.columns, 5))

    def time_discretize_quantile(self, tier):
        utils.discretize(self.gaussian, dict.fromkeys(self.gaussian.columns, 5), method="quantile")

    def time_preprocess_data(self, tier):
        utils.preprocess_data(self.mixed)

    def time_get_dataset_type(self, tier):
        utils.get_dataset_type(self.mixed)

    def time_cartesian(self, tier):
        utils.cartesian([np.arange(CARD[tier])] * ARRAYS[tier])

    def time_sample_discrete(self, tier):
        utils.sample_discrete(np.arange(STATES), pmfs(tier)[0], size=ROWS[tier], rng=0)

    def time_sample_discrete_per_draw(self, tier):
        utils.sample_discrete(np.arange(STATES), pmfs(tier)[1], size=ROWS[tier], rng=0)


class TimeToTimeseriesFormat(Bench):
    params = [["fast", "slow"], ["pd-multiindex", "numpy3d", "pd-list", "sorted"]]
    param_names = ["tier", "return_format"]

    def prepare(self, tier, return_format):
        self.data = timeseries_frame(tier)

    def time_to_timeseries_format(self, tier, return_format):
        utils.to_timeseries_format(self.data, return_format=return_format)
