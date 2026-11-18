"""Benchmarks for ``pgmpy.base.SimpleCausalModel``."""

from benchmarks._bench import Bench

from pgmpy.base import SimpleCausalModel


class TimeSimpleCausalModel(Bench):
    def time_init(self):
        SimpleCausalModel(exposures=2, outcomes=2, confounders=4, mediators=4, instruments=4)
