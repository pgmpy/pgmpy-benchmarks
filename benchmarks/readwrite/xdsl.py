"""Benchmarks for ``pgmpy.readwrite.XDSL``: only ``XDSLReader.read`` and ``str(XDSLWriter)``."""

import random

from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.readwrite import XDSLReader, XDSLWriter

DISCRETE = "asia"


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


class TimeXDSLReader(Bench):
    def setup(self):
        random.seed(0)  # XDSLWriter draws node positions from the global random module
        self.text = str(XDSLWriter(discrete_bn(DISCRETE)))

    def time_read(self):
        XDSLReader(string=self.text).read()


class TimeXDSLWriter(Bench):
    def setup(self):
        self.model = discrete_bn.shared(DISCRETE)

    def time_str(self):
        str(XDSLWriter(self.model))
