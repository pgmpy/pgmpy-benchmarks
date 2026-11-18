"""Benchmarks for ``pgmpy.readwrite.XMLBeliefNetwork``: only ``XBNReader.read`` and ``str(XBNWriter)``."""

from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.readwrite import XBNReader, XBNWriter

DISCRETE = "asia"


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


class TimeXBNReader(Bench):
    def setup(self):
        self.text = str(XBNWriter(discrete_bn(DISCRETE)))

    def time_read(self):
        XBNReader(string=self.text).read()


class TimeXBNWriter(Bench):
    def setup(self):
        self.model = discrete_bn.shared(DISCRETE)

    def time_str(self):
        str(XBNWriter(self.model))
