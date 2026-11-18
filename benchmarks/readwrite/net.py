"""Benchmarks for ``pgmpy.readwrite.NET``: only ``NETReader.read`` and ``str(NETWriter)``."""

from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.readwrite import NETReader, NETWriter

DISCRETE = "asia"


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


class TimeNETReader(Bench):
    def setup(self):
        self.text = str(NETWriter(discrete_bn(DISCRETE)))

    def time_read(self):
        NETReader(string=self.text).read()


class TimeNETWriter(Bench):
    def setup(self):
        self.model = discrete_bn.shared(DISCRETE)

    def time_str(self):
        str(NETWriter(self.model))
