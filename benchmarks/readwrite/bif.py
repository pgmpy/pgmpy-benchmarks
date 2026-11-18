"""Benchmarks for ``pgmpy.readwrite.BIF``: only ``BIFReader.read`` and ``str(BIFWriter)``."""

from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.readwrite import BIFReader, BIFWriter

DISCRETE = "asia"


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


class TimeBIFReader(Bench):
    def setup(self):
        self.text = str(BIFWriter(discrete_bn(DISCRETE)))

    def time_read(self):
        BIFReader(string=self.text).read()


class TimeBIFWriter(Bench):
    def setup(self):
        self.model = discrete_bn.shared(DISCRETE)

    def time_str(self):
        str(BIFWriter(self.model))
