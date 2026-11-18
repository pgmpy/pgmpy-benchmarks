"""Benchmarks for ``pgmpy.readwrite.UAI``: only ``UAIReader.read`` and ``str(UAIWriter)``."""

from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.readwrite import UAIReader, UAIWriter

DISCRETE = "asia"


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


class TimeUAIReader(Bench):
    def setup(self):
        self.text = str(UAIWriter(discrete_bn(DISCRETE), round_values=4))  # UAIReader cannot parse "1e-05"

    def time_read(self):
        UAIReader(string=self.text).read()


class TimeUAIWriter(Bench):
    def setup(self):
        self.model = discrete_bn.shared(DISCRETE)

    def time_str(self):
        str(UAIWriter(self.model))
