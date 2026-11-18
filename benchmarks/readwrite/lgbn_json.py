"""Benchmarks for ``pgmpy.readwrite.LGBNJSON``: only ``LGBNJSONReader.read`` and ``str(LGBNJSONWriter)``."""

from benchmarks._bench import Bench, fresh

from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.readwrite import LGBNJSONReader, LGBNJSONWriter

GAUSSIAN = 8


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


class TimeLGBNJSONReader(Bench):
    def setup(self):
        self.text = str(LGBNJSONWriter(gaussian_bn(GAUSSIAN)))

    def time_read(self):
        LGBNJSONReader(string=self.text).read()


class TimeLGBNJSONWriter(Bench):
    def setup(self):
        self.model = gaussian_bn.shared(GAUSSIAN)

    def time_str(self):
        str(LGBNJSONWriter(self.model))
