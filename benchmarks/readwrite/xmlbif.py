"""Benchmarks for ``pgmpy.readwrite.XMLBIF``: only ``XMLBIFReader.read`` and ``str(XMLBIFWriter)``."""

from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.readwrite import XMLBIFReader, XMLBIFWriter

DISCRETE = "asia"


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


class TimeXMLBIFReader(Bench):
    def setup(self):
        self.text = str(XMLBIFWriter(discrete_bn(DISCRETE)))

    def time_read(self):
        XMLBIFReader(string=self.text).read()


class TimeXMLBIFWriter(Bench):
    def setup(self):
        self.model = discrete_bn.shared(DISCRETE)

    def time_str(self):
        str(XMLBIFWriter(self.model))
