"""Benchmarks for ``pgmpy.readwrite``: one read and one write for each format. The readers read the text of pgmpy's
writers (PomdpX: ``benchmarks/data/rocksample.pomdpx``).

Skipped:
    BaseReader.get_model, write_bif / write_uai / write_net / write_xmlbif / write_xbn / write_xdsl: deprecated
    write: trivial (str and a file write)
    the readers' and writers' get_* / set_* / add_* (most are timed through read and str), indent, BIF_templates,
    NET_templates, net_cpd, PomdpXWriter.add_conditions, .add_parameter_dd, .__str__: internal
    XMLBeliefNetwork: alias of the XBN classes
"""

import random
from functools import cache
from pathlib import Path

from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.models import LinearGaussianBayesianNetwork
from pgmpy.readwrite import (
    BIFReader,
    BIFWriter,
    LGBNJSONReader,
    LGBNJSONWriter,
    NETReader,
    NETWriter,
    PomdpXReader,
    PomdpXWriter,
    UAIReader,
    UAIWriter,
    XBNReader,
    XBNWriter,
    XDSLReader,
    XDSLWriter,
    XMLBIFReader,
    XMLBIFWriter,
)

DISCRETE = "asia"
GAUSSIAN = 8


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


@fresh
def gaussian_bn(n):
    return LinearGaussianBayesianNetwork.get_random(n_nodes=n, edge_prob=0.3, rng=0)


@cache
def readwrite_text(fmt, name):
    """A reader's input: ``name`` written by pgmpy's writer for ``fmt``."""
    if fmt == "pomdpx":
        return (Path(__file__).parent / "data" / f"{name}.pomdpx").read_text(encoding="utf-8")
    if fmt == "json":
        return str(LGBNJSONWriter(gaussian_bn(name)))
    writers = {"bif": BIFWriter, "xmlbif": XMLBIFWriter, "xbn": XBNWriter, "xdsl": XDSLWriter, "net": NETWriter,
               "uai": lambda model: UAIWriter(model, round_values=4)}  # UAIReader cannot parse "1e-05"
    random.seed(0)  # XDSLWriter draws node positions from the global random module
    return str(writers[fmt](discrete_bn(name)))


def read_pomdpx(text):
    """PomdpX has no ``read``: the constructor and every getter."""
    reader = PomdpXReader(string=text)
    return {
        "description": reader.get_description(),
        "discount": str(reader.get_discount()),
        "variables": reader.get_variables(),
        "initial_state_belief": reader.get_initial_beliefs(),
        "state_transition_function": reader.get_state_transition_function(),
        "obs_function": reader.get_obs_function(),
        "reward_function": reader.get_reward_function(),
    }


@fresh
def pomdpx_model_data(name):
    return read_pomdpx(readwrite_text("pomdpx", name))


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


NAME = {"json": GAUSSIAN, "pomdpx": "rocksample"}


class ReaderBench(Bench):
    def setup(self):
        self.text = readwrite_text(self.fmt, NAME.get(self.fmt, DISCRETE))


class _Read(ReaderBench):
    def time_read(self):
        self.reader_class(string=self.text).read()


class _Writer(Bench):
    def setup(self):
        self.model = discrete_bn.shared(DISCRETE)

    def time_str(self):
        str(self.writer(self.model))


class TimeBIFReader(_Read):
    fmt, reader_class = "bif", BIFReader


class TimeBIFWriter(_Writer):
    writer = BIFWriter


class TimeXMLBIFReader(_Read):
    fmt, reader_class = "xmlbif", XMLBIFReader


class TimeXMLBIFWriter(_Writer):
    writer = XMLBIFWriter


class TimeXBNReader(_Read):
    fmt, reader_class = "xbn", XBNReader


class TimeXBNWriter(_Writer):
    writer = XBNWriter


class TimeXDSLReader(_Read):
    fmt, reader_class = "xdsl", XDSLReader


class TimeXDSLWriter(_Writer):
    writer = XDSLWriter


class TimeUAIReader(_Read):
    fmt, reader_class = "uai", UAIReader


class TimeUAIWriter(_Writer):
    writer = UAIWriter


class TimeNETReader(_Read):
    fmt, reader_class = "net", NETReader


class TimeNETWriter(_Writer):
    writer = NETWriter


class TimeLGBNJSONReader(_Read):
    fmt, reader_class = "json", LGBNJSONReader


class TimeLGBNJSONWriter(Bench):
    def setup(self):
        self.model = gaussian_bn.shared(GAUSSIAN)

    def time_str(self):
        str(LGBNJSONWriter(self.model))


class TimePomdpXReader(ReaderBench):
    fmt = "pomdpx"

    def time_read(self):
        read_pomdpx(self.text)


class TimePomdpXWriter(Bench):
    def setup(self):
        self.data = pomdpx_model_data("rocksample")  # fresh: the writer is given the dict, not a copy

    def time_write(self):
        """PomdpX has no ``str``: the constructor and every ``add_*``."""
        writer = PomdpXWriter(model_data=self.data)
        writer.get_variables()
        writer.add_initial_belief()
        writer.add_state_transition_function()
        writer.add_obs_function()
        writer.add_reward_function()
