"""Benchmarks for ``pgmpy.readwrite``, on text from pgmpy's writers (PomdpX: ``benchmarks/data/rocksample.pomdpx``).

Skipped:
    BaseReader.get_model, write_bif / write_uai / write_net / write_xmlbif / write_xbn / write_xdsl: deprecated
    XBNReader.get_analysisnotebook_values, .get_bnmodel_name, XMLBIFReader / XDSLReader / NETReader.get_edges,
    UAIReader.get_variables, PomdpXReader.get_description, .get_discount: trivial
    PomdpXReader.get_parameter_dd: no decision-diagram input; XMLBeliefNetwork: alias of the XBN classes
    writers' get_* / set_* (timed through the constructor), indent, BIF_templates, NET_templates, net_cpd,
    PomdpXWriter.add_conditions, .add_parameter_dd, .__str__: internal
"""

import random
import tempfile
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


@fresh
def pomdpx_model_data(name):
    reader = PomdpXReader(string=readwrite_text("pomdpx", name))
    return {
        "description": reader.get_description(),
        "discount": str(reader.get_discount()),
        "variables": reader.get_variables(),
        "initial_state_belief": reader.get_initial_beliefs(),
        "state_transition_function": reader.get_state_transition_function(),
        "obs_function": reader.get_obs_function(),
        "reward_function": reader.get_reward_function(),
    }


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


NAME = {"json": GAUSSIAN, "pomdpx": "rocksample"}


class ReaderBench(Bench):
    def setup(self):
        self.text = readwrite_text(self.fmt, NAME.get(self.fmt, DISCRETE))


class GettersBench(ReaderBench):
    def setup(self):
        super().setup()
        self.reader = self.reader_class(string=self.text)


class _Read(ReaderBench):
    def time_read(self):
        self.reader_class(string=self.text).read()


class _Writer(Bench):
    def setup(self):
        self.model = discrete_bn.shared(DISCRETE)
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "model"

    def teardown(self):
        self.dir.cleanup()

    def time_str(self):
        str(self.writer(self.model))


class _FileWriter(_Writer):
    def time_write(self):
        self.writer(self.model).write(self.path)


class TimeBIFReader(_Read):
    fmt, reader_class = "bif", BIFReader


class TimeBIFReaderGetters(GettersBench):
    fmt, reader_class = "bif", BIFReader

    def time_get_variable_grammar(self):
        self.reader.get_variable_grammar()

    def time_get_probability_grammar(self):
        self.reader.get_probability_grammar()


class TimeBIFWriter(_FileWriter):
    writer = BIFWriter


class TimeXMLBIFReader(_Read):
    fmt, reader_class = "xmlbif", XMLBIFReader


class TimeXMLBIFReaderGetters(GettersBench):
    fmt, reader_class = "xmlbif", XMLBIFReader

    def time_get_variables(self):
        self.reader.get_variables()

    def time_get_states(self):
        self.reader.get_states()

    def time_get_parents(self):
        self.reader.get_parents()

    def time_get_values(self):
        self.reader.get_values()

    def time_get_property(self):
        self.reader.get_property()


class TimeXMLBIFWriter(_FileWriter):
    writer = XMLBIFWriter


class TimeXBNReader(_Read):
    fmt, reader_class = "xbn", XBNReader


class TimeXBNReaderGetters(GettersBench):
    fmt, reader_class = "xbn", XBNReader

    def time_get_static_properties(self):
        self.reader.get_static_properties()

    def time_get_variables(self):
        self.reader.get_variables()

    def time_get_edges(self):
        self.reader.get_edges()

    def time_get_distributions(self):
        self.reader.get_distributions()


class TimeXBNWriter(_FileWriter):
    writer = XBNWriter


class TimeXDSLReader(_Read):
    fmt, reader_class = "xdsl", XDSLReader


class TimeXDSLReaderGetters(GettersBench):
    fmt, reader_class = "xdsl", XDSLReader

    def time_get_variables(self):
        self.reader.get_variables()

    def time_get_parents(self):
        self.reader.get_parents()

    def time_get_states(self):
        self.reader.get_states()

    def time_get_values(self):
        self.reader.get_values()


class TimeXDSLWriter(_FileWriter):
    writer = XDSLWriter


class TimeUAIReader(_Read):
    fmt, reader_class = "uai", UAIReader


class TimeUAIReaderGetters(GettersBench):
    fmt, reader_class = "uai", UAIReader

    def time_get_grammar(self):
        self.reader.get_grammar()

    def time_get_network_type(self):
        self.reader.get_network_type()

    def time_get_domain(self):
        self.reader.get_domain()

    def time_get_edges(self):
        self.reader.get_edges()

    def time_get_tables(self):
        self.reader.get_tables()


class TimeUAIWriter(_Writer):
    writer = UAIWriter


class TimeNETReader(_Read):
    fmt, reader_class = "net", NETReader


class TimeNETReaderGetters(GettersBench):
    fmt, reader_class = "net", NETReader

    def time_get_variable_grammar(self):
        self.reader.get_variable_grammar()

    def time_get_probability_grammar(self):
        self.reader.get_probability_grammar()

    def time_get_network_name(self):
        self.reader.get_network_name()

    def time_get_variables(self):
        self.reader.get_variables()

    def time_get_states(self):
        self.reader.get_states()

    def time_get_property(self):
        self.reader.get_property()

    def time_get_parents(self):
        self.reader.get_parents()

    def time_get_values(self):
        self.reader.get_values()


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

    def time_init(self):
        PomdpXReader(string=self.text)


class TimePomdpXReaderGetters(GettersBench):
    fmt, reader_class = "pomdpx", PomdpXReader

    def setup(self):
        super().setup()
        self.cond_prob = self.reader.network.find("StateTransitionFunction").find("CondProb")
        self.parameter = self.cond_prob.find("Parameter")

    def time_get_variables(self):
        self.reader.get_variables()

    def time_get_initial_beliefs(self):
        self.reader.get_initial_beliefs()

    def time_get_state_transition_function(self):
        self.reader.get_state_transition_function()

    def time_get_obs_function(self):
        self.reader.get_obs_function()

    def time_get_reward_function(self):
        self.reader.get_reward_function()

    def time_get_parameter(self):
        self.reader.get_parameter(self.cond_prob)

    def time_get_parameter_tbl(self):
        self.reader.get_parameter_tbl(self.parameter)


class TimePomdpXWriter(Bench):
    def setup(self):
        self.data = pomdpx_model_data("rocksample")  # fresh: the writer is given the dict, not a copy

    def time_init(self):
        PomdpXWriter(model_data=self.data)

    def time_get_variables(self):
        PomdpXWriter(model_data=self.data).get_variables()

    def time_add_initial_belief(self):
        PomdpXWriter(model_data=self.data).add_initial_belief()

    def time_add_state_transition_function(self):
        PomdpXWriter(model_data=self.data).add_state_transition_function()

    def time_add_obs_function(self):
        PomdpXWriter(model_data=self.data).add_obs_function()

    def time_add_reward_function(self):
        PomdpXWriter(model_data=self.data).add_reward_function()
