"""Benchmarks for ``pgmpy.readwrite``, on text from pgmpy's writers (PomdpX: ``benchmarks/data/rocksample.pomdpx``).

``Time<Format>Reader.time_read``: construct from ``string=`` and ``read()``; ``Time<Format>ReaderGetters``: the parsing ``get_*`` methods.
``Time<Format>Writer``: ``time_str`` and ``time_write`` (into a temp dir). Inherited ``BaseWriter.write`` is timed once, on BIFWriter.
UAI files use ``round_values=4`` (the reader cannot parse "1e-05"); ``XDSLWriter`` output is random but its work is not.

Skipped:
    BaseReader.get_model, write_bif / write_uai / write_net / write_xmlbif / write_xbn / write_xdsl: deprecated
    XMLBeliefNetwork: alias of the XBN classes
    XBNReader.get_analysisnotebook_values, .get_bnmodel_name, XMLBIFReader.get_edges, XDSLReader.get_edges,
    NETReader.get_edges, UAIReader.get_variables, PomdpXReader.get_description, .get_discount: trivial
    PomdpXReader.get_parameter_dd: no decision-diagram input
    writers' get_* / set_* methods: timed through the constructor
    indent, BIF_templates, NET_templates, net_cpd, PomdpXWriter.add_conditions, .add_parameter_dd, .__str__: internal
"""

import tempfile
from functools import cache
from pathlib import Path

from benchmarks._fixtures import discrete_bn, gaussian_bn, pomdpx_model_data, readwrite_text
from benchmarks._bench import DISCRETE, GAUSSIAN, Bench

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

FILE = {
    "bif": "asia",
    "xmlbif": "asia",
    "xbn": "asia",
    "xdsl": "asia",
    "uai": "asia",
    "net": "asia",
    "json": "random_gaussian_8",
    "pomdpx": "rocksample",
}


@cache
def _discrete():
    return discrete_bn(DISCRETE)


@cache
def _gaussian():
    return gaussian_bn(GAUSSIAN)


class ReaderBench(Bench):
    """A reader class's input: the text of its file."""

    fmt = None

    def setup(self):
        self.text = readwrite_text(self.fmt, FILE[self.fmt])


class GettersBench(ReaderBench):
    """A reader constructed once per call, for the timing of its ``get_*`` methods."""

    reader_class = None

    def setup(self):
        super().setup()
        self.reader = self.reader_class(string=self.text)


class WriterBench(Bench):
    """A writer class's input: the model and a directory to write into."""

    def setup(self):
        self.model = _discrete()
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "model"

    def teardown(self):
        self.dir.cleanup()


class TimeBIFReader(ReaderBench):
    fmt = "bif"

    def time_read(self):
        BIFReader(string=self.text).read()


class TimeBIFReaderGetters(GettersBench):
    fmt = "bif"
    reader_class = BIFReader

    def time_get_variable_grammar(self):
        self.reader.get_variable_grammar()

    def time_get_probability_grammar(self):
        self.reader.get_probability_grammar()


class TimeBIFWriter(WriterBench):
    def time_str(self):
        str(BIFWriter(self.model))

    def time_write(self):
        BIFWriter(self.model).write(self.path)


class TimeXMLBIFReader(ReaderBench):
    fmt = "xmlbif"

    def time_read(self):
        XMLBIFReader(string=self.text).read()


class TimeXMLBIFReaderGetters(GettersBench):
    fmt = "xmlbif"
    reader_class = XMLBIFReader

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


class TimeXMLBIFWriter(WriterBench):
    def time_str(self):
        str(XMLBIFWriter(self.model))

    def time_write(self):
        XMLBIFWriter(self.model).write(self.path)


class TimeXBNReader(ReaderBench):
    fmt = "xbn"

    def time_read(self):
        XBNReader(string=self.text).read()


class TimeXBNReaderGetters(GettersBench):
    fmt = "xbn"
    reader_class = XBNReader

    def time_get_static_properties(self):
        self.reader.get_static_properties()

    def time_get_variables(self):
        self.reader.get_variables()

    def time_get_edges(self):
        self.reader.get_edges()

    def time_get_distributions(self):
        self.reader.get_distributions()


class TimeXBNWriter(WriterBench):
    def time_str(self):
        str(XBNWriter(self.model))

    def time_write(self):
        XBNWriter(self.model).write(self.path)


class TimeXDSLReader(ReaderBench):
    fmt = "xdsl"

    def time_read(self):
        XDSLReader(string=self.text).read()


class TimeXDSLReaderGetters(GettersBench):
    fmt = "xdsl"
    reader_class = XDSLReader

    def time_get_variables(self):
        self.reader.get_variables()

    def time_get_parents(self):
        self.reader.get_parents()

    def time_get_states(self):
        self.reader.get_states()

    def time_get_values(self):
        self.reader.get_values()


class TimeXDSLWriter(WriterBench):
    def time_str(self):
        str(XDSLWriter(self.model))

    def time_write(self):
        XDSLWriter(self.model).write(self.path)


class TimeUAIReader(ReaderBench):
    fmt = "uai"

    def time_read(self):
        UAIReader(string=self.text).read()


class TimeUAIReaderGetters(GettersBench):
    fmt = "uai"
    reader_class = UAIReader

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


class TimeUAIWriter(WriterBench):
    def time_str(self):
        str(UAIWriter(self.model))


class TimeNETReader(ReaderBench):
    fmt = "net"

    def time_read(self):
        NETReader(string=self.text).read()


class TimeNETReaderGetters(GettersBench):
    fmt = "net"
    reader_class = NETReader

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


class TimeNETWriter(WriterBench):
    def time_str(self):
        str(NETWriter(self.model))


class TimeLGBNJSONReader(ReaderBench):
    fmt = "json"

    def time_read(self):
        LGBNJSONReader(string=self.text).read()


class TimeLGBNJSONWriter(Bench):
    def setup(self):
        self.model = _gaussian()

    def time_str(self):
        str(LGBNJSONWriter(self.model))


class TimePomdpXReader(ReaderBench):
    fmt = "pomdpx"

    def time_init(self):
        PomdpXReader(string=self.text)


class TimePomdpXReaderGetters(GettersBench):
    fmt = "pomdpx"
    reader_class = PomdpXReader

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
        self.data = pomdpx_model_data(FILE["pomdpx"])  # fresh: the writer is given the dict, not a copy

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
