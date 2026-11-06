"""Benchmarks for ``pgmpy.readwrite``.

Inputs are frozen files in ``benchmarks/data/readwrite`` (``tools/build_fixtures.py --only-readwrite``), loaded
through ``_fixtures.readwrite_text`` / ``pomdpx_model_data``; no reader or writer produces an input here.

Readers and writers
-------------------
* **Readers** are timed by construction from ``string=`` plus ``read()`` (``Time<Format>Reader.time_read``): the
  BIF, UAI and NET constructors do all the parsing, the XML ones most of it. The public ``get_*`` methods that
  parse are timed on a reader built in ``prepare`` (``Time<Format>ReaderGetters``); the reader is not changed by
  them. ``BaseReader.get_model`` is deprecated (``read`` replaces it), so ``read`` is the one timed.
* **Writers** are timed by construction plus serialization (``Time<Format>Writer``): ``time_str`` is
  ``str(Writer(model))`` and ``time_write`` is ``Writer(model).write(path)`` into a ``TemporaryDirectory`` made in
  ``prepare`` and removed in ``teardown``. The model is the cached ``discrete_bn`` / ``gaussian_bn`` fixture, which no
  writer changes. The writer constructors already call the writers' ``get_*`` / ``set_*`` methods, so those are
  timed through them. ``write`` is inherited from ``BaseWriter`` by BIFWriter, NETWriter, UAIWriter and
  LGBNJSONWriter, and is timed once, on BIFWriter; XMLBIFWriter, XBNWriter and XDSLWriter override it.

Fixtures (``FILE`` below: fast / slow)
--------------------------------------
* BIF, XMLBIF, XBN, XDSL: ``asia`` / ``andes``.
* UAI and NET: ``asia`` / ``alarm``. At andes the readers take 18 s (UAI) and 11 s (NET) because their pyparsing
  grammars are rescanned per function / node (quadratic); alarm takes 0.9 s and 0.3 s. So the slow reader rule is
  "the largest bnlearn fixture whose read stays under about 1 s", which is alarm. Their writers still use andes.
  The UAI files are written with ``round_values=4`` because ``UAIReader`` cannot parse scientific notation
  ("1e-05"); the UAI writer benchmark uses the default (no rounding).
* LGBNJSON: ``random_gaussian_8`` / ``arth150``.
* PomdpX: ``rocksample``, the document of pgmpy's own test suite, / ``state50``, a synthetic document with 50
  state variables (3 values each), 5 observation variables and 1 action. ``PomdpXWriter`` takes a ``model_data``
  dict, which is frozen next to each document (``<name>.model.json``).

Slow tier, one call (PYTHONHASHSEED=0): UAI reader 0.9 s (its ``get_tables`` 1.4 s), BIF reader 0.75 s, NET reader
0.3 s, everything else (XML formats at andes, LGBNJSON at arth150, PomdpX at state50, all writers) a few ms.

Notes
-----
* ``XDSLWriter`` draws node positions from the global ``random`` module, so its output differs from call to call;
  the work done does not.
* ``PomdpXReader.get_parameter_dd`` is not timed: neither document has a decision-diagram parameter.

Skipped:
    ``BaseReader.get_model`` (deprecated), the ``write_bif`` / ``write_uai`` / ``write_net`` / ``write_xmlbif`` /
    ``write_xbn`` / ``write_xdsl`` aliases (deprecated), ``XMLBeliefNetwork`` (alias of the XBN classes),
    ``XBNReader.get_analysisnotebook_values`` and ``get_bnmodel_name``, ``XMLBIFReader.get_edges``,
    ``XDSLReader.get_edges``, ``NETReader.get_edges``, ``UAIReader.get_variables`` and ``PomdpXReader.get_description``
    / ``get_discount`` (trivial: they return stored attributes or one tag), ``PomdpXReader.get_parameter_dd`` (no
    decision-diagram input), the writers' ``get_*`` / ``set_*`` methods (constructor, timed by it), ``indent``,
    ``BIF_templates``, ``NET_templates``, ``net_cpd``, ``PomdpXWriter.add_conditions`` / ``add_parameter_dd`` /
    ``__str__`` (internal helpers of the methods that are timed).
"""

import tempfile
from functools import cache
from pathlib import Path

from benchmarks._fixtures import discrete_bn, gaussian_bn, pomdpx_model_data, readwrite_text
from benchmarks._tiers import Bench

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

FILE = {  # format: {tier: file name} of the files the readers parse
    "bif": {"fast": "asia", "slow": "andes"},
    "xmlbif": {"fast": "asia", "slow": "andes"},
    "xbn": {"fast": "asia", "slow": "andes"},
    "xdsl": {"fast": "asia", "slow": "andes"},
    "uai": {"fast": "asia", "slow": "alarm"},
    "net": {"fast": "asia", "slow": "alarm"},
    "json": {"fast": "random_gaussian_8", "slow": "arth150"},
    "pomdpx": {"fast": "rocksample", "slow": "state50"},
}
DISCRETE_MODEL = {"fast": "asia", "slow": "andes"}  # the model every discrete writer serializes
GAUSSIAN_MODEL = {"fast": "random_gaussian_8", "slow": "arth150"}


@cache
def _discrete(tier):
    return discrete_bn(DISCRETE_MODEL[tier])  # writers only read the model


@cache
def _gaussian(tier):
    return gaussian_bn(GAUSSIAN_MODEL[tier])


class ReaderBench(Bench):
    """A reader class's input: the text of its file."""

    fmt = None

    def prepare(self, tier):
        self.text = readwrite_text(self.fmt, FILE[self.fmt][tier])


class GettersBench(ReaderBench):
    """A reader constructed once per call, for the timing of its ``get_*`` methods."""

    reader_class = None

    def prepare(self, tier):
        super().prepare(tier)
        self.reader = self.reader_class(string=self.text)


class WriterBench(Bench):
    """A writer class's input: the model and a directory to write into."""

    def prepare(self, tier):
        self.model = _discrete(tier)
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "model"

    def teardown(self, tier):
        self.dir.cleanup()


# --- BIF ---------------------------------------------------------------------------------------------------


class TimeBIFReader(ReaderBench):
    fmt = "bif"

    def time_read(self, tier):
        BIFReader(string=self.text).read()


class TimeBIFReaderGetters(GettersBench):
    fmt = "bif"
    reader_class = BIFReader

    def time_get_variable_grammar(self, tier):
        self.reader.get_variable_grammar()

    def time_get_probability_grammar(self, tier):
        self.reader.get_probability_grammar()


class TimeBIFWriter(WriterBench):
    def time_str(self, tier):
        str(BIFWriter(self.model))

    def time_write(self, tier):
        BIFWriter(self.model).write(self.path)


# --- XMLBIF ------------------------------------------------------------------------------------------------


class TimeXMLBIFReader(ReaderBench):
    fmt = "xmlbif"

    def time_read(self, tier):
        XMLBIFReader(string=self.text).read()


class TimeXMLBIFReaderGetters(GettersBench):
    fmt = "xmlbif"
    reader_class = XMLBIFReader

    def time_get_variables(self, tier):
        self.reader.get_variables()

    def time_get_states(self, tier):
        self.reader.get_states()

    def time_get_parents(self, tier):
        self.reader.get_parents()

    def time_get_values(self, tier):
        self.reader.get_values()

    def time_get_property(self, tier):
        self.reader.get_property()


class TimeXMLBIFWriter(WriterBench):
    def time_str(self, tier):
        str(XMLBIFWriter(self.model))

    def time_write(self, tier):
        XMLBIFWriter(self.model).write(self.path)


# --- XBN (XMLBeliefNetwork) --------------------------------------------------------------------------------


class TimeXBNReader(ReaderBench):
    fmt = "xbn"

    def time_read(self, tier):
        XBNReader(string=self.text).read()


class TimeXBNReaderGetters(GettersBench):
    fmt = "xbn"
    reader_class = XBNReader

    def time_get_static_properties(self, tier):
        self.reader.get_static_properties()

    def time_get_variables(self, tier):
        self.reader.get_variables()

    def time_get_edges(self, tier):
        self.reader.get_edges()

    def time_get_distributions(self, tier):
        self.reader.get_distributions()


class TimeXBNWriter(WriterBench):
    def time_str(self, tier):
        str(XBNWriter(self.model))

    def time_write(self, tier):
        XBNWriter(self.model).write(self.path)


# --- XDSL --------------------------------------------------------------------------------------------------


class TimeXDSLReader(ReaderBench):
    fmt = "xdsl"

    def time_read(self, tier):
        XDSLReader(string=self.text).read()


class TimeXDSLReaderGetters(GettersBench):
    fmt = "xdsl"
    reader_class = XDSLReader

    def time_get_variables(self, tier):
        self.reader.get_variables()

    def time_get_parents(self, tier):
        self.reader.get_parents()

    def time_get_states(self, tier):
        self.reader.get_states()

    def time_get_values(self, tier):
        self.reader.get_values()


class TimeXDSLWriter(WriterBench):
    def time_str(self, tier):
        str(XDSLWriter(self.model))

    def time_write(self, tier):
        XDSLWriter(self.model).write(self.path)


# --- UAI ---------------------------------------------------------------------------------------------------


class TimeUAIReader(ReaderBench):
    fmt = "uai"

    def time_read(self, tier):
        UAIReader(string=self.text).read()


class TimeUAIReaderGetters(GettersBench):
    fmt = "uai"
    reader_class = UAIReader

    def time_get_grammar(self, tier):
        self.reader.get_grammar()

    def time_get_network_type(self, tier):
        self.reader.get_network_type()

    def time_get_domain(self, tier):
        self.reader.get_domain()

    def time_get_edges(self, tier):
        self.reader.get_edges()

    def time_get_tables(self, tier):
        self.reader.get_tables()


class TimeUAIWriter(WriterBench):
    """``write`` is inherited from BaseWriter (timed on BIFWriter)."""

    def time_str(self, tier):
        str(UAIWriter(self.model))


# --- NET ---------------------------------------------------------------------------------------------------


class TimeNETReader(ReaderBench):
    fmt = "net"

    def time_read(self, tier):
        NETReader(string=self.text).read()


class TimeNETReaderGetters(GettersBench):
    fmt = "net"
    reader_class = NETReader

    def time_get_variable_grammar(self, tier):
        self.reader.get_variable_grammar()

    def time_get_probability_grammar(self, tier):
        self.reader.get_probability_grammar()

    def time_get_network_name(self, tier):
        self.reader.get_network_name()

    def time_get_variables(self, tier):
        self.reader.get_variables()

    def time_get_states(self, tier):
        self.reader.get_states()

    def time_get_property(self, tier):
        self.reader.get_property()

    def time_get_parents(self, tier):
        self.reader.get_parents()

    def time_get_values(self, tier):
        self.reader.get_values()


class TimeNETWriter(WriterBench):
    """``write`` is inherited from BaseWriter (timed on BIFWriter)."""

    def time_str(self, tier):
        str(NETWriter(self.model))


# --- LGBNJSON ----------------------------------------------------------------------------------------------


class TimeLGBNJSONReader(ReaderBench):
    fmt = "json"

    def time_read(self, tier):
        LGBNJSONReader(string=self.text).read()


class TimeLGBNJSONWriter(Bench):
    def prepare(self, tier):
        self.model = _gaussian(tier)

    def time_str(self, tier):
        str(LGBNJSONWriter(self.model))


# --- PomdpX ------------------------------------------------------------------------------------------------


class TimePomdpXReader(ReaderBench):
    fmt = "pomdpx"

    def time_init(self, tier):
        PomdpXReader(string=self.text)


class TimePomdpXReaderGetters(GettersBench):
    fmt = "pomdpx"
    reader_class = PomdpXReader

    def prepare(self, tier):
        super().prepare(tier)
        # the first transition table: the element get_parameter / get_parameter_tbl parse
        self.cond_prob = self.reader.network.find("StateTransitionFunction").find("CondProb")
        self.parameter = self.cond_prob.find("Parameter")

    def time_get_variables(self, tier):
        self.reader.get_variables()

    def time_get_initial_beliefs(self, tier):
        self.reader.get_initial_beliefs()

    def time_get_state_transition_function(self, tier):
        self.reader.get_state_transition_function()

    def time_get_obs_function(self, tier):
        self.reader.get_obs_function()

    def time_get_reward_function(self, tier):
        self.reader.get_reward_function()

    def time_get_parameter(self, tier):
        self.reader.get_parameter(self.cond_prob)

    def time_get_parameter_tbl(self, tier):
        self.reader.get_parameter_tbl(self.parameter)


class TimePomdpXWriter(Bench):
    """Each method builds a writer from the frozen ``model_data`` and runs one section of the document."""

    def prepare(self, tier):
        self.data = pomdpx_model_data(FILE["pomdpx"][tier])  # fresh: the writer is given the dict, not a copy

    def time_init(self, tier):
        PomdpXWriter(model_data=self.data)

    def time_get_variables(self, tier):
        PomdpXWriter(model_data=self.data).get_variables()

    def time_add_initial_belief(self, tier):
        PomdpXWriter(model_data=self.data).add_initial_belief()

    def time_add_state_transition_function(self, tier):
        PomdpXWriter(model_data=self.data).add_state_transition_function()

    def time_add_obs_function(self, tier):
        PomdpXWriter(model_data=self.data).add_obs_function()

    def time_add_reward_function(self, tier):
        PomdpXWriter(model_data=self.data).add_reward_function()
