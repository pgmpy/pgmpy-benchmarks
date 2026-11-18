"""Benchmarks for ``pgmpy.readwrite.PomdpX`` on ``benchmarks/data/rocksample.pomdpx``. PomdpX has no ``read`` and no
``str``: the reader benchmark runs the constructor and every getter, the writer benchmark the constructor and every
``add_*``.
"""

from pathlib import Path

from benchmarks._bench import Bench, fresh

from pgmpy.readwrite import PomdpXReader, PomdpXWriter

TEXT = (Path(__file__).parents[1] / "data" / "rocksample.pomdpx").read_text(encoding="utf-8")


def read_pomdpx(text):
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
def model_data():
    return read_pomdpx(TEXT)


class TimePomdpXReader(Bench):
    def setup(self):
        self.text = TEXT

    def time_read(self):
        read_pomdpx(self.text)


class TimePomdpXWriter(Bench):
    def setup(self):
        self.data = model_data()  # fresh: the writer is given the dict, not a copy

    def time_write(self):
        writer = PomdpXWriter(model_data=self.data)
        writer.get_variables()
        writer.add_initial_belief()
        writer.add_state_transition_function()
        writer.add_obs_function()
        writer.add_reward_function()
