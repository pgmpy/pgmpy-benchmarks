"""Benchmarks for ``pgmpy.models.JunctionTree``.

Skipped:
    JunctionTree.states: trivial
"""

from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model

MODEL = "alarm"


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


def junction_tree(name):
    return discrete_bn(name).to_junction_tree()


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(MODEL)


class TimeJunctionTree(Bench):
    def setup(self):
        self.model = junction_tree(MODEL)
        self.edge = list(self.model.edges)[-1]
        self.cut = junction_tree(MODEL)
        self.cut.remove_edge(*self.edge)

    def time_add_edge(self):
        self.cut.add_edge(*self.edge)

    def time_check_model(self):
        self.model.check_model()

    def time_copy(self):
        self.model.copy()
