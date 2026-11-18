"""Benchmarks for ``pgmpy.models.NaiveBayes``."""

import networkx as nx
from benchmarks._bench import Bench, fresh

from pgmpy.example_models import load_model
from pgmpy.models import NaiveBayes

ROWS = 1_000
DISCRETE = "asia"


def topological(model):
    return list(nx.topological_sort(model))


@fresh
def discrete_bn(name):
    return load_model(f"bnlearn/{name}")


@fresh
def discrete_data(name, n_rows, seed=0):
    model = discrete_bn(name)
    return model.simulate(n_rows, rng=seed, show_progress=False)[topological(model)]


# Load at import: asv's forkserver imports the benchmarks once and forks each benchmark process from it.
discrete_bn(DISCRETE)


def class_and_features(name):
    columns = topological(discrete_bn(name))
    return columns[0], columns[1:]


class TimeNaiveBayes(Bench):
    def setup(self):
        self.cls, self.features = class_and_features(DISCRETE)
        self.model = NaiveBayes(self.features, self.cls)
        self.almost = NaiveBayes(self.features[:-1], self.cls)
        self.empty = NaiveBayes()
        self.edges = [(self.cls, f) for f in self.features]

    def time_active_trail_nodes(self):
        self.model.active_trail_nodes(self.features[0], observed=self.features[1:3])

    def time_add_edge(self):
        self.almost.add_edge(self.cls, self.features[-1])

    def time_add_edges_from(self):
        self.empty.add_edges_from(self.edges)

    def time_local_independencies(self):
        self.model.local_independencies(self.features)


class TimeNaiveBayesFit(Bench):
    def setup(self):
        name = DISCRETE
        cls, features = class_and_features(name)
        self.model = NaiveBayes(features, cls)
        self.data = discrete_data(name, ROWS)

    def time_fit(self):
        self.model.fit(self.data)
