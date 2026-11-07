import networkx as nx
import numpy as np
import pandas as pd
import pytest

from benchmarks import _fixtures
from benchmarks._fixtures import (
    DBN_SLICES,
    bfs_prefix,
    dbn,
    dbn_data,
    discrete_bn,
    discrete_data,
    factors,
    gaussian_bn,
    graph,
    joint_chain,
    load_spec,
    markov_chain,
    markov_chain_matrices,
    markov_chain_sample,
    pairwise_markov_network,
    sem_inputs,
    syntax,
    tree_factor_graph,
)
from pgmpy.inference import BeliefPropagation, BeliefPropagationWithMessagePassing, DBNInference, Mplp, VariableElimination
from pgmpy.models import SEM, DiscreteBayesianNetwork, DynamicBayesianNetwork, SEMAlg, SEMGraph
from pgmpy.sampling import BayesianModelSampling


def test_discrete_data_is_deterministic():
    pd.testing.assert_frame_equal(discrete_data("asia", 500), discrete_data("asia", 500))


def test_discrete_data_does_not_use_pgmpy_sampling(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("pgmpy sampling used")

    monkeypatch.setattr(DiscreteBayesianNetwork, "simulate", boom)
    monkeypatch.setattr(BayesianModelSampling, "forward_sample", boom)
    _fixtures._discrete_array.cache_clear()
    assert discrete_data("asia", 100, seed=1).shape == (100, 8)


def test_discrete_sampler_matches_exact_marginals():
    data = discrete_data("asia", 100_000)
    infer = VariableElimination(discrete_bn("asia"))
    for var in data.columns:
        q = infer.query([var], show_progress=False)
        for state in q.state_names[var]:
            assert abs((data[var] == state).mean() - q.get_value(**{var: state})) < 0.01


def test_gaussian_sampler_recovers_parameters():
    spec = load_spec("random_gaussian_8")
    data = _fixtures.gaussian_data("random_gaussian_8", 100_000)
    for node in spec["nodes"]:
        X = np.column_stack([np.ones(len(data))] + [data[p] for p in node["parents"]])
        beta, *_ = np.linalg.lstsq(X, data[node["name"]], rcond=None)
        np.testing.assert_allclose(beta, [node["intercept"], *node["coefficients"]], atol=0.05)


def test_models_are_valid_and_sized():
    alarm = discrete_bn("alarm")
    assert alarm.check_model()
    assert (len(alarm.nodes()), len(alarm.edges())) == (37, 46)
    assert gaussian_bn("random_gaussian_8").check_model()
    assert len(graph("dag_20").nodes()) == 20


def test_builders_return_fresh_objects():
    assert discrete_bn("asia") is not discrete_bn("asia")
    df = discrete_data("asia", 10)
    df.iloc[0, 0] = "MUTATED"
    assert discrete_data("asia", 10).iloc[0, 0] != "MUTATED"


def test_syntax_fixtures_describe_the_dag_fixtures():
    lavaan, dagitty = syntax("syntax_20")
    assert dagitty.count("->") == graph("dag_20").number_of_edges()
    assert "~" in lavaan


def test_joint_chain_is_a_valid_deterministic_imap():
    jpd, model = joint_chain(5)
    assert model.check_model()
    assert jpd.is_imap(model)
    np.testing.assert_array_equal(jpd.values, joint_chain(5)[0].values)
    assert jpd.variables == [f"X{i}" for i in range(5)]


def test_factors_match_the_cpd_families():
    fs = factors("alarm")
    assert [f.variables[0] for f in fs] == [c.variable for c in discrete_bn("alarm").cpds]
    assert factors("alarm")[0] is not fs[0]


def test_dbn_is_valid_and_sized():
    model = dbn("dbn_5")
    assert model.check_model()
    assert len(model.cpds) == 10
    assert len(model.get_slice_nodes()) == 5
    assert model is not dbn("dbn_5")


def test_dbn_data_is_deterministic_and_unrolled():
    data = dbn_data("dbn_5", 500)
    pd.testing.assert_frame_equal(data, dbn_data("dbn_5", 500))
    assert data.shape == (500, 5 * DBN_SLICES)
    assert set(data.columns) == {(f"V{i}", t) for i in range(5) for t in range(DBN_SLICES)}
    assert set(np.unique(data.values)) <= {0, 1}
    data.iloc[0, 0] = 7
    assert dbn_data("dbn_5", 500).iloc[0, 0] != 7


def test_dbn_sampler_is_recovered_by_fit():
    model = DynamicBayesianNetwork()
    reference = dbn("dbn_5")
    model.add_nodes_from(["V0", "V1", "V2", "V3", "V4"])
    model.add_edges_from([(u.to_tuple(), v.to_tuple()) for u, v in reference.edges()])
    model.fit(dbn_data("dbn_5", 200_000))
    # Slice-0 CPDs are fitted from the first two slices together, so only the transition CPDs are compared.
    for cpd in reference.cpds[len(reference.cpds) // 2 :]:
        fitted = model.get_cpds(cpd.variable).reorder_parents(list(cpd.variables[1:]), inplace=False)
        np.testing.assert_allclose(fitted, cpd.get_values(), atol=0.03)


def test_markov_chain_is_deterministic_and_valid():
    matrices = markov_chain_matrices(4, 3)
    assert list(matrices) == ["X0", "X1", "X2", "X3"]
    for matrix in matrices.values():
        np.testing.assert_allclose(matrix.sum(axis=1), 1)
    np.testing.assert_array_equal(matrices["X1"], markov_chain_matrices(4, 3)["X1"])
    chain = markov_chain(4, 3)
    assert chain.sample(size=5, rng=0).shape == (5, 4)
    assert chain is not markov_chain(4, 3)


def test_markov_chain_sample_follows_the_transition_matrices():
    data = markov_chain_sample(2, 3, 20_000)
    pd.testing.assert_frame_equal(data, markov_chain_sample(2, 3, 20_000))
    assert (data.iloc[0] == 0).all()
    matrix = markov_chain_matrices(2, 3)["X0"]
    previous, current = data["X0"].values[:-1], data["X0"].values[1:]
    for state in range(3):
        freq = np.bincount(current[previous == state], minlength=3) / (previous == state).sum()
        np.testing.assert_allclose(freq, matrix[state], atol=0.05)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # pyparsing, used by the lavaan parser
def test_sem_inputs_agree_with_pgmpy_conversions():
    name, n_err_corr = "random_gaussian_8", 1
    sem = sem_inputs(name, n_err_corr)
    graph_model = SEMGraph(ebunch=sem["ebunch"], latents=sem["latents"], err_corr=sem["err_corr"])
    var_names, params, _ = graph_model.to_standard_lisrel()
    assert {k: set(v) for k, v in var_names.items()} == {k: set(v) for k, v in sem["var_names"].items()}
    for key in ("B", "gamma", "wedge_y", "wedge_x"):
        assert params[key].sum() == sem["params"][key].sum()
    plain = {("v" + u.replace("_", ""), "v" + v.replace("_", "")) for u, v in sem["ebunch"]}
    assert set(SEM.from_lavaan(string=sem["lavaan"]).graph.edges) == plain
    alg = SEMAlg(eta=sem["variables"], B=sem["B"], zeta=sem["zeta"], wedge_y=sem["wedge_y"])
    assert set(alg.to_SEMGraph().graph.edges) == set(sem["ebunch"])
    alg.set_params(sem["B_values"], sem["zeta_values"])
    assert alg.generate_samples(n_samples=5).shape == (5, len(sem["observed"]))
    assert sem["ebunch"] is not sem_inputs(name, n_err_corr)["ebunch"]


def test_dbn_fixtures_are_usable_by_dbn_inference():
    # dbn_5 must answer queries (backward inference needs the same interface in both slices).
    inference = DBNInference(dbn("dbn_5"))
    variables = [("V0", 2), ("V3", 2)]
    for result in (inference.forward_inference(variables), inference.query(variables)):
        for var in variables:
            np.testing.assert_allclose(result[var].values.sum(), 1)
    assert inference.backward_inference([("V0", 1), ("V3", 1)])


@pytest.mark.filterwarnings("ignore:At least one state name was not found")  # Mplp's own decoding
def test_pairwise_markov_network_has_node_and_edge_factors_only():
    model = pairwise_markov_network("asia")
    bn = discrete_bn("asia")
    assert {frozenset(e) for e in model.edges()} == {frozenset(e) for e in bn.edges()}
    sizes = sorted(len(f.variables) for f in model.get_factors())
    assert sizes == [1] * len(bn) + [2] * bn.number_of_edges()
    assert model.check_model()
    assert len(Mplp(model).map_query()) == len(bn)
    assert pairwise_markov_network("asia") is not model


def test_tree_factor_graph_is_a_forest_and_answers_queries_exactly():
    fg = tree_factor_graph("asia")
    assert nx.is_forest(nx.Graph(fg.edges()))
    assert len(fg.get_factors()) == len(discrete_bn("asia"))
    exact = BeliefPropagation(fg).query(["dysp", "lung"], evidence={"bronc": "yes"}, joint=False, show_progress=False)
    message_passing = BeliefPropagationWithMessagePassing(fg).query(["dysp", "lung"], evidence={"bronc": 0})
    for var in ("dysp", "lung"):
        np.testing.assert_allclose(message_passing[var].values, exact[var].values)
    assert tree_factor_graph("asia") is not fg


def test_bfs_prefix_of_a_graph_fixture_is_deterministic_sized_and_induced():
    dag = graph("dag_20")
    sub = bfs_prefix(dag.edges(), ["X_10", "X_8"], 12)
    assert len(sub) == 12
    assert {"X_10", "X_8"} <= set(sub.nodes)
    assert set(sub.edges()) == set(bfs_prefix(graph("dag_20").edges(), ["X_10", "X_8"], 12).edges())
    assert set(sub.edges()) == {e for e in dag.edges() if set(e) <= set(sub.nodes)}


def test_perturbed_dag_is_a_valid_acyclic_perturbation_of_the_true_dag():
    true, est = graph("dag_20"), graph("dag_20_perturbed")
    assert nx.is_directed_acyclic_graph(est)
    assert set(est.nodes) == set(true.nodes)
    t, e = set(true.edges), set(est.edges)
    reversed_ = {(u, v) for u, v in t - e if (v, u) in e}
    removed = t - e - reversed_
    assert len(removed) == round(0.1 * len(t))
    assert 0 < len(reversed_) <= round(0.1 * len(t))
    assert e - {(v, u) for u, v in reversed_} <= t  # nothing is added
    assert est is not graph("dag_20_perturbed")


def test_perturbed_dag_spec_is_deterministic():
    from tools.build_fixtures import perturbed_dag_spec

    assert perturbed_dag_spec(load_spec("dag_20")) == load_spec("dag_20_perturbed")


def test_independencies_are_deterministic_valid_and_sized():
    from benchmarks._fixtures import independencies
    from tools.build_fixtures import independencies_spec

    name, n_vars, n_assertions, seed = "indep_small", 7, 5, 0
    assert independencies_spec(n_vars, n_assertions, rng=seed) == load_spec(name)
    ind = independencies(name)
    assert len(ind.get_assertions()) == n_assertions
    assert len(set(ind.get_assertions())) == n_assertions
    assert ind.get_all_variables() <= set(load_spec(name)["nodes"])
    assert all(a.event1 and a.event2 and not (a.event1 & a.event2 or a.event1 & a.event3 or a.event2 & a.event3)
               for a in ind.get_assertions())
    assert independencies(name) is not ind


READWRITE_FILES = [(fmt, "asia") for fmt in ("bif", "xmlbif", "xbn", "xdsl", "uai", "net")]


@pytest.mark.parametrize("fmt, name", READWRITE_FILES)
def test_readwrite_discrete_files_read_back_to_their_model(fmt, name):
    from pgmpy.readwrite import FORMATS

    model = discrete_bn(name)
    read = FORMATS[fmt][0](string=_fixtures.readwrite_text(fmt, name)).read()
    if fmt == "uai":  # UAI has no names: variable i is the i-th node by (cardinality, name), as UAIWriter orders them
        by_card = sorted(model.nodes, key=lambda n: (model.get_cardinality(n), n))
        names_ = {f"var_{i}": n for i, n in enumerate(by_card)}
        assert {(names_[u], names_[v]) for u, v in read.edges} == set(model.edges)
        assert len(read.nodes) == len(model.nodes)
    else:
        assert set(read.nodes) == set(model.nodes)
        assert set(read.edges) == set(model.edges)


def test_readwrite_json_files_read_back_to_their_model():
    from pgmpy.readwrite import LGBNJSONReader

    model = gaussian_bn("random_gaussian_8")
    read = LGBNJSONReader(string=_fixtures.readwrite_text("json", "random_gaussian_8")).read()
    assert set(read.nodes) == set(model.nodes)
    assert set(read.edges) == set(model.edges)


def test_readwrite_pomdpx_files_read_back_and_match_their_model_data():
    from pgmpy.readwrite import PomdpXReader, PomdpXWriter

    name, n_states = "rocksample", 2
    reader = PomdpXReader(string=_fixtures.readwrite_text("pomdpx", name))
    variables = reader.get_variables()
    assert len(variables["StateVar"]) == n_states
    data = _fixtures.pomdpx_model_data(name)
    assert data["variables"] == variables
    assert data["initial_state_belief"] == reader.get_initial_beliefs()
    assert data["state_transition_function"] == reader.get_state_transition_function()
    assert data["obs_function"] == reader.get_obs_function()
    assert data["reward_function"] == reader.get_reward_function()
    declared = {v for kind in variables.values() for var in kind for k, v in var.items() if k.startswith("vname")}
    for table in data["initial_state_belief"] + data["state_transition_function"] + data["obs_function"]:
        assert {table["Var"], *table["Parent"]} - {"null"} <= declared
    assert _fixtures.pomdpx_model_data(name) is not _fixtures.pomdpx_model_data(name)
    assert PomdpXWriter(model_data=_fixtures.pomdpx_model_data(name)).get_variables()


def test_readwrite_generation_is_deterministic():
    from tools.build_fixtures import build_readwrite_files

    first = build_readwrite_files(with_rocksample=False)
    assert first and first == build_readwrite_files(with_rocksample=False)
    assert "rocksample.pomdpx" not in first
