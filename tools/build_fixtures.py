"""Generate the frozen benchmark fixtures in benchmarks/data/*.json.gz.

Run once (needs network for load_model) and commit the output, so base and head
commits are always benchmarked on identical inputs:

    python tools/build_fixtures.py --out benchmarks/data

To add a fixture, write a ``spec_<name>`` function returning a spec dict and
register it in ``SPECS``. Specs are stored with nodes in topological order.

Non-DAG graph derivations (pgmpy conversions at build time, then frozen):
  pdag: ``DAG.to_pdag()`` (the CPDAG).
  ug:   ``DAG.moralize()`` (moral graph).
  admg: ``DAG.to_admg()`` (latent projection) of a seeded random DAG with
        10% more nodes, a seeded subset of which is marked latent, so exactly
        20/500 observed nodes remain.
  mag:  the admg above, with each bidirected edge between a node and one of its
        ancestors dropped (keeps the graph ancestral). Maximality is checked
        (and printed) only for the small graph; it is too slow at 500 nodes.
"""

import argparse
import ast
import gzip
import json
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np

from pgmpy.base import DAG
from pgmpy.example_models import load_model
from pgmpy.models import DiscreteBayesianNetwork, LinearGaussianBayesianNetwork

LATENT_FRACTION = 0.1


def _topo(model):
    return list(nx.topological_sort(model))


def discrete_spec(model):
    nodes = []
    for name in _topo(model):
        cpd = model.get_cpds(name)
        evidence = list(cpd.variables[1:])
        nodes.append(
            {
                "name": name,
                "states": list(cpd.state_names[name]),
                "parents": evidence,
                "values": cpd.get_values().tolist(),
            }
        )
    return {"kind": "discrete", "nodes": nodes}


def gaussian_spec(model):
    nodes = []
    for name in _topo(model):
        cpd = model.get_cpds(name)
        beta = [float(b) for b in cpd.beta]
        nodes.append(
            {
                "name": name,
                "parents": list(cpd.evidence),
                "intercept": beta[0],
                "coefficients": beta[1:],
                "std": float(cpd.std),
            }
        )
    return {"kind": "gaussian", "nodes": nodes}


def graph_spec(kind, nodes, directed=(), undirected=(), bidirected=(), latents=()):
    return {
        "kind": kind,
        "nodes": list(nodes),
        "directed": [list(e) for e in directed],
        "undirected": [list(e) for e in undirected],
        "bidirected": [list(e) for e in bidirected],
        "latents": list(latents),
    }


def dag_spec(dag):
    return graph_spec("dag", dag.nodes(), directed=dag.edges())


def pdag_spec(dag):
    pdag = dag.to_pdag()
    return graph_spec(
        "pdag", pdag.nodes(), directed=sorted(pdag.directed_edges), undirected=sorted(pdag.undirected_edges)
    )


def ug_spec(dag):
    ug = dag.moralize()
    return graph_spec("ug", ug.nodes(), undirected=ug.edges())


def _total_nodes(n):
    """Size of the bigger DAG to project, so that exactly ``n`` observed nodes remain."""
    return n + max(1, round(n * LATENT_FRACTION))


def _admg(dag, n_latent, rng=0):
    rng = np.random.default_rng(seed)
    nodes = list(dag.nodes())
    latents = set(rng.choice(nodes, size=n_latent, replace=False).tolist())
    marked = DAG(dag.edges())
    marked.add_nodes_from(nodes)  # isolated nodes are not in edges()
    marked.latents = latents
    return marked.to_admg()


def admg_spec(dag, n_latent):
    admg = _admg(dag, n_latent)
    edges = admg.get_edges(data=True)
    return graph_spec(
        "admg",
        admg.nodes(),
        directed=[(u, v) for u, v, t in edges if t == "->"],
        bidirected=[(u, v) for u, v, t in edges if t == "<>"],
    )


def mag_spec(dag, n_latent):
    from pgmpy.base import MAG

    admg = _admg(dag, n_latent)
    edges = admg.get_edges(data=True)
    directed = [(u, v) for u, v, t in edges if t == "->"]
    ancestors = {n: admg.get_ancestors(n) for n in admg.nodes()}
    bidirected = [(u, v) for u, v, t in edges if t == "<>" and u not in ancestors[v] and v not in ancestors[u]]
    spec = graph_spec("mag", admg.nodes(), directed=directed, bidirected=bidirected)
    mag = MAG(edge_list=[(u, v, "->") for u, v in directed] + [(u, v, "<>") for u, v in bidirected])
    if len(spec["nodes"]) <= 50:  # all-pairs inducing-path check is far too slow at 500
        print(f"  mag n={len(spec['nodes'])} is_maximal={mag.is_maximal()}")
    return spec


def junction_tree_spec(model):
    # pgmpy builds a junction tree only for connected models (andes has 4 components): keep the largest.
    keep = max(nx.weakly_connected_components(model), key=len)
    if len(keep) < len(model.nodes()):
        print(f"  jt: using largest component, {len(keep)} of {len(model.nodes())} nodes")
        sub = DiscreteBayesianNetwork([(u, v) for u, v in model.edges() if u in keep])
        sub.add_cpds(*[model.get_cpds(n) for n in keep])
        model = sub
    jt = model.to_junction_tree()
    cliques = [list(c) for c in jt.nodes()]
    index = {tuple(c): i for i, c in enumerate(jt.nodes())}
    return {
        "kind": "junction_tree",
        "cliques": cliques,
        "edges": [[index[u], index[v]] for u, v in jt.edges()],
        "factors": [
            {
                "scope": list(f.scope()),
                "cardinality": [int(c) for c in f.cardinality],
                "values": f.values.ravel().tolist(),
                "state_names": {k: list(v) for k, v in f.state_names.items()},
            }
            for f in jt.factors
        ],
    }


def syntax_spec(dag):
    """The DAG written as lavaan and dagitty text, frozen so parser benchmarks have fixed input."""
    return {"kind": "syntax", "lavaan": dag.to_lavaan(), "dagitty": dag.to_dagitty()}


def dbn_spec(n_vars, intra_prob, inter_prob, interface_every=2, n_states=2, rng=0, band=None):
    """A 2-TBN of ``n_vars`` variables ``V0..`` with numpy CPDs (never pgmpy's ``get_random``).

    Built so that ``DBNInference`` works on it:

    * Every ``V(j-1) -> Vj`` intra-slice edge is present (a DBN only learns of both slices of a variable
      through its edges, and a junction tree needs a connected slice), plus ``Vi -> Vj`` (i < j) with
      ``intra_prob``. With ``band`` set, only variables at most ``band`` indices apart are linked, which
      keeps the junction tree small.
    * Only every ``interface_every``-th variable (the interface) has inter-slice edges: ``Vi(0) -> Vi(1)``
      always, and ``Vi(0) -> Vk(1)`` between interface variables with ``inter_prob``. The sources and the
      targets of the inter-slice edges are then the same variables, which ``backward_inference`` needs.
      ``DBNInference`` joins the interface nodes of a slice into one clique, so the interface must be small.
    * Variables outside the interface have the same parents in both slices, so
      ``initialize_initial_state`` can derive their slice-1 CPDs from the slice-0 ones.

    Nodes are listed in topological order, slice 0 then slice 1. A parent is ``[name, slice]``; ``values``
    has one column per parent assignment (first parent varies slowest).
    """
    rng = np.random.default_rng(seed)

    def near(i, j):
        return band is None or abs(i - j) <= band

    names = [f"V{i}" for i in range(n_vars)]
    intra = {
        j: [i for i in range(j) if (near(i, j) and rng.random() < intra_prob) or i == j - 1] for j in range(n_vars)
    }
    interface = range(0, n_vars, interface_every)
    inter = {
        j: [i for i in interface if i == j or rng.random() < inter_prob] if j in interface else []
        for j in range(n_vars)
    }
    nodes = []
    for t in (0, 1):
        for j in range(n_vars):
            parents = [[names[i], t] for i in intra[j]]
            if t == 1:
                parents += [[names[i], 0] for i in inter[j]]
            values = rng.dirichlet(np.ones(n_states), size=n_states ** len(parents)).T
            nodes.append({"name": names[j], "t": t, "parents": parents, "values": values.tolist()})
    return {"kind": "dbn", "n_states": n_states, "variables": names, "nodes": nodes}


def perturbed_dag_spec(dag_spec_, fraction=0.1, rng=0):
    """A frozen "estimated" DAG: ``fraction`` of the edges reversed and ``fraction`` of the others removed.

    Edges are taken in sorted order and chosen with ``numpy.random.default_rng(rng)``. Reversals are applied
    in sorted order; a reversal ``u -> v`` that would close a cycle (another directed path ``u ~> v`` exists)
    is skipped and the edge stays as it was. Removals are drawn from the edges that were not selected for
    reversal, so the two sets never overlap.
    """
    edges = sorted(tuple(e) for e in dag_spec_["directed"])
    n = round(fraction * len(edges))
    rng = np.random.default_rng(rng)
    picked = rng.permutation(len(edges))[: 2 * n]
    reverse = sorted(edges[i] for i in picked[:n])
    remove = {edges[i] for i in picked[n:]}
    graph = nx.DiGraph(e for e in edges if e not in remove)
    graph.add_nodes_from(dag_spec_["nodes"])
    skipped = 0
    for u, v in reverse:
        graph.remove_edge(u, v)
        if nx.has_path(graph, u, v):
            graph.add_edge(u, v)
            skipped += 1
        else:
            graph.add_edge(v, u)
    print(f"perturbed dag: {len(edges)} edges, {n} reversal picks ({skipped} skipped: cycle), {n} removed")
    return graph_spec("dag", dag_spec_["nodes"], directed=sorted(graph.edges()))


def independencies_spec(n_vars, n_assertions, rng=0):
    """Random conditional independence assertions ``[X, Y, Z]`` over ``V0..V<n_vars-1>``.

    ``X`` has 1-2 variables, ``Y`` 1-3 and ``Z`` 0-2, drawn disjoint with ``default_rng(rng)``; duplicate
    assertions are redrawn. The sizes are tuned so that ``Independencies.closure`` takes ~0.05 s (small) /
    ~1.6 s (large).
    """
    rng = np.random.default_rng(rng)
    names = [f"V{i}" for i in range(n_vars)]
    assertions = []
    while len(assertions) < n_assertions:
        nx_, ny, nz = (int(k) for k in (rng.integers(1, 3), rng.integers(1, 4), rng.integers(0, 3)))
        pick = [str(v) for v in rng.choice(names, size=nx_ + ny + nz, replace=False)]
        assertion = [sorted(pick[:nx_]), sorted(pick[nx_ : nx_ + ny]), sorted(pick[nx_ + ny :])]
        if assertion not in assertions:
            assertions.append(assertion)
    return {"kind": "independencies", "nodes": names, "assertions": assertions}


def build_derived_specs(out):
    """Specs derived from committed ones, needing no network: the perturbed DAGs and the independencies."""
    specs = {}
    for n in (20, 500):
        with gzip.open(out / f"dag_{n}.json.gz", "rt") as f:
            specs[f"dag_{n}_perturbed"] = perturbed_dag_spec(json.load(f))
    specs["indep_small"] = independencies_spec(7, 5, rng=0)
    specs["indep_large"] = independencies_spec(12, 12, rng=1)
    return specs


def build_dbn_specs():
    # DBNInference multiplies a whole slice into one table, so dbn_50 (2 ** 50 cells) can be built but not
    # queried, and dbn_20 is the largest slice whose forward / backward inference takes about 0.3 s.
    return {
        "dbn_5": dbn_spec(5, 0.4, 0.3, interface_every=2),
        "dbn_20": dbn_spec(20, 0.2, 0.1, interface_every=7, band=2),
        "dbn_50": dbn_spec(50, 0.2, 0.1, interface_every=10, band=2),
    }


def build_specs():
    specs = {}
    discrete = {}
    for name in ["asia", "alarm", "andes", "munin", "pathfinder"]:
        discrete[name] = load_model(f"bnlearn/{name}")
        specs[name] = discrete_spec(discrete[name])
    specs["random_discrete_200"] = discrete_spec(
        DiscreteBayesianNetwork.get_random(n_nodes=200, edge_prob=0.02, n_states=3, rng=0)
    )

    specs["random_gaussian_8"] = gaussian_spec(
        LinearGaussianBayesianNetwork.get_random(n_nodes=8, edge_prob=0.3, rng=0)
    )
    for name in ["ecoli70", "arth150"]:
        specs[name] = gaussian_spec(load_model(f"bnlearn/{name}"))

    for n, p in [(20, 0.15), (500, 0.005)]:
        dag = DAG.get_random(n_nodes=n, edge_prob=p, rng=0)
        print(f"graphs n={n}")
        specs[f"dag_{n}"] = dag_spec(dag)
        specs[f"pdag_{n}"] = pdag_spec(dag)
        # Project a larger DAG so that exactly n observed nodes remain.
        big = DAG.get_random(n_nodes=_total_nodes(n), edge_prob=p, rng=0)
        n_latent = _total_nodes(n) - n
        specs[f"admg_{n}"] = admg_spec(big, n_latent)
        specs[f"mag_{n}"] = mag_spec(big, n_latent)
        specs[f"ug_{n}"] = ug_spec(dag)
        specs[f"syntax_{n}"] = syntax_spec(dag)

    for name in ["alarm", "andes"]:
        specs[f"jt_{name}"] = junction_tree_spec(discrete[name])

    specs.update(build_dbn_specs())
    return specs


# --- readwrite files (benchmarks/data/readwrite/<name>.<format>.gz) --------------------------------------------------
#
# Frozen inputs for benchmarks/readwrite.py, written by the pgmpy writers (fine here: this is the one-off generator)
# and gzip-compressed with mtime=0 so that regenerating them is byte-identical.
#   discrete formats: asia (fast) and andes (slow) in bif / xmlbif / xbn / xdsl. The UAI and NET readers are too slow
#     on andes (18 s and 11 s, quadratic pyparsing scans), so their slow file is alarm (0.9 s and 0.3 s).
#     UAI is written with round_values=4: UAIReader cannot parse scientific notation ("1e-05").
#   json (LGBNJSON): random_gaussian_8 (fast) and arth150 (slow).
#   pomdpx: ``rocksample``, the document of pgmpy's own test suite, and ``state50``, a synthetic one with 50 state
#     variables. ``<name>.model.json`` holds the dict PomdpXWriter takes as ``model_data`` (PomdpX has no model class).

READWRITE_DISCRETE = {
    "bif": ("asia", "andes"),
    "xmlbif": ("asia", "andes"),
    "xbn": ("asia", "andes"),
    "xdsl": ("asia", "andes"),
    "uai": ("asia", "alarm"),
    "net": ("asia", "alarm"),
}
POMDPX_TEST_FILE = "pgmpy/tests/test_readwrite/test_PomdpX.py"  # relative to the pgmpy checkout; its ``string`` fixture


def pomdpx_rocksample(pgmpy_dir):
    """The RockSample document of pgmpy's test suite: the return value of the ``string`` fixture."""
    tree = ast.parse((Path(pgmpy_dir) / POMDPX_TEST_FILE).read_text(encoding="utf-8"))
    (fixture,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "string"]
    return ast.literal_eval(fixture.body[-1].value)


def _probs(rng, n):
    p = rng.random(n) + 0.05
    return " ".join(f"{v:.3f}" for v in p / p.sum())


def pomdpx_state50(n_states=50, n_values=3, n_actions=4, n_obs=5, rng=0):
    """A synthetic PomdpX document with ``n_states`` state variables of ``n_values`` values each.

    One action variable, ``n_obs`` observation variables and one reward. Every state variable's next value depends
    on the action, its own previous value and one other (seeded) previous state variable; each observation on the
    action and two (seeded) current state variables; the reward on the action and two previous states. All
    conditional tables are full ``TBL`` ones with random normalised rows.
    """
    rng = np.random.default_rng(rng)
    actions = [f"a{k}" for k in range(n_actions)]
    values = [f"s{k}" for k in range(n_values)]
    obs_values = [f"o{k}" for k in range(n_values)]

    def entries(n_parents, tag, table):
        out = []
        for combo in np.ndindex(*([n_actions] + [n_values] * n_parents)):
            instance = " ".join([actions[combo[0]]] + [values[v] for v in combo[1:]])
            out.append(f"<Entry><Instance>{instance}</Instance><{tag}>{table()}</{tag}></Entry>")
        return "".join(out)

    def cond(var, parents, tag="CondProb", table_tag="ProbTable", table=lambda: _probs(rng, n_values)):
        return (
            f"<{tag}><Var>{var}</Var><Parent>{' '.join(parents)}</Parent><Parameter type=\"TBL\">"
            f"{entries(len(parents) - 1, table_tag, table)}</Parameter></{tag}>"
        )

    state_vars = "".join(
        f'<StateVar vnamePrev="s{i}_0" vnameCurr="s{i}_1"' + (' fullyObs="true"' if i % 10 == 0 else "")
        + f"><NumValues>{n_values}</NumValues></StateVar>"
        for i in range(n_states)
    )
    obs_vars = "".join(f'<ObsVar vname="o{k}"><ValueEnum>{" ".join(obs_values)}</ValueEnum></ObsVar>' for k in range(n_obs))
    belief = "".join(
        f'<CondProb><Var>s{i}_0</Var><Parent>null</Parent><Parameter type="TBL"><Entry><Instance>-</Instance>'
        f"<ProbTable>{_probs(rng, n_values)}</ProbTable></Entry></Parameter></CondProb>"
        for i in range(n_states)
    )
    transition = "".join(
        cond(f"s{i}_1", ["action_0", f"s{i}_0", f"s{rng.choice([j for j in range(n_states) if j != i])}_0"])
        for i in range(n_states)
    )
    observation = "".join(
        cond(f"o{k}", ["action_0", *(f"s{j}_1" for j in rng.choice(n_states, size=2, replace=False))])
        for k in range(n_obs)
    )
    reward = cond(
        "reward_0",
        ["action_0", *(f"s{j}_0" for j in rng.choice(n_states, size=2, replace=False))],
        tag="Func",
        table_tag="ValueTable",
        table=lambda: str(int(rng.integers(-10, 11))),
    )
    return (
        '<pomdpx version="1.0" id="state50">'
        f"<Description>Synthetic problem with {n_states} state variables.</Description><Discount>0.95</Discount>"
        f"<Variable>{state_vars}{obs_vars}"
        f'<ActionVar vname="action_0"><ValueEnum>{" ".join(actions)}</ValueEnum></ActionVar>'
        '<RewardVar vname="reward_0"/></Variable>'
        f"<InitialStateBelief>{belief}</InitialStateBelief>"
        f"<StateTransitionFunction>{transition}</StateTransitionFunction>"
        f"<ObsFunction>{observation}</ObsFunction>"
        f"<RewardFunction>{reward}</RewardFunction></pomdpx>"
    )


def pomdpx_model_data(string):
    """The dict PomdpXWriter takes as ``model_data``, read from a document by PomdpXReader (generator only)."""
    from pgmpy.readwrite import PomdpXReader

    reader = PomdpXReader(string=string)
    return {
        "description": reader.get_description(),
        "discount": str(reader.get_discount()),
        "variables": reader.get_variables(),
        "initial_state_belief": reader.get_initial_beliefs(),
        "state_transition_function": reader.get_state_transition_function(),
        "obs_function": reader.get_obs_function(),
        "reward_function": reader.get_reward_function(),
    }


REPO_ROOT = Path(__file__).resolve().parent.parent


def build_readwrite_files(pgmpy_dir=None, with_rocksample=True):
    """{file name: text} for benchmarks/data/readwrite, from the committed specs (no network).

    ``with_rocksample=False`` leaves out the document copied from pgmpy's test suite (and so needs no checkout).
    """
    sys.path.insert(0, str(REPO_ROOT))
    from benchmarks._fixtures import discrete_bn, gaussian_bn
    from pgmpy.readwrite import (
        BIFWriter,
        LGBNJSONWriter,
        NETWriter,
        UAIWriter,
        XBNWriter,
        XDSLWriter,
        XMLBIFWriter,
    )

    writers = {
        "bif": BIFWriter,
        "xmlbif": XMLBIFWriter,
        "xbn": XBNWriter,
        "xdsl": XDSLWriter,
        "uai": lambda model: UAIWriter(model, round_values=4),
        "net": NETWriter,
    }
    files = {}
    for fmt, names in READWRITE_DISCRETE.items():
        for name in names:
            random.seed(0)  # XDSLWriter draws node positions from the global ``random`` module
            files[f"{name}.{fmt}"] = str(writers[fmt](discrete_bn(name)))
    for name in ("random_gaussian_8", "arth150"):
        files[f"{name}.json"] = str(LGBNJSONWriter(gaussian_bn(name)))
    documents = {"state50": pomdpx_state50()}
    if with_rocksample:
        documents["rocksample"] = pomdpx_rocksample(pgmpy_dir or REPO_ROOT.parent / "pgmpy")
    for name, text in documents.items():
        files[f"{name}.pomdpx"] = text
        files[f"{name}.model.json"] = json.dumps(pomdpx_model_data(text), separators=(",", ":"))
    return files


def write_readwrite(out, pgmpy_dir):
    out = out / "readwrite"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in build_readwrite_files(pgmpy_dir).items():
        (out / f"{name}.gz").write_bytes(gzip.compress(text.encode("utf-8"), compresslevel=9, mtime=0))
        print("wrote readwrite", name)


def write(out, specs):
    for name, spec in specs.items():
        with gzip.open(out / f"{name}.json.gz", "wt") as f:
            json.dump(spec, f, separators=(",", ":"))
        print("wrote", name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="benchmarks/data")
    parser.add_argument("--only-dbn", action="store_true", help="write only the dbn_* specs (no network needed)")
    parser.add_argument(
        "--only-derived",
        action="store_true",
        help="write only the *_perturbed and indep_* specs (no network; needs dag_20/dag_500 in --out)",
    )
    parser.add_argument(
        "--only-readwrite",
        action="store_true",
        help="write only the readwrite/* files (no network; needs the discrete / gaussian specs in --out)",
    )
    parser.add_argument("--pgmpy", default=str(REPO_ROOT.parent / "pgmpy"), help="pgmpy checkout holding the PomdpX test document")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.only_readwrite:
        write_readwrite(out, args.pgmpy)
        return
    if args.only_dbn:
        specs = build_dbn_specs()
    elif args.only_derived:
        specs = build_derived_specs(out)
    else:
        write(out, build_specs())
        specs = build_derived_specs(out)
        write(out, specs)
        write_readwrite(out, args.pgmpy)
        return
    write(out, specs)


if __name__ == "__main__":
    main()
